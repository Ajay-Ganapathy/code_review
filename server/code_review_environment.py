# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Review Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CodeReviewAction,
        CodeReviewObservation,
        CodeReviewReward,
        CodeReviewPullRequest,
    )
except ImportError:
    from models import (
        CodeReviewAction,
        CodeReviewObservation,
        CodeReviewReward,
        CodeReviewPullRequest,
    )

import json
from pathlib import Path

dataset_path = Path(__file__).parent.parent / "dataset" / "dataset.json"

class CodeReviewEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = CodeReviewEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Code Review environment ready!"
        >>>
        >>> obs = env.step(CodeReviewAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the code_review environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.max_steps = 5
        self.task_index = 0
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        self.reset()

    def reset(self) -> CodeReviewObservation:
        """
        Reset the environment.

        Returns:
            CodeReviewObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        self.sample = self.dataset[self.task_index % len(self.dataset)]

        self.pr = CodeReviewPullRequest(**self.sample["pr"])
        self.gt = self.sample["ground_truth"]
        self.task_type = self.sample.get("task_type", "unknown")

        self.history = []
        self.step_count = 0
        self.done = False

        # State evolution variables
        self.issues_identified = []
        self.fix_attempted = False

        return CodeReviewObservation(
            echoed_message="Code Review environment ready!",
            pr=self.pr,
            previous_comments=self.history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=0.0,
            done=False,
        )

    def step(self, action: CodeReviewAction) -> CodeReviewObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: CodeReviewAction containing the message to echo

        Returns:
            CodeReviewObservation with the echoed message and its length
        """
        self._state.step_count += 1

        try:
            action = CodeReviewAction(**action)
        except Exception as e:
            print(f"Error occurred while processing action: {e}")
            return self._invalid_step()

        self.step_count += 1
        self.history.append(action)

        if action.action_type == "comment" and action.comment:
            self.issues_identified.append(action.comment)

        if action.action_type == "suggest_fix":
            self.fix_attempted = True

        score = self.grade_action(action, self.gt)
        print(f"Step {self.step_count} - Score: {score:.4f}")

        bonus = 0.0

        # Encourage meaningful comments
        if action.comment and len(action.comment) > 30:
            bonus += 0.1

        # Encourage early correct decisions
        if action.action_type == "final_decision" and self.step_count <= 2:
            bonus += 0.1

        # Penalize useless steps
        if not action.comment and action.action_type != "final_decision":
            bonus -= 0.1

        # Penalize long trajectories
        if self.step_count > 3:
            bonus -= 0.05

        score += bonus
        score = max(0.0, min(score, 1.0))

        done = (
            action.action_type == "final_decision" or self.step_count >= self.max_steps
        )

        if done:
            score = max([self.grade_action(a, self.gt) for a in self.history] or [0.0])

        return (
            CodeReviewObservation(
                echoed_message="Code Review Step Completed",
                pr=self.pr,
                previous_comments=[a.comment for a in self.history if a.comment],
                step_count=self.step_count,
                max_steps=self.max_steps,
            ),
            CodeReviewReward(score=score, feedback="graded"),
            done,
            {
                "task_type": self.task_type,
                "issues_identified": len(self.issues_identified),
                "fix_attempted": self.fix_attempted,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _invalid_step(self):
        return (
            CodeReviewObservation(
                echoed_message="Invalid action format. Please send a valid CodeReviewAction.",
                pr=self.pr,
                previous_comments=[a.comment for a in self.history if a.comment],
                step_count=self.step_count,
                max_steps=self.max_steps,
            ),
            CodeReviewReward(score=0.0, feedback="invalid action"),
            True,  # Terminate step on invalid action
            {"error": "invalid_action"},
        )

    def grade_action(self, action, ground_truth):
        score = 0.0

        print("Action === ", action)
        print("Ground truth === ", ground_truth)

        # ------------------------------
        # ISSUE DETECTION (40%)
        # ------------------------------
        issue_score = self.score_issues(action.comment, ground_truth)
        score += 0.4 * issue_score
        print("After Issue Score == ", issue_score)

        # ------------------------------
        # FIX QUALITY (30%)
        # ------------------------------
        fix_score = self.score_fix(action.suggested_code, ground_truth)
        score += 0.3 * fix_score

        print("After Fix Score == ", fix_score)

        # ------------------------------
        # DECISION (30%)
        # ------------------------------
        decision_score = self.score_decision(action, ground_truth)
        score += 0.3 * decision_score

        print("After Decision Score == ", decision_score)

        # ------------------------------
        # CLAMP SCORE
        # ------------------------------
        score = max(0.0, min(score, 1.0))

        return score

    def normalize(self, text):
        return (text or "").lower().strip()

    # ==============================
    # ISSUE MATCH (PARTIAL CREDIT)
    # ==============================
    def score_issues(self, comment, ground_truth):
        issues = ground_truth.get("issues", [])
        if not comment or not issues:
            return 0.0

        comment = self.normalize(comment)

        matches = sum(1 for issue in issues if self.normalize(issue) in comment)

        return matches / len(issues)

    # ==============================
    # FIX MATCH (FUZZY)
    # ==============================
    def score_fix(self, suggested_code, ground_truth):
        if not suggested_code:
            return 0.0

        expected_fix = self.normalize(ground_truth.get("fix", ""))
        suggested_code = self.normalize(suggested_code)

        # direct match
        if expected_fix in suggested_code:
            return 1.0

        # partial keyword match
        keywords = expected_fix.split()
        if not keywords:
            return 0.0

        matches = sum(1 for word in keywords if word in suggested_code)

        return matches / len(keywords)

    # ==============================
    # DECISION MATCH
    # ==============================
    def score_decision(self, action, ground_truth):
        expected = ground_truth.get("decision")

        # Not a decision step → no contribution
        if action.action_type != "final_decision":
            return 0.0

        # Missing decision → small penalty
        if not action.decision:
            return 0.0

        # Correct decision
        if action.decision == expected:
            return 1.0

        # Wrong decision → partial penalty (not negative)
        return 0.2
