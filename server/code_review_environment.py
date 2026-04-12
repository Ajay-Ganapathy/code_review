# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Review Environment Implementation.

Supports three grader difficulty levels: "easy", "medium", "hard".
Pass `grader_level` to the constructor to select the desired tier.
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
        CodeReviewStepResponse,
    )
except ImportError:
    from models import (
        CodeReviewAction,
        CodeReviewObservation,
        CodeReviewReward,
        CodeReviewPullRequest,
        CodeReviewStepResponse,
    )

import json
from pathlib import Path

try:
    from .graders import get_grader
except ImportError:
    from graders import get_grader

dataset_path = Path(__file__).parent.parent / "dataset" / "dataset.json"


class CodeReviewEnvironment(Environment):
    """
    Code Review environment with configurable grading difficulty.

    Args:
        grader_level: Grading difficulty — one of "easy", "medium", "hard".
                      Defaults to "medium".

    Example:
        >>> env = CodeReviewEnvironment(grader_level="hard")
        >>> obs = env.reset()
        >>> obs = env.step(CodeReviewAction(action_type="final_decision", decision="approve"))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, grader_level: str = "medium"):
        """Initialise the environment with the chosen grader tier."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.max_steps = 5
        self.task_index = 0

        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.reset()

    def reset(self) -> CodeReviewObservation:
        """Reset the environment and advance to the next task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.task_index += 1

        self.sample = self.dataset[self.task_index % len(self.dataset)]

        self.pr        = CodeReviewPullRequest(**self.sample["pr"])
        self.gt        = self.sample["ground_truth"]
        self.task_type = self.sample.get("task_type", "unknown")
        grader_level = self.task_type if self.task_type in ("easy", "medium", "hard") else "medium"
        self.grader = get_grader(grader_level)
        self.grader_level = grader_level

        self.history            = []
        self.step_count         = 0
        self.done               = False
        self.issues_identified  = []
        self.fix_attempted      = False

        return CodeReviewObservation(
            pr=self.pr,
            previous_comments=self.history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=0.0,
            done=False,
        )

    def step(self, action: CodeReviewAction) -> CodeReviewStepResponse:  # type: ignore[override]
        """Execute one step: grade the action and return an observation + reward."""
        self._state.step_count += 1

        # ------------------------------------------------------------------
        # Normalise action into a CodeReviewAction object
        # ------------------------------------------------------------------
        try:
            if isinstance(action, dict):
                action = CodeReviewAction(**action)
            elif isinstance(action, (list, tuple)):
                action = CodeReviewAction(
                    action_type=action[0],
                    comment=action[1]      if len(action) > 1 else None,
                    suggested_code=action[2] if len(action) > 2 else None,
                    decision=action[3]     if len(action) > 3 else None,
                )
            elif isinstance(action, CodeReviewAction):
                pass
            else:
                raise ValueError(f"Unsupported action type: {type(action)}")
        except Exception as e:
            print(f"Error processing action: {e}")
            return self._invalid_step()

        # ------------------------------------------------------------------
        # Update state
        # ------------------------------------------------------------------
        self.step_count += 1
        self.history.append(action)

        if action.action_type == "comment" and action.comment:
            self.issues_identified.append(action.comment)

        if action.action_type == "suggest_fix":
            self.fix_attempted = True

        # ------------------------------------------------------------------
        # Score via the active grader
        # ------------------------------------------------------------------
        score = self.grader.grade_action(action, self.gt)
        bonus = self.grader.compute_step_bonus(action, self.step_count, self.history)

        score = max(0.01, min(score + bonus, 0.99))

        done = (
            action.action_type == "final_decision"
            or self.step_count >= self.max_steps
        )

        if done:
            score = self.grader.compute_done_score(self.history, self.gt)

        # ------------------------------------------------------------------
        # Build response
        # ------------------------------------------------------------------
        obs = CodeReviewObservation(
            pr=self.pr,
            previous_comments=[a.comment for a in self.history if a.comment],
            step_count=self.step_count,
            max_steps=self.max_steps,
        )

        rew = CodeReviewReward(score=score, feedback="graded")
        # print(f"[{self.grader_level.upper()}] Step {self.step_count} — Score: {rew.score:.4f}")

        return CodeReviewStepResponse(
            observation=obs,
            reward=rew.score,
            done=done,
            info={
                "grader_level":      self.grader_level,
                "task_type":         self.task_type,
                "issues_identified": len(self.issues_identified),
                "fix_attempted":     self.fix_attempted,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def _invalid_step(self) -> CodeReviewStepResponse:
        rew = CodeReviewReward(score=0.0, feedback="invalid action")
        obs = CodeReviewObservation(
            pr=self.pr,
            previous_comments=[a.comment for a in self.history if a.comment],
            step_count=self.step_count,
            max_steps=self.max_steps,
        )
        return CodeReviewStepResponse(
            observation=obs,
            reward=rew,
            done=True,
            info={"error": "invalid_action"},
        )