import json
from typing import Dict, Any

from models import Observation, Action, Reward, PullRequest
from grader import grade_action


class CodeReviewEnv:
    """
    Production-ready OpenEnv environment for code review tasks.
    Deterministic, validated, and trajectory-aware.
    """

    def __init__(self):
        with open("data/dataset.json") as f:
            self.dataset = json.load(f)

        self.max_steps = 5
        self.task_index = 0  # deterministic
        self.reset()

    # ==============================
    # RESET
    # ==============================
    def reset(self):
        # Deterministic dataset selection
        self.sample = self.dataset[self.task_index % len(self.dataset)]

        self.pr = PullRequest(**self.sample["pr"])
        self.gt = self.sample["ground_truth"]
        self.task_type = self.sample.get("task_type", "unknown")

        self.history = []
        self.step_count = 0
        self.done = False

        # State evolution variables
        self.issues_identified = []
        self.fix_attempted = False

        return Observation(
            pr=self.pr,
            previous_comments=[],
            step_count=0,
            max_steps=self.max_steps,
        )

    # ==============================
    # VALIDATION
    # ==============================
    def validate_action(self, action: Action) -> bool:
        valid_types = ["comment", "suggest_fix", "final_decision"]
        return action.action_type in valid_types

    def _invalid_step(self):
        return (
            Observation(
                pr=self.pr,
                previous_comments=[a.comment for a in self.history if a.comment],
                step_count=self.step_count,
                max_steps=self.max_steps,
            ),
            Reward(score=0.0, feedback="invalid action"),
            True,
            {"error": "invalid_action"},
        )

    # ==============================
    # STEP
    # ==============================
    def step(self, action_dict: Dict[str, Any]):
        # Parse safely
        try:
            action = Action(**action_dict)
        except Exception:
            return self._invalid_step()

        # Validate
        if not self.validate_action(action):
            return self._invalid_step()

        self.step_count += 1
        self.history.append(action)

        # ------------------------------
        # STATE EVOLUTION (IMPORTANT)
        # ------------------------------
        if action.action_type == "comment" and action.comment:
            self.issues_identified.append(action.comment)

        if action.action_type == "suggest_fix":
            self.fix_attempted = True

        # ------------------------------
        # BASE REWARD
        # ------------------------------
        score = grade_action(action, self.gt)

        # ------------------------------
        # REWARD SHAPING
        # ------------------------------
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

        # ------------------------------
        # DONE CONDITION
        # ------------------------------
        done = (
            action.action_type == "final_decision"
            or self.step_count >= self.max_steps
        )

        # ------------------------------
        # EPISODE-LEVEL SCORING
        # ------------------------------
        if done:
            score = max(
                [grade_action(a, self.gt) for a in self.history] or [0.0]
            )

        # ------------------------------
        # RETURN
        # ------------------------------
        return (
            Observation(
                pr=self.pr,
                previous_comments=[a.comment for a in self.history if a.comment],
                step_count=self.step_count,
                max_steps=self.max_steps,
            ),
            Reward(score=score, feedback="graded"),
            done,
            {
                "task_type": self.task_type,
                "issues_identified": len(self.issues_identified),
                "fix_attempted": self.fix_attempted,
            },
        )

    # ==============================
    # STATE
    # ==============================
    def state(self):
        return {
            "pr": self.pr.dict(),
            "history": [a.dict() for a in self.history],
            "step": self.step_count,
            "issues_identified": self.issues_identified,
            "fix_attempted": self.fix_attempted,
        }