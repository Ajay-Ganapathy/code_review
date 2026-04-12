# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders for the Code Review Environment.

Three difficulty tiers:
  - EasyGrader   : Forgiving. Substring matching, partial credit for wrong decisions.
  - MediumGrader : Balanced. Token overlap, line-level fix matching, recency weighting.
  - HardGrader   : Strict.   No wrong-decision credit, final-step-only done scoring.
"""

import re
from difflib import SequenceMatcher

STOP_WORDS = {
    "use", "the", "a", "an", "to", "and", "or", "of", "in",
    "for", "with", "is", "it", "on", "at", "by", "from", "that",
}


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def _code_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z_]\w*|\d+|[=<>!+\-*/]+", text)
    return [t for t in tokens if t.lower() not in STOP_WORDS]


# ==============================================================================
# Base Grader
# ==============================================================================

class BaseGrader:
    """
    Shared helpers. Subclasses override score_* and compute_* methods
    to implement their difficulty level.
    """

    # Subclasses set these to configure weights (must sum to 1.0)
    ISSUE_WEIGHT: float = 0.40
    FIX_WEIGHT: float = 0.30
    DECISION_WEIGHT: float = 0.30

    def grade_action(self, action, ground_truth: dict) -> float:
        score = (
            self.ISSUE_WEIGHT   * self.score_issues(action.comment, ground_truth)
            + self.FIX_WEIGHT   * self.score_fix(action.suggested_code, ground_truth)
            + self.DECISION_WEIGHT * self.score_decision(action, ground_truth)
        )
        return max(0.01, min(score, 0.99))

    def score_issues(self, comment: str, ground_truth: dict) -> float:
        raise NotImplementedError

    def score_fix(self, suggested_code: str, ground_truth: dict) -> float:
        raise NotImplementedError

    def score_decision(self, action, ground_truth: dict) -> float:
        raise NotImplementedError

    def compute_step_bonus(self, action, step_count: int, history: list) -> float:
        raise NotImplementedError

    def compute_done_score(self, history: list, ground_truth: dict) -> float:
        raise NotImplementedError


# ==============================================================================
# Easy Grader
# ==============================================================================

class EasyGrader(BaseGrader):
    """
    Lenient grader. Best for round-1 filtering / warm-up tasks.

    - Issue detection : simple substring match
    - Fix quality     : token overlap + sequence similarity
    - Wrong decision  : 0.2 partial credit
    - Done scoring    : max over entire history (most forgiving)
    - Bonuses         : generous, long trajectories are acceptable

    Weights: issues=40%, fix=30%, decision=30%
    """

    ISSUE_WEIGHT    = 0.40
    FIX_WEIGHT      = 0.30
    DECISION_WEIGHT = 0.30

    def score_issues(self, comment: str, ground_truth: dict) -> float:
        issues = ground_truth.get("issues", [])
        if not comment or not issues:
            return 0.0
        comment_norm = _normalize(comment)
        matches = sum(1 for issue in issues if _normalize(issue) in comment_norm)
        return matches / len(issues)

    def score_fix(self, suggested_code: str, ground_truth: dict) -> float:
        if not suggested_code:
            return 0.0
        expected  = _normalize(ground_truth.get("fix", ""))
        suggested = _normalize(suggested_code)
        if not expected:
            return 0.0
        if expected in suggested:
            return 1.0
        exp_tok = _code_tokens(expected)
        sug_tok = set(_code_tokens(suggested))
        token_score = (
            sum(1 for t in exp_tok if t in sug_tok) / len(exp_tok) if exp_tok else 0.0
        )
        seq_score = SequenceMatcher(None, expected, suggested).ratio()
        return round(0.7 * token_score + 0.3 * seq_score, 4)

    def score_decision(self, action, ground_truth: dict) -> float:
        if action.action_type != "final_decision" or not action.decision:
            return 0.0
        if action.decision == ground_truth.get("decision"):
            return 1.0
        return 0.2  # generous partial credit for wrong decision

    def compute_step_bonus(self, action, step_count: int, history: list) -> float:
        bonus = 0.0
        if action.comment and len(action.comment) > 30:
            bonus += 0.15
        if action.action_type == "final_decision" and step_count <= 3:
            bonus += 0.10
        if not action.comment and action.action_type != "final_decision":
            bonus -= 0.05
        return bonus

    def compute_done_score(self, history: list, ground_truth: dict) -> float:
        """Most forgiving: best single action across all of history."""
        scores = [self.grade_action(a, ground_truth) for a in history] or [0.0]
        return max(0.01, min(max(scores), 0.99))


# ==============================================================================
# Medium Grader
# ==============================================================================

class MediumGrader(BaseGrader):
    """
    Balanced grader. Suitable for main competition rounds.

    - Issue detection : token overlap + substring fallback
    - Fix quality     : token overlap + line-level + sequence similarity
    - Wrong decision  : 0.1 partial credit
    - Done scoring    : recency-weighted (recent actions matter more)
    - Bonuses         : moderate, efficiency is rewarded

    Weights: issues=42%, fix=30%, decision=28%
    """

    ISSUE_WEIGHT    = 0.42
    FIX_WEIGHT      = 0.30
    DECISION_WEIGHT = 0.28

    def score_issues(self, comment: str, ground_truth: dict) -> float:
        issues = ground_truth.get("issues", [])
        if not comment or not issues:
            return 0.0
        comment_text   = _normalize(comment)
        comment_tokens = set(re.findall(r"[a-zA-Z_]\w*", comment_text)) - STOP_WORDS
        best_scores = []
        for issue in issues:
            issue_text   = _normalize(issue)
            issue_tokens = set(re.findall(r"[a-zA-Z_]\w*", issue_text)) - STOP_WORDS
            if not issue_tokens:
                continue
            overlap      = len(issue_tokens & comment_tokens) / len(issue_tokens)
            substring    = 1.0 if issue_text in comment_text else 0.0
            best_scores.append(max(overlap, substring))
        return round(sum(best_scores) / len(issues), 4) if best_scores else 0.0

    def score_fix(self, suggested_code: str, ground_truth: dict) -> float:
        if not suggested_code:
            return 0.0
        expected  = _normalize(ground_truth.get("fix", ""))
        suggested = _normalize(suggested_code)
        if not expected:
            return 0.0
        if expected in suggested:
            return 1.0
        exp_lines = [l.strip() for l in expected.splitlines()  if l.strip()]
        sug_lines = [l.strip() for l in suggested.splitlines() if l.strip()]
        line_score = (
            sum(1 for l in exp_lines if l in sug_lines) / len(exp_lines)
            if exp_lines else 0.0
        )
        exp_tok = _code_tokens(expected)
        sug_tok = set(_code_tokens(suggested))
        token_score = (
            sum(1 for t in exp_tok if t in sug_tok) / len(exp_tok) if exp_tok else 0.0
        )
        seq_score = SequenceMatcher(None, expected, suggested).ratio()
        return round(0.4 * token_score + 0.3 * seq_score + 0.3 * line_score, 4)

    def score_decision(self, action, ground_truth: dict) -> float:
        if action.action_type != "final_decision" or not action.decision:
            return 0.0
        if action.decision == ground_truth.get("decision"):
            return 1.0
        return 0.1  # reduced partial credit

    def compute_step_bonus(self, action, step_count: int, history: list) -> float:
        bonus = 0.0
        if action.comment and len(action.comment) > 40:
            bonus += 0.10
        if action.action_type == "final_decision":
            if step_count == 1:
                bonus += 0.10
            elif step_count == 2:
                bonus += 0.05
        if step_count > 3:
            bonus -= 0.04
        if not action.comment and action.action_type != "final_decision":
            bonus -= 0.08
        return bonus

    def compute_done_score(self, history: list, ground_truth: dict) -> float:
        """Recency-weighted: later actions in history count for more."""
        n = max(len(history), 1)
        weighted = [
            self.grade_action(a, ground_truth) * (0.6 + 0.4 * (i / n))
            for i, a in enumerate(history)
        ]
        return max(0.01, min(max(weighted), 0.99))


# ==============================================================================
# Hard Grader
# ==============================================================================

class HardGrader(BaseGrader):
    """
    Strict grader. For finals / advanced rounds.

    - Issue detection : token overlap + seq similarity with a minimum threshold
    - Fix quality     : line-level match dominant, no free token credit
    - Wrong decision  : 0.0 (no credit at all)
    - Done scoring    : final step only (harshest)
    - Bonuses         : minimal, escalating penalty for long trajectories

    Weights: issues=45%, fix=28%, decision=27%
    """

    ISSUE_WEIGHT    = 0.45
    FIX_WEIGHT      = 0.28
    DECISION_WEIGHT = 0.27

    # Minimum combined score an issue match must clear to get any credit
    ISSUE_THRESHOLD = 0.30

    def score_issues(self, comment: str, ground_truth: dict) -> float:
        issues = ground_truth.get("issues", [])
        if not comment or not issues:
            return 0.0
        comment_text   = _normalize(comment)
        comment_tokens = set(re.findall(r"[a-zA-Z_]\w*", comment_text)) - STOP_WORDS
        scores = []
        for issue in issues:
            issue_text   = _normalize(issue)
            issue_tokens = set(re.findall(r"[a-zA-Z_]\w*", issue_text)) - STOP_WORDS
            if not issue_tokens:
                continue
            token_overlap = len(issue_tokens & comment_tokens) / len(issue_tokens)
            seq_sim       = SequenceMatcher(None, issue_text, comment_text).ratio()
            combined      = 0.7 * token_overlap + 0.3 * seq_sim
            # Must clear threshold to get any credit — no partial reward for vague hints
            scores.append(combined if combined >= self.ISSUE_THRESHOLD else 0.0)
        return round(sum(scores) / len(issues), 4) if scores else 0.0

    def score_fix(self, suggested_code: str, ground_truth: dict) -> float:
        if not suggested_code:
            return 0.0
        expected  = _normalize(ground_truth.get("fix", ""))
        suggested = _normalize(suggested_code)
        if not expected:
            return 0.0
        if expected in suggested:
            return 1.0
        exp_lines = [l.strip() for l in expected.splitlines()  if l.strip()]
        sug_lines = set(l.strip() for l in suggested.splitlines() if l.strip())
        line_score = (
            sum(1 for l in exp_lines if l in sug_lines) / len(exp_lines)
            if exp_lines else 0.0
        )
        exp_tok = _code_tokens(expected)
        sug_tok = set(_code_tokens(suggested))
        token_score = (
            sum(1 for t in exp_tok if t in sug_tok) / len(exp_tok) if exp_tok else 0.0
        )
        seq_score = SequenceMatcher(None, expected, suggested).ratio()
        # Line-level match is dominant in hard mode
        return round(0.5 * line_score + 0.3 * token_score + 0.2 * seq_score, 4)

    def score_decision(self, action, ground_truth: dict) -> float:
        if action.action_type != "final_decision" or not action.decision:
            return 0.0
        return 1.0 if action.decision == ground_truth.get("decision") else 0.0

    def compute_step_bonus(self, action, step_count: int, history: list) -> float:
        bonus = 0.0
        if action.action_type == "final_decision" and step_count == 1:
            bonus += 0.05  # only reward decisive first-step finishes
        if step_count > 2:
            bonus -= 0.05 * (step_count - 2)  # escalating penalty
        if not action.comment and action.action_type != "final_decision":
            bonus -= 0.12
        return bonus

    def compute_done_score(self, history: list, ground_truth: dict) -> float:
        """Strictest: only the final action in the episode counts."""
        if not history:
            return 0.01
        return max(0.01, min(self.grade_action(history[-1], ground_truth), 0.99))


# ==============================================================================
# Factory
# ==============================================================================

GRADER_REGISTRY: dict[str, type[BaseGrader]] = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}


def get_grader(level: str = "medium") -> BaseGrader:
    """
    Return a grader instance for the given difficulty level.

    Args:
        level: One of "easy", "medium", or "hard".

    Returns:
        An instantiated grader.

    Raises:
        ValueError: If the level is not recognised.
    """
    level = level.lower()
    if level not in GRADER_REGISTRY:
        raise ValueError(
            f"Unknown grader level '{level}'. Choose from: {list(GRADER_REGISTRY)}"
        )
    return GRADER_REGISTRY[level]()