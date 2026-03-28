def normalize(text):
    return (text or "").lower().strip()


# ==============================
# ISSUE MATCH (PARTIAL CREDIT)
# ==============================
def score_issues(comment, ground_truth):
    issues = ground_truth.get("issues", [])
    if not comment or not issues:
        return 0.0

    comment = normalize(comment)

    matches = sum(
        1 for issue in issues if normalize(issue) in comment
    )

    return matches / len(issues)


# ==============================
# FIX MATCH (FUZZY)
# ==============================
def score_fix(suggested_code, ground_truth):
    if not suggested_code:
        return 0.0

    expected_fix = normalize(ground_truth.get("fix", ""))
    suggested_code = normalize(suggested_code)

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
def score_decision(action, ground_truth):
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


# ==============================
# FINAL GRADER
# ==============================
def grade_action(action, ground_truth):
    score = 0.0

    print("Action === " , action)
    print("Ground truth === " , ground_truth)
   

    # ------------------------------
    # ISSUE DETECTION (40%)
    # ------------------------------
    issue_score = score_issues(action.comment, ground_truth)
    score += 0.4 * issue_score
    print("After Issue Score == " , issue_score)

    # ------------------------------
    # FIX QUALITY (30%)
    # ------------------------------
    fix_score = score_fix(action.suggested_code, ground_truth)
    score += 0.3 * fix_score

    print("After Fix Score == " , fix_score)

    # ------------------------------
    # DECISION (30%)
    # ------------------------------
    decision_score = score_decision(action, ground_truth)
    score += 0.3 * decision_score

    print("After Decision Score == " , decision_score)

    # ------------------------------
    # CLAMP SCORE
    # ------------------------------
    score = max(0.0, min(score, 1.0))

    return score