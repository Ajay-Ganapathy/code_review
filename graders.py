class CodeReviewGrader:
    def grade(self, action, ground_truth) -> float:
        from server.code_review_environment import CodeReviewEnvironment

        env = CodeReviewEnvironment()
        return env.grade_action(action, ground_truth)
