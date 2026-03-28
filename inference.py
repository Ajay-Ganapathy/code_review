import os
import json
from typing import Dict, Any

from openai import OpenAI
from environment import CodeReviewEnv

# ==============================
# ENV VARIABLES (MANDATORY)
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

TEMPERATURE = 0.0
MAX_TOKENS = 300
MAX_STEPS = 30

# ==============================
# INIT CLIENT
# ==============================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ==============================
# PROMPT
# ==============================
SYSTEM_PROMPT = """
You are a senior software engineer reviewing a pull request.

Your responsibilities:
1. Identify all issues in the code:
   - bugs
   - edge cases
   - performance issues
   - security vulnerabilities
2. Suggest fixes where appropriate
3. Decide whether the PR should be approved or rejected

Guidelines:
- Be precise and use clear technical terminology
- Mention specific issues explicitly
- Consider correctness, safety, and best practices
- Reject if there are bugs or risks

Output ONLY valid JSON:
{
  "action_type": "comment | suggest_fix | final_decision",
  "comment": "string or null",
  "suggested_code": "string or null",
  "decision": "approve | reject | null"
}
"""

# ==============================
# SAFE COMPLETION (RETRY)
# ==============================
def safe_completion(messages):
    for _ in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception:
            continue
    return None


# ==============================
# PARSE RESPONSE
# ==============================
def parse_action(text: str) -> Dict[str, Any]:
    if not text:
        return fallback_action()

    text = text.strip().replace("```json", "").replace("```", "")

    try:
        return json.loads(text)
    except Exception:
        return fallback_action()


def fallback_action():
    return {
        "action_type": "comment",
        "comment": "fallback: invalid response",
        "suggested_code": None,
        "decision": None,
    }


# ==============================
# BUILD PROMPT
# ==============================
def build_prompt(obs):
    return f"""
Review this Pull Request carefully.

Title: {obs.pr.title}
Description: {obs.pr.description}

Code Diffs:
{obs.pr.diffs}

Previous Comments:
{obs.previous_comments}

Instructions:
- Identify ALL issues
- Suggest a fix if possible
- Decide approve or reject

Return JSON only.
"""


# ==============================
# MAIN LOOP
# ==============================
def run_episode(env):
    obs = env.reset()
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        completion = safe_completion(messages)

        if completion is None:
            action = fallback_action()
        else:
            response_text = completion.choices[0].message.content or ""
            action = parse_action(response_text)

        obs, reward, done, _ = env.step(action)

        final_score = max(final_score, reward.score)

        #print(f"Step {step} | Action: {action} | Reward: {reward.score:.2f}")

        if done:
            print(f"Done in {step} steps")
            break

    return final_score


# ==============================
# MAIN
# ==============================
def main():
    env = CodeReviewEnv()

    scores = []
    print(env.dataset )

    # Run multiple episodes (deterministic dataset)
    for i in range(len(env.dataset)):
        
        env.task_index = i  # ensure deterministic selection
        score = run_episode(env)
        scores.append(score)
        print(scores)

    final_score = sum(scores) / len(scores)

    print("\n====================")
    print(f"Final Score: {final_score:.3f}")
    print("====================")


if __name__ == "__main__":
    main()