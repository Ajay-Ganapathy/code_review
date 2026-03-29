"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict, Any

from openai import OpenAI
import numpy as np
import json

from code_review import CodeReviewAction, CodeReviewObservation, CodeReviewEnvironment

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 30
TEMPERATURE = 0.0
MAX_TOKENS = 300

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


SYSTEM_PROMPT = textwrap.dedent(
    """
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
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def build_prompt(step: int, observation) -> str:
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Review this Pull Request carefully.

        Title: {observation.pr.title}
        Description: {observation.pr.description}

        Code Diffs:
        {observation.pr.diffs}

        Previous Comments:
        {observation.previous_comments}

        Instructions:
        - Identify ALL issues
        - Suggest a fix if possible
        - Decide approve or reject

        Return JSON only.
        """
    ).strip()
    return prompt


def safe_completion(client, messages):
    for _ in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            print("Error during completion, retrying...")
            print(e)
            continue
    return None


def fallback_action():
    return {
        "action_type": "comment",
        "comment": "fallback: invalid response",
        "suggested_code": None,
        "decision": None,
    }


def parse_action(text: str) -> Dict[str, Any]:
    if not text:
        return fallback_action()

    text = text.strip().replace("```json", "").replace("```", "")

    try:
        return json.loads(text)
    except Exception as e:
        print(e)
        return fallback_action()


def run_episode(client, env):
    obs = env.reset()
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(step, obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        completion = safe_completion(client,messages)

        if completion is None:
            print("Not able to send messages to LLM:")
            action = fallback_action()
        else:
            print("Sending messages to LLM:")
            response_text = completion.choices[0].message.content or ""
            # print(f"[INFO] Response: {response_text}")
            action = parse_action(response_text)
            print(action)

        obs, reward, done, _ = env.step(action)

        final_score = max(final_score, reward.score)

        print(f"Step {step} | Action: {action} | Reward: {reward.score:.2f}")

        if done:
            print(f"Done in {step} steps")
            break

    return final_score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # env = BrowserGymEnv.from_docker_image(
    #     image="browsergym-env:latest",
    #     env_vars={
    #         "BROWSERGYM_BENCHMARK": "miniwob",
    #         "BROWSERGYM_TASK_NAME": "click-test",
    #     },
    # )

    env = CodeReviewEnvironment()
    scores = []

    try:
        for i in range(len(env.dataset)):
            env.task_index = i
            score = run_episode(client, env)
            scores.append(score)
            print(scores)
        else:
            print(f"Reached max steps ({MAX_STEPS}).")

    finally:
        env.close()


if __name__ == "__main__":
    main()
