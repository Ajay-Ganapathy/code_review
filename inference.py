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
MAX_STEPS = 3
TEMPERATURE = 0.2
MAX_TOKENS = 512

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


SYSTEM_PROMPT = textwrap.dedent(
    """
You are a senior software engineer reviewing a pull request.

You MUST follow this workflow:

Step 1:
Identify all issues in the code.
List them clearly in the comment.

Step 2:
Provide a suggested fix with corrected code.

Step 3:
Make a final decision:
- reject if any bug, security risk, or incorrect logic exists
- approve only if the code is safe and correct

Rules:
- Mention every issue explicitly
- Use precise technical language
- Write detailed comments (>30 characters)

Return ONLY JSON:

{
  "action_type": "comment | suggest_fix | final_decision",
  "comment": "...",
  "suggested_code": "...",
  "decision": "approve | reject | null"
}
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def build_prompt(step: int, max_steps: int, observation) -> str:
    if step == 1:
        instruction = (
            "Carefully analyze the diff. List EVERY issue you find in the comment field. "
            "Use exact technical terms (e.g. 'sql injection', 'null handling', 'hardcoded password'). "
            "Set action_type to 'comment'."
            "If the code looks correct with no issues, still output a comment like: 'No issues found. Code is clean.' and prepare to approve."
        )
    elif step == 2:
        instruction = (
            "Now provide the fix. Set action_type to 'suggest_fix'. "
            "Write the corrected code in suggested_code. "
            "Also repeat the issues in the comment field."
        )
    else:
        instruction = (
            "Make your final decision. Set action_type to 'final_decision'. "
            "Set decision to 'reject' if any bug, security issue, or bad logic exists. "
            "Set decision to 'approve' only if the code is clean and correct."
        )

    diff_text = "\n\n".join(
        f"File: {d.file_name}\n{d.diff}" for d in observation.pr.diffs
    )

    return textwrap.dedent(
        f"""
        Step {step}/{max_steps}

        Title: {observation.pr.title}
        Description: {observation.pr.description}

        Code Diffs:
        {diff_text}

        Previous Comments:
        {build_history_lines(observation.previous_comments)}

        Your task: {instruction}

        Return ONLY valid JSON:
        {{
          "action_type": "...",
          "comment": "...",
          "suggested_code": "...",
          "decision": "approve | reject | null"
        }}
    """
    ).strip()


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
    conversation = []  # persists across steps

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(step, MAX_STEPS, obs)
        conversation.append({"role": "user", "content": prompt})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
        completion = safe_completion(client, messages)

        if completion is None:
            action = fallback_action()
            response_text = json.dumps(action)
        else:
            response_text = completion.choices[0].message.content or ""
            action = parse_action(response_text)

        # Add model reply to history so it knows what it already said
        conversation.append({"role": "assistant", "content": response_text})

        obs, reward, done, info = env.step(action)
        final_score = max(final_score, reward.score)
        print(
            f"Step {step} | Action type: {action.get('action_type')} | Score: {reward.score:.2f}"
        )

        if done:
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
