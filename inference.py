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

from code_review import CodeReviewAction, CodeReviewObservation
from code_review.client import CodeReviewEnv
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


def build_prompt(step: int, max_steps: int, observation) -> str:
    # print("Obeservation === " , observation)
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

async def run_episode(client, env):
    result = await env.reset()
    
    obs = result.observation

    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):

        prompt = build_prompt(step, MAX_STEPS, obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        completion = safe_completion(client, messages)  # still sync
        # print(completion)
        if completion is None:
            action = fallback_action()
        else:
            response_text = completion.choices[0].message.content or ""
            action_dict = parse_action(response_text)

            # print(response_text)

            action = CodeReviewAction(
                action_type=action_dict.get("action_type"),
                comment=action_dict.get("comment"),
                suggested_code=action_dict.get("suggested_code"),
                decision=action_dict.get("decision"),
            )

        result = await env.step(action)
        # print("Result === " , result)

        obs = result.observation
        reward = result.reward
        done = result.done

        final_score = max(final_score, reward.score if reward else 0.0)

        print(f"Step {step} | Action: {action} | Reward: {reward}")

        if done:
            print(f"Done in {step} steps")
            break

    return final_score

import asyncio

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = []
    

    async with CodeReviewEnv(base_url="http://localhost:8000") as env:
        # print(env)
        NUM_EPISODES=6
        # print(NUM_EPISODES)
        
        for i in range(NUM_EPISODES):
            print(f"\n=== Episode {i+1} ===")
            env.task_index = i

            score = await run_episode(client, env)
            scores.append(score)

            print(f"Scores so far: {scores}")
            #return 0

    print("\nFinished all episodes")
    print(f"Final Scores: {scores}")


if __name__ == "__main__":
    asyncio.run(main())
