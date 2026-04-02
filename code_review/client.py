# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code Review Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CodeReviewAction, CodeReviewObservation, CodeReviewReward , CodeReviewPullRequest


class CodeReviewEnv(
    EnvClient[CodeReviewAction, CodeReviewObservation, State]
):
    """
    Client for the Code Review Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CodeReviewEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CodeReviewAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodeReviewEnv.from_docker_image("code_review-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CodeReviewAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CodeReviewAction) -> Dict:
        # print("Action == ", action)

        # Handle dict input
        if isinstance(action, dict):
            act = {
                "action_type": action.get("action_type"),
                "comment": action.get("comment"),
                "suggested_code": action.get("suggested_code"),
                "decision": action.get("decision"),
            }
        else:
            act = {
                "action_type": action.action_type,
                "comment": action.comment,
                "suggested_code": action.suggested_code,
                "decision": action.decision,
            }

        # print("Act == ", act)
        return act

    def _parse_result(self, payload: Dict) -> StepResult[CodeReviewObservation]:
        """
        Parse server response into StepResult[CodeReviewObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CodeReviewObservation
        """

        """
         return CodeReviewObservation(
            #echoed_message="Code Review environment ready!",
            pr=self.pr,
            previous_comments=self.history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=0.0,
            done=False,
        )
        """
        # print("Payload ====== ", payload)

      
        obs_data = payload.get("observation") or {}

        if "observation" in obs_data:  # nested case
            obs_data = obs_data["observation"]

     

      
        if not obs_data or "pr" not in obs_data:
            raise ValueError(f"Invalid observation payload: {payload}")

      
        pr_data = obs_data["pr"]

        observation = CodeReviewObservation(
            pr=CodeReviewPullRequest(**pr_data),
            previous_comments=obs_data.get("previous_comments") or [],
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 3),
        )

        # Handle reward (reset vs step)
        reward_data = payload.get("reward")
        reward = None

        if isinstance(reward_data, dict):
            reward = CodeReviewReward(**reward_data)
        # else: float/None → ignore (reset case)

        return StepResult(
            observation=observation,
            reward=reward,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
