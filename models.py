# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Data models for the My Env Environment.

# The my_env environment is a simple test environment that echoes back messages.
# """

# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# class MyAction(Action):
#     """Action for the My Env environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class MyObservation(Observation):
#     """Observation from the My Env environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")


# from dataclasses import dataclass
# from typing import Optional, Dict

# @dataclass
# class MyAction:
#     code: str

# @dataclass
# class MyObservation:
#     problem: str
#     last_error: str
#     output: Optional[str]
#     done: bool
#     reward: float
#     metadata: Dict


from pydantic import BaseModel
from typing import List, Optional

class CodeDiff(BaseModel):
    file_name: str
    diff: str

class PullRequest(BaseModel):
    id: str
    title: str
    description: str
    diffs: List[CodeDiff]
    language: str

class Observation(BaseModel):
    pr: PullRequest
    previous_comments: List[str]
    step_count: int
    max_steps: int

class Action(BaseModel):
    action_type: str  # comment / suggest_fix / final_decision
    comment: Optional[str] = None
    suggested_code: Optional[str] = None
    decision: Optional[str] = None

class Reward(BaseModel):
    score: float
    feedback: str