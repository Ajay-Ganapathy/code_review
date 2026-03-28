# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Code Review Environment.

The code_review environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import Optional, List

class CodeReviewAction(Action):
    """Action for the Code Review environment - just a message to echo."""

    # message: str = Field(..., description="Message to echo back")
    action_type: str  # comment / suggest_fix / final_decision
    comment: Optional[str] = None
    suggested_code: Optional[str] = None
    decision: Optional[str] = None

class CodeDiff(BaseModel):
    file_name: str
    diff: str


class CodeReviewPullRequest(BaseModel):
    id: str
    title: str
    description: str
    diffs: List[CodeDiff]
    language: str

class CodeReviewObservation(Observation):
    """Observation from the Code Review environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    pr: CodeReviewPullRequest
    previous_comments: List[str]
    step_count: int
    max_steps: int


class CodeReviewReward(BaseModel):
    score: float
    feedback: str
