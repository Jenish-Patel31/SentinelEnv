# Copyright (c) 2026 Your Team Name
# All rights reserved.

"""
Data models for the SentinelEnv environment.

The environment exposes a governance-auditor style action space over a fixed set
of deterministic benchmark tasks.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, ValidationInfo, field_validator

ActionType = Literal["pass", "redact", "intervene", "escalate"]


class ChatTurn(BaseModel):
    """A single message in the conversation transcript."""

    role: Literal["user", "assistant"] = Field(
        ..., description="The speaker role for the chat message."
    )
    content: str = Field(..., description="The text content of the chat message.")


class SentinelAction(Action):
    """Action emitted by the Sentinel auditor."""

    action_type: ActionType = Field(
        default="pass",
        description=(
            "The governance decision: pass, redact (PII), intervene (policy), "
            "or escalate (hallucination)."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Short deterministic rationale for the audit log.",
    )
    modified_text: Optional[str] = Field(
        default=None,
        description=(
            "Required when action_type is 'redact' or 'intervene'. Contains the "
            "safe replacement text."
        ),
    )

    @field_validator("modified_text")
    @classmethod
    def check_modified_text(
        cls, value: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Require a rewrite payload when the action edits unsafe content."""
        action_type = info.data.get("action_type")
        if action_type in {"redact", "intervene"} and not value:
            raise ValueError(
                f"modified_text is strictly required when action_type is '{action_type}'"
            )
        return value


class SentinelObservation(Observation):
    """Observation returned to the governance auditor."""

    transcript: List[ChatTurn] = Field(
        default_factory=list,
        description="Recent conversation history under review.",
    )
    source_of_truth: str = Field(
        default="",
        description="Policy or factual reference the agent must ground on.",
    )
    task_goal: str = Field(
        default="",
        description="Instructions for the active benchmark tier.",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Validation feedback for the last invalid semantic action.",
    )


class SentinelState(State):
    """Minimal externally visible runtime state for the environment."""

    task_id: str = Field(
        default="pii_shield_easy", description="Current benchmark task id."
    )
    step_count: int = Field(default=0, description="Number of steps in this episode.")
    is_done: bool = Field(default=False, description="Whether the episode has ended.")


# Backward-compatible aliases for the starter template naming.
SentinelenvAction = SentinelAction
SentinelenvObservation = SentinelObservation
SentinelenvState = SentinelState


__all__ = [
    "ChatTurn",
    "ActionType",
    "SentinelAction",
    "SentinelObservation",
    "SentinelState",
    "SentinelenvAction",
    "SentinelenvObservation",
    "SentinelenvState",
]
