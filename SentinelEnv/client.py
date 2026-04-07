# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client wrapper for the SentinelEnv benchmark."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SentinelAction, SentinelObservation, SentinelState


class SentinelEnv(EnvClient[SentinelAction, SentinelObservation, SentinelState]):
    """
    OpenEnv client for the Sentinel governance benchmark.

    The client keeps transport handling intentionally thin: actions are dumped via
    Pydantic, and server payloads are validated back into the strict contracts.
    """

    def _step_payload(self, action: SentinelAction) -> Dict[str, Any]:
        """Serialize an action using the Pydantic contract's JSON field names."""
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SentinelObservation]:
        """
        Parse a step/reset result from the OpenEnv transport envelope.

        OpenEnv commonly places ``reward`` and ``done`` at the top level of the
        envelope, so we merge them back into the observation before validation.
        """
        observation_payload = payload.get("observation", {}) or {}
        merged_observation = dict(observation_payload)

        if "reward" in payload:
            merged_observation["reward"] = payload["reward"]
        elif "reward" not in merged_observation:
            merged_observation["reward"] = None

        if "done" in payload:
            merged_observation["done"] = payload["done"]
        else:
            merged_observation.setdefault("done", False)

        observation = SentinelObservation.model_validate(merged_observation)
        reward = payload.get("reward", observation.reward)
        done = bool(payload.get("done", observation.done))

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SentinelState:
        """Validate environment state directly against the strict Sentinel model."""
        return SentinelState.model_validate(payload)


# Backward-compatible alias for the starter template.
SentinelenvEnv = SentinelEnv


__all__ = ["SentinelEnv", "SentinelenvEnv"]
