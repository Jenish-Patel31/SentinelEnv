# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backward-compatible import shim for the original starter template path."""

from .env import SentinelEnvironment, SentinelenvEnvironment

__all__ = ["SentinelEnvironment", "SentinelenvEnvironment"]
