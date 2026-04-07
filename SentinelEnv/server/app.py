# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the SentinelEnv environment."""

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SentinelAction, SentinelObservation
    from .env import SentinelEnvironment
except ImportError:
    from models import SentinelAction, SentinelObservation
    from server.env import SentinelEnvironment


# OpenEnv's web UI reads README content from /app/README.md or ENV_README_PATH.
# Our Docker image stores the environment under /app/env, so set an explicit
# fallback before create_app() loads environment metadata.
os.environ.setdefault(
    "ENV_README_PATH", str(Path(__file__).resolve().parents[1] / "README.md")
)

# Create the app with web interface and README integration
app = create_app(
    SentinelEnvironment,
    SentinelAction,
    SentinelObservation,
    env_name="SentinelEnv",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m SentinelEnv.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn SentinelEnv.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
