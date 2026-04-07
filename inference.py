"""Root inference entrypoint for hackathon validators.

The full SentinelEnv inference implementation lives in ``SentinelEnv/inference.py``.
This thin wrapper keeps the required repository-root ``inference.py`` path without
duplicating logic.
"""

import asyncio

from SentinelEnv.inference import main


if __name__ == "__main__":
    asyncio.run(main())
