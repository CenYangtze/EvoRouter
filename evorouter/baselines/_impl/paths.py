"""Artifact locations shared by baseline implementations."""

import os
from pathlib import Path

ASSETS = Path(
    os.environ.get(
        "EVOROUTER_ASSET_ROOT", Path(__file__).resolve().parents[3] / "assets"
    )
).resolve()
WEIGHTS = ASSETS / "weights/baselines"
UPSTREAM = ASSETS / "upstream"
OUTPUT = Path(os.environ.get("EVOROUTER_OUTPUT_DIR", "outputs")).resolve()
