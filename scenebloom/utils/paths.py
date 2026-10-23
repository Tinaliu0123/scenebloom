"""Fixed folders of the repository: weights, third-party sources and data."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO_ROOT / "checkpoints"
THIRD_PARTY = REPO_ROOT / "third_party"
DATA = REPO_ROOT / "data"
