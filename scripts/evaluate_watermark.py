#!/usr/bin/env python3
"""Standalone watermark detection benchmark."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main import evaluate_watermark


if __name__ == "__main__":
    evaluate_watermark(Path.cwd())
