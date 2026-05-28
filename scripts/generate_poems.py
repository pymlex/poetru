#!/usr/bin/env python3
"""Standalone poem generation with digital watermarking."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main import generate_poems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    generate_poems(args.root.resolve(), count=args.count)
