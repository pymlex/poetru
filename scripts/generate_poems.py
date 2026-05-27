#!/usr/bin/env python3
"""Standalone poem generation with digital watermarking."""

import argparse
from pathlib import Path

from main import generate_poems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    generate_poems(args.root.resolve(), count=args.count)
