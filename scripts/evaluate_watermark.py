#!/usr/bin/env python3
"""Standalone watermark detection benchmark."""

from pathlib import Path

from main import evaluate_watermark


if __name__ == "__main__":
    evaluate_watermark(Path.cwd())
