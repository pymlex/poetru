#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

TRACKED_PATHS=()

if [[ -d "artifacts/metrics" ]]; then
  TRACKED_PATHS+=("artifacts/metrics")
fi
if [[ -f "artifacts/logs/train_history.csv" ]]; then
  TRACKED_PATHS+=("artifacts/logs/train_history.csv")
fi
if [[ -f "artifacts/generated_poems.jsonl" ]]; then
  TRACKED_PATHS+=("artifacts/generated_poems.jsonl")
fi
if [[ -d "docs/experiments" ]]; then
  TRACKED_PATHS+=("docs/experiments")
fi

if [[ "${#TRACKED_PATHS[@]}" -eq 0 ]]; then
  echo "No result files found to push."
  exit 0
fi

git add -A "${TRACKED_PATHS[@]}"

if git diff --cached --quiet; then
  echo "No result changes to commit."
  exit 0
fi

MSG="${1:-Update training results $(date -u +"%Y-%m-%d %H:%M:%S UTC")}"
git commit -m "$MSG"
git push origin HEAD
