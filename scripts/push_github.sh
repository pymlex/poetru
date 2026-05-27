#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

git add -A
git diff --cached --quiet && echo "Nothing to commit" && exit 0

MSG="${1:-Update poetru pipeline}"
git commit -m "$MSG"
git push origin HEAD
