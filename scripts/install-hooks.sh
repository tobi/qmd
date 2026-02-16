#!/usr/bin/env bash
set -euo pipefail

# Self-installing git hooks for qmd
# Called from package.json "prepare" script after bun install

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

if [[ ! -d "$HOOKS_DIR" ]]; then
  echo "Not a git repository, skipping hook install"
  exit 0
fi

# Install pre-push hook
cp "$REPO_ROOT/scripts/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"

echo "Installed git hooks: pre-push"
