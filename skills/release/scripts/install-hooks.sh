#!/usr/bin/env bash
set -euo pipefail

# Install git hooks for release validation.
# Idempotent — safe to run multiple times.

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [[ -z "$REPO_ROOT" ]]; then
  echo "Error: not in a git repository" >&2
  exit 1
fi

HOOKS_DIR="$REPO_ROOT/.git/hooks"
SOURCE="$REPO_ROOT/scripts/pre-push"

if [[ ! -f "$SOURCE" ]]; then
  echo "Error: scripts/pre-push not found at $SOURCE" >&2
  exit 1
fi

# Install pre-push hook
if [[ -L "$HOOKS_DIR/pre-push" ]] && [[ "$(readlink "$HOOKS_DIR/pre-push")" == "$SOURCE" ]]; then
  echo "pre-push hook: already installed (symlink)"
elif [[ -f "$HOOKS_DIR/pre-push" ]]; then
  # Existing hook that isn't our symlink — back it up
  BACKUP="$HOOKS_DIR/pre-push.backup.$(date +%s)"
  echo "pre-push hook: backing up existing hook to $(basename "$BACKUP")"
  mv "$HOOKS_DIR/pre-push" "$BACKUP"
  ln -sf "$SOURCE" "$HOOKS_DIR/pre-push"
  echo "pre-push hook: installed (symlink → scripts/pre-push)"
else
  ln -sf "$SOURCE" "$HOOKS_DIR/pre-push"
  echo "pre-push hook: installed (symlink → scripts/pre-push)"
fi

# Ensure the source is executable
chmod +x "$SOURCE"
echo "Done."
