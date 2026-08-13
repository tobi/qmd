#!/usr/bin/env bash
set -euo pipefail

# Gather release context for the /release skill (skills/release/SKILL.md step 1).
# Silently installs git hooks, then prints version info, working-tree status,
# commits and files since the last release tag, the current [Unreleased]
# changelog block, and the previous release entry for style reference.
#
# Usage: skills/release/scripts/release-context.sh [patch|minor|major|<version>]

VERSION_ARG="${1:?Usage: release-context.sh [patch|minor|major|<version>]}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$REPO_ROOT" ]]; then
  echo "Error: not in a git repository" >&2
  exit 1
fi
cd "$REPO_ROOT"

# Hooks are installed silently — the skill documents this as the auto-install path.
if [[ -x "$SCRIPT_DIR/install-hooks.sh" ]]; then
  "$SCRIPT_DIR/install-hooks.sh" >/dev/null
fi

bump_version() {
  local current="$1" type="$2"
  IFS='.' read -r major minor patch <<< "$current"
  case "$type" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "$major.$((minor + 1)).0" ;;
    patch) echo "$major.$minor.$((patch + 1))" ;;
    *)     echo "$type" ;;
  esac
}

if [[ ! -f package.json ]]; then
  echo "Error: package.json not found in $REPO_ROOT" >&2
  exit 1
fi

CURRENT=$(jq -r .version package.json)
NEXT=$(bump_version "$CURRENT" "$VERSION_ARG")
BRANCH=$(git branch --show-current)
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || true)

echo "=== Version ==="
echo "Current:   $CURRENT"
echo "Requested: $VERSION_ARG"
echo "Next:      $NEXT"
echo "Branch:    ${BRANCH:-"(detached)"}"
if [[ -n "$LAST_TAG" ]]; then
  echo "Last tag:  $LAST_TAG"
else
  echo "Last tag:  (none)"
fi
echo

echo "=== Working tree ==="
STATUS=$(git status --short)
if [[ -z "$STATUS" ]]; then
  echo "(clean)"
else
  printf '%s\n' "$STATUS"
fi
echo

echo "=== Commits since last release ==="
if [[ -n "$LAST_TAG" ]]; then
  COMMITS=$(git log --oneline "${LAST_TAG}..HEAD")
  if [[ -z "$COMMITS" ]]; then
    echo "(none since $LAST_TAG)"
  else
    printf '%s\n' "$COMMITS"
  fi
else
  echo "(no tags)"
  git log --oneline
fi
echo

echo "=== Files changed since last release ==="
if [[ -n "$LAST_TAG" ]]; then
  FILES=$(git diff --name-only "${LAST_TAG}..HEAD")
  if [[ -z "$FILES" ]]; then
    echo "(none since $LAST_TAG)"
  else
    printf '%s\n' "$FILES"
  fi
else
  echo "(no tags)"
fi
echo

echo "=== CHANGELOG [Unreleased] ==="
if [[ -f CHANGELOG.md ]]; then
  UNRELEASED=$(awk '
    /^## \[Unreleased\]/ { p=1; next }
    /^## \[/ { if (p) exit }
    p { print }
  ' CHANGELOG.md)
  TRIMMED=$(printf '%s' "$UNRELEASED" | sed '/^[[:space:]]*$/d')
  if [[ -z "$TRIMMED" ]]; then
    echo "(empty)"
  else
    printf '%s\n' "$UNRELEASED"
  fi
else
  echo "(CHANGELOG.md not found)"
fi
echo

echo "=== Previous release entry ==="
if [[ -f CHANGELOG.md ]]; then
  PREVIOUS=$(awk '
    /^## \[Unreleased\]/ { next }
    /^## \[/ { if (p) exit; p=1 }
    p { print }
  ' CHANGELOG.md)
  TRIMMED=$(printf '%s' "$PREVIOUS" | sed '/^[[:space:]]*$/d')
  if [[ -z "$TRIMMED" ]]; then
    echo "(none)"
  else
    printf '%s\n' "$PREVIOUS"
  fi
else
  echo "(CHANGELOG.md not found)"
fi
