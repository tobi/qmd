#!/usr/bin/env bash
set -euo pipefail

# QMD Release Script
# Usage: ./scripts/release.sh [patch|minor|major|<version>]
# Examples:
#   ./scripts/release.sh patch     # 0.9.0 -> 0.9.1
#   ./scripts/release.sh minor     # 0.9.0 -> 0.10.0
#   ./scripts/release.sh major     # 0.9.0 -> 1.0.0
#   ./scripts/release.sh 1.0.0     # explicit version

BUMP="${1:?Usage: release.sh [patch|minor|major|<version>]}"

# Ensure we're on main and clean
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
  echo "Error: must be on main branch (currently on $BRANCH)" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working directory not clean" >&2
  git status --short
  exit 1
fi

# Read current version
CURRENT=$(jq -r .version package.json)
echo "Current version: $CURRENT"

# Calculate new version
bump_version() {
  local current="$1" type="$2"
  IFS='.' read -r major minor patch <<< "$current"
  case "$type" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "$major.$((minor + 1)).0" ;;
    patch) echo "$major.$minor.$((patch + 1))" ;;
    *)     echo "$type" ;; # explicit version
  esac
}

NEW=$(bump_version "$CURRENT" "$BUMP")
echo "New version:     $NEW"
echo ""

# Confirm
read -p "Release v$NEW? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

# Gather commits since last tag (or all if no tags)
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [[ -n "$LAST_TAG" ]]; then
  RANGE="$LAST_TAG..HEAD"
else
  RANGE="HEAD"
fi

echo ""
echo "Commits since ${LAST_TAG:-beginning}:"
git log "$RANGE" --oneline --no-decorate
echo ""

# Generate changelog entry
DATE=$(date +%Y-%m-%d)
ENTRY="## [$NEW] - $DATE"$'\n'$'\n'

# Collect conventional commits
FEATS=$(git log "$RANGE" --oneline --no-decorate --grep="^feat" | sed 's/^[a-f0-9]* feat[:(]/- /' | sed 's/)$//' || true)
FIXES=$(git log "$RANGE" --oneline --no-decorate --grep="^fix" | sed 's/^[a-f0-9]* fix[:(]/- /' | sed 's/)$//' || true)
OTHER=$(git log "$RANGE" --oneline --no-decorate --grep="^feat" --grep="^fix" --grep="^docs" --grep="^chore" --grep="^refactor" --invert-grep | sed 's/^[a-f0-9]* /- /' || true)

if [[ -n "$FEATS" ]]; then
  ENTRY+="### Features"$'\n'$'\n'"$FEATS"$'\n'$'\n'
fi
if [[ -n "$FIXES" ]]; then
  ENTRY+="### Fixes"$'\n'$'\n'"$FIXES"$'\n'$'\n'
fi
if [[ -n "$OTHER" ]]; then
  ENTRY+="### Other"$'\n'$'\n'"$OTHER"$'\n'$'\n'
fi

# Add link reference
LINK="[$NEW]: https://github.com/tobi/qmd/compare/v$CURRENT...v$NEW"

# Show what will be added
echo "--- Changelog entry ---"
echo "$ENTRY"
echo "$LINK"
echo "--- End ---"
echo ""
read -p "Looks good? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

# Update package.json version
jq --arg v "$NEW" '.version = $v' package.json > package.json.tmp && mv package.json.tmp package.json

# Prepend changelog entry (after the header line)
if [[ -f CHANGELOG.md ]]; then
  # Insert after "# Changelog" header and any blank lines
  awk -v entry="$ENTRY$LINK" '
    /^# Changelog/ { print; getline; print; print ""; print entry; print ""; next }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
else
  echo "# Changelog"$'\n'$'\n'"$ENTRY$LINK" > CHANGELOG.md
fi

# Commit and tag
git add package.json CHANGELOG.md
git commit -m "release: v$NEW"
git tag -a "v$NEW" -m "v$NEW"

echo ""
echo "Created commit and tag v$NEW"
echo ""
echo "Next steps:"
echo "  git push origin main --tags   # push to GitHub"
echo "  npm publish                   # publish to npm"
echo ""
echo "Or both at once:"
echo "  git push origin main --tags && npm publish"
