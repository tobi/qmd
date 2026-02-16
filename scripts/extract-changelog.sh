#!/usr/bin/env bash
set -euo pipefail

# Extract cumulative release notes from CHANGELOG.md.
#
# For a given version (e.g. 1.0.5), extracts all entries from the current
# minor series back to x.x.0 (e.g. 1.0.0 through 1.0.5). This means each
# GitHub release restates the full arc of changes for the minor series.
#
# The [Unreleased] section is included — it contains the content that will
# become [X.Y.Z] when the release script runs. If the version is already
# released, [Unreleased] may be empty and is omitted.
#
# Fails if neither [Unreleased] nor [X.Y.Z] has content in the changelog.
#
# Usage: scripts/extract-changelog.sh <version>
# Example: scripts/extract-changelog.sh 1.0.5
#   -> extracts [Unreleased] + [1.0.5], [1.0.4], ..., [1.0.0]

VERSION="${1:?Usage: extract-changelog.sh <version>}"

# Parse major.minor.patch from version
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

if [[ ! -f CHANGELOG.md ]]; then
  echo "CHANGELOG.md not found" >&2
  exit 1
fi

# Extract [Unreleased] section and all [X.Y.Z] sections matching our minor series.
OUTPUT=""
CAPTURING=false
UNRELEASED_CONTENT=""
IN_UNRELEASED=false

while IFS= read -r line; do
  if [[ "$line" =~ ^##\ \[Unreleased\] ]]; then
    CAPTURING=true
    IN_UNRELEASED=true
  elif [[ "$line" =~ ^##\ \[([0-9]+\.[0-9]+\.[0-9]+)\] ]]; then
    IN_UNRELEASED=false
    ENTRY_VERSION="${BASH_REMATCH[1]}"
    IFS='.' read -r E_MAJOR E_MINOR E_PATCH <<< "$ENTRY_VERSION"
    if [[ "$E_MAJOR" == "$MAJOR" && "$E_MINOR" == "$MINOR" ]]; then
      CAPTURING=true
      OUTPUT+="$line"$'\n'
    else
      CAPTURING=false
    fi
  elif [[ "$line" =~ ^##\  ]]; then
    IN_UNRELEASED=false
    CAPTURING=false
  elif $CAPTURING; then
    if $IN_UNRELEASED; then
      UNRELEASED_CONTENT+="$line"$'\n'
    else
      OUTPUT+="$line"$'\n'
    fi
  fi
done < CHANGELOG.md

# Only include [Unreleased] if it has non-blank content
TRIMMED=$(echo "$UNRELEASED_CONTENT" | sed '/^[[:space:]]*$/d')
if [[ -n "$TRIMMED" ]]; then
  OUTPUT="## [Unreleased]"$'\n'"$UNRELEASED_CONTENT$OUTPUT"
fi

# Fail if we got nothing
TRIMMED_OUTPUT=$(echo "$OUTPUT" | sed '/^[[:space:]]*$/d')
if [[ -z "$TRIMMED_OUTPUT" ]]; then
  echo "error: no changelog content found for $VERSION" >&2
  echo "Expected either:" >&2
  echo "  ## [Unreleased]  (with content)" >&2
  echo "  ## [$VERSION] - YYYY-MM-DD" >&2
  exit 1
fi

printf '%s' "$OUTPUT"
