#!/usr/bin/env bash
# Re-pull the wiki-bench v0 corpus pages from private tobi/wiki.
# Requires: gh auth with read access to tobi/wiki.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/test/wiki-bench-docs"
FIXTURE_LIST="$ROOT/test/wiki-bench-docs/.expected-files.txt"

# Prefer the checked-in list next to this script's sibling fixture copy if present,
# else derive from current tree, else fall back to a hard-coded list file.
LIST_FILE="$ROOT/scripts/wiki-bench-expected-files.txt"
if [[ ! -f "$LIST_FILE" ]]; then
  LIST_FILE="$(mktemp)"
  # Reconstruct from existing vendored paths when the manifest isn't checked in.
  if [[ -d "$DEST" ]]; then
    (cd "$DEST" && find . -type f -name '*.md' | sed 's|^\./||' | sort > "$LIST_FILE")
  else
    echo "error: no expected-files manifest and no existing corpus at $DEST" >&2
    exit 1
  fi
fi

mkdir -p "$DEST"
count=0
while IFS= read -r path; do
  [[ -z "$path" || "$path" =~ ^# ]] && continue
  dir="$(dirname "$path")"
  mkdir -p "$DEST/$dir"
  gh api "repos/tobi/wiki/contents/wiki/$path" --jq .content | base64 -d > "$DEST/$path"
  bytes="$(wc -c < "$DEST/$path" | tr -d ' ')"
  if [[ "$bytes" -eq 0 ]]; then
    echo "error: empty file for $path" >&2
    exit 1
  fi
  echo "OK $path ($bytes bytes)"
  count=$((count + 1))
done < "$LIST_FILE"

echo "Synced $count files into $DEST"
empty="$(find "$DEST" -type f -name '*.md' -empty | wc -l | tr -d ' ')"
total="$(find "$DEST" -type f -name '*.md' | wc -l | tr -d ' ')"
echo "total=$total empty=$empty"
if [[ "$empty" != "0" ]]; then
  echo "error: found empty markdown files" >&2
  exit 1
fi
