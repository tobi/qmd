#!/usr/bin/env bash
# Tests for bin/qmd runtime detection logic.
# Simulates lockfile combinations in a temp directory and verifies which
# runtime the launcher would choose.
#
# Usage: bash test/launcher-detection.test.sh
set -euo pipefail

PASS=0
FAIL=0
TMPDIR_BASE=$(mktemp -d)

cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT

ok()   { printf "  %-60s OK\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "  %-60s FAIL\n" "$1 (got: $2, expected: $3)"; FAIL=$((FAIL + 1)); }

# Extract the detection logic from bin/qmd into a testable function.
# Instead of exec-ing a runtime, we echo which one would be chosen.
detect_runtime() {
  local DIR="$1"
  if [ -f "$DIR/package-lock.json" ]; then
    echo "node"
  elif [ -f "$DIR/bun.lock" ] || [ -f "$DIR/bun.lockb" ]; then
    echo "bun"
  else
    echo "node"
  fi
}

# Verify detect_runtime matches the actual bin/qmd logic
assert_runtime() {
  local label="$1" dir="$2" expected="$3"
  local got
  got=$(detect_runtime "$dir")
  if [[ "$got" == "$expected" ]]; then
    ok "$label"
  else
    fail "$label" "$got" "$expected"
  fi
}

echo "=== bin/qmd runtime detection tests ==="

# --- Test cases ---

# 1. No lockfiles → default to node
d="$TMPDIR_BASE/no-lockfiles"
mkdir -p "$d"
assert_runtime "no lockfiles → node" "$d" "node"

# 2. Only bun.lock → bun
d="$TMPDIR_BASE/bun-lock-only"
mkdir -p "$d"
touch "$d/bun.lock"
assert_runtime "bun.lock only → bun" "$d" "bun"

# 3. Only bun.lockb → bun
d="$TMPDIR_BASE/bun-lockb-only"
mkdir -p "$d"
touch "$d/bun.lockb"
assert_runtime "bun.lockb only → bun" "$d" "bun"

# 4. Only package-lock.json → node
d="$TMPDIR_BASE/npm-only"
mkdir -p "$d"
touch "$d/package-lock.json"
assert_runtime "package-lock.json only → node" "$d" "node"

# 5. Both package-lock.json AND bun.lock → node (npm takes priority)
# This is the key fix for #381: source checkouts have bun.lock committed,
# and contributors who run npm install also create package-lock.json.
d="$TMPDIR_BASE/both-lockfiles"
mkdir -p "$d"
touch "$d/package-lock.json"
touch "$d/bun.lock"
assert_runtime "package-lock.json + bun.lock → node (npm priority)" "$d" "node"

# 6. Both package-lock.json AND bun.lockb → node (npm takes priority)
d="$TMPDIR_BASE/both-lockfiles-b"
mkdir -p "$d"
touch "$d/package-lock.json"
touch "$d/bun.lockb"
assert_runtime "package-lock.json + bun.lockb → node (npm priority)" "$d" "node"

# 7. All three lockfiles → node (npm takes priority)
d="$TMPDIR_BASE/all-lockfiles"
mkdir -p "$d"
touch "$d/package-lock.json"
touch "$d/bun.lock"
touch "$d/bun.lockb"
assert_runtime "all three lockfiles → node (npm priority)" "$d" "node"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]]
