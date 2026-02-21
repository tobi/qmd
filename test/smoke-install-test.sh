#!/usr/bin/env bash
# Smoke test: install @tobilu/qmd from tarball and verify it runs under node and bun.
# Both runtimes need node on PATH (the bin uses #!/usr/bin/env node shebang).
set -uo pipefail

TARBALL=$(ls /tmp/tobilu-qmd-*.tgz | head -1)
PASS=0
FAIL=0
TMP=$(mktemp)

ok()   { printf "  %-44s OK\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "  %-44s FAIL\n" "$1"; FAIL=$((FAIL + 1)); cat "$TMP" | sed 's/^/    /'; }

NODE_BIN="$(mise where node@latest)/bin"
BUN_BIN="$(mise where bun@latest)/bin"
BASE_PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin"

# ---------------------------------------------------------------------------
# Node: install via npm, runs with node (via shebang)
# ---------------------------------------------------------------------------
echo "=== Node $($NODE_BIN/node --version) ==="
export PATH="$NODE_BIN:$BASE_PATH"

if npm install -g "$TARBALL" >"$TMP" 2>&1; then ok "npm install -g"
else fail "npm install -g"; fi

timeout 10 qmd >"$TMP" 2>&1 || true
if grep -q "Usage:" "$TMP"; then ok "qmd shows help"
else fail "qmd shows help"; fi

if timeout 10 qmd collection list >"$TMP" 2>&1; then ok "qmd collection list"
else fail "qmd collection list"; fi

# ---------------------------------------------------------------------------
# Bun: install via bun, still runs with node (shebang)
# ---------------------------------------------------------------------------
echo ""
echo "=== Bun $($BUN_BIN/bun --version) ==="
export PATH="$BUN_BIN:$HOME/.bun/bin:$NODE_BIN:$BASE_PATH"

if bun install -g "$TARBALL" >"$TMP" 2>&1; then ok "bun install -g"
else fail "bun install -g"; fi

timeout 10 "$HOME/.bun/bin/qmd" >"$TMP" 2>&1 || true
if grep -q "Usage:" "$TMP"; then ok "qmd shows help (bun-installed)"
else fail "qmd shows help (bun-installed)"; fi

if timeout 10 "$HOME/.bun/bin/qmd" collection list >"$TMP" 2>&1; then ok "qmd collection list (bun-installed)"
else fail "qmd collection list (bun-installed)"; fi

rm -f "$TMP"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]]
