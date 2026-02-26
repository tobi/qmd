#!/usr/bin/env bash
# Build a container image with qmd installed via npm and bun, then run smoke tests.
# Works with docker or podman (whichever is available).
#
# Usage:
#   test/smoke-install.sh              # build + run all smoke tests
#   test/smoke-install.sh --build      # build image only
#   test/smoke-install.sh --shell      # drop into container shell
#   test/smoke-install.sh -- CMD...    # run arbitrary command in container
set -euo pipefail

cd "$(dirname "$0")/.."

# Pick container runtime
if command -v podman &>/dev/null; then
  CTR=podman
elif command -v docker &>/dev/null; then
  CTR=docker
else
  echo "Error: neither podman nor docker found" >&2
  exit 1
fi

IMAGE=qmd-smoke

build_image() {
  echo "==> Building TypeScript..."
  npm run build --silent

  echo "==> Packing tarball..."
  rm -f test/tobilu-qmd-*.tgz
  TARBALL=$(npm pack --pack-destination test/ 2>/dev/null | tail -1)
  echo "    $TARBALL"

  # Copy project files into build context so vitest/bun tests can run inside
  rm -rf test/test-src
  mkdir -p test/test-src/src test/test-src/test
  cp src/*.ts test/test-src/src/
  cp -r dist test/test-src/
  cp test/*.test.ts test/test-src/test/
  cp package.json tsconfig.json tsconfig.build.json test/test-src/

  echo "==> Building container image ($CTR)..."
  $CTR build -f test/Containerfile -t "$IMAGE" test/

  # Clean up
  rm -f test/tobilu-qmd-*.tgz
  rm -rf test/test-src
  echo "==> Image ready: $IMAGE"
}

run() {
  $CTR run --rm "$IMAGE" bash -c "$*"
}

PASS=0
FAIL=0

ok()   { printf "  %-50s OK\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "  %-50s FAIL\n" "$1"; FAIL=$((FAIL + 1)); echo "$2" | sed 's/^/    /'; }

smoke_test() {
  local label="$1"; shift
  local out
  if out=$(run "$@" 2>&1); then
    ok "$label"
  else
    fail "$label" "$out"
  fi
}

smoke_test_output() {
  local label="$1"; local expect="$2"; shift 2
  local out
  out=$(run "$@" 2>&1) || true
  if echo "$out" | grep -q "$expect"; then
    ok "$label"
  else
    fail "$label" "$out"
  fi
}

run_smoke_tests() {
  # ------------------------------------------------------------------
  # Node (npm-installed qmd)
  # ------------------------------------------------------------------
  local NODE_BIN='$(mise where node@latest)/bin'
  echo "=== Node (npm install) ==="

  smoke_test_output "qmd shows help" "Usage:" \
    "export PATH=$NODE_BIN:\$PATH; qmd"

  smoke_test "qmd collection list" \
    "export PATH=$NODE_BIN:\$PATH; qmd collection list"

  smoke_test "qmd status" \
    "export PATH=$NODE_BIN:\$PATH; qmd status"

  smoke_test "sqlite-vec loads" \
    "export PATH=$NODE_BIN:\$PATH;
     NPM_GLOBAL=\$(npm root -g);
     node -e \"
      const {openDatabase, loadSqliteVec} = await import('\$NPM_GLOBAL/@tobilu/qmd/dist/db.js');
      const db = openDatabase(':memory:');
      loadSqliteVec(db);
      const r = db.prepare('SELECT vec_version() as v').get();
      console.log('sqlite-vec', r.v);
      if (!r.v) process.exit(1);
    \""

  smoke_test "vitest (node)" \
    "export PATH=$NODE_BIN:\$PATH; cd /opt/qmd && npx vitest run --reporter=verbose test/store.test.ts 2>&1 | tail -5"

  # ------------------------------------------------------------------
  # Bun (bun-installed qmd)
  # ------------------------------------------------------------------
  local BUN_BIN='$(mise where bun@latest)/bin'
  echo ""
  echo "=== Bun (bun install) ==="

  smoke_test_output "qmd shows help" "Usage:" \
    "export PATH=$BUN_BIN:$NODE_BIN:\$PATH; \$HOME/.bun/bin/qmd"

  smoke_test "qmd collection list" \
    "export PATH=$BUN_BIN:$NODE_BIN:\$PATH; \$HOME/.bun/bin/qmd collection list"

  smoke_test "qmd status" \
    "export PATH=$BUN_BIN:$NODE_BIN:\$PATH; \$HOME/.bun/bin/qmd status"

  smoke_test "sqlite-vec loads (bun)" \
    "export PATH=$BUN_BIN:\$PATH; bun -e \"
      const {openDatabase, loadSqliteVec} = await import('\$HOME/.bun/install/global/node_modules/@tobilu/qmd/dist/db.js');
      const db = openDatabase(':memory:');
      loadSqliteVec(db);
      const r = db.prepare('SELECT vec_version() as v').get();
      console.log('sqlite-vec', r.v);
      if (!r.v) process.exit(1);
    \""

  smoke_test "bun test store" \
    "export PATH=$BUN_BIN:\$PATH; cd /opt/qmd && bun test --preload ./src/test-preload.ts --timeout 30000 test/store.test.ts 2>&1 | tail -10"

  # ------------------------------------------------------------------
  echo ""
  echo "=== Results: $PASS passed, $FAIL failed ==="
  [[ $FAIL -eq 0 ]]
}

# Parse arguments
case "${1:-}" in
  --build)
    build_image
    ;;
  --shell)
    build_image
    echo "==> Dropping into container shell..."
    $CTR run --rm -it "$IMAGE" bash
    ;;
  --)
    shift
    run "$@"
    ;;
  *)
    build_image
    echo ""
    echo "==> Running smoke tests..."
    run_smoke_tests
    ;;
esac
