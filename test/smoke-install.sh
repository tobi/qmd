#!/usr/bin/env bash
# Build a clean container image from the current checkout package and exercise
# install/runtime scenarios under npm, npx, and Bun. Supports optional qmd embed
# and GPU probes, but keeps those expensive/device-specific checks opt-in.
#
# Usage:
#   test/smoke-install.sh                         # build + run default smoke scenarios
#   test/smoke-install.sh --build                 # build image only
#   test/smoke-install.sh --shell                 # drop into container shell
#   test/smoke-install.sh --scenario node         # run one scenario (node|npx|bun|all)
#   test/smoke-install.sh --with-embed            # also run tiny qmd embed smoke tests
#   test/smoke-install.sh --with-gpu              # also probe GPU in doctor/embed scenarios
#   QMD_SMOKE_GPU_BACKEND=cuda|vulkan|auto        # backend for --with-gpu (default: auto)
#   test/smoke-install.sh --no-build              # reuse existing image
#   test/smoke-install.sh -- CMD...               # run arbitrary command in container
#
# GPU notes:
#   Docker uses:  --gpus all
#   Podman uses:  --device nvidia.com/gpu=all
#   If your podman setup uses a different CDI device name, override with:
#     QMD_SMOKE_GPU_ARGS='--device nvidia.com/gpu=all' test/smoke-install.sh --with-gpu
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v podman &>/dev/null; then
  CTR=podman
elif command -v docker &>/dev/null; then
  CTR=docker
else
  echo "Error: neither podman nor docker found" >&2
  exit 1
fi

IMAGE=${QMD_SMOKE_IMAGE:-qmd-smoke}
SCENARIO=all
DO_BUILD=1
WITH_EMBED=0
WITH_GPU=0
GPU_BACKEND=${QMD_SMOKE_GPU_BACKEND:-auto}
declare -a ARBITRARY_CMD=()

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) DO_BUILD=1; BUILD_ONLY=1; shift ;;
    --no-build) DO_BUILD=0; shift ;;
    --shell) SHELL_ONLY=1; shift ;;
    --scenario) SCENARIO="${2:-}"; shift 2 ;;
    --with-embed) WITH_EMBED=1; shift ;;
    --with-gpu) WITH_GPU=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; ARBITRARY_CMD=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

BUILD_ONLY=${BUILD_ONLY:-0}
SHELL_ONLY=${SHELL_ONLY:-0}

gpu_args() {
  if [[ $WITH_GPU -ne 1 ]]; then return 0; fi
  if [[ -n "${QMD_SMOKE_GPU_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    echo ${QMD_SMOKE_GPU_ARGS}
    return 0
  fi
  case "$CTR" in
    docker) echo "--gpus all" ;;
    podman) echo "--device nvidia.com/gpu=all" ;;
  esac
}

build_image() {
  echo "==> Building TypeScript package..."
  npm run build --silent

  echo "==> Packing tarball..."
  rm -f test/tobilu-qmd-*.tgz
  TARBALL=$(npm pack --pack-destination test/ 2>/dev/null | tail -1)
  echo "    $TARBALL"

  echo "==> Preparing container test project..."
  rm -rf test/test-src
  mkdir -p test/test-src/test
  cp -r src test/test-src/
  cp -r dist test/test-src/
  cp -r test/*.test.ts test/test-src/test/
  cp package.json tsconfig.json tsconfig.build.json test/test-src/

  echo "==> Building container image ($CTR): $IMAGE"
  $CTR build -f test/Containerfile -t "$IMAGE" test/

  rm -f test/tobilu-qmd-*.tgz
  rm -rf test/test-src
  echo "==> Image ready: $IMAGE"
}

run() {
  local args=()
  # Intentionally word-split GPU args: container CLIs expect separate flags.
  # shellcheck disable=SC2206
  args=( $(gpu_args) )
  $CTR run --rm "${args[@]}" "$IMAGE" bash -lc "$*"
}

PASS=0
FAIL=0

ok()   { printf "  %-58s OK\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "  %-58s FAIL\n" "$1"; FAIL=$((FAIL + 1)); echo "$2" | sed 's/^/    /'; }

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
  if grep -q "$expect" <<<"$out"; then
    ok "$label"
  else
    fail "$label" "$out"
  fi
}

fixture_setup='rm -rf /tmp/qmd-fixture /tmp/qmd-cache /tmp/qmd-config /tmp/qmd-models; mkdir -p /tmp/qmd-fixture; printf "# Smoke Doc\n\nGPU and CPU embedding smoke test.\n" > /tmp/qmd-fixture/doc.md; export XDG_CACHE_HOME=/tmp/qmd-cache QMD_CONFIG_DIR=/tmp/qmd-config'

gpu_env() {
  case "$GPU_BACKEND" in
    auto|"") echo "" ;;
    cuda|vulkan|metal) echo "QMD_LLAMA_GPU=$GPU_BACKEND" ;;
    *) echo "Unsupported QMD_SMOKE_GPU_BACKEND=$GPU_BACKEND" >&2; exit 1 ;;
  esac
}

run_doctor_smoke() {
  local label="$1" bin="$2" extra_env="${3:-}"
  smoke_test_output "$label doctor" "QMD Doctor" \
    "$fixture_setup; $extra_env $bin doctor"
}

run_collection_smoke() {
  local label="$1" bin="$2" extra_env="${3:-}"
  smoke_test "$label collection add/list/status" \
    "$fixture_setup; cd /tmp/qmd-fixture; $extra_env $bin collection add . --name smoke; $extra_env $bin collection list; $extra_env $bin status"
}

run_embed_smoke() {
  local label="$1" bin="$2" extra_env="${3:-}"
  [[ $WITH_EMBED -eq 1 ]] || return 0
  smoke_test "$label qmd embed tiny fixture" \
    "$fixture_setup; cd /tmp/qmd-fixture; $extra_env $bin collection add . --name smoke; $extra_env $bin embed --max-docs-per-batch 1 --max-batch-mb 1; $extra_env $bin doctor"
}

run_runtime_matrix() {
  local label="$1" bin="$2" path_env="$3"
  smoke_test_output "$label qmd help" "Usage:" "$path_env; $bin"
  run_doctor_smoke "$label auto" "$path_env; $bin"
  run_doctor_smoke "$label force-cpu" "$path_env; $bin" "QMD_FORCE_CPU=1"
  run_collection_smoke "$label" "$path_env; $bin" "QMD_FORCE_CPU=1"
  run_embed_smoke "$label force-cpu" "$path_env; $bin" "QMD_FORCE_CPU=1"
  run_embed_smoke "$label auto" "$path_env; $bin"
  if [[ $WITH_GPU -eq 1 ]]; then
    local ge
    ge=$(gpu_env)
    run_doctor_smoke "$label gpu-$GPU_BACKEND" "$path_env; $bin" "$ge"
    run_embed_smoke "$label gpu-$GPU_BACKEND" "$path_env; $bin" "$ge"
  fi
}

run_node_scenario() {
  local NODE_BIN='$(mise where node@latest)/bin'
  local bin='qmd'
  echo "=== Node: npm install -g packed tarball ==="
  run_runtime_matrix "node" "$bin" "export PATH=$NODE_BIN:\$PATH"
  smoke_test "node sqlite-vec loads" \
    "export PATH=$NODE_BIN:\$PATH; NPM_GLOBAL=\$(npm root -g); node -e \"
      const {openDatabase, loadSqliteVec} = await import('\\$NPM_GLOBAL/@tobilu/qmd/dist/db.js');
      const db = openDatabase(':memory:');
      loadSqliteVec(db);
      const r = db.prepare('SELECT vec_version() as v').get();
      console.log('sqlite-vec', r.v);
      if (!r.v) process.exit(1);
    \""
  smoke_test "node vitest store subset" \
    "export PATH=$NODE_BIN:\$PATH; cd /opt/qmd && npx vitest run --reporter=verbose test/store.test.ts 2>&1 | tail -5"
}

run_npx_scenario() {
  local NODE_BIN='$(mise where node@latest)/bin'
  local bin='npm exec --yes --package /tmp/tobilu-qmd.tgz -- qmd'
  echo "=== Node: npm exec/npx-style packed tarball ==="
  run_runtime_matrix "npx-style" "$bin" "export PATH=$NODE_BIN:\$PATH"
}

run_bun_scenario() {
  local NODE_BIN='$(mise where node@latest)/bin'
  local BUN_BIN='$(mise where bun@latest)/bin'
  local bin='$HOME/.bun/bin/qmd'
  echo "=== Bun: bun install -g packed tarball ==="
  run_runtime_matrix "bun" "$bin" "export PATH=$BUN_BIN:$NODE_BIN:\$PATH"
  smoke_test "bun sqlite-vec loads" \
    "export PATH=$BUN_BIN:\$PATH; bun -e \"
      const {openDatabase, loadSqliteVec} = await import('\\$HOME/.bun/install/global/node_modules/@tobilu/qmd/dist/db.js');
      const db = openDatabase(':memory:');
      loadSqliteVec(db);
      const r = db.prepare('SELECT vec_version() as v').get();
      console.log('sqlite-vec', r.v);
      if (!r.v) process.exit(1);
    \""
  smoke_test "bun test store subset" \
    "export PATH=$BUN_BIN:\$PATH; cd /opt/qmd && bun test --preload ./src/test-preload.ts --timeout 30000 test/store.test.ts 2>&1 | tail -10"
}

run_smoke_tests() {
  case "$SCENARIO" in
    node) run_node_scenario ;;
    npx) run_npx_scenario ;;
    bun) run_bun_scenario ;;
    all) run_node_scenario; echo; run_npx_scenario; echo; run_bun_scenario ;;
    *) echo "Unknown scenario: $SCENARIO" >&2; exit 1 ;;
  esac
  echo ""
  echo "=== Results: $PASS passed, $FAIL failed ==="
  [[ $FAIL -eq 0 ]]
}

if [[ $DO_BUILD -eq 1 ]]; then
  build_image
fi

if [[ ${#ARBITRARY_CMD[@]} -gt 0 ]]; then
  run "${ARBITRARY_CMD[*]}"
  exit $?
fi

if [[ $BUILD_ONLY -eq 1 ]]; then
  exit 0
fi

if [[ $SHELL_ONLY -eq 1 ]]; then
  echo "==> Dropping into container shell..."
  # shellcheck disable=SC2206
  gpu=( $(gpu_args) )
  $CTR run --rm -it "${gpu[@]}" "$IMAGE" bash
  exit $?
fi

echo ""
echo "==> Running smoke tests..."
run_smoke_tests
