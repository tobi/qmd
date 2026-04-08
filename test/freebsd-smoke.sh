#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: test/freebsd-smoke.sh [--quick|--full] [--keep-temp]

  --quick      BM25 + sqlite-vec validation only
  --full       Search + embed + lifecycle/maintenance validation (requires working node-llama-cpp; default)
  --keep-temp  Keep the temporary corpus and state directory
EOF
}

MODE="full"
KEEP_TEMP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)
      MODE="quick"
      ;;
    --full)
      MODE="full"
      ;;
    --keep-temp)
      KEEP_TEMP=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [ "$(uname -s)" != "FreeBSD" ]; then
  echo "Error: test/freebsd-smoke.sh is intended for FreeBSD hosts." >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' was not found in PATH." >&2
    exit 1
  fi
}

for cmd in node corepack git cmake ninja python3 gmake bash sqlite3 envsubst; do
  require_cmd "$cmd"
done

if [ ! -d node_modules ]; then
  echo "Error: node_modules is missing. Run 'corepack pnpm install --frozen-lockfile' first." >&2
  exit 1
fi

resolve_sqlite_vec_path() {
  if [ -n "${QMD_SQLITE_VEC_PATH:-}" ]; then
    if [ ! -f "${QMD_SQLITE_VEC_PATH}" ]; then
      echo "Error: QMD_SQLITE_VEC_PATH points to a missing file: ${QMD_SQLITE_VEC_PATH}" >&2
      exit 1
    fi
    printf '%s\n' "${QMD_SQLITE_VEC_PATH}"
    return 0
  fi

  if [ -f "../sqlite-vec/dist/vec0.so" ]; then
    printf '%s\n' "$(cd ../sqlite-vec && pwd)/dist/vec0.so"
    return 0
  fi

  if [ -d "../sqlite-vec" ]; then
    echo "==> Building ../sqlite-vec/dist/vec0.so" >&2
    (
      cd ../sqlite-vec
      gmake sqlite-vec.h loadable
    ) >&2
    if [ -f "../sqlite-vec/dist/vec0.so" ]; then
      printf '%s\n' "$(cd ../sqlite-vec && pwd)/dist/vec0.so"
      return 0
    fi
  fi

  cat >&2 <<'EOF'
Error: could not locate vec0.so.

Set QMD_SQLITE_VEC_PATH=/absolute/path/to/vec0.so, or clone sqlite-vec next to qmd:

  git clone https://github.com/asg017/sqlite-vec ../sqlite-vec
  cd ../sqlite-vec
  gmake sqlite-vec.h loadable
EOF
  exit 1
}

SQLITE_VEC_PATH="$(resolve_sqlite_vec_path)"

TMP_ROOT="$(mktemp -d -t qmd-freebsd-smoke)"
STATE_DIR="${TMP_ROOT}/state"
CORPUS_DIR="${TMP_ROOT}/notes"
mkdir -p "${STATE_DIR}/config" "${CORPUS_DIR}"

cleanup() {
  if [ "${KEEP_TEMP}" -eq 0 ]; then
    rm -rf "${TMP_ROOT}"
  fi
}

trap cleanup EXIT INT TERM

cat > "${CORPUS_DIR}/auth.md" <<'EOF'
# Authentication Flow

Use bearer tokens for API authentication.
Rotate secrets every 90 days.
EOF

cat > "${CORPUS_DIR}/db.md" <<'EOF'
# Database Notes

Use connection pooling to avoid timeout spikes.
Prepared statements prevent SQL injection.
EOF

mkdir -p "${CORPUS_DIR}/security"
cat > "${CORPUS_DIR}/security/tokens.md" <<'EOF'
# Token Handling

Store API tokens in a dedicated secrets manager.
Never commit production credentials.
EOF

ARCHIVE_DIR="${TMP_ROOT}/archive"
mkdir -p "${ARCHIVE_DIR}"
cat > "${ARCHIVE_DIR}/legacy.md" <<'EOF'
# Legacy Notes

This collection is only used to validate collection removal.
EOF

INDEX_DB="${STATE_DIR}/smoke.sqlite"
CONFIG_DIR="${STATE_DIR}/config"

run_qmd() {
  env \
    INDEX_PATH="${INDEX_DB}" \
    QMD_CONFIG_DIR="${CONFIG_DIR}" \
    QMD_SQLITE_VEC_PATH="${SQLITE_VEC_PATH}" \
    QMD_LLAMA_GPU=false \
    corepack pnpm qmd "$@"
}

run_qmd_capture() {
  set +e
  RUN_OUTPUT="$(
    env \
      INDEX_PATH="${INDEX_DB}" \
      QMD_CONFIG_DIR="${CONFIG_DIR}" \
      QMD_SQLITE_VEC_PATH="${SQLITE_VEC_PATH}" \
      QMD_LLAMA_GPU=false \
      corepack pnpm qmd "$@" 2>&1
  )"
  RUN_STATUS=$?
  set -e
}

run_qmd_expect_success() {
  run_qmd_capture "$@"
  printf '%s\n' "$RUN_OUTPUT"
  if [ "${RUN_STATUS}" -ne 0 ]; then
    echo "Command failed: corepack pnpm qmd $*" >&2
    exit "${RUN_STATUS}"
  fi
}

assert_contains() {
  haystack="$1"
  needle="$2"
  if ! printf '%s\n' "$haystack" | grep -F "$needle" >/dev/null 2>&1; then
    echo "Expected output to contain: $needle" >&2
    exit 1
  fi
}

assert_not_contains() {
  haystack="$1"
  needle="$2"
  if printf '%s\n' "$haystack" | grep -F "$needle" >/dev/null 2>&1; then
    echo "Expected output not to contain: $needle" >&2
    exit 1
  fi
}

assert_status() {
  actual="$1"
  expected="$2"
  if [ "$actual" -ne "$expected" ]; then
    echo "Expected exit status $expected, got $actual" >&2
    exit 1
  fi
}

echo "==> Using vec0.so: ${SQLITE_VEC_PATH}"
echo "==> Temporary root: ${TMP_ROOT}"

run_qmd_expect_success collection add "${CORPUS_DIR}" --name notes
assert_contains "$RUN_OUTPUT" "Collection 'notes' created successfully"

run_qmd_expect_success status
assert_contains "$RUN_OUTPUT" "Collections"

run_qmd_expect_success search authentication
assert_contains "$RUN_OUTPUT" "qmd://notes/auth.md"

run_qmd_expect_success ls notes
assert_contains "$RUN_OUTPUT" "qmd://notes/auth.md"
assert_contains "$RUN_OUTPUT" "qmd://notes/security/tokens.md"

run_qmd_expect_success get qmd://notes/auth.md
assert_contains "$RUN_OUTPUT" "Authentication Flow"
assert_contains "$RUN_OUTPUT" "Rotate secrets every 90 days."

OUT="$(sqlite3 :memory: ".load ${SQLITE_VEC_PATH}" "select vec_version();" 2>&1)"
printf '%s\n' "$OUT"
assert_contains "$OUT" "v0."

if [ "${MODE}" = "full" ]; then
  run_qmd_expect_success context add qmd://notes/ "Engineering knowledge base"
  assert_contains "$RUN_OUTPUT" "Added context for: qmd://notes/"

  run_qmd_expect_success context list
  assert_contains "$RUN_OUTPUT" "Configured Contexts"
  assert_contains "$RUN_OUTPUT" "Engineering knowledge base"

  run_qmd_expect_success get qmd://notes/auth.md
  assert_contains "$RUN_OUTPUT" "Folder Context: Engineering knowledge base"

  run_qmd_expect_success embed
  assert_contains "$RUN_OUTPUT" "Embedded"

  run_qmd_expect_success vsearch "secure api authentication"
  assert_contains "$RUN_OUTPUT" "qmd://notes/auth.md"

  run_qmd_expect_success query "secure api authentication" --explain
  assert_contains "$RUN_OUTPUT" "qmd://notes/auth.md"
  assert_contains "$RUN_OUTPUT" "Explain:"

  OUT="$(
    env \
      INDEX_PATH="${INDEX_DB}" \
      QMD_CONFIG_DIR="${CONFIG_DIR}" \
      QMD_SQLITE_VEC_PATH="${SQLITE_VEC_PATH}" \
      QMD_LLAMA_GPU=false \
      QMD_STATUS_LLM_PROBE=1 \
      corepack pnpm qmd status 2>&1
  )"
  printf '%s\n' "$OUT"
  assert_contains "$OUT" "Device"

  run_qmd_expect_success collection update-cmd notes "printf 'update-hook-ran\n'"
  assert_contains "$RUN_OUTPUT" "Set update command for 'notes'"

  cat > "${CORPUS_DIR}/db.md" <<'EOF'
# Database Notes

Use connection pooling to avoid timeout spikes.
Prepared statements prevent SQL injection.
Enable WAL mode for better concurrent writes.
EOF

  rm -f "${CORPUS_DIR}/auth.md"

  cat > "${CORPUS_DIR}/ops.md" <<'EOF'
# Operations Notes

Monitoring alerts should page the on-call engineer.
Runbook links belong next to each alert.
EOF

  run_qmd_expect_success update
  assert_contains "$RUN_OUTPUT" "Running update command: printf 'update-hook-ran"
  assert_contains "$RUN_OUTPUT" "update-hook-ran"
  assert_contains "$RUN_OUTPUT" "1 new, 1 updated, 1 unchanged, 1 removed"
  assert_contains "$RUN_OUTPUT" "Run 'qmd embed' to update embeddings"

  run_qmd_capture get qmd://notes/auth.md
  printf '%s\n' "$RUN_OUTPUT"
  assert_status "$RUN_STATUS" 1
  assert_contains "$RUN_OUTPUT" "Document not found"

  run_qmd_expect_success search monitoring
  assert_contains "$RUN_OUTPUT" "qmd://notes/ops.md"
  assert_not_contains "$RUN_OUTPUT" "qmd://notes/auth.md"

  run_qmd_expect_success cleanup
  assert_contains "$RUN_OUTPUT" "Cleared"
  assert_contains "$RUN_OUTPUT" "Removed"
  assert_contains "$RUN_OUTPUT" "orphaned embedding chunks"
  assert_contains "$RUN_OUTPUT" "Database vacuumed"

  run_qmd_expect_success embed
  assert_contains "$RUN_OUTPUT" "Embedded"

  run_qmd_expect_success vsearch "on-call monitoring alerts"
  assert_contains "$RUN_OUTPUT" "qmd://notes/ops.md"

  run_qmd_expect_success collection rename notes knowledge
  assert_contains "$RUN_OUTPUT" "Renamed collection 'notes' to 'knowledge'"
  assert_contains "$RUN_OUTPUT" "qmd://knowledge/"

  run_qmd_expect_success collection list
  assert_contains "$RUN_OUTPUT" "qmd://knowledge/"
  assert_not_contains "$RUN_OUTPUT" "qmd://notes/"

  run_qmd_expect_success get qmd://knowledge/db.md
  assert_contains "$RUN_OUTPUT" "Folder Context: Engineering knowledge base"
  assert_contains "$RUN_OUTPUT" "Enable WAL mode for better concurrent writes."

  run_qmd_expect_success update
  assert_contains "$RUN_OUTPUT" "[1/1]"
  assert_contains "$RUN_OUTPUT" "knowledge"
  assert_contains "$RUN_OUTPUT" "update-hook-ran"

  run_qmd_expect_success context rm qmd://knowledge/
  assert_contains "$RUN_OUTPUT" "Removed context for: qmd://knowledge/"

  run_qmd_expect_success context list
  assert_contains "$RUN_OUTPUT" "No contexts configured"

  run_qmd_expect_success collection add "${ARCHIVE_DIR}" --name archive
  assert_contains "$RUN_OUTPUT" "Collection 'archive' created successfully"

  run_qmd_expect_success collection list
  assert_contains "$RUN_OUTPUT" "qmd://archive/"
  assert_contains "$RUN_OUTPUT" "qmd://knowledge/"

  run_qmd_expect_success collection remove archive
  assert_contains "$RUN_OUTPUT" "Removed collection 'archive'"
  assert_contains "$RUN_OUTPUT" "Deleted 1 documents"

  run_qmd_expect_success collection list
  assert_contains "$RUN_OUTPUT" "qmd://knowledge/"
  assert_not_contains "$RUN_OUTPUT" "qmd://archive/"
fi

echo "==> FreeBSD smoke passed (${MODE})"

if [ "${KEEP_TEMP}" -eq 1 ]; then
  echo "==> Kept temp dir: ${TMP_ROOT}"
fi
