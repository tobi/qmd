#!/usr/bin/env bash
# Build, pack, and smoke-test qmd in a container with mise + node + bun.
# Works with docker or podman (whichever is available).
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
echo "Using: $CTR"

# Build TypeScript
echo "==> Building TypeScript..."
npm run build --silent

# Pack tarball into test/ (the build context)
echo "==> Packing tarball..."
rm -f test/tobilu-qmd-*.tgz
TARBALL=$(npm pack --pack-destination test/ 2>/dev/null | tail -1)
echo "    $TARBALL"

# Build container image
echo "==> Building container..."
$CTR build -f test/Containerfile -t qmd-smoke test/

# Run smoke tests
echo "==> Running smoke tests..."
$CTR run --rm qmd-smoke

# Clean up tarball
rm -f test/tobilu-qmd-*.tgz
