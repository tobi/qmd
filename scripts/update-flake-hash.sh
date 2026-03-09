#!/usr/bin/env bash
# Recompute the Bun dependency hash in flake.nix after bun.lock changes.
#
# Usage: ./scripts/update-flake-hash.sh
#
# How it works:
#   1. Temporarily sets the outputHash to a known-wrong value
#   2. Runs `nix build`, which fails but prints the correct hash
#   3. Patches the correct hash back into flake.nix
#
# Requires: nix with flakes enabled, sed, grep

set -uo pipefail

FLAKE="flake.nix"
FAKE_HASH="sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

# 1. Replace current hash with a known-wrong one
sed -i 's|outputHash = "sha256-[^"]*";|outputHash = "'"$FAKE_HASH"'";|' "$FLAKE"

echo "Computing dependency hash (this will download deps in a sandbox)..."

# 2. Build and capture the expected hash from the error output.
#    nix build will exit non-zero (hash mismatch), which is expected.
BUILD_OUTPUT=$(nix build 2>&1 || true)
HASH=$(echo "$BUILD_OUTPUT" | grep -oP 'got:\s+\Ksha256-[A-Za-z0-9+/]+=*' | head -1)

if [ -z "$HASH" ]; then
	echo "ERROR: could not extract hash from nix build output." >&2
	echo "" >&2
	echo "nix build output:" >&2
	echo "$BUILD_OUTPUT" >&2
	# Restore the fake hash so it's obvious something went wrong
	exit 1
fi

# 3. Write the real hash back
sed -i 's|outputHash = "'"$FAKE_HASH"'";|outputHash = "'"$HASH"'";|' "$FLAKE"

echo "Updated flake.nix with hash: $HASH"
