#!/usr/bin/env nix
#! nix shell --inputs-from .# nixpkgs#nodejs nixpkgs#jq nixpkgs#gnused nixpkgs#git --command bash

# Update nix/package-lock.json and npmDepsHash
# Usage: .github/actions/update-lockfile.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "Generating package-lock.json..."
npm install --package-lock-only

echo "Removing win32 dependency and copying to nix/..."
jq 'del(.packages["node_modules/sqlite-vec-win32-x64"])' package-lock.json > nix/package-lock.json
rm package-lock.json

echo "Calculating new npmDepsHash..."
if hash=$(nix build .#qmd 2>&1 | sed -n 's/.*got:[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -1) && [ -n "$hash" ]; then
  echo "Updating npmDepsHash to: $hash"
  sed -i "s|npmDepsHash = \".*\"|npmDepsHash = \"$hash\"|" nix/package.nix
  echo "Done. Verify with: nix build .#qmd"
else
  echo "Hash already up to date or build succeeded"
fi
