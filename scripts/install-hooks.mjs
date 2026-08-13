#!/usr/bin/env node
// Self-installing git hooks for qmd
// Called from the package.json "prepare" script after install
import { chmodSync, copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL("..", import.meta.url)));
const hooksDir = join(root, ".git", "hooks");

if (!existsSync(hooksDir)) {
  console.log("Not a git repository, skipping hook install");
  process.exit(0);
}

copyFileSync(join(root, "scripts", "pre-push"), join(hooksDir, "pre-push"));
chmodSync(join(hooksDir, "pre-push"), 0o755);
console.log("Installed git hooks: pre-push");
