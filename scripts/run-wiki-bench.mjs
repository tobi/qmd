#!/usr/bin/env node
/**
 * One-shot wiki-bench runner: index vendored docs into a temp DB and run
 * runBenchmark with the bm25 (lex) backend only.
 *
 * Usage:
 *   node scripts/run-wiki-bench.mjs
 *   bun scripts/run-wiki-bench.mjs
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const runner = join(root, "scripts", "_run-wiki-bench.ts");

if (!existsSync(runner)) {
  console.error(`Missing ${runner}`);
  process.exit(1);
}

const bun = spawnSync("bun", ["--version"], { encoding: "utf8" });
const useBun = bun.status === 0;
const cmd = useBun ? "bun" : "npx";
const args = useBun ? [runner] : ["tsx", runner];

const result = spawnSync(cmd, args, {
  cwd: root,
  stdio: "inherit",
  env: process.env,
});
process.exit(result.status ?? 1);
