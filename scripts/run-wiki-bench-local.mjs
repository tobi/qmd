#!/usr/bin/env node
/**
 * wiki-bench v0 — private/local BM25 runner.
 *
 * Requires a local corpus via env (never vendors wiki bodies into the repo):
 *   QMD_WIKI_PATH       — checkout of tobi/wiki; uses $QMD_WIKI_PATH/wiki
 *   QMD_WIKI_BENCH_DOCS — markdown root with concepts|sources|entities/...
 *
 * Builds a temporary QMD index/collection named `wiki-bench`, runs BM25 against
 * src/bench/fixtures/wiki-v0.json via runBenchmark, prints metrics, and exits
 * non-zero if exact R@3 < 0.85, alias R@3 < 0.80, or exact+alias MRR < 0.70.
 *
 * Usage:
 *   QMD_WIKI_PATH=~/src/wiki node scripts/run-wiki-bench-local.mjs
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const helper = join(root, "scripts", "wiki-bench-local.ts");

function resolveRunner() {
  const tsx = join(root, "node_modules", ".bin", "tsx");
  if (existsSync(tsx)) {
    return { command: tsx, args: [helper], label: "tsx" };
  }
  const bun = spawnSync("bun", ["--version"], { encoding: "utf8" });
  if (bun.status === 0) {
    return { command: "bun", args: [helper], label: "bun" };
  }
  console.error(
    "wiki-bench-local: need local TypeScript runner (node_modules/.bin/tsx or bun).",
  );
  process.exit(1);
}

if (!existsSync(helper)) {
  console.error(`wiki-bench-local: missing helper ${helper}`);
  process.exit(1);
}

const runner = resolveRunner();
const result = spawnSync(runner.command, runner.args, {
  cwd: root,
  stdio: "inherit",
  env: process.env,
  shell: process.platform === "win32",
});

if (result.error) {
  console.error(`wiki-bench-local: failed to launch ${runner.label}:`, result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 1);
