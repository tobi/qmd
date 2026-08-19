#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

// Mirror bin/qmd's safe Darwin residency default for test subprocesses. This
// must be set before a subprocess imports node-llama-cpp. Preserve either
// explicit operator choice so tests can still exercise both modes.
const darwinMetalEnv =
  process.platform === "darwin"
    && process.env.QMD_METAL_KEEP_RESIDENCY === undefined
    && process.env.GGML_METAL_NO_RESIDENCY === undefined
    ? { QMD_METAL_KEEP_RESIDENCY: "1" }
    : {};

function run(label, command, args, options = {}) {
  console.log(`==> ${label}`);
  const { env: extraEnv, ...spawnOptions } = options;
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
    env: { ...process.env, ...darwinMetalEnv, ...(extraEnv ?? {}) },
    ...spawnOptions,
  });
  if (result.status !== 0) {
    console.error(`Test task failed: ${label}`);
    process.exit(result.status ?? 1);
  }
}

run("TypeScript build typecheck", process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "tsconfig.build.json", "--noEmit"]);
run("Vitest suite under Node", process.execPath, [join(root, "node_modules", "vitest", "vitest.mjs"), "run", "--reporter=verbose", "--testTimeout", "60000", "test/"], { env: { CI: "true" } });
run("Bun test suite", "bun", ["test", "--timeout", "60000", "--preload", "./src/test-preload.ts", "test/"], { env: { CI: "true" } });
run("Package smoke", process.execPath, ["scripts/package-smoke.mjs"]);
