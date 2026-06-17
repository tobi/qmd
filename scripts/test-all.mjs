#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

// Mirror bin/qmd's darwin Metal residency mitigation for test subprocesses.
// libggml-metal asserts on a non-empty residency set during its static
// destructor (ggml-org/llama.cpp#22593, fix open as #22595) and dumps a
// multi-kB backtrace at process exit even when tests pass. The env var must
// be set BEFORE the subprocess starts because libggml-metal reads it via
// libc getenv at module-load time. Opt out with QMD_METAL_KEEP_RESIDENCY=1.
const darwinMetalEnv =
  process.platform === "darwin" && process.env.QMD_METAL_KEEP_RESIDENCY !== "1"
    ? { GGML_METAL_NO_RESIDENCY: "1" }
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
