/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to dispose of llama.cpp Metal resources before
 * the process exits — necessary on darwin to avoid the upstream rsets
 * destructor assertion (ggml-org/llama.cpp#22593, fix open as #22595).
 *
 * The runner-level mitigation `GGML_METAL_NO_RESIDENCY=1` must be set
 * BEFORE bun/node starts (libggml-metal reads it via libc getenv at
 * module load). Bun does not propagate `process.env` writes to libc
 * setenv, so setting it from here would be a no-op for the native
 * binding. The env var is injected by:
 *   - bin/qmd for production CLI runs
 *   - scripts/test-all.mjs for `npm test`
 *   - package.json test:bun / test:unit scripts for direct invocation
 * See CLAUDE.md for invoking `bun test` manually on darwin.
 */
import { afterAll } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
