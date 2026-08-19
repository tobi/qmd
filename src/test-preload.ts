/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to dispose of llama.cpp Metal resources before
 * the process exits — necessary on darwin to avoid the upstream rsets
 * destructor assertion (ggml-org/llama.cpp#22593, fix open as #22595).
 *
 * The runner sets `QMD_METAL_KEEP_RESIDENCY=1` before bun/node starts so
 * affected Apple Silicon runtimes do not enter the crashing no-residency
 * path. The preload still performs orderly resource disposal after tests.
 */
import { afterAll } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
