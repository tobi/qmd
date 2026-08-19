/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to dispose of llama.cpp Metal resources before
 * the process exits — necessary on darwin to avoid the upstream rsets
 * destructor assertion (ggml-org/llama.cpp#22593, fix open as #22595).
 */
import { afterAll } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
