/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to properly dispose of llama.cpp Metal
 * resources before the process exits, avoiding GGML_ASSERT failures.
 */
import { afterAll, setDefaultTimeout } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Match vitest's testTimeout (default 5s is too short for CLI subprocess tests)
setDefaultTimeout(30_000);

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
