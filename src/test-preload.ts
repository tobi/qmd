/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to properly dispose of llama.cpp Metal
 * resources before the process exits, avoiding GGML_ASSERT failures.
 */
import { afterAll } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
