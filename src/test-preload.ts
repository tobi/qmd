/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Properly disposes llama.cpp Metal resources before the process exits,
 * avoiding GGML_ASSERT failures.
 */
import { afterAll } from "vitest";
import { disposeDefaultLlamaCpp } from "./llm.js";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
