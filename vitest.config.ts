import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    testTimeout: process.platform === "win32" ? 60000 : 30000,
    include: ["test/**/*.test.ts"],
    // Native LLM integration tests share process-global state and are flaky on
    // Windows when files run in parallel.
    fileParallelism: process.platform !== "win32",
  },
});
