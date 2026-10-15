import { defineConfig } from "vitest/config";

// When QMD_REMOTE_URL is set in the environment, tests route LLM operations
// through that `qmd serve` instance, sharing its resident models instead of
// spawning their own LlamaCpp (which would collide with serve on the GPU).
// Unset → hermetic local LlamaCpp. CI environments leave it unset.
export default defineConfig({
  test: {
    testTimeout: 30000,
    fileParallelism: false,
    include: ["test/**/*.test.ts"],
    typecheck: {
      enabled: true,
      include: ["test/**/*.test-d.ts"],
      // Judge only the .test-d.ts assertions; tsc covers src/ separately.
      ignoreSourceErrors: true,
    },
    setupFiles: ["./src/test-setup-remote.ts"],
    env: {
      QMD_REMOTE_URL: process.env.QMD_REMOTE_URL ?? "",
    },
  },
});
