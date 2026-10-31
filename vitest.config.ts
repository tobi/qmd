import { defineConfig } from "vitest/config";

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
  },
});
