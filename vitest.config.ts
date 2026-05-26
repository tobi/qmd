import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    testTimeout: 30000,
    fileParallelism: false,
    include: ["test/**/*.test.ts"],
    // Unset shell-provided remote-server config so tests stay hermetic and
    // don't accidentally route through a developer's local `qmd serve`.
    env: {
      QMD_REMOTE_URL: "",
    },
  },
});
