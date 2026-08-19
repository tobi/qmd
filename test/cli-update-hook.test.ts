import { describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const cliPath = join(repoRoot, "src", "cli", "qmd.ts");
const tsxLoader = join(repoRoot, "node_modules", "tsx", "dist", "loader.mjs");

/**
 * `process.exit()` is replaced by a throwing CliExit shim during owned CLI
 * dispatch. The update hook used to call it from inside a try/catch that
 * treated any throw as a spawn failure, so a hook exiting 2 was reported twice
 * and downgraded to exit 1.
 */
describe("qmd update hook exit code", () => {
  test.skipIf(process.platform === "win32")("a hook exiting 2 exits 2 with exactly one failure line", () => {
    const workDir = mkdtempSync(join(tmpdir(), "qmd-update-hook-"));
    try {
      const configDir = join(workDir, "config");
      const docsDir = join(workDir, "docs");
      mkdirSync(configDir, { recursive: true });
      mkdirSync(docsDir, { recursive: true });
      writeFileSync(join(docsDir, "note.md"), "# note\n");
      writeFileSync(
        join(configDir, "index.yml"),
        [
          "collections:",
          "  notes:",
          `    path: ${docsDir}`,
          '    pattern: "**/*.md"',
          '    update: "exit 2"',
          "",
        ].join("\n"),
      );

      const result = spawnSync(
        process.execPath,
        [
          ...(process.versions.bun ? [] : ["--import", tsxLoader]),
          cliPath,
          "update",
        ],
        {
          cwd: workDir,
          env: {
            ...process.env,
            QMD_CONFIG_DIR: configDir,
            XDG_CACHE_HOME: join(workDir, "cache"),
            CI: "true",
            NO_COLOR: "1",
          },
          encoding: "utf8",
          timeout: 120_000,
        },
      );

      const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
      expect(result.status, output).toBe(2);
      expect(output.match(/Update command failed/g)?.length, output).toBe(1);
      expect(output).toContain("exit code 2");
    } finally {
      rmSync(workDir, { recursive: true, force: true });
    }
  }, 150_000);
});
