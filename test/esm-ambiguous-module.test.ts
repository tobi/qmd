import { describe, expect, test } from "vitest";
import { execFileSync } from "child_process";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("Node ESM entrypoints", () => {
  test("CLI --index path normalizes via setIndexName/setConfigIndexName under Node 22+", () => {
    execFileSync(process.execPath, ["scripts/build.mjs"], {
      cwd: repoRoot,
      encoding: "utf-8",
      stdio: "pipe",
    });

    const indexPath = join(mkdtempSync(join(tmpdir(), "qmd-index-")), "nested", "idx");
    const output = execFileSync(process.execPath, ["dist/cli/qmd.js", "--index", indexPath, "--version"], {
      cwd: repoRoot,
      encoding: "utf-8",
      stdio: "pipe",
    });

    expect(output).toContain("qmd ");
  }, 120_000);
});
