import { afterEach, describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const buildSrc = readFileSync(join(repoRoot, "scripts", "build.mjs"), "utf8");

const fixtures: string[] = [];

afterEach(() => {
  for (const dir of fixtures.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("scripts/build.mjs Windows execPath spaces (#681)", () => {
  test("does not enable a Windows cmd.exe shell for spawnSync", () => {
    // cmd.exe splits unquoted C:\\Program Files\\nodejs\\node.exe at the space.
    expect(buildSrc).not.toMatch(/shell:\s*process\.platform\s*===\s*["']win32["']/);
    expect(buildSrc).toMatch(/shell:\s*false/);
    expect(buildSrc).toMatch(/result\.error/);
  });

  test("spawnSync without a shell can run a binary whose path contains a space", () => {
    const root = mkdtempSync(join(tmpdir(), "qmd-build-execpath-"));
    fixtures.push(root);
    const spacedDir = join(root, "Program Files", "nodejs");
    mkdirSync(spacedDir, { recursive: true });
    const spacedBin = join(spacedDir, "node");
    // Stand-in for C:\\Program Files\\nodejs\\node.exe: a real executable
    // whose path contains a space. We own this file so chmod is allowed.
    writeFileSync(
      spacedBin,
      `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} "$@"\n`,
    );
    chmodSync(spacedBin, 0o755);

    const result = spawnSync(spacedBin, ["-e", "process.stdout.write('ok')"], {
      encoding: "utf8",
      shell: false,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stdout).toBe("ok");
  });

  test("run() treats a spawn error as a failed build, not status 0", () => {
    // Mirrors scripts/build.mjs run(): a missing binary must not look successful.
    const result = spawnSync(join(tmpdir(), "no-such-qmd-node-binary"), ["-e", "0"], {
      encoding: "utf8",
      shell: false,
    });
    expect(result.error).toBeDefined();
    expect(result.status).not.toBe(0);
    const exitCode = result.error || result.status !== 0 ? (result.status ?? 1) : 0;
    expect(exitCode).not.toBe(0);
  });
});
