import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const installer = join(repoRoot, "scripts", "install-hooks.mjs");

describe("scripts/install-hooks.mjs", () => {
  let tempRoot: string;
  beforeEach(() => { tempRoot = mkdtempSync(join(tmpdir(), "qmd-install-hooks-")); });
  afterEach(() => { rmSync(tempRoot, { recursive: true, force: true }); });

  test("skips cleanly when .git/hooks is absent", () => {
    const fakeRoot = join(tempRoot, "pkg");
    mkdirSync(join(fakeRoot, "scripts"), { recursive: true });
    writeFileSync(join(fakeRoot, "scripts", "pre-push"), "#!/bin/sh\necho hook\n");
    writeFileSync(join(fakeRoot, "scripts", "install-hooks.mjs"), readFileSync(installer, "utf8"));
    const result = spawnSync(process.execPath, [join(fakeRoot, "scripts", "install-hooks.mjs")], { cwd: fakeRoot, encoding: "utf8" });
    expect(result.status).toBe(0);
    expect(result.stdout).toMatch(/Not a git repository, skipping hook install/);
    expect(existsSync(join(fakeRoot, ".git", "hooks", "pre-push"))).toBe(false);
  });

  test("copies pre-push into .git/hooks and makes it executable", () => {
    const fakeRoot = join(tempRoot, "repo");
    mkdirSync(join(fakeRoot, "scripts"), { recursive: true });
    mkdirSync(join(fakeRoot, ".git", "hooks"), { recursive: true });
    writeFileSync(join(fakeRoot, "scripts", "pre-push"), "#!/bin/sh\necho pre-push-ok\n");
    writeFileSync(join(fakeRoot, "scripts", "install-hooks.mjs"), readFileSync(installer, "utf8"));
    const result = spawnSync(process.execPath, [join(fakeRoot, "scripts", "install-hooks.mjs")], { cwd: fakeRoot, encoding: "utf8" });
    expect(result.status).toBe(0);
    expect(result.stdout).toMatch(/Installed git hooks: pre-push/);
    const dest = join(fakeRoot, ".git", "hooks", "pre-push");
    expect(readFileSync(dest, "utf8")).toContain("pre-push-ok");
    if (process.platform !== "win32") {
      expect(statSync(dest).mode & 0o111).toBeTruthy();
    }
  });
});

describe("package.json prepare script", () => {
  test("is Windows-safe Node and still builds dist", () => {
    const pkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8"));
    const prepare: string = pkg.scripts.prepare;
    expect(prepare.includes("[ -d")).toBe(false);
    expect(prepare.includes("install-hooks.sh")).toBe(false);
    expect(prepare.includes("node scripts/install-hooks.mjs")).toBe(true);
    expect(prepare.includes("node scripts/build.mjs")).toBe(true);
  });
});
