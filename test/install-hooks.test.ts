import { describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const scriptSrc = join(repoRoot, "scripts", "install-hooks.mjs");
const prePushSrc = join(repoRoot, "scripts", "pre-push");

function runInstallHooks(cwd: string) {
  return spawnSync(process.execPath, [join(cwd, "scripts", "install-hooks.mjs")], {
    cwd,
    encoding: "utf8",
  });
}

function makeFixture(withGitHooks: boolean) {
  const root = mkdtempSync(join(tmpdir(), "qmd-install-hooks-"));
  mkdirSync(join(root, "scripts"), { recursive: true });
  writeFileSync(join(root, "scripts", "install-hooks.mjs"), readFileSync(scriptSrc));
  writeFileSync(join(root, "scripts", "pre-push"), readFileSync(prePushSrc));
  chmodSync(join(root, "scripts", "pre-push"), 0o755);
  if (withGitHooks) {
    mkdirSync(join(root, ".git", "hooks"), { recursive: true });
  }
  return root;
}

describe("scripts/install-hooks.mjs", () => {
  test("skips when .git/hooks is missing (non-git / packaged install)", () => {
    const root = makeFixture(false);
    try {
      const result = runInstallHooks(root);
      expect(result.status).toBe(0);
      expect(result.stdout).toMatch(/Not a git repository, skipping hook install/);
      expect(existsSync(join(root, ".git", "hooks", "pre-push"))).toBe(false);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("copies pre-push into .git/hooks when present", () => {
    const root = makeFixture(true);
    try {
      const result = runInstallHooks(root);
      expect(result.status).toBe(0);
      expect(result.stdout).toMatch(/Installed git hooks: pre-push/);
      const installed = join(root, ".git", "hooks", "pre-push");
      expect(existsSync(installed)).toBe(true);
      expect(readFileSync(installed, "utf8")).toBe(readFileSync(prePushSrc, "utf8"));
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("package.json prepare keeps Windows-safe hooks AND dist build", () => {
    const pkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8"));
    expect(pkg.scripts.prepare).toBe(
      "node scripts/install-hooks.mjs && node scripts/build.mjs",
    );
    // Must not use a POSIX shell test / .sh fragment (breaks cmd.exe).
    expect(pkg.scripts.prepare).not.toMatch(/\[ -d|\.sh/);
  });
});
