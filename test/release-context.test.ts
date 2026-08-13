import { afterEach, describe, expect, test } from "vitest";
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
const scriptSrc = join(repoRoot, "skills", "release", "scripts", "release-context.sh");
const installHooksSrc = join(repoRoot, "skills", "release", "scripts", "install-hooks.sh");
const prePushSrc = join(repoRoot, "scripts", "pre-push");
const skillSrc = join(repoRoot, "skills", "release", "SKILL.md");

const fixtures: string[] = [];

afterEach(() => {
  for (const dir of fixtures.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function git(cwd: string, args: string[]) {
  const result = spawnSync("git", ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", ...args], {
    cwd,
    encoding: "utf8",
    env: {
      ...process.env,
      GIT_AUTHOR_NAME: "qmd-test",
      GIT_AUTHOR_EMAIL: "qmd-test@example.com",
      GIT_COMMITTER_NAME: "qmd-test",
      GIT_COMMITTER_EMAIL: "qmd-test@example.com",
      GIT_EDITOR: "true",
    },
  });
  if (result.status !== 0) {
    throw new Error(`git ${args.join(" ")} failed: ${result.stderr || result.stdout}`);
  }
  return result;
}

function makeReleaseFixture(opts?: { dirty?: boolean }) {
  const root = mkdtempSync(join(tmpdir(), "qmd-release-context-"));
  fixtures.push(root);

  mkdirSync(join(root, "scripts"), { recursive: true });
  mkdirSync(join(root, "skills", "release", "scripts"), { recursive: true });

  writeFileSync(join(root, "skills", "release", "scripts", "release-context.sh"), readFileSync(scriptSrc));
  writeFileSync(join(root, "skills", "release", "scripts", "install-hooks.sh"), readFileSync(installHooksSrc));
  writeFileSync(join(root, "scripts", "pre-push"), readFileSync(prePushSrc));
  chmodSync(join(root, "skills", "release", "scripts", "release-context.sh"), 0o755);
  chmodSync(join(root, "skills", "release", "scripts", "install-hooks.sh"), 0o755);
  chmodSync(join(root, "scripts", "pre-push"), 0o755);

  writeFileSync(join(root, "package.json"), JSON.stringify({ name: "qmd", version: "2.6.3" }) + "\n");
  writeFileSync(
    join(root, "CHANGELOG.md"),
    [
      "# Changelog",
      "",
      "## [Unreleased]",
      "",
      "### Fixed",
      "",
      "- pending fix for the next cut",
      "",
      "## [2.6.3] - 2026-08-12",
      "",
      "### Fixed",
      "",
      "- previous release style reference",
      "",
    ].join("\n"),
  );
  writeFileSync(join(root, "README.md"), "hello\n");

  git(root, ["init", "-b", "main"]);
  git(root, ["config", "user.name", "qmd-test"]);
  git(root, ["config", "user.email", "qmd-test@example.com"]);
  git(root, ["config", "commit.gpgsign", "false"]);
  git(root, ["config", "tag.gpgsign", "false"]);
  git(root, ["add", "package.json", "CHANGELOG.md", "README.md", "scripts", "skills"]);
  git(root, ["commit", "-m", "initial"]);
  git(root, ["tag", "-a", "v2.6.3", "-m", "v2.6.3"]);

  writeFileSync(join(root, "NEW.md"), "post-tag change\n");
  git(root, ["add", "NEW.md"]);
  git(root, ["commit", "-m", "add NEW.md after tag"]);

  if (opts?.dirty) {
    writeFileSync(join(root, "dirty.txt"), "unstaged\n");
  }

  return root;
}

function runContext(cwd: string, versionArg: string) {
  return spawnSync("bash", [join(cwd, "skills", "release", "scripts", "release-context.sh"), versionArg], {
    cwd,
    encoding: "utf8",
  });
}

describe("skills/release/scripts/release-context.sh (#796)", () => {
  test("SKILL.md step 1 points at a script that exists in the repo", () => {
    const skill = readFileSync(skillSrc, "utf8");
    expect(skill).toMatch(/skills\/release\/scripts\/release-context\.sh/);
    expect(existsSync(scriptSrc)).toBe(true);
  });

  test("SKILL.md process steps are uniquely numbered", () => {
    const skill = readFileSync(skillSrc, "utf8");
    const process = skill.split("## Dependency Policy")[0];
    const nums = [...process.matchAll(/^(\d+)\. \*\*/gm)].map(m => m[1]);
    expect(nums).toEqual(["1", "2", "3", "4", "5", "6", "7", "8"]);
  });

  test("fails without a version argument", () => {
    const root = makeReleaseFixture();
    const result = spawnSync("bash", [join(root, "skills", "release", "scripts", "release-context.sh")], {
      cwd: root,
      encoding: "utf8",
    });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/Usage: release-context\.sh/);
  });

  test("prints version, status, commits, files, unreleased, and previous entry", () => {
    const root = makeReleaseFixture();
    const result = runContext(root, "patch");
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toMatch(/=== Version ===/);
    expect(result.stdout).toMatch(/Current:\s+2\.6\.3/);
    expect(result.stdout).toMatch(/Requested:\s+patch/);
    expect(result.stdout).toMatch(/Next:\s+2\.6\.4/);
    expect(result.stdout).toMatch(/Last tag:\s+v2\.6\.3/);
    expect(result.stdout).toMatch(/=== Working tree ===/);
    expect(result.stdout).toMatch(/\(clean\)/);
    expect(result.stdout).toMatch(/=== Commits since last release ===/);
    expect(result.stdout).toMatch(/add NEW\.md after tag/);
    expect(result.stdout).toMatch(/=== Files changed since last release ===/);
    expect(result.stdout).toMatch(/^NEW\.md$/m);
    expect(result.stdout).toMatch(/=== CHANGELOG \[Unreleased\] ===/);
    expect(result.stdout).toMatch(/pending fix for the next cut/);
    expect(result.stdout).toMatch(/=== Previous release entry ===/);
    expect(result.stdout).toMatch(/## \[2\.6\.3\] - 2026-08-12/);
    expect(result.stdout).toMatch(/previous release style reference/);
  });

  test("silently installs the pre-push hook", () => {
    const root = makeReleaseFixture();
    const hook = join(root, ".git", "hooks", "pre-push");
    expect(existsSync(hook)).toBe(false);
    const result = runContext(root, "2.6.4");
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).not.toMatch(/pre-push hook/);
    expect(result.stdout).not.toMatch(/^Done\.$/m);
    expect(existsSync(hook)).toBe(true);
  });

  test("shows a dirty working tree", () => {
    const root = makeReleaseFixture({ dirty: true });
    const result = runContext(root, "minor");
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toMatch(/Next:\s+2\.7\.0/);
    expect(result.stdout).toMatch(/dirty\.txt/);
    expect(result.stdout).not.toMatch(/\(clean\)/);
  });
});
