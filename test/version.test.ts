/**
 * Version/commit identification tests (issue #717's sibling, #787).
 *
 * `qmd --version` used to run `git -C <scriptDir> rev-parse --short HEAD`.
 * `git -C` walks *up*, so any install nested inside an unrelated repository
 * reported that repository's HEAD as qmd's — the reason bug reports for the
 * same published tarball carried three different "commits".
 *
 * These tests pin the two rules that replace it: a build stamp wins, and a
 * git lookup is only trusted when the enclosing repository is the package we
 * are actually running from.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { execFileSync } from "child_process";
import { mkdir, mkdtemp, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import {
  BUILD_INFO_FILENAME,
  gitCommitForCheckout,
  readStampedCommit,
  resolveCommit,
} from "../src/cli/version.js";

let testDir: string;

function git(args: string[], cwd: string): string {
  return execFileSync("git", args, {
    cwd,
    encoding: "utf-8",
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      GIT_AUTHOR_NAME: "qmd test",
      GIT_AUTHOR_EMAIL: "test@example.com",
      GIT_COMMITTER_NAME: "qmd test",
      GIT_COMMITTER_EMAIL: "test@example.com",
    },
  }).trim();
}

/**
 * A git repo whose working tree is clean: everything already present in `dir`
 * is committed, so later assertions can distinguish clean from dirty.
 * Returns the short HEAD.
 */
function initRepo(dir: string): string {
  git(["init", "-q", "--initial-branch=main", "."], dir);
  git(["add", "-A"], dir);
  git(["commit", "-q", "--allow-empty", "-m", "root"], dir);
  return git(["rev-parse", "--short", "HEAD"], dir);
}

/** package root + the directory the CLI runs from (dist/cli or src/cli). */
async function makePackage(root: string): Promise<{ packageRoot: string; scriptDir: string }> {
  const scriptDir = join(root, "dist", "cli");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(root, "package.json"), JSON.stringify({ name: "@tobilu/qmd", version: "9.9.9" }));
  return { packageRoot: root, scriptDir };
}

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-version-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

describe("gitCommitForCheckout", () => {
  test("returns nothing when the enclosing repo is not this package (#787)", async () => {
    // The reported scenario: a global install living inside an unrelated
    // checkout (e.g. npm prefix under a git-managed /opt/homebrew).
    const outer = join(testDir, "unrelated-repo");
    await mkdir(outer, { recursive: true });
    const outerHead = initRepo(outer);

    const { packageRoot, scriptDir } = await makePackage(join(outer, "lib", "node_modules", "@tobilu", "qmd"));

    const commit = gitCommitForCheckout(scriptDir, packageRoot);
    expect(commit).toBe("");
    // The bug was reporting exactly this.
    expect(commit).not.toBe(outerHead);
  });

  test("returns the real HEAD when the package root is the repo", async () => {
    const repo = join(testDir, "qmd-checkout");
    const { packageRoot, scriptDir } = await makePackage(repo);
    const head = initRepo(repo);

    expect(gitCommitForCheckout(scriptDir, packageRoot)).toBe(head);
  });

  test("marks a dirty working tree", async () => {
    const repo = join(testDir, "dirty-checkout");
    const { packageRoot, scriptDir } = await makePackage(repo);
    const head = initRepo(repo);
    await writeFile(join(repo, "uncommitted.txt"), "edit\n");

    expect(gitCommitForCheckout(scriptDir, packageRoot)).toBe(`${head}-dirty`);
  });

  test("works when the install path contains a space", async () => {
    // The old code interpolated the directory into a shell string unquoted, so
    // a space made git fail and the commit silently vanished.
    const repo = join(testDir, "a dir", "qmd checkout");
    const { packageRoot, scriptDir } = await makePackage(repo);
    const head = initRepo(repo);

    expect(gitCommitForCheckout(scriptDir, packageRoot)).toBe(head);
  });

  test("returns nothing outside any repository", async () => {
    const plain = join(testDir, "no-repo");
    await mkdir(plain, { recursive: true });
    const { packageRoot, scriptDir } = await makePackage(plain);

    expect(gitCommitForCheckout(scriptDir, packageRoot)).toBe("");
  });
});

describe("readStampedCommit", () => {
  test("reads the commit stamped beside the running script", async () => {
    const { scriptDir } = await makePackage(join(testDir, "stamped"));
    await writeFile(
      join(scriptDir, BUILD_INFO_FILENAME),
      JSON.stringify({ commit: "abc1234", builtAt: "2026-01-01T00:00:00.000Z" })
    );

    expect(readStampedCommit(scriptDir)).toBe("abc1234");
  });

  test("returns nothing when there is no stamp", async () => {
    const { scriptDir } = await makePackage(join(testDir, "unstamped"));
    expect(readStampedCommit(scriptDir)).toBe("");
  });

  test("tolerates a malformed or commit-less stamp", async () => {
    const { scriptDir } = await makePackage(join(testDir, "malformed"));
    await writeFile(join(scriptDir, BUILD_INFO_FILENAME), "{not json");
    expect(readStampedCommit(scriptDir)).toBe("");

    await writeFile(join(scriptDir, BUILD_INFO_FILENAME), JSON.stringify({ builtAt: "whenever" }));
    expect(readStampedCommit(scriptDir)).toBe("");
  });
});

describe("resolveCommit", () => {
  test("prefers the build stamp over the enclosing checkout", async () => {
    const repo = join(testDir, "stamp-wins");
    const { packageRoot, scriptDir } = await makePackage(repo);
    const head = initRepo(repo);
    await writeFile(join(scriptDir, BUILD_INFO_FILENAME), JSON.stringify({ commit: "stamped1" }));

    expect(resolveCommit(scriptDir, packageRoot)).toBe("stamped1");
    expect(resolveCommit(scriptDir, packageRoot)).not.toBe(head);
  });

  test("falls back to the verified checkout when unstamped", async () => {
    const repo = join(testDir, "fallback");
    const { packageRoot, scriptDir } = await makePackage(repo);
    const head = initRepo(repo);

    expect(resolveCommit(scriptDir, packageRoot)).toBe(head);
  });

  test("reports nothing rather than a foreign commit", async () => {
    const outer = join(testDir, "foreign");
    await mkdir(outer, { recursive: true });
    initRepo(outer);
    const { packageRoot, scriptDir } = await makePackage(join(outer, "node_modules", "@tobilu", "qmd"));

    expect(resolveCommit(scriptDir, packageRoot)).toBe("");
  });
});
