import { afterEach, describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const hookSrc = join(repoRoot, "scripts", "pre-push");

const VERSION = "2.9.0";
const TAG = `v${VERSION}`;
const ZERO_SHA = "0".repeat(40);

/** Fake `gh` CLI: logs its argv and answers from per-workflow JSON fixtures. */
const FAKE_GH = `#!/usr/bin/env bash
set -uo pipefail

if [[ -n "\${FAKE_GH_LOG:-}" ]]; then
  printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
fi

if [[ "\${1:-}" != "api" ]]; then
  echo "fake gh: unsupported command: $*" >&2
  exit 64
fi

url="\${2:-}"
if [[ ! "$url" =~ workflows/([^/]+)/runs ]]; then
  echo "fake gh: no workflow in URL: $url" >&2
  exit 65
fi
workflow="\${BASH_REMATCH[1]}"

if [[ -n "\${FAKE_GH_FAIL:-}" ]]; then
  echo "fake gh: simulated API failure" >&2
  exit 1
fi

fixture="\${FAKE_GH_DIR:?}/$workflow.json"
if [[ ! -f "$fixture" ]]; then
  echo "fake gh: no fixture for $workflow" >&2
  exit 1
fi
cat "$fixture"
`;

interface RunFixture {
  status?: string;
  conclusion?: string | null;
  event?: string;
  head_sha?: string;
}

interface Fixture {
  root: string;
  commit: string;
  tagSha: string;
  binDir: string;
  ghDir: string;
  logPath: string;
}

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
  return result.stdout.trim();
}

function makeFixture(opts?: { packageVersion?: string; changelogVersion?: string }): Fixture {
  const root = mkdtempSync(join(tmpdir(), "qmd-pre-push-"));
  fixtures.push(root);

  mkdirSync(join(root, "scripts"), { recursive: true });
  const hook = join(root, "scripts", "pre-push");
  writeFileSync(hook, readFileSync(hookSrc));
  chmodSync(hook, 0o755);

  writeFileSync(
    join(root, "package.json"),
    JSON.stringify({ name: "qmd", version: opts?.packageVersion ?? VERSION }, null, 2) + "\n",
  );
  writeFileSync(
    join(root, "CHANGELOG.md"),
    [
      "# Changelog",
      "",
      "## [Unreleased]",
      "",
      `## [${opts?.changelogVersion ?? VERSION}] - 2026-11-17`,
      "",
      "### Added",
      "",
      "- deterministic release hook tests",
      "",
    ].join("\n"),
  );

  git(root, ["init", "-b", "main"]);
  git(root, ["config", "user.name", "qmd-test"]);
  git(root, ["config", "user.email", "qmd-test@example.com"]);
  git(root, ["add", "package.json", "CHANGELOG.md", "scripts"]);
  git(root, ["commit", "-m", "release 2.9.0"]);
  git(root, ["tag", "-a", TAG, "-m", TAG]);

  const commit = git(root, ["rev-parse", "HEAD"]);
  const tagSha = git(root, ["rev-parse", TAG]);

  const binDir = join(root, "bin");
  const ghDir = join(root, "gh");

  // Restricted PATH without `gh`, so the "gh absent" case is deterministic
  // even on a machine that has the real CLI installed.
  const binDirRoot = join(root, "bin-no-gh");
  mkdirSync(binDirRoot, { recursive: true });
  for (const tool of ["bash", "git", "jq", "grep", "date", "cat"]) {
    const target = spawnSync("bash", ["-c", `command -v ${tool}`], { encoding: "utf8" }).stdout.trim();
    if (target) symlinkSync(target, join(binDirRoot, tool));
  }

  mkdirSync(binDir, { recursive: true });
  mkdirSync(ghDir, { recursive: true });
  const ghPath = join(binDir, "gh");
  writeFileSync(ghPath, FAKE_GH);
  chmodSync(ghPath, 0o755);

  return { root, commit, tagSha, binDir, ghDir, logPath: join(root, "gh.log") };
}

function writeRuns(f: Fixture, workflow: string, commit: string, runs: RunFixture[]) {
  const body = {
    total_count: runs.length,
    workflow_runs: runs.map((run) => ({
      head_sha: commit,
      event: "push",
      status: "completed",
      conclusion: "success",
      ...run,
    })),
  };
  writeFileSync(join(f.ghDir, `${workflow}.json`), JSON.stringify(body));
}

function writeBothPassing(f: Fixture) {
  for (const workflow of ["ci.yml", "nix.yml"]) {
    writeRuns(f, workflow, f.commit, [{}]);
  }
}

function runHook(f: Fixture, stdin: string, opts?: { path?: "with-gh" | "without-gh"; env?: Record<string, string> }) {
  return spawnSync("bash", [join(f.root, "scripts", "pre-push")], {
    cwd: f.root,
    encoding: "utf8",
    input: stdin,
    env: {
      ...process.env,
      PATH: opts?.path === "without-gh" ? join(f.root, "bin-no-gh") : `${f.binDir}:${process.env.PATH ?? "/usr/bin:/bin"}`,
      FAKE_GH_DIR: f.ghDir,
      FAKE_GH_LOG: f.logPath,
      ...opts?.env,
    },
  });
}

const tagPush = (f: Fixture) =>
  [`refs/tags/${TAG} ${f.tagSha} refs/tags/${TAG} ${ZERO_SHA}`, ""].join("\n");

describe("scripts/pre-push release gate", () => {
  test("ignores non-release pushes (including when gh is absent)", () => {
    const f = makeFixture();
    const result = runHook(f, [`refs/heads/main ${f.commit} refs/heads/main ${ZERO_SHA}`, ""].join("\n"), {
      path: "without-gh",
    });
    expect(result.status, result.stderr).toBe(0);
    expect(existsSync(f.logPath)).toBe(false);
  });

  test("ignores tag deletions (including when gh is absent)", () => {
    const f = makeFixture();
    const result = runHook(f, [`refs/tags/${TAG} ${ZERO_SHA} refs/tags/${TAG} ${f.tagSha}`, ""].join("\n"), {
      path: "without-gh",
    });
    expect(result.status, result.stderr).toBe(0);
    expect(existsSync(f.logPath)).toBe(false);
  });

  test("rejects a tag whose package.json version does not match", () => {
    const f = makeFixture({ packageVersion: "2.8.9" });
    writeBothPassing(f);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/package\.json version is 2\.8\.9 but tag is v2\.9\.0/);
    expect(result.stderr).not.toMatch(/All checks passed/);
  });

  test("rejects a tag with no CHANGELOG entry", () => {
    const f = makeFixture({ changelogVersion: "2.8.9" });
    writeBothPassing(f);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/CHANGELOG\.md has no entry for \[2\.9\.0\]/);
    expect(result.stderr).not.toMatch(/All checks passed/);
  });

  test("fails closed when gh is absent", () => {
    const f = makeFixture();
    writeBothPassing(f);
    const result = runHook(f, tagPush(f), { path: "without-gh" });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/gh CLI is required/);
    expect(result.stderr).not.toMatch(/passed/);
  });

  test("fails closed when the GitHub API is unreachable", () => {
    const f = makeFixture();
    writeBothPassing(f);
    const result = runHook(f, tagPush(f), { env: { FAKE_GH_FAIL: "1" } });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/cannot read GitHub Actions 'ci\.yml' runs/);
    expect(result.stderr).not.toMatch(/All checks passed/);
  });

  test("fails closed when there are no CI runs for the tagged commit", () => {
    const f = makeFixture();
    writeBothPassing(f);
    writeRuns(f, "ci.yml", f.commit, []);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/no GitHub Actions 'ci\.yml' push run found/);
    expect(result.stderr).not.toMatch(/All checks passed/);
  });

  test("ignores runs for other commits or non-push events", () => {
    const f = makeFixture();
    writeBothPassing(f);
    writeRuns(f, "nix.yml", f.commit, [{ head_sha: "a".repeat(40), event: "push" }]);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/no GitHub Actions 'nix\.yml' push run found/);
  });

  test("rejects pending CI", () => {
    const f = makeFixture();
    writeBothPassing(f);
    writeRuns(f, "ci.yml", f.commit, [{ status: "in_progress", conclusion: null }]);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/GitHub Actions 'ci\.yml' is not green.*in_progress\/none/);
    expect(result.stderr).not.toMatch(/All checks passed/);
  });

  for (const conclusion of ["failure", "cancelled", "skipped", "action_required"]) {
    test(`rejects CI with conclusion ${conclusion}`, () => {
      const f = makeFixture();
      writeBothPassing(f);
      writeRuns(f, "nix.yml", f.commit, [{ conclusion }]);
      const result = runHook(f, tagPush(f));
      expect(result.status).not.toBe(0);
      expect(result.stderr).toMatch(new RegExp(`'nix\\.yml' is not green.*completed/${conclusion}`));
      expect(result.stderr).not.toMatch(/All checks passed/);
    });
  }

  test("rejects a run that is not completed even if conclusion looks successful", () => {
    const f = makeFixture();
    writeBothPassing(f);
    writeRuns(f, "ci.yml", f.commit, [{ status: "queued", conclusion: "success" }]);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/queued\/success/);
  });

  test("fails closed on malformed GitHub JSON", () => {
    const f = makeFixture();
    writeBothPassing(f);
    writeFileSync(join(f.ghDir, "ci.yml.json"), "{not json");
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/unexpected GitHub Actions 'ci\.yml' response/);
  });

  test("rejects when only ci.yml passed and nix.yml is missing", () => {
    const f = makeFixture();
    writeRuns(f, "ci.yml", f.commit, [{}]);
    const result = runHook(f, tagPush(f));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/cannot read GitHub Actions 'nix\.yml' runs/);
  });

  test("passes when both workflows have successful completed push runs", () => {
    const f = makeFixture();
    writeBothPassing(f);
    const result = runHook(f, tagPush(f));
    expect(result.status, result.stderr).toBe(0);
    expect(result.stderr).toMatch(/CI \(ci\.yml\): 1 run\(s\) passed/);
    expect(result.stderr).toMatch(/CI \(nix\.yml\): 1 run\(s\) passed/);
    expect(result.stderr).toMatch(/All checks passed for v2\.9\.0/);
  });

  test("queries both workflows by exact commit and push event without a hardcoded owner", () => {
    const f = makeFixture();
    writeBothPassing(f);
    const result = runHook(f, tagPush(f));
    expect(result.status, result.stderr).toBe(0);

    const calls = readFileSync(f.logPath, "utf8").trim().split("\n");
    expect(calls).toHaveLength(2);
    for (const workflow of ["ci.yml", "nix.yml"]) {
      const call = calls.find((line) => line.includes(`/workflows/${workflow}/runs`));
      expect(call, `gh called for ${workflow}`).toBeDefined();
      expect(call).toContain("repos/{owner}/{repo}/actions/workflows/");
      expect(call).toContain(`head_sha=${f.commit}`);
      expect(call).toContain("event=push");
      expect(call).not.toMatch(/github\.com|tobi\//);
    }
  });
});
