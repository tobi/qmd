import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, mkdir, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import {
  buildPostSyncCommands,
  buildCollectionPlans,
  buildRsyncArgs,
  getDefaultSyncOptions,
  detectConflicts,
  includePatternsForCollection,
  parseConfigYaml,
  parseRsyncItemized,
  remoteRsyncPath,
  runQmdSync,
  shellQuote,
  type CommandRunner,
} from "../src/sync.js";

const originalEnv = { ...process.env };

afterEach(() => {
  process.env = { ...originalEnv };
});

describe("qmd sync config and collection planning", () => {
  test("parses empty or missing collection config", () => {
    expect(parseConfigYaml("", "empty")).toEqual({ collections: {} });
    expect(parseConfigYaml("global_context: hello\n", "ctx")).toEqual({
      global_context: "hello",
      collections: {},
    });
  });

  test("builds bidirectional and one-sided mirror plans", () => {
    const plans = buildCollectionPlans({
      host: "root@example.com",
      remoteHome: "/home/ubuntu",
      localConfig: {
        collections: {
          docs: { path: "/local/docs", pattern: "**/*.md" },
          localOnly: { path: "/local/only", pattern: "**/*.md" },
        },
      },
      remoteConfig: {
        collections: {
          docs: { path: "/remote/docs", pattern: "**/*.md" },
          remoteOnly: { path: "/remote/only", pattern: "**/*.md" },
        },
      },
    });

    expect(plans.find(p => p.name === "docs")).toMatchObject({
      direction: "bidirectional",
      localPath: "/local/docs",
      remotePath: "/remote/docs",
      pattern: "**/*.md",
      localConfigured: true,
      remoteConfigured: true,
    });
    expect(plans.find(p => p.name === "remoteOnly")).toMatchObject({
      direction: "download-mirror",
      remotePath: "/remote/only",
      localConfigured: false,
      remoteConfigured: true,
    });
    expect(plans.find(p => p.name === "localOnly")).toMatchObject({
      direction: "upload-mirror",
      localPath: "/local/only",
      localConfigured: true,
      remoteConfigured: false,
    });
  });
});

describe("qmd sync collection masks", () => {
  test("maps markdown collection masks to rsync includes", () => {
    expect(includePatternsForCollection("**/*.md")).toEqual(["*/", "*.md"]);
    expect(includePatternsForCollection("**/*.txt")).toEqual([]);
  });

  test("uses include rules before exclude-all for markdown collections", () => {
    const args = buildRsyncArgs({
      source: "/local/docs",
      destination: "root@example.com:/remote/docs",
      remoteQmdUser: "ubuntu",
      includes: includePatternsForCollection("**/*.md"),
      dryRun: true,
    });

    expect(args).toContain("--include");
    expect(args).toContain("*/");
    expect(args).toContain("*.md");
    const excludeAllIndex = args.findIndex((arg, index) => arg === "--exclude" && args[index + 1] === "*");
    const includeMdIndex = args.findIndex((arg) => arg === "*.md");
    expect(includeMdIndex).toBeGreaterThan(-1);
    expect(excludeAllIndex).toBeGreaterThan(includeMdIndex);
  });
});

describe("qmd sync rsync command generation", () => {
  test("quotes remote rsync path under the QMD user", () => {
    expect(remoteRsyncPath("ubuntu")).toBe("sudo -u 'ubuntu' rsync");
    expect(shellQuote("a'b")).toBe("'a'\\''b'");
  });

  test("uses resumable rsync options and remote user switching in dry-run", () => {
    const args = buildRsyncArgs({
      source: "/local/docs",
      destination: "root@example.com:/remote/docs",
      remoteQmdUser: "ubuntu",
      dryRun: true,
      delete: true,
      excludeFrom: "/tmp/conflicts",
    });

    expect(args).toContain("--dry-run");
    expect(args).toContain("--delete");
    expect(args).toContain("--partial");
    expect(args).toContain("--partial-dir=.qmd-rsync-partial");
    expect(args).toContain("--delay-updates");
    expect(args).not.toContain("--temp-dir");
    expect(args).toContain("--rsync-path");
    expect(args).toContain("sudo -u 'ubuntu' rsync");
    expect(args).toContain("--exclude-from");
    expect(args).toContain("/tmp/conflicts");
    expect(args.at(-2)).toBe("/local/docs/");
    expect(args.at(-1)).toBe("root@example.com:/remote/docs/");
  });

  test("uses an explicit temp directory for apply mode", () => {
    const args = buildRsyncArgs({
      source: "/local/docs",
      destination: "root@example.com:/remote/docs",
      remoteQmdUser: "ubuntu",
      tempDir: "/remote/docs/.qmd-rsync-tmp",
    });

    expect(args).toContain("--temp-dir");
    expect(args).toContain("/remote/docs/.qmd-rsync-tmp");
  });

  test("preserves exact file paths for conflict copies", () => {
    const args = buildRsyncArgs({
      source: "/local/docs/file.md",
      destination: "root@example.com:/remote/docs/file.md.conflict.local.20260525Z",
      remoteQmdUser: "ubuntu",
      preserveFilePath: true,
    });

    expect(args.at(-2)).toBe("/local/docs/file.md");
    expect(args.at(-1)).toBe("root@example.com:/remote/docs/file.md.conflict.local.20260525Z");
  });

  test("shell-quotes remote endpoints with spaces without requiring modern rsync -s", () => {
    const args = buildRsyncArgs({
      source: "/local/Obsidian Vault",
      destination: "root@example.com:/remote/Obsidian Vault",
      remoteQmdUser: "ubuntu",
      dryRun: true,
    });

    expect(args).not.toContain("-s");
    expect(args.at(-2)).toBe("/local/Obsidian Vault/");
    expect(args.at(-1)).toBe("root@example.com:'/remote/Obsidian Vault/'");
  });
});

describe("qmd sync dry-run parsing and conflicts", () => {
  test("parses rsync itemize output into relative paths", () => {
    const output = [
      ">f.st...... notes/a.md",
      "cd+++++++++ new-dir/",
      ">f+++++++++ new-dir/b.md",
      ">f.st...... .qmd-rsync-partial/tmp",
      "",
    ].join("\n");

    expect(parseRsyncItemized(output)).toEqual(["notes/a.md", "new-dir/b.md"]);
  });

  test("detects two-way modified paths and names conflict copies", () => {
    const conflicts = detectConflicts(
      "docs",
      ["a.md", "same.md"],
      ["same.md", "b.md"],
      "20260525T010203Z",
    );

    expect(conflicts).toEqual([{
      collection: "docs",
      path: "same.md",
      localConflictPath: "same.md.conflict.remote.20260525T010203Z",
      remoteConflictPath: "same.md.conflict.local.20260525T010203Z",
    }]);
  });
});

describe("qmd sync update freshness", () => {
  test("builds local and remote post-sync commands with sudo remote user", () => {
    const opts = getDefaultSyncOptions({
      host: "root@example.com",
      remoteQmdUser: "ubuntu",
      update: true,
      embed: true,
      localQmdCommand: ["bun", "src/cli/qmd.ts"],
    });

    expect(buildPostSyncCommands(opts).map(step => ({
      side: step.side,
      action: step.action,
      command: step.command,
      skipped: step.skipped,
    }))).toEqual([
      { side: "local", action: "update", command: ["bun", "src/cli/qmd.ts", "update"], skipped: false },
      { side: "remote", action: "update", command: ["ssh", "root@example.com", "sudo -u 'ubuntu' sh -lc 'qmd update'"], skipped: false },
      { side: "local", action: "embed", command: ["bun", "src/cli/qmd.ts", "embed"], skipped: false },
      { side: "remote", action: "embed", command: ["ssh", "root@example.com", "sudo -u 'ubuntu' sh -lc 'qmd embed'"], skipped: false },
    ]);
  });

  test("dry-run --update plans update commands without executing them", async () => {
    const env = await createSyncTestEnv();
    const calls: Array<{ command: string; args: string[] }> = [];
    const summary = await runQmdSync({
      host: "root@example.com",
      remoteQmdUser: "ubuntu",
      remoteHome: "/home/ubuntu",
      dryRun: true,
      update: true,
      localQmdCommand: ["qmd-test"],
      runCommand: fakeRunner(calls),
    });

    expect(summary.failed).toBe(false);
    expect(summary.postSync).toHaveLength(2);
    expect(summary.postSync.every(step => step.skipped && step.reason === "dry-run")).toBe(true);
    expect(calls.some(call => call.command === "qmd-test")).toBe(false);
    expect(calls.some(call => call.command === "ssh" && call.args.join(" ").includes("qmd update"))).toBe(false);
    await env.cleanup();
  });

  test("--collection limits sync to collection paths and skips config apply", async () => {
    const env = await createSyncTestEnv();
    const calls: Array<{ command: string; args: string[] }> = [];
    const summary = await runQmdSync({
      host: "root@example.com",
      remoteQmdUser: "ubuntu",
      remoteHome: "/home/ubuntu",
      collection: ["docs"],
      runCommand: fakeRunner(calls),
    });

    expect(summary.rsync.map(result => result.label)).toEqual(["docs", "docs", "docs", "docs"]);
    expect(calls
      .filter(call => call.command === "rsync")
      .some(call => call.args.join(" ").includes(".config/qmd"))).toBe(false);
    await env.cleanup();
  });

  test("apply rsync failure marks sync failed and skips update/embed", async () => {
    const env = await createSyncTestEnv();
    const calls: Array<{ command: string; args: string[] }> = [];
    const summary = await runQmdSync({
      host: "root@example.com",
      remoteQmdUser: "ubuntu",
      remoteHome: "/home/ubuntu",
      update: true,
      embed: true,
      localQmdCommand: ["qmd-test"],
      runCommand: fakeRunner(calls, { failApplyRsync: true }),
    });

    expect(summary.failed).toBe(true);
    expect(summary.postSync).toHaveLength(4);
    expect(summary.postSync.every(step => step.skipped && step.reason === "sync failed; update/embed not run")).toBe(true);
    expect(calls.some(call => call.command === "qmd-test")).toBe(false);
    await env.cleanup();
  });

  test("successful apply runs local and remote update before embed", async () => {
    const env = await createSyncTestEnv();
    const calls: Array<{ command: string; args: string[] }> = [];
    const summary = await runQmdSync({
      host: "root@example.com",
      remoteQmdUser: "ubuntu",
      remoteHome: "/home/ubuntu",
      update: true,
      embed: true,
      localQmdCommand: ["qmd-test"],
      runCommand: fakeRunner(calls),
    });

    expect(summary.failed).toBe(false);
    expect(summary.postSync.map(step => `${step.side}:${step.action}:${step.exitCode}`)).toEqual([
      "local:update:0",
      "remote:update:0",
      "local:embed:0",
      "remote:embed:0",
    ]);
    const executed = calls
      .filter(call => call.command === "qmd-test" || call.args.join(" ").includes("qmd update") || call.args.join(" ").includes("qmd embed"))
      .map(call => [call.command, ...call.args].join(" "));
    expect(executed).toEqual([
      "qmd-test update",
      "ssh root@example.com sudo -u 'ubuntu' sh -lc 'qmd update'",
      "qmd-test embed",
      "ssh root@example.com sudo -u 'ubuntu' sh -lc 'qmd embed'",
    ]);
    await env.cleanup();
  });
});

async function createSyncTestEnv(): Promise<{ cleanup: () => Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), "qmd-sync-test-"));
  const configDir = join(root, "config");
  const cacheDir = join(root, "cache");
  const dataDir = join(root, "data");
  const docsDir = join(root, "docs");
  await mkdir(configDir, { recursive: true });
  await mkdir(cacheDir, { recursive: true });
  await mkdir(dataDir, { recursive: true });
  await mkdir(docsDir, { recursive: true });
  await writeFile(join(docsDir, "local.md"), "# Local\n");
  await writeFile(join(configDir, "index.yml"), `collections:\n  docs:\n    path: ${JSON.stringify(docsDir)}\n    pattern: "**/*.md"\n`);
  process.env.QMD_CONFIG_DIR = configDir;
  process.env.XDG_CACHE_HOME = cacheDir;
  process.env.XDG_DATA_HOME = dataDir;
  return {
    cleanup: async () => {
      await rm(root, { recursive: true, force: true });
    },
  };
}

function fakeRunner(
  calls: Array<{ command: string; args: string[] }>,
  options: { failApplyRsync?: boolean } = {},
): CommandRunner {
  return async (command, args) => {
    calls.push({ command, args });
    if (command === "ssh") {
      const remoteCommand = args.join(" ");
      if (remoteCommand.includes("command -v rsync")) {
        return { exitCode: 0, stdout: "rsync=1\nflock=1\nqmd 2.1.0\n", stderr: "" };
      }
      if (remoteCommand.includes("cat") && remoteCommand.includes("index.yml")) {
        return { exitCode: 0, stdout: "collections:\n  docs:\n    path: /remote/docs\n    pattern: \"**/*.md\"\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "remote ok\n", stderr: "" };
    }
    if (command === "rsync") {
      const isDryRun = args.includes("--dry-run");
      if (!isDryRun && options.failApplyRsync) {
        return { exitCode: 23, stdout: "", stderr: "rsync failed" };
      }
      return { exitCode: 0, stdout: "", stderr: "" };
    }
    if (command === "qmd-test") {
      return { exitCode: 0, stdout: `${args[0]} ok\n`, stderr: "" };
    }
    return { exitCode: 0, stdout: "", stderr: "" };
  };
}
