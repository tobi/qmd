import { afterEach, describe, expect, test } from "vitest";
import { chmodSync, copyFileSync, existsSync, mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { setTimeout as sleep } from "node:timers/promises";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const fixtures: string[] = [];

// Shebang-forward for bin/qmd must exec a real Node binary, not bun.
// `bun test` sets process.execPath to bun; interpolating that would run the
// trampoline under bun and hit the PATH-`node` fallback, which is the
// opposite of the NODE_MODULE_VERSION case this suite pins.
function realNodeExecPath(): string {
  if (typeof process.versions.bun !== "string") return process.execPath;
  const result = spawnSync("node", ["-p", "process.execPath"], { encoding: "utf8" });
  const resolved = result.stdout?.trim();
  if (result.status === 0 && resolved) return resolved;
  return "node";
}
const REAL_NODE = realNodeExecPath();

function makeTempFixture() {
  const root = mkdtempSync(join(tmpdir(), "qmd-bin-wrapper-"));
  fixtures.push(root);
  const capturePath = join(root, "capture.txt");
  const runtimeBin = join(root, "runtime-bin");
  mkdirSync(runtimeBin, { recursive: true });

  for (const runtime of ["node", "bun"]) {
    const runtimePath = join(runtimeBin, runtime);
    if (runtime === "node") {
      // Shebang of bin/qmd is `#!/usr/bin/env node`, so PATH `node` must
      // still launch the trampoline. The trampoline itself must NOT
      // re-resolve `node` from PATH for the child (#577 leftover): that
      // is the NODE_MODULE_VERSION bug. Exit 42 if it does.
      writeFileSync(
        runtimePath,
        `#!/bin/sh
if [ "$(basename "$1")" = "qmd" ]; then
  exec "${REAL_NODE}" "$@"
else
  echo "qmd launcher must not re-resolve node from PATH" >&2
  exit 42
fi
`,
      );
    } else {
      writeFileSync(
        runtimePath,
        `#!/bin/sh\n{\n  printf '%s\\n' '${runtime}'\n  printf '%s\\n' "$1"\n  shift\n  printf '%s\\n' "$@"\n} > "$QMD_WRAPPER_CAPTURE"\n`,
      );
    }
    chmodSync(runtimePath, 0o755);
  }

  return { root, capturePath, runtimeBin };
}

function makePackage(root: string, packagePath: string, lockfiles: string[] = [], options: { dist?: boolean; source?: boolean; tsx?: boolean; git?: boolean } = {}) {
  const packageRoot = join(root, packagePath);
  const includeDist = options.dist ?? true;
  mkdirSync(join(packageRoot, "bin"), { recursive: true });
  copyFileSync(join(repoRoot, "bin", "qmd"), join(packageRoot, "bin", "qmd"));
  chmodSync(join(packageRoot, "bin", "qmd"), 0o755);
  if (includeDist) {
    mkdirSync(join(packageRoot, "dist", "cli"), { recursive: true });
    writeFileSync(
      join(packageRoot, "dist", "cli", "qmd.js"),
      [
        'const { writeFileSync } = require("node:fs");',
        'const capture = process.env.QMD_WRAPPER_CAPTURE;',
        'if (capture) {',
        '  writeFileSync(capture, ["node", process.argv[1], ...process.argv.slice(2)].join("\\n") + "\\n");',
        '}',
        '',
      ].join("\n"),
    );
  }
  if (options.source) {
    mkdirSync(join(packageRoot, "src", "cli"), { recursive: true });
    writeFileSync(join(packageRoot, "src", "cli", "qmd.ts"), "// source fixture\n");
  }
  if (options.tsx) {
    mkdirSync(join(packageRoot, "node_modules", "tsx", "dist"), { recursive: true });
    writeFileSync(
      join(packageRoot, "node_modules", "tsx", "dist", "cli.mjs"),
      [
        'import { writeFileSync } from "node:fs";',
        'const capture = process.env.QMD_WRAPPER_CAPTURE;',
        'if (capture) {',
        '  writeFileSync(capture, ["node", process.argv[1], ...process.argv.slice(2)].join("\\n") + "\\n");',
        '}',
        '',
      ].join("\n"),
    );
  }
  if (options.git) {
    mkdirSync(join(packageRoot, ".git"), { recursive: true });
  }
  for (const lockfile of lockfiles) {
    writeFileSync(join(packageRoot, lockfile), "");
  }
  return packageRoot;
}

function symlinkRelative(target: string, linkPath: string) {
  mkdirSync(dirname(linkPath), { recursive: true });
  symlinkSync(relative(dirname(linkPath), target), linkPath);
}

function runWrapper(commandPath: string, runtimeBin: string, capturePath: string, env: Record<string, string> = {}) {
  rmSync(capturePath, { force: true });
  execFileSync(commandPath, ["--version"], {
    env: {
      ...process.env,
      ...env,
      PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
      QMD_WRAPPER_CAPTURE: capturePath,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const [runtime, scriptPath, ...args] = readFileSync(capturePath, "utf8").trimEnd().split("\n");
  return { runtime, scriptPath, args };
}

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 2000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for process state");
    await sleep(25);
  }
}

afterEach(() => {
  for (const fixture of fixtures.splice(0)) {
    rmSync(fixture, { recursive: true, force: true });
  }
});

describe("bin/qmd package wrapper", () => {
  test("direct package invocation resolves dist/cli/qmd.js from the package root", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd");

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
    expect(result.args).toEqual(["--version"]);
  });

  test("npm/Homebrew global bin symlink resolves scoped package path", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "opt/homebrew/lib/node_modules/@tobilu/qmd");
    const globalBin = join(root, "opt", "homebrew", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), globalBin);

    const result = runWrapper(globalBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("multi-hop global bin symlink chain resolves to the real package root", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "opt/homebrew/lib/node_modules/@tobilu/qmd");
    const globalBin = join(root, "opt", "homebrew", "bin", "qmd");
    const shim = join(root, "opt", "homebrew", "Cellar", "qmd", "current", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), shim);
    symlinkRelative(shim, globalBin);

    const result = runWrapper(globalBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("linuxbrew global bin symlink resolves lib/node_modules scoped package path", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "home/linuxbrew/.linuxbrew/lib/node_modules/@tobilu/qmd");
    const globalBin = join(root, "home", "linuxbrew", ".linuxbrew", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), globalBin);

    const result = runWrapper(globalBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("npx scoped package .bin symlink resolves @tobilu/qmd package path", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "npm/_npx/abc123/node_modules/@tobilu/qmd");
    const npxBin = join(root, "npm", "_npx", "abc123", "node_modules", ".bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), npxBin);

    const result = runWrapper(npxBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("bun global symlink uses bun when package-local bun lockfile exists", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "home/user/.bun/install/global/node_modules/@tobilu/qmd", ["bun.lock"]);
    const bunBin = join(root, "home", "user", ".bun", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), bunBin);

    const result = runWrapper(bunBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("bun");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("bun global install with bun.lock at the install root uses bun", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "home/user/.bun/install/global/node_modules/@tobilu/qmd");
    writeFileSync(join(root, "home", "user", ".bun", "install", "global", "bun.lock"), "");
    const bunBin = join(root, "home", "user", ".bun", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), bunBin);

    const result = runWrapper(bunBin, runtimeBin, capturePath);

    expect(result.runtime).toBe("bun");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("package-lock.json at the install root keeps npm priority", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "project/node_modules/@tobilu/qmd");
    writeFileSync(join(root, "project", "package-lock.json"), "");
    writeFileSync(join(root, "project", "bun.lock"), "");

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("ambient BUN_INSTALL alone does not select bun for an npm-installed package", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "opt/homebrew/lib/node_modules/@tobilu/qmd");
    const globalBin = join(root, "opt", "homebrew", "bin", "qmd");
    symlinkRelative(join(packageRoot, "bin", "qmd"), globalBin);

    const result = runWrapper(globalBin, runtimeBin, capturePath, { BUN_INSTALL: join(root, ".bun") });

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("package-lock.json takes priority over bun lockfiles", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["package-lock.json", "bun.lock"]);

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("packaged tree uses dist even if source files are present", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["bun.lock"], { source: true });

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("bun");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
  });

  test("prefers source with bun in a Bun checkout even when dist exists", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "qmd", ["bun.lock"], { source: true, git: true });

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("bun");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "src", "cli", "qmd.ts")));
    expect(result.args).toEqual(["--version"]);
  });

  test("prefers source through tsx in a Node checkout even when dist exists", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "qmd", [], { source: true, tsx: true, git: true });

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "node_modules", "tsx", "dist", "cli.mjs")));
    expect(result.args).toEqual([realpathSync(join(packageRoot, "src", "cli", "qmd.ts")), "--version"]);
  });

  test("source checkout with both bun.lock and package-lock.json prefers node+tsx", () => {
    // Mirrors the dist-mode "npm priority" rule: a working tree that has both
    // lockfiles (because the user ran `npm install` against a repo that also
    // ships bun.lock) installed native modules for Node's ABI, so source mode
    // must route through tsx to avoid better-sqlite3 / sqlite-vec mismatches.
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "qmd", ["bun.lock", "package-lock.json"], { source: true, tsx: true, git: true });

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "node_modules", "tsx", "dist", "cli.mjs")));
    expect(result.args).toEqual([realpathSync(join(packageRoot, "src", "cli", "qmd.ts")), "--version"]);
  });

  test("node child uses process.execPath, not a different PATH node (#577 leftover)", () => {
    const { root, runtimeBin, capturePath } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd");

    const result = runWrapper(join(packageRoot, "bin", "qmd"), runtimeBin, capturePath);

    expect(result.runtime).toBe("node");
    expect(result.scriptPath).toBe(realpathSync(join(packageRoot, "dist", "cli", "qmd.js")));
    expect(result.args).toEqual(["--version"]);
  });

  test("explains how to build when dist is missing and source cannot run", () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "qmd", [], { dist: false });

    const result = spawnSync(join(packageRoot, "bin", "qmd"), ["--version"], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
      },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("qmd is not built");
    expect(result.stderr).toContain("bun install && bun run build");
    expect(result.stderr).toContain("npm install && npm run build");
    expect(result.stderr).toContain("qmd doctor");
  });

  test.skipIf(process.platform === "win32")("terminating the wrapper also terminates a running query child", async () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["bun.lock"]);
    const childPidPath = join(root, "query-child.pid");
    const bunPath = join(runtimeBin, "bun");
    writeFileSync(
      bunPath,
      `#!/bin/sh
printf '%s\n' "$$" > "$QMD_CHILD_PID"
trap '' TERM INT HUP
while :; do sleep 1; done
`,
    );
    chmodSync(bunPath, 0o755);

    const wrapper = spawn(join(packageRoot, "bin", "qmd"), [
      "query",
      "intent: find fixture result\nlex: existing indexed fixture",
    ], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
        QMD_CHILD_PID: childPidPath,
        QMD_SHUTDOWN_GRACE_MS: "200",
      },
      stdio: "ignore",
    });

    let childPid: number | undefined;
    try {
      await waitFor(() => existsSync(childPidPath));
      childPid = Number(readFileSync(childPidPath, "utf8").trim());
      expect(processExists(childPid)).toBe(true);

      wrapper.kill("SIGTERM");
      await waitFor(() => !processExists(childPid!));
      await waitFor(() => wrapper.exitCode !== null || wrapper.signalCode !== null);

      expect(processExists(childPid)).toBe(false);
      expect(wrapper.exitCode ?? wrapper.signalCode).not.toBeNull();
    } finally {
      if (wrapper.exitCode === null && wrapper.signalCode === null) wrapper.kill("SIGKILL");
      if (childPid !== undefined && processExists(childPid)) process.kill(childPid, "SIGKILL");
    }
  });
});

describe("bin/qmd shutdown supervisor", () => {
  function writeChildFixture(packageRoot: string, source: string) {
    writeFileSync(join(packageRoot, "dist", "cli", "qmd.js"), source);
  }

  test.skipIf(process.platform === "win32")("first SIGTERM waits for child cleanup and reports 143", async () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["package-lock.json"]);
    const breadcrumb = join(root, "breadcrumb.txt");
    const childPidPath = join(root, "child.pid");
    writeChildFixture(packageRoot, `
const { appendFileSync, writeFileSync } = require("node:fs");
writeFileSync(process.env.QMD_CHILD_PID, String(process.pid));
process.on("SIGTERM", () => {
  appendFileSync(process.env.QMD_CHILD_BREADCRUMB, "cleanup-started\\n");
  setTimeout(() => process.exit(143), 400);
});
setInterval(() => {}, 1 << 30);
`);

    const wrapper = spawn(join(packageRoot, "bin", "qmd"), ["query", "fixture"], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
        QMD_CHILD_BREADCRUMB: breadcrumb,
        QMD_CHILD_PID: childPidPath,
        QMD_SHUTDOWN_GRACE_MS: "5000",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let childPid: number | undefined;
    try {
      await waitFor(() => existsSync(childPidPath));
      childPid = Number(readFileSync(childPidPath, "utf8").trim());
      wrapper.kill("SIGTERM");
      await waitFor(() => existsSync(breadcrumb), 2000);
      expect(processExists(childPid)).toBe(true);
      await sleep(150);
      expect(wrapper.exitCode).toBeNull();
      expect(processExists(childPid)).toBe(true);
      await waitFor(() => wrapper.exitCode !== null || wrapper.signalCode !== null, 3000);
      expect(wrapper.exitCode).toBe(143);
      expect(readFileSync(breadcrumb, "utf8")).toContain("cleanup-started");
    } finally {
      if (wrapper.exitCode === null && wrapper.signalCode === null) wrapper.kill("SIGKILL");
      if (childPid !== undefined && processExists(childPid)) process.kill(childPid, "SIGKILL");
    }
  });

  test.skipIf(process.platform === "win32")("second SIGTERM SIGKILLs immediately and still reports 143", async () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["package-lock.json"]);
    const breadcrumb = join(root, "breadcrumb.txt");
    const childPidPath = join(root, "child.pid");
    writeChildFixture(packageRoot, `
const { appendFileSync, writeFileSync } = require("node:fs");
writeFileSync(process.env.QMD_CHILD_PID, String(process.pid));
process.on("SIGTERM", () => {
  appendFileSync(process.env.QMD_CHILD_BREADCRUMB, "cleanup-started\\n");
});
setInterval(() => {}, 1 << 30);
`);

    const wrapper = spawn(join(packageRoot, "bin", "qmd"), ["query", "fixture"], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
        QMD_CHILD_BREADCRUMB: breadcrumb,
        QMD_CHILD_PID: childPidPath,
        QMD_SHUTDOWN_GRACE_MS: "5000",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let childPid: number | undefined;
    try {
      await waitFor(() => existsSync(childPidPath));
      childPid = Number(readFileSync(childPidPath, "utf8").trim());
      wrapper.kill("SIGTERM");
      await waitFor(() => existsSync(breadcrumb));
      wrapper.kill("SIGTERM");
      await waitFor(() => !processExists(childPid!), 2000);
      await waitFor(() => wrapper.exitCode !== null || wrapper.signalCode !== null, 2000);
      expect(wrapper.exitCode).toBe(143);
      expect(processExists(childPid)).toBe(false);
    } finally {
      if (wrapper.exitCode === null && wrapper.signalCode === null) wrapper.kill("SIGKILL");
      if (childPid !== undefined && processExists(childPid)) process.kill(childPid, "SIGKILL");
    }
  });

  test.skipIf(process.platform === "win32")("grace expiry SIGKILLs an ignoring child and reports 130", async () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["package-lock.json"]);
    const childPidPath = join(root, "child.pid");
    writeChildFixture(packageRoot, `
const { writeFileSync } = require("node:fs");
writeFileSync(process.env.QMD_CHILD_PID, String(process.pid));
process.on("SIGINT", () => {});
process.on("SIGTERM", () => {});
setInterval(() => {}, 1 << 30);
`);

    const wrapper = spawn(join(packageRoot, "bin", "qmd"), ["query", "fixture"], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
        QMD_CHILD_PID: childPidPath,
        QMD_SHUTDOWN_GRACE_MS: "200",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let childPid: number | undefined;
    try {
      await waitFor(() => existsSync(childPidPath));
      childPid = Number(readFileSync(childPidPath, "utf8").trim());
      wrapper.kill("SIGINT");
      await waitFor(() => !processExists(childPid!), 2000);
      await waitFor(() => wrapper.exitCode !== null || wrapper.signalCode !== null, 2000);
      expect(wrapper.exitCode).toBe(130);
      expect(processExists(childPid)).toBe(false);
    } finally {
      if (wrapper.exitCode === null && wrapper.signalCode === null) wrapper.kill("SIGKILL");
      if (childPid !== undefined && processExists(childPid)) process.kill(childPid, "SIGKILL");
    }
  });
});

describe("bin/qmd supervision marker", () => {
  test.skipIf(process.platform === "win32")("marks the child QMD_SUPERVISED=1 and leaves Metal env alone", async () => {
    const { root, runtimeBin } = makeTempFixture();
    const packageRoot = makePackage(root, "node_modules/@tobilu/qmd", ["package-lock.json"]);
    const envDump = join(root, "child-env.json");
    writeFileSync(join(packageRoot, "dist", "cli", "qmd.js"), `
const { writeFileSync } = require("node:fs");
writeFileSync(process.env.QMD_CHILD_ENV, JSON.stringify({
  supervised: process.env.QMD_SUPERVISED ?? null,
  keep: process.env.QMD_METAL_KEEP_RESIDENCY ?? null,
  noResidency: process.env.GGML_METAL_NO_RESIDENCY ?? null,
}));
`);

    const wrapper = spawn(join(packageRoot, "bin", "qmd"), ["query", "fixture"], {
      env: {
        ...process.env,
        PATH: `${runtimeBin}:${process.env.PATH ?? ""}`,
        QMD_CHILD_ENV: envDump,
        QMD_METAL_KEEP_RESIDENCY: undefined,
        GGML_METAL_NO_RESIDENCY: undefined,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    try {
      await waitFor(() => existsSync(envDump), 3000);
      await waitFor(() => wrapper.exitCode !== null || wrapper.signalCode !== null, 3000);
      expect(JSON.parse(readFileSync(envDump, "utf8"))).toEqual({
        supervised: "1",
        keep: null,
        noResidency: null,
      });
    } finally {
      if (wrapper.exitCode === null && wrapper.signalCode === null) wrapper.kill("SIGKILL");
    }
  });
});
