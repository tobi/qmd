import { expect, test } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const project = fileURLToPath(new URL("..", import.meta.url));
const script = join(project, "src", "cli", "qmd.ts");
const runner = "Bun" in globalThis ? [script] : [join(project, "node_modules", "tsx", "dist", "cli.mjs"), script];
function run(root: string, command: string) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(process.execPath, [...runner, command], {
      cwd: root,
      env: {
        PATH: process.env.PATH, HOME: root, XDG_CONFIG_HOME: join(root, "config"), XDG_CACHE_HOME: join(root, "cache"),
        QMD_CONFIG_DIR: join(root, "config"), INDEX_PATH: join(root, "index.sqlite"), CI: "true", NO_COLOR: "1",
        QMD_DOCTOR_DEVICE_PROBE: "0", TYPESAFE_API_KEY: "synthetic-cli-key-never-send",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", data => { stdout += data; });
    child.stderr.on("data", data => { stderr += data; });
    child.on("error", reject);
    child.on("close", code => resolve({ code, stdout, stderr }));
  });
}

test("CLI status/doctor describe configured TypeSafe without inference or suppressing local embedding device checks", async () => {
  const root = mkdtempSync(join(tmpdir(), "qmd-jev-cli-"));
  mkdirSync(join(root, "config"));
  writeFileSync(join(root, "config", "index.yml"), "collections: {}\nmodels:\n  rerank: typesafe:jev-test\n  rerankOptions:\n    revision: test-v1\n    apiKeyEnv: QMD_TEST_JEV_MISSING\n");
  try {
    const status = await run(root, "status");
    expect(status.code, status.stderr).toBe(0);
    expect(status.stdout).toContain("configured remote TypeSafe (not probed; no local download)");
    const doctor = await run(root, "doctor");
    expect(doctor.code, doctor.stderr).toBe(0);
    expect(doctor.stdout).toContain("configured remote TypeSafe (not probed; no local download)");
    expect(doctor.stdout).toContain("skipped by QMD_DOCTOR_DEVICE_PROBE=0");
    expect(doctor.stdout).not.toContain("skipped for remote embedding configuration");
    expect(status.stdout + status.stderr + doctor.stdout + doctor.stderr).not.toContain("synthetic-cli-key-never-send");
    expect(status.stderr + doctor.stderr).not.toMatch(/Loading model|Downloading|LLM operations are disabled/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
