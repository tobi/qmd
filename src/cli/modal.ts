/**
 * Handler for the `qmd modal` CLI subcommand group.
 *
 * Provides deploy, status, destroy, and test subcommands for managing
 * the Modal inference backend. Pre-flight checks validate python3,
 * the modal pip package, and ~/.modal.toml before deploy/destroy.
 */

import { execSync, exec } from "child_process";
import { existsSync } from "fs";
import { join } from "path";
import { homedir, platform } from "os";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { promisify } from "util";
import { getModalConfig, setModalConfig } from "../collections.js";
import { ModalBackend } from "../modal.js";

const execAsync = promisify(exec);

const REGION_ENDPOINTS: Record<string, string> = {
  us: "ec2.us-east-1.amazonaws.com",
  eu: "ec2.eu-central-1.amazonaws.com",
  ap: "ec2.ap-northeast-1.amazonaws.com",
  uk: "ec2.eu-west-2.amazonaws.com",
  ca: "ec2.ca-central-1.amazonaws.com",
};

const VALID_REGIONS = ["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx", "default"];

async function pingEndpoint(endpoint: string, count: number = 3): Promise<number> {
  const isWindows = platform() === "win32";
  const countFlag = isWindows ? "-n" : "-c";
  const cmd = `ping ${countFlag} ${count} ${endpoint}`;

  try {
    const { stdout } = await execAsync(cmd, { timeout: 30000 });
    const linuxMatch = stdout.match(/rtt min\/avg\/max\/mdev = [\d.]+\/([\d.]+)/);
    const winMatch = stdout.match(/Average = (\d+)ms/);
    const match = linuxMatch ?? winMatch;
    return match?.[1] ? parseFloat(match[1]) : Infinity;
  } catch {
    return Infinity;
  }
}

async function detectRegion(): Promise<string> {
  console.log("Detecting fastest Modal region...");

  const entries = Object.entries(REGION_ENDPOINTS);
  const results = await Promise.all(
    entries.map(async ([region, endpoint]) => {
      const latency = await pingEndpoint(endpoint);
      return { region, latency };
    }),
  );

  const valid = results.filter(r => isFinite(r.latency));
  if (valid.length === 0) {
    console.warn("Warning: Region detection failed, using US default");
    return "us";
  }

  const best = valid.reduce((a, b) => a.latency < b.latency ? a : b);
  console.log(`Detected fastest region: ${best.region} (${best.latency.toFixed(1)}ms median)`);
  return best.region;
}

function isValidRegion(value: string): boolean {
  return VALID_REGIONS.includes(value);
}

// ============================================================================
// Types
// ============================================================================

/** Result of a modal CLI command (for testability — no process.exit). */
export interface ModalCommandResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

/** Options for handleModalCommand (overrides for testing). */
export interface ModalCommandOptions {
  /** Override ~/.modal.toml path (for testing). */
  tomlPath?: string;
  /** Override serve.py path (for testing). */
  servePyPath?: string;
}

// ============================================================================
// Pre-flight checks
// ============================================================================

function checkPython3(): string | null {
  try {
    execSync("python3 --version", { stdio: "pipe" });
    return null;
  } catch {
    return (
      "Error: python3 not found on PATH.\n" +
      "Modal deployment requires Python 3.10+.\n" +
      "Install it from https://python.org or via your package manager."
    );
  }
}

function checkModalPip(): string | null {
  try {
    execSync('python3 -c "import modal"', { stdio: "pipe" });
    return null;
  } catch {
    return (
      "Error: Python 'modal' package not found.\n" +
      "Install it with: pip install modal\n" +
      "Then authenticate with: modal token set"
    );
  }
}

function checkModalToml(tomlPath: string): string | null {
  if (!existsSync(tomlPath)) {
    return (
      "Error: Modal not authenticated. No ~/.modal.toml found.\n" +
      "Run: modal token set\n" +
      "to authenticate with your Modal account."
    );
  }
  return null;
}

/**
 * Run all pre-flight checks required for deploy/destroy.
 * Returns an error string if any check fails, null if all pass.
 */
function preflightChecks(tomlPath: string): string | null {
  return checkPython3() ?? checkModalPip() ?? checkModalToml(tomlPath);
}

// ============================================================================
// Resolve serve.py path
// ============================================================================

function resolveServePyPath(): string {
  // When running from dist/cli/modal.js or src/cli/modal.ts,
  // serve.py is at ../../modal/serve.py relative to the file.
  const thisDir = dirname(fileURLToPath(import.meta.url));
  return join(thisDir, "..", "..", "modal", "serve.py");
}

// ============================================================================
// Subcommand handlers
// ============================================================================

async function handleDeploy(
  tomlPath: string,
  servePyPath: string,
): Promise<ModalCommandResult> {
  const preflight = preflightChecks(tomlPath);
  if (preflight) {
    return { exitCode: 1, stdout: "", stderr: preflight };
  }

  const modalConfig = getModalConfig();
  const gpu = modalConfig.gpu;
  const scaledownWindow = modalConfig.scaledown_window;

  try {
    execSync(
      `python3 "${servePyPath}" deploy --gpu ${gpu} --scaledown-window ${scaledownWindow}`,
      { stdio: "pipe" },
    );
  } catch (err: unknown) {
    const stderr =
      err instanceof Error && "stderr" in err
        ? (err as any).stderr?.toString() ?? ""
        : "";
    return {
      exitCode: 1,
      stdout: "",
      stderr: `Error: Modal deployment failed.\n${stderr}`,
    };
  }

  setModalConfig({ inference: true });

  // Trigger snapshot creation by calling ping() — this forces a container
  // to spin up and load models onto GPU. Without this, the first user call
  // would pay the ~40s snapshot creation cost.
  let snapshotNote = "";
  try {
    process.stdout.write("Creating GPU snapshot (loading models onto GPU, ~1 minute)...\n");
    const backend = new ModalBackend();
    await backend.ping();
    snapshotNote = "\nGPU snapshot created — subsequent cold starts will be ~6s.";
  } catch {
    snapshotNote = "\nWarning: could not create GPU snapshot. First call may be slow (~40s).";
  }

  const costNote =
    `Modal inference deployed successfully.\n` +
    `GPU: ${gpu} (~$0.59/hr, billed per second, scales to zero when idle)` +
    snapshotNote;

  return { exitCode: 0, stdout: costNote, stderr: "" };
}

async function handleDestroy(
  tomlPath: string,
  servePyPath: string,
): Promise<ModalCommandResult> {
  const preflight = preflightChecks(tomlPath);
  if (preflight) {
    return { exitCode: 1, stdout: "", stderr: preflight };
  }

  try {
    execSync(`python3 "${servePyPath}" destroy`, { stdio: "pipe" });
  } catch (err: unknown) {
    const stderr =
      err instanceof Error && "stderr" in err
        ? (err as any).stderr?.toString() ?? ""
        : "";
    return {
      exitCode: 1,
      stdout: "",
      stderr: `Error: Modal destroy failed.\n${stderr}`,
    };
  }

  setModalConfig({ inference: false });

  return { exitCode: 0, stdout: "Modal inference function destroyed.", stderr: "" };
}

async function handleStatus(
  tomlPath: string,
): Promise<ModalCommandResult> {
  const backend = new ModalBackend({ tomlPath });
  try {
    await backend.ping();
    return { exitCode: 0, stdout: "Modal inference function is deployed and reachable.", stderr: "" };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return { exitCode: 1, stdout: "", stderr: `Error: ${message}` };
  } finally {
    backend.dispose();
  }
}

async function handleTest(
  tomlPath: string,
): Promise<ModalCommandResult> {
  const backend = new ModalBackend({ tomlPath });
  const lines: string[] = [];
  let failed = false;

  // Test embed
  try {
    const vectors = await backend.embed(["test"]);
    if (Array.isArray(vectors) && vectors.length > 0 && Array.isArray(vectors[0])) {
      lines.push("embed: PASS");
    } else {
      lines.push("embed: FAIL (unexpected response shape)");
      failed = true;
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    lines.push(`embed: FAIL (${message})`);
    failed = true;
  }

  // Test generate
  try {
    const text = await backend.generate("Hello", null, 5);
    if (typeof text === "string" && text.length > 0) {
      lines.push("generate: PASS");
    } else {
      lines.push("generate: FAIL (empty or non-string response)");
      failed = true;
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    lines.push(`generate: FAIL (${message})`);
    failed = true;
  }

  backend.dispose();

  return {
    exitCode: failed ? 1 : 0,
    stdout: lines.join("\n"),
    stderr: "",
  };
}

// ============================================================================
// Main handler
// ============================================================================

/**
 * Handle `qmd modal <subcommand>` CLI invocation.
 *
 * Returns a result object instead of calling process.exit directly,
 * making it testable without mocking process.exit.
 */
export async function handleModalCommand(
  args: string[],
  options?: ModalCommandOptions,
): Promise<ModalCommandResult> {
  const tomlPath = options?.tomlPath ?? join(homedir(), ".modal.toml");
  const servePyPath = options?.servePyPath ?? resolveServePyPath();
  const subcommand = args[0];

  if (!subcommand) {
    const usage = [
      "Usage: qmd modal <command>",
      "",
      "Commands:",
      "  deploy   Deploy the Modal inference function",
      "  status   Check if the Modal function is deployed and reachable",
      "  destroy  Tear down the deployed Modal function",
      "  test     Run a smoke test against the deployed function",
    ].join("\n");
    return { exitCode: 1, stdout: "", stderr: usage };
  }

  switch (subcommand) {
    case "deploy":
      return handleDeploy(tomlPath, servePyPath);
    case "destroy":
      return handleDestroy(tomlPath, servePyPath);
    case "status":
      return handleStatus(tomlPath);
    case "test":
      return handleTest(tomlPath);
    default:
      return {
        exitCode: 1,
        stdout: "",
        stderr: `Unknown subcommand: ${subcommand}\nAvailable: deploy, status, destroy, test`,
      };
  }
}
