/**
 * Unit tests for the `qmd modal` CLI subcommand group.
 *
 * Tests pre-flight checks (python3, modal pip package, ~/.modal.toml),
 * deploy/destroy config toggling, usage help, and unknown subcommand errors.
 *
 * Mocks: child_process.execSync, ModalBackend, getModalConfig/setModalConfig,
 * and process.exit/console.error/console.log for output capture.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, unlinkSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

// ---------------------------------------------------------------------------
// Shared mocks
// ---------------------------------------------------------------------------

// Mock child_process.execSync
const mockExecSync = vi.fn();
vi.mock("child_process", () => ({
  execSync: (...args: any[]) => mockExecSync(...args),
  // Preserve other exports the module might need
  spawn: vi.fn(),
}));

// Mock ModalBackend
const mockPing = vi.fn();
const mockEmbed = vi.fn();
const mockGenerate = vi.fn();
const mockDispose = vi.fn();
vi.mock("../src/modal.js", () => ({
  ModalBackend: vi.fn(() => ({
    ping: mockPing,
    embed: mockEmbed,
    generate: mockGenerate,
    dispose: mockDispose,
  })),
}));

// Mock getModalConfig / setModalConfig
const mockGetModalConfig = vi.fn(() => ({
  inference: false,
  gpu: "T4",
  scaledown_window: 15,
}));
const mockSetModalConfig = vi.fn();
vi.mock("../src/collections.js", () => ({
  getModalConfig: (...args: any[]) => mockGetModalConfig(...args),
  setModalConfig: (...args: any[]) => mockSetModalConfig(...args),
}));

// ---------------------------------------------------------------------------
// Import the handler under test
// ---------------------------------------------------------------------------

// We need to extract and test the modal CLI logic. Since the CLI is a giant
// switch statement in the main module, we'll import the module's handleModal
// function. But since it's not exported, we need to test via a helper that
// mimics what the CLI switch does.
//
// Strategy: Extract the modal handler into a separate function we can import.
// For now, we'll create and import a focused handler module.
import { handleModalCommand } from "../src/cli/modal.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Temp dir for fake ~/.modal.toml
const tempDir = mkdtempSync(join(tmpdir(), "modal-cli-test-"));
const fakeTomlPath = join(tempDir, ".modal.toml");
const missingTomlPath = join(tempDir, "nonexistent", ".modal.toml");

function setupFakeToml(): void {
  writeFileSync(fakeTomlPath, "[default]\ntoken_id = fake\ntoken_secret = fake\n");
}

afterEach(() => {
  try { unlinkSync(fakeTomlPath); } catch { /* ignore */ }
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("qmd modal CLI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetModalConfig.mockReturnValue({
      inference: false,
      gpu: "T4",
      scaledown_window: 15,
    });
  });

  // -------------------------------------------------------------------------
  // No subcommand → usage help
  // -------------------------------------------------------------------------

  test("no subcommand prints usage and exits with code 1", async () => {
    const result = await handleModalCommand([], { tomlPath: fakeTomlPath });
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toMatch(/Usage: qmd modal/);
    expect(result.stderr).toMatch(/deploy/);
    expect(result.stderr).toMatch(/status/);
    expect(result.stderr).toMatch(/destroy/);
    expect(result.stderr).toMatch(/test/);
  });

  // -------------------------------------------------------------------------
  // Unknown subcommand → error
  // -------------------------------------------------------------------------

  test("unknown subcommand prints error with available subcommands", async () => {
    const result = await handleModalCommand(["unknown"], { tomlPath: fakeTomlPath });
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toMatch(/Unknown subcommand: unknown/);
    expect(result.stderr).toMatch(/deploy, status, destroy, test/);
  });

  // -------------------------------------------------------------------------
  // deploy: pre-flight checks
  // -------------------------------------------------------------------------

  describe("deploy", () => {
    test("python3 not found → specific error message", async () => {
      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") {
          throw new Error("command not found");
        }
      });

      const result = await handleModalCommand(["deploy"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("python3 not found on PATH");
      expect(result.stderr).toContain("Python 3.10+");
      expect(result.stderr).toContain("https://python.org");
    });

    test("modal pip package missing → specific error message", async () => {
      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") return "Python 3.11.0";
        if (cmd.includes("import modal")) {
          throw new Error("ModuleNotFoundError");
        }
      });

      const result = await handleModalCommand(["deploy"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("Python 'modal' package not found");
      expect(result.stderr).toContain("pip install modal");
    });

    test("~/.modal.toml missing → specific error message", async () => {
      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") return "Python 3.11.0";
        if (cmd.includes("import modal")) return "";
      });

      const result = await handleModalCommand(["deploy"], { tomlPath: missingTomlPath });
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("Modal not authenticated");
      expect(result.stderr).toContain("modal token set");
    });

    test("success → sets modal.inference=true in config", async () => {
      setupFakeToml();

      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") return "Python 3.11.0";
        if (cmd.includes("import modal")) return "";
        // python3 serve.py deploy — succeed silently
        return "";
      });

      const result = await handleModalCommand(["deploy"], {
        tomlPath: fakeTomlPath,
        servePyPath: "/fake/modal/serve.py",
      });
      expect(result.exitCode).toBe(0);
      expect(mockSetModalConfig).toHaveBeenCalledWith({ inference: true });
      expect(result.stdout).toContain("T4");
      expect(result.stdout).toContain("$0.59/hr");
    });

    test("deploy failure → shows error", async () => {
      setupFakeToml();

      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") return "Python 3.11.0";
        if (cmd.includes("import modal")) return "";
        if (cmd.includes("serve.py")) {
          throw Object.assign(new Error("deploy failed"), {
            stderr: Buffer.from("Modal deploy error details"),
          });
        }
      });

      const result = await handleModalCommand(["deploy"], {
        tomlPath: fakeTomlPath,
        servePyPath: "/fake/modal/serve.py",
      });
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("Modal deployment failed");
    });
  });

  // -------------------------------------------------------------------------
  // destroy
  // -------------------------------------------------------------------------

  describe("destroy", () => {
    test("success → sets modal.inference=false in config", async () => {
      setupFakeToml();

      mockExecSync.mockImplementation((cmd: string) => {
        if (cmd === "python3 --version") return "Python 3.11.0";
        if (cmd.includes("import modal")) return "";
        return "";
      });

      const result = await handleModalCommand(["destroy"], {
        tomlPath: fakeTomlPath,
        servePyPath: "/fake/modal/serve.py",
      });
      expect(result.exitCode).toBe(0);
      expect(mockSetModalConfig).toHaveBeenCalledWith({ inference: false });
    });
  });

  // -------------------------------------------------------------------------
  // status
  // -------------------------------------------------------------------------

  describe("status", () => {
    test("ping success → shows reachable", async () => {
      mockPing.mockResolvedValue(true);

      const result = await handleModalCommand(["status"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toMatch(/reachable|deployed|ok/i);
    });

    test("ping failure → shows error", async () => {
      mockPing.mockRejectedValue(new Error("not reachable"));

      const result = await handleModalCommand(["status"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain("not reachable");
    });
  });

  // -------------------------------------------------------------------------
  // test
  // -------------------------------------------------------------------------

  describe("test", () => {
    test("embed and generate pass → shows pass for each", async () => {
      mockEmbed.mockResolvedValue([[0.1, 0.2, 0.3]]);
      mockGenerate.mockResolvedValue("Hello world");

      const result = await handleModalCommand(["test"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toMatch(/embed.*pass/i);
      expect(result.stdout).toMatch(/generate.*pass/i);
    });

    test("embed fails → shows fail", async () => {
      mockEmbed.mockRejectedValue(new Error("embed error"));
      mockGenerate.mockResolvedValue("Hello world");

      const result = await handleModalCommand(["test"], { tomlPath: fakeTomlPath });
      expect(result.exitCode).toBe(1);
      expect(result.stdout).toMatch(/embed.*fail/i);
    });
  });
});
