import { mkdtempSync, readFileSync } from "node:fs";
import { rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test } from "vitest";
import { callCSharpSidecar } from "../src/csharp-sidecar.js";

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitForExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!processExists(pid)) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  return !processExists(pid);
}

describe("callCSharpSidecar", () => {
  test("returns null when sidecar command is missing", async () => {
    const result = await callCSharpSidecar(
      "InventoryService.cs",
      "public class Foo {}",
      {
        command: "missing-sidecar-command",
        timeoutMs: 50,
      },
    );

    expect(result).toBeNull();
  });

  test("returns null on timeout and terminates the spawned sidecar process", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "qmd-csharp-sidecar-"));
    const pidFile = join(tempDir, "sidecar.pid");
    const script = `require("node:fs").writeFileSync(${JSON.stringify(pidFile)}, String(process.pid)); setInterval(function () {}, 1000);`;
    const command = `"${process.execPath}" -e ${JSON.stringify(script)}`;

    try {
      const startedAt = Date.now();
      const result = await callCSharpSidecar(
        "InventoryService.cs",
        "public class Foo {}",
        {
          command,
          timeoutMs: 100,
        },
      );
      const elapsedMs = Date.now() - startedAt;

      expect(result).toBeNull();
      expect(elapsedMs).toBeLessThan(1500);

      const childPid = Number.parseInt(readFileSync(pidFile, "utf8"), 10);
      expect(Number.isNaN(childPid)).toBe(false);
      expect(await waitForExit(childPid, 2000)).toBe(true);
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  });
});
