import { describe, expect, test } from "vitest";
import { callCSharpSidecar } from "../src/csharp-sidecar.js";

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
});
