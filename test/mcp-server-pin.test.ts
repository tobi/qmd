/**
 * @modelcontextprotocol/server below 2.3.0 carries advisories in the HTTP
 * transport (2 high, 1 moderate). The manifest and bun.lock must resolve to a
 * fixed release.
 */

import { describe, test, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..");
const read = (f: string) => readFileSync(join(root, f), "utf8");
const MIN = [2, 3, 0];

function atLeastMin(version: string): boolean {
  const v = version.split(".").map((n) => parseInt(n, 10));
  for (let i = 0; i < MIN.length; i++) {
    if ((v[i] ?? 0) !== MIN[i]!) return (v[i] ?? 0) > MIN[i]!;
  }
  return true;
}

describe("@modelcontextprotocol/server pin", () => {
  test("package.json pins a fixed release", () => {
    const pkg = JSON.parse(read("package.json"));
    expect(atLeastMin(pkg.dependencies["@modelcontextprotocol/server"])).toBe(true);
  });

  test("bun.lock resolves a fixed release", () => {
    const m = read("bun.lock").match(/"@modelcontextprotocol\/server@([^"]+)"/);
    expect(m).not.toBeNull();
    expect(atLeastMin(m![1]!)).toBe(true);
  });
});
