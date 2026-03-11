import { describe, it, expect } from "vitest";
import { normalizeMcpHost, validateMcpHostInput } from "../src/mcp-host.js";

// =============================================================================
// normalizeMcpHost — 13 tests
// =============================================================================

describe("mcp-host normalization", () => {
  it("falls back to localhost for undefined or blank host", () => {
    expect(normalizeMcpHost(undefined)).toEqual({ bindHost: "localhost", displayHost: "localhost" });
    expect(normalizeMcpHost("   ")).toEqual({ bindHost: "localhost", displayHost: "localhost" });
  });

  it("normalizes bracketed IPv6 input and trims whitespace", () => {
    expect(normalizeMcpHost("[::1]  ")).toEqual({ bindHost: "::1", displayHost: "[::1]" });
  });

  it("brackets plain IPv6 for display only", () => {
    expect(normalizeMcpHost("2001:db8::1")).toEqual({
      bindHost: "2001:db8::1",
      displayHost: "[2001:db8::1]",
    });
  });

  it("preserves plain IPv4 address unchanged", () => {
    expect(normalizeMcpHost("127.0.0.1")).toEqual({
      bindHost: "127.0.0.1",
      displayHost: "127.0.0.1",
    });
  });

  it("preserves 0.0.0.0 (wildcard) unchanged", () => {
    expect(normalizeMcpHost("0.0.0.0")).toEqual({
      bindHost: "0.0.0.0",
      displayHost: "0.0.0.0",
    });
  });

  it("preserves hostname strings unchanged", () => {
    expect(normalizeMcpHost("my-server.local")).toEqual({
      bindHost: "my-server.local",
      displayHost: "my-server.local",
    });
    expect(normalizeMcpHost("localhost")).toEqual({
      bindHost: "localhost",
      displayHost: "localhost",
    });
  });

  it("handles null the same as undefined", () => {
    expect(normalizeMcpHost(null)).toEqual({ bindHost: "localhost", displayHost: "localhost" });
  });

  it("handles empty string as localhost fallback", () => {
    expect(normalizeMcpHost("")).toEqual({ bindHost: "localhost", displayHost: "localhost" });
  });

  it("trims whitespace from non-IPv6 hosts", () => {
    expect(normalizeMcpHost("  127.0.0.1  ")).toEqual({
      bindHost: "127.0.0.1",
      displayHost: "127.0.0.1",
    });
  });

  it("handles bracketed IPv6 with inner whitespace", () => {
    expect(normalizeMcpHost("[ ::1 ]")).toEqual({ bindHost: "::1", displayHost: "[::1]" });
  });

  it("handles full IPv6 address without abbreviation", () => {
    const full = "0000:0000:0000:0000:0000:0000:0000:0001";
    const result = normalizeMcpHost(full);
    expect(result.bindHost).toBe(full);
    expect(result.displayHost).toBe(`[${full}]`);
  });

  it("handles IPv4-mapped IPv6 address", () => {
    const result = normalizeMcpHost("::ffff:127.0.0.1");
    expect(result.bindHost).toBe("::ffff:127.0.0.1");
    expect(result.displayHost).toBe("[::ffff:127.0.0.1]");
  });

  it("return shape has exactly bindHost and displayHost keys", () => {
    const result = normalizeMcpHost("127.0.0.1");
    expect(Object.keys(result).sort()).toEqual(["bindHost", "displayHost"]);
  });
});

// =============================================================================
// validateMcpHostInput — 9 tests
// =============================================================================

describe("mcp-host validation", () => {
  it("accepts hostnames and IPs", () => {
    for (const valid of ["localhost", "127.0.0.1", "[::1]", "::1", "0.0.0.0", "my-server.local"]) {
      expect(() => validateMcpHostInput(valid)).not.toThrow();
    }
  });

  it("rejects URL-like hosts (://)", () => {
    for (const invalid of ["http://127.0.0.1", "/foo", "localhost/path"]) {
      expect(() => validateMcpHostInput(invalid)).toThrow(/Invalid --host value/);
    }
  });

  it("rejects flag-like values (value-stealing from parseArgs)", () => {
    for (const invalid of ["--daemon", "--port", "-p"]) {
      expect(() => validateMcpHostInput(invalid)).toThrow(/Invalid --host value/);
    }
  });

  it("rejects host:port patterns", () => {
    for (const invalid of ["localhost:8181", "127.0.0.1:8080", "[::1]:8080"]) {
      expect(() => validateMcpHostInput(invalid)).toThrow(/Invalid --host value/);
    }
  });

  it("rejects https and ftp URLs", () => {
    expect(() => validateMcpHostInput("https://localhost")).toThrow(/not a URL/);
    expect(() => validateMcpHostInput("ftp://server")).toThrow(/not a URL/);
  });

  it("rejects relative paths", () => {
    expect(() => validateMcpHostInput("./path")).toThrow(/not a URL/);
    expect(() => validateMcpHostInput("../parent")).toThrow(/not a URL/);
  });

  it("rejects --flag=value patterns", () => {
    expect(() => validateMcpHostInput("--port=8181")).toThrow(/requires a hostname/);
  });

  it("accepts link-local IPv6 address", () => {
    expect(() => validateMcpHostInput("fe80::1")).not.toThrow();
  });

  it("rejects empty string", () => {
    expect(() => validateMcpHostInput("")).toThrow(/non-empty/);
  });

  it("rejects whitespace-only string", () => {
    expect(() => validateMcpHostInput("   ")).toThrow(/non-empty/);
  });

  it("rejects empty brackets", () => {
    expect(() => validateMcpHostInput("[]")).toThrow(/brackets.*IPv6/);
  });

  it("rejects bracketed hostname", () => {
    expect(() => validateMcpHostInput("[localhost]")).toThrow(/brackets.*IPv6/);
  });

  it("rejects bracketed non-IPv6", () => {
    expect(() => validateMcpHostInput("[foo]")).toThrow(/brackets.*IPv6/);
  });

  it("rejects bracketed IPv4", () => {
    expect(() => validateMcpHostInput("[127.0.0.1]")).toThrow(/brackets.*IPv6/);
  });

  it("rejects whitespace in host", () => {
    expect(() => validateMcpHostInput("foo bar")).toThrow(/whitespace/);
  });

  it("rejects tab in host", () => {
    expect(() => validateMcpHostInput("foo\tbar")).toThrow(/whitespace/);
  });

  it("error messages include the raw input value", () => {
    try {
      validateMcpHostInput("http://bad");
      expect.unreachable("should have thrown");
    } catch (e: any) {
      expect(e.message).toContain("http://bad");
    }
  });
});
