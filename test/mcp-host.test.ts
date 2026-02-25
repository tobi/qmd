import { describe, expect, test } from "vitest";
import { normalizeMcpHost, validateMcpHostInput } from "../src/mcp-host";

describe("mcp-host normalization", () => {
  test("falls back to localhost for undefined or blank host", () => {
    expect(normalizeMcpHost(undefined)).toEqual({ bindHost: "localhost", displayHost: "localhost" });
    expect(normalizeMcpHost("   ")).toEqual({ bindHost: "localhost", displayHost: "localhost" });
  });

  test("normalizes bracketed IPv6 input and trims whitespace", () => {
    expect(normalizeMcpHost("[::1]  ")).toEqual({ bindHost: "::1", displayHost: "[::1]" });
  });

  test("brackets plain IPv6 for display only", () => {
    expect(normalizeMcpHost("2001:db8::1")).toEqual({
      bindHost: "2001:db8::1",
      displayHost: "[2001:db8::1]",
    });
  });
});

describe("mcp-host validation", () => {
  test("accepts hostnames and IPs", () => {
    expect(() => validateMcpHostInput("localhost")).not.toThrow();
    expect(() => validateMcpHostInput("127.0.0.1")).not.toThrow();
    expect(() => validateMcpHostInput("[::1]")).not.toThrow();
  });

  test("rejects URL-like hosts", () => {
    expect(() => validateMcpHostInput("http://127.0.0.1")).toThrow("Invalid --host value");
    expect(() => validateMcpHostInput("/foo")).toThrow("Invalid --host value");
    expect(() => validateMcpHostInput("localhost/path")).toThrow("Invalid --host value");
  });
});
