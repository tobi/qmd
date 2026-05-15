import { describe, it, expect } from "vitest";
import { parseDurationMs, runWatchLoop } from "../src/watch.js";

describe("parseDurationMs", () => {
  it("parses seconds suffix", () => {
    expect(parseDurationMs("30s")).toBe(30_000);
    expect(parseDurationMs("1s")).toBe(1_000);
  });

  it("parses minutes suffix", () => {
    expect(parseDurationMs("5m")).toBe(300_000);
    expect(parseDurationMs("1m")).toBe(60_000);
  });

  it("parses hours suffix", () => {
    expect(parseDurationMs("1h")).toBe(3_600_000);
    expect(parseDurationMs("2h")).toBe(7_200_000);
  });

  it("parses millisecond suffix", () => {
    expect(parseDurationMs("500ms")).toBe(500);
    expect(parseDurationMs("1ms")).toBe(1);
  });

  it("treats bare numbers as seconds", () => {
    expect(parseDurationMs("30")).toBe(30_000);
  });

  it("accepts fractional values", () => {
    expect(parseDurationMs("1.5m")).toBe(90_000);
    expect(parseDurationMs("0.5h")).toBe(1_800_000);
  });

  it("trims surrounding whitespace", () => {
    expect(parseDurationMs("  30s  ")).toBe(30_000);
  });

  it("is case-insensitive for the unit", () => {
    expect(parseDurationMs("30S")).toBe(30_000);
    expect(parseDurationMs("5M")).toBe(300_000);
    expect(parseDurationMs("1H")).toBe(3_600_000);
    expect(parseDurationMs("500MS")).toBe(500);
  });

  it("rejects empty input", () => {
    expect(() => parseDurationMs("")).toThrow(/Invalid duration/);
    expect(() => parseDurationMs("   ")).toThrow(/Invalid duration/);
  });

  it("rejects unknown units", () => {
    expect(() => parseDurationMs("5d")).toThrow(/Invalid duration/);
    expect(() => parseDurationMs("10w")).toThrow(/Invalid duration/);
  });

  it("rejects zero and negative durations", () => {
    expect(() => parseDurationMs("0s")).toThrow(/must be > 0/);
    expect(() => parseDurationMs("-1s")).toThrow(/Invalid duration/);
  });

  it("rejects non-numeric input", () => {
    expect(() => parseDurationMs("abc")).toThrow(/Invalid duration/);
    expect(() => parseDurationMs("s")).toThrow(/Invalid duration/);
  });
});

describe("runWatchLoop", () => {
  it("runs the tick function once per cycle and stops after maxTicks", async () => {
    let count = 0;
    const result = await runWatchLoop({
      intervalMs: 5,
      maxTicks: 3,
      installSignalHandlers: false,
      tick: async () => {
        count += 1;
      },
    });

    expect(count).toBe(3);
    expect(result.ticks).toBe(3);
    expect(result.failures).toBe(0);
    expect(result.stoppedBy).toBe("max-ticks");
  });

  it("counts failures but keeps running below the threshold", async () => {
    let attempts = 0;
    const result = await runWatchLoop({
      intervalMs: 1,
      maxTicks: 4,
      maxConsecutiveFailures: 5,
      installSignalHandlers: false,
      log: () => {},
      tick: async () => {
        attempts += 1;
        if (attempts === 2) throw new Error("boom");
      },
    });

    expect(attempts).toBe(4);
    expect(result.failures).toBe(1);
    expect(result.stoppedBy).toBe("max-ticks");
  });

  it("stops after maxConsecutiveFailures and returns max-failures", async () => {
    let attempts = 0;
    const result = await runWatchLoop({
      intervalMs: 1,
      maxConsecutiveFailures: 2,
      installSignalHandlers: false,
      log: () => {},
      tick: async () => {
        attempts += 1;
        throw new Error(`fail ${attempts}`);
      },
    });

    expect(attempts).toBe(2);
    expect(result.failures).toBe(2);
    expect(result.stoppedBy).toBe("max-failures");
  });

  it("resets the consecutive-failure counter on a successful tick", async () => {
    let attempts = 0;
    const result = await runWatchLoop({
      intervalMs: 1,
      maxTicks: 5,
      maxConsecutiveFailures: 2,
      installSignalHandlers: false,
      log: () => {},
      tick: async () => {
        attempts += 1;
        // Fail, succeed, fail, succeed, succeed — never two failures in a row.
        if (attempts === 1 || attempts === 3) throw new Error("transient");
      },
    });

    expect(attempts).toBe(5);
    expect(result.failures).toBe(2);
    expect(result.stoppedBy).toBe("max-ticks");
  });

  it("rejects an invalid intervalMs", async () => {
    await expect(
      runWatchLoop({
        intervalMs: 0,
        installSignalHandlers: false,
        tick: async () => {},
      }),
    ).rejects.toThrow(/Invalid intervalMs/);
  });
});
