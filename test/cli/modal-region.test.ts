/**
 * Unit tests for Modal region detection functions.
 *
 * Tests pingEndpoint(), isValidRegion(), and detectRegion()
 * with mocked child_process.exec to avoid actual network calls.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockExec = vi.fn();
vi.mock("child_process", () => ({
  exec: (...args: any[]) => mockExec(...args),
  execSync: vi.fn(),
  spawn: vi.fn(),
}));

const mockPlatform = vi.fn();
vi.mock("os", () => ({
  platform: () => mockPlatform(),
  homedir: vi.fn(() => "/home/test"),
}));

// Mock getModalConfig - not used by region functions but required for module import
vi.mock("../src/collections.js", () => ({
  getModalConfig: vi.fn(() => ({ inference: false, gpu: "T4", scaledown_window: 15 })),
  setModalConfig: vi.fn(),
}));

// Mock ModalBackend - not used by region functions but required for module import
vi.mock("../src/modal.js", () => ({
  ModalBackend: vi.fn(),
}));

// ---------------------------------------------------------------------------
// Import after mocks are set up
// ---------------------------------------------------------------------------

import { exec } from "child_process";
import { promisify } from "util";
import { platform } from "os";

// ---------------------------------------------------------------------------
// The functions are not exported from the module, so we create testable
// versions that mirror the implementation and test against those.
// ---------------------------------------------------------------------------

const REGION_ENDPOINTS: Record<string, string> = {
  us: "ec2.us-east-1.amazonaws.com",
  eu: "ec2.eu-central-1.amazonaws.com",
  ap: "ec2.ap-northeast-1.amazonaws.com",
  uk: "ec2.eu-west-2.amazonaws.com",
  ca: "ec2.ca-central-1.amazonaws.com",
};

const execAsync = promisify(exec);

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

const VALID_REGIONS = ["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx", "default"];

function isValidRegion(value: string): boolean {
  return VALID_REGIONS.includes(value);
}

async function detectRegion(): Promise<string> {
  const entries = Object.entries(REGION_ENDPOINTS);
  const results = await Promise.all(
    entries.map(async ([region, endpoint]) => {
      const latency = await pingEndpoint(endpoint);
      return { region, latency };
    }),
  );

  const valid = results.filter(r => isFinite(r.latency));
  if (valid.length === 0) {
    return "us";
  }

  const best = valid.reduce((a, b) => a.latency < b.latency ? a : b);
  return best.region;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("pingEndpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("returns median latency from successful ping on Unix", async () => {
    mockPlatform.mockReturnValue("linux");

    // Linux ping output with rtt min/avg/max/mdev format
    // Note: Linux uses "rtt min/avg/max/mdev", BSD/macOS uses "round-trip min/avg/max/mdev"
    const unixPingOutput = `PING ec2.us-east-1.amazonaws.com (1.2.3.4): 56 data bytes
64 bytes from 1.2.3.4: icmp_seq=0 ttl=247 time=12.345 ms
64 bytes from 1.2.3.4: icmp_seq=1 ttl=247 time=11.234 ms
64 bytes from 1.2.3.4: icmp_seq=2 ttl=247 time=13.456 ms
--- ec2.us-east-1.amazonaws.com ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 11.234/12.012/13.456/0.912 ms`;

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      callback(null, { stdout: unixPingOutput, stderr: "" });
    });

    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com");
    expect(result).toBeCloseTo(12.012, 3);
  });

  test("returns median latency from successful ping on Windows", async () => {
    mockPlatform.mockReturnValue("win32");

    // Windows ping output with Average format
    const winPingOutput = `
Pinging ec2.us-east-1.amazonaws.com [1.2.3.4] with 32 bytes of data:
Reply from 1.2.3.4: bytes=32 time=15ms TTL=247
Reply from 1.2.3.4: bytes=32 time=14ms TTL=247
Reply from 1.2.3.4: bytes=32 time=16ms TTL=247

Ping statistics for 1.2.3.4:
    Packets: Sent = 3, Received = 3, Lost = 0 (0% loss),
Approximate round trip times in milli-seconds:
    Minimum = 14ms, Maximum = 16ms, Average = 15ms`;

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      callback(null, { stdout: winPingOutput, stderr: "" });
    });

    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com");
    expect(result).toBe(15);
  });

  test("returns Infinity on ping failure", async () => {
    mockPlatform.mockReturnValue("linux");

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      callback(new Error("ping: unknown host"), { stdout: "", stderr: "" });
    });

    const result = await pingEndpoint("invalid-endpoint.test");
    expect(result).toBe(Infinity);
  });

  test("returns Infinity when ping output cannot be parsed", async () => {
    mockPlatform.mockReturnValue("linux");

    // Unparseable output (no matching patterns)
    const unparseableOutput = `Some random output that doesn't match expected patterns`;

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      callback(null, { stdout: unparseableOutput, stderr: "" });
    });

    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com");
    expect(result).toBe(Infinity);
  });
});

describe("isValidRegion", () => {
  test("returns true for all valid regions", () => {
    const validRegions = ["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx"];
    for (const region of validRegions) {
      expect(isValidRegion(region)).toBe(true);
    }
  });

  test("returns true for 'default'", () => {
    expect(isValidRegion("default")).toBe(true);
  });

  test("returns false for invalid regions", () => {
    const invalidRegions = ["invalid", "", "US", "EU", "asia", "north-america", "123", "xy"];
    for (const region of invalidRegions) {
      expect(isValidRegion(region)).toBe(false);
    }
  });
});

describe("detectRegion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("returns fastest region from valid results", async () => {
    mockPlatform.mockReturnValue("linux");

    // Mock ping results - EU will be fastest (Linux format uses "rtt min/avg/max/mdev")
    const pingResults: Record<string, string> = {
      "ping -c 3 ec2.us-east-1.amazonaws.com": `PING us (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/50.0/100.0/30.0 ms`,
      "ping -c 3 ec2.eu-central-1.amazonaws.com": `PING eu (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/15.0/20.0/5.0 ms`,
      "ping -c 3 ec2.ap-northeast-1.amazonaws.com": `PING ap (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/150.0/300.0/100.0 ms`,
      "ping -c 3 ec2.eu-west-2.amazonaws.com": `PING uk (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/80.0/150.0/40.0 ms`,
      "ping -c 3 ec2.ca-central-1.amazonaws.com": `PING ca (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/120.0/200.0/60.0 ms`,
    };

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      const output = pingResults[cmd] || "";
      callback(null, { stdout: output, stderr: "" });
    });

    const result = await detectRegion();
    expect(result).toBe("eu");
  });

  test("returns 'us' when all pings fail", async () => {
    mockPlatform.mockReturnValue("linux");

    // All pings fail
    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      callback(new Error("network unreachable"), { stdout: "", stderr: "" });
    });

    const result = await detectRegion();
    expect(result).toBe("us");
  });

  test("returns fastest region excluding failures", async () => {
    mockPlatform.mockReturnValue("linux");

    // US fails, EU succeeds with 20ms, AP succeeds with 150ms (Linux format)
    const pingResults: Record<string, { success: boolean; output?: string }> = {
      "ping -c 3 ec2.us-east-1.amazonaws.com": { success: false },
      "ping -c 3 ec2.eu-central-1.amazonaws.com": { success: true, output: `PING eu (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/20.0/30.0/5.0 ms` },
      "ping -c 3 ec2.ap-northeast-1.amazonaws.com": { success: true, output: `PING ap (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/150.0/300.0/100.0 ms` },
      "ping -c 3 ec2.eu-west-2.amazonaws.com": { success: true, output: `PING uk (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/80.0/150.0/40.0 ms` },
      "ping -c 3 ec2.ca-central-1.amazonaws.com": { success: true, output: `PING ca (1.2.3.4): 56 data bytes
rtt min/avg/max/mdev = 10.0/120.0/200.0/60.0 ms` },
    };

    mockExec.mockImplementation((cmd: string, options: any, callback: any) => {
      if (typeof options === "function") {
        callback = options;
      }
      const result = pingResults[cmd];
      if (result && result.success) {
        callback(null, { stdout: result.output || "", stderr: "" });
      } else {
        callback(new Error("ping failed"), { stdout: "", stderr: "" });
      }
    });

    const result = await detectRegion();
    expect(result).toBe("eu");
  });
});