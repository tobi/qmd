# Modal Region Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically detect the fastest Modal region on first deployment based on network latency.

**Architecture:** Ping AWS endpoints for each Modal region, calculate median latency, select fastest region. Persist to config. Support manual override via `--region` flag.

**Tech Stack:** TypeScript, Node.js `child_process.exec`, Python argparse

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `src/cli/modal.ts` | Modify | Add `--region`, `--detect-region` flags, detection logic |
| `modal/serve.py` | Modify | Add `--region` CLI argument |
| `src/collections.ts` | Modify | Add `region?: string` to `ModalConfig` |
| `test/cli/modal-region.test.ts` | Create | Unit tests for region detection |
| `docs/modal-latency.md` | Modify | Document region detection |

---

### Task 1: Add region to ModalConfig

**Files:**
- Modify: `src/collections.ts:39-50, 486-490`

- [ ] **Step 1: Add region field to ModalConfig interface**

```typescript
// src/collections.ts line 39
export interface ModalConfig {
  inference?: boolean;       // Whether to use Modal for inference (default: false)
  gpu?: string;              // GPU type to use (default: "T4")
  scaledown_window?: number; // Seconds before idle container scales down (default: 15)
  region?: string;           // Modal region for worker deployment (default: "" = auto-detect)
}
```

- [ ] **Step 2: Add region to MODAL_DEFAULTS**

```typescript
// src/collections.ts line 486
const MODAL_DEFAULTS: Required<ModalConfig> = {
  inference: false,
  gpu: "T4",
  scaledown_window: 15,
  region: "",  // Empty string = not set, will auto-detect on first deploy
};
```

- [ ] **Step 3: Run typecheck**

Run: `cd /home/ubuntu/repos/qmd && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/collections.ts
git commit -m "feat(modal): add region field to ModalConfig"
```

---

### Task 2: Add region detection function

**Files:**
- Modify: `src/cli/modal.ts`

- [ ] **Step 1: Add imports and constants**

```typescript
// Add at top of src/cli/modal.ts after imports
import { platform } from "os";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const REGION_ENDPOINTS: Record<string, string> = {
  us: "ec2.us-east-1.amazonaws.com",
  eu: "ec2.eu-central-1.amazonaws.com",
  ap: "ec2.ap-northeast-1.amazonaws.com",
  uk: "ec2.eu-west-2.amazonaws.com",
  ca: "ec2.ca-central-1.amazonaws.com",
};

const VALID_REGIONS = ["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx", "default"];
```

- [ ] **Step 2: Add pingEndpoint function**

```typescript
// Add after constants
async function pingEndpoint(endpoint: string, count: number = 3): Promise<number> {
  const isWindows = platform() === "win32";
  const countFlag = isWindows ? "-n" : "-c";
  const cmd = `ping ${countFlag} ${count} ${endpoint}`;
  
  try {
    const { stdout } = await execAsync(cmd, { timeout: 30000 });
    // Linux/Mac: rtt min/avg/max/mdev = 0.5/1.2/2.0/0.5
    // Windows: Minimum = 0ms, Maximum = 2ms, Average = 1ms
    const match = stdout.match(/rtt min\/avg\/max\/mdev = [\d.]+\/([\d.]+)/) ||
                  stdout.match(/Average = (\d+)ms/);
    return match ? parseFloat(match[1]) : Infinity;
  } catch {
    return Infinity;
  }
}
```

- [ ] **Step 3: Add detectRegion function**

```typescript
// Add after pingEndpoint
async function detectRegion(): Promise<string> {
  console.log("Detecting fastest Modal region...");
  
  const entries = Object.entries(REGION_ENDPOINTS);
  const results = await Promise.all(
    entries.map(async ([region, endpoint]) => {
      const latency = await pingEndpoint(endpoint);
      return { region, latency };
    })
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
```

- [ ] **Step 4: Add isValidRegion helper**

```typescript
// Add after detectRegion
function isValidRegion(value: string): boolean {
  return VALID_REGIONS.includes(value);
}
```

- [ ] **Step 5: Run typecheck**

Run: `cd /home/ubuntu/repos/qmd && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/cli/modal.ts
git commit -m "feat(modal): add region detection functions"
```

---

### Task 3: Update handleDeploy to use region

**Files:**
- Modify: `src/cli/modal.ts:101-152`

- [ ] **Step 1: Add region flag and detection to handleDeploy**

Find the existing `handleDeploy` function and update it:

```typescript
async function handleDeploy(
  tomlPath: string,
  servePyPath: string,
  region?: string,
  detectRegion?: boolean,
): Promise<ModalCommandResult> {
  const preflight = preflightChecks(tomlPath);
  if (preflight) {
    return { exitCode: 1, stdout: "", stderr: preflight };
  }

  const modalConfig = getModalConfig();
  
  // Determine region (but DON'T save to config yet)
  let selectedRegion = region;
  if (detectRegion || (!selectedRegion && !modalConfig.region)) {
    selectedRegion = await detectRegion();
  } else if (!selectedRegion) {
    selectedRegion = modalConfig.region || "us";
  }
  
  // Handle "default" = clear region
  if (selectedRegion === "default") {
    selectedRegion = undefined;
    console.log("Region cleared, using Modal default (US)");
  }
  
  // Validate region
  if (selectedRegion && !isValidRegion(selectedRegion)) {
    return {
      exitCode: 1,
      stdout: "",
      stderr: `Invalid region '${selectedRegion}'. Valid: ${VALID_REGIONS.join(", ")}`,
    };
  }

  const gpu = modalConfig.gpu;
  const scaledownWindow = modalConfig.scaledown_window;

  try {
    const regionArg = selectedRegion ? ` --region ${selectedRegion}` : "";
    execSync(
      `python3 "${servePyPath}" deploy --gpu ${gpu} --scaledown-window ${scaledownWindow}${regionArg}`,
      { stdio: "pipe" },
    );
  } catch (err: unknown) {
    // ... existing error handling
    // Note: Config NOT saved on failure - allows clean retry
  }

  // Save config AFTER successful deploy
  setModalConfig({ inference: true });
  if (selectedRegion) {
    setModalConfig({ region: selectedRegion });
  }

  // ... rest of existing code (GPU snapshot creation)
}
```

- [ ] **Step 2: Update ModalCommandOptions interface**

```typescript
export interface ModalCommandOptions {
  tomlPath?: string;
  servePyPath?: string;
  region?: string;
  detectRegion?: boolean;
}
```

- [ ] **Step 3: Update handleModalCommand switch**

```typescript
case "deploy":
  return handleDeploy(tomlPath, servePyPath, options?.region, options?.detectRegion);
```

- [ ] **Step 4: Update usage string**

```typescript
const usage = [
  "Usage: qmd modal <command>",
  "",
  "Commands:",
  "  deploy [--region <region>] [--detect-region]  Deploy the Modal inference function",
  "  status                                          Check if Modal is deployed",
  "  destroy                                         Tear down the deployed function",
  "  test                                            Run a smoke test",
  "",
  "Regions: us, eu, ap, uk, ca, me, sa, af, mx",
  "  --region default  Clear saved region (use US default)",
  "  --detect-region   Force re-detection of fastest region",
].join("\n");
```

- [ ] **Step 5: Run typecheck**

Run: `cd /home/ubuntu/repos/qmd && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add src/cli/modal.ts
git commit -m "feat(modal): integrate region detection into deploy command"
```

---

### Task 4: Update modal/serve.py for region argument

**Files:**
- Modify: `modal/serve.py`

- [ ] **Step 1: Add region argument to argparse**

Find the deploy subparser and add:

```python
# In the deploy subparser section (~line 199)
deploy_parser.add_argument(
    "--region",
    type=str,
    default=None,
    choices=["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx"],
    help="Region for Modal deployment (default: Modal's default, typically US)",
)
```

- [ ] **Step 2: Pass region to @app.cls()**

Find the `@app.cls()` decorator and update:

```python
# Get region from args or env
region = args.region or os.environ.get("QMD_MODAL_REGION")

@app.cls(
    gpu=gpu_config,
    region=region,  # Add this line
    scaledown_window=idle_timeout,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
```

- [ ] **Step 3: Commit**

```bash
git add modal/serve.py
git commit -m "feat(modal): add --region argument to serve.py deploy"
```

---

### Task 5: Write unit tests

**Files:**
- Create: `test/cli/modal-region.test.ts`

- [ ] **Step 1: Create test file with complete implementations**

```typescript
// test/cli/modal-region.test.ts
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { exec } from "child_process";
import { promisify } from "util";

// Mock child_process
vi.mock("child_process", () => ({
  exec: vi.fn(),
}));

const execMock = vi.mocked(exec);

describe("pingEndpoint", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("returns median latency from successful ping on Unix", async () => {
    const mockOutput = "PING host\nrtt min/avg/max/mdev = 0.5/1.2/2.0/0.5 ms";
    execMock.mockImplementation(((_cmd: string, _opts: any, cb: any) => {
      cb(null, { stdout: mockOutput, stderr: "" });
      return {} as any;
    }));
    
    // Import after mock is set up
    const { pingEndpoint } = await import("../../src/cli/modal.js");
    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com", 3);
    expect(result).toBe(1.2);
  });

  test("returns median latency from successful ping on Windows", async () => {
    const mockOutput = "Pinging host\nMinimum = 0ms, Maximum = 2ms, Average = 1ms";
    vi.stubGlobal("process", { platform: "win32" });
    
    execMock.mockImplementation(((_cmd: string, _opts: any, cb: any) => {
      cb(null, { stdout: mockOutput, stderr: "" });
      return {} as any;
    }));
    
    const { pingEndpoint } = await import("../../src/cli/modal.js");
    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com", 3);
    expect(result).toBe(1);
  });

  test("returns Infinity on ping failure", async () => {
    execMock.mockImplementation(((_cmd: string, _opts: any, cb: any) => {
      cb(new Error("ping failed"), { stdout: "", stderr: "error" });
      return {} as any;
    }));
    
    const { pingEndpoint } = await import("../../src/cli/modal.js");
    const result = await pingEndpoint("invalid.host", 3);
    expect(result).toBe(Infinity);
  });

  test("returns Infinity when ping output cannot be parsed", async () => {
    execMock.mockImplementation(((_cmd: string, _opts: any, cb: any) => {
      cb(null, { stdout: "unparseable output", stderr: "" });
      return {} as any;
    }));
    
    const { pingEndpoint } = await import("../../src/cli/modal.js");
    const result = await pingEndpoint("ec2.us-east-1.amazonaws.com", 3);
    expect(result).toBe(Infinity);
  });
});

describe("isValidRegion", () => {
  test("returns true for valid regions", async () => {
    const { isValidRegion, VALID_REGIONS } = await import("../../src/cli/modal.js");
    expect(isValidRegion("us")).toBe(true);
    expect(isValidRegion("eu")).toBe(true);
    expect(isValidRegion("ap")).toBe(true);
    expect(isValidRegion("uk")).toBe(true);
    expect(isValidRegion("ca")).toBe(true);
    expect(isValidRegion("me")).toBe(true);
    expect(isValidRegion("sa")).toBe(true);
    expect(isValidRegion("af")).toBe(true);
    expect(isValidRegion("mx")).toBe(true);
  });

  test("returns true for 'default'", async () => {
    const { isValidRegion } = await import("../../src/cli/modal.js");
    expect(isValidRegion("default")).toBe(true);
  });

  test("returns false for invalid regions", async () => {
    const { isValidRegion } = await import("../../src/cli/modal.js");
    expect(isValidRegion("invalid")).toBe(false);
    expect(isValidRegion("unknown")).toBe(false);
    expect(isValidRegion("")).toBe(false);
  });
});

describe("detectRegion", () => {
  test("returns fastest region from valid results", async () => {
    // Mock pingEndpoint to return known values
    vi.mock("../../src/cli/modal.js", async () => {
      const actual = await vi.importActual("../../src/cli/modal.js");
      return {
        ...actual,
        pingEndpoint: vi.fn()
          .mockResolvedValueOnce(100) // us
          .mockResolvedValueOnce(5)  // eu - fastest
          .mockResolvedValueOnce(150) // ap
          .mockResolvedValueOnce(10) // uk
          .mockResolvedValueOnce(200), // ca
      };
    });
    
    const { detectRegion } = await import("../../src/cli/modal.js");
    const result = await detectRegion();
    expect(result).toBe("eu");
  });

  test("returns 'us' when all pings fail", async () => {
    vi.mock("../../src/cli/modal.js", async () => {
      const actual = await vi.importActual("../../src/cli/modal.js");
      return {
        ...actual,
        pingEndpoint: vi.fn().mockResolvedValue(Infinity),
      };
    });
    
    const { detectRegion } = await import("../../src/cli/modal.js");
    const result = await detectRegion();
    expect(result).toBe("us");
  });

  test("returns fastest region excluding failures", async () => {
    vi.mock("../../src/cli/modal.js", async () => {
      const actual = await vi.importActual("../../src/cli/modal.js");
      return {
        ...actual,
        pingEndpoint: vi.fn()
          .mockResolvedValueOnce(Infinity) // us fails
          .mockResolvedValueOnce(10)  // eu - fastest
          .mockResolvedValueOnce(Infinity) // ap fails
          .mockResolvedValueOnce(50)  // uk
          .mockResolvedValueOnce(Infinity), // ca fails
      };
    });
    
    const { detectRegion } = await import("../../src/cli/modal.js");
    const result = await detectRegion();
    expect(result).toBe("eu");
  });
});
```

- [ ] **Step 2: Run tests**

Run: `cd /home/ubuntu/repos/qmd && npx vitest run test/cli/modal-region.test.ts`
Expected: Tests pass

- [ ] **Step 3: Commit**

```bash
git add test/cli/modal-region.test.ts
git commit -m "test(modal): add tests for region detection"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `docs/modal-latency.md`

- [ ] **Step 1: Add region detection section**

Add to `docs/modal-latency.md`:

```markdown
## Region Auto-Detection

`qmd modal deploy` automatically detects the fastest Modal region on first deployment:

1. Pings AWS endpoints for each Modal region (3 pings, median)
2. Selects region with lowest latency
3. Stores in `~/.config/qmd/index.yml`

### CLI Commands

```bash
# Auto-detect (first deploy)
qmd modal deploy

# Manual override
qmd modal deploy --region eu

# Force re-detection
qmd modal deploy --detect-region

# Clear saved region
qmd modal deploy --region default
```

### Supported Regions

| Region | Description | Endpoint |
|--------|-------------|----------|
| us | United States | ec2.us-east-1.amazonaws.com |
| eu | European Economic Area | ec2.eu-central-1.amazonaws.com |
| ap | Asia-Pacific | ec2.ap-northeast-1.amazonaws.com |
| uk | United Kingdom | ec2.eu-west-2.amazonaws.com |
| ca | Canada | ec2.ca-central-1.amazonaws.com |
```

- [ ] **Step 2: Commit**

```bash
git add docs/modal-latency.md
git commit -m "docs: add region auto-detection documentation"
```

---

### Task 7: Integration test

- [ ] **Step 1: Run full test suite**

Run: `cd /home/ubuntu/repos/qmd && npx vitest run`
Expected: All tests pass

- [ ] **Step 2: Manual deploy test with explicit region**

```bash
qmd modal deploy --region eu
```

Expected: Deploy succeeds with EU region

- [ ] **Step 3: Verify config after successful deploy**

```bash
cat ~/.config/qmd/index.yml
```

Expected: `region: eu` in modal config

- [ ] **Step 4: Test auto-detection**

```bash
# Clear region from config first
qmd modal deploy --region default
# Then test auto-detect
qmd modal deploy --detect-region
```

Expected: Detects fastest region, deploys, saves to config

- [ ] **Step 5: Test deploy uses saved config**

```bash
# After previous deploy, verify config region is used
qmd modal status
cat ~/.config/qmd/index.yml
```

Expected: Uses saved region fromconfig, no re-detection

- [ ] **Step 6: Test config NOT saved on deploy failure**

```bash
# Save current config
cp ~/.config/qmd/index.yml /tmp/config-backup.yml

# Simulate deploy failure (destroy first, then try deploy with invalid creds)
# Deploy should fail but config should remain unchanged
# Restore config
```

Expected: Config unchanged after failed deploy

---

### Task 8: Final commit and push

- [ ] **Step 1: Ensure all changes committed**

Run: `git status`
Expected: No uncommitted changes

- [ ] **Step 2: Push to branch**

```bash
git push origin feat/modal-inference
```

- [ ] **Step 3: Update PR description**

Add region auto-detection feature to PR #444 description.