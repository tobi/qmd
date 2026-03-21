# Modal Region Auto-Detection Design

**Date:** 2026-03-21
**Status:** Draft
**Related:** `docs/modal-latency.md`

## Summary

Add automatic region detection to `qmd modal deploy` that selects the fastest Modal region based on network latency. This reduces network overhead for EU/AP users by ~40%.

## Background

Modal workers run in AWS regions worldwide. The current deployment defaults to US (no region specified), which causes significant latency for users in other regions:

- EU users: ~97ms RTT per Modal call (40% overhead)
- AP users: Similar latency penalty

By detecting the fastest region, users automatically get optimal latency without manual configuration.

## Requirements

1. **Auto-detect fastest region** on first `qmd modal deploy`
2. **Manual override** via `--region` flag
3. **Persist selection** in config for subsequent deploys
4. **Re-detect** via `--detect-region` flag
5. **No breaking changes** to existing deployments

## Design

### Region Mapping

Static mapping of Modal regions to AWS endpoints for latency probing:

```typescript
const REGION_ENDPOINTS: Record<string, string> = {
  'us': 'ec2.us-east-1.amazonaws.com',
  'eu': 'ec2.eu-central-1.amazonaws.com',
  'ap': 'ec2.ap-northeast-1.amazonaws.com',
  'uk': 'ec2.eu-west-2.amazonaws.com',
  'ca': 'ec2.ca-central-1.amazonaws.com',
};
```

**Note:** This is an intentional scope limitation. These 5 regions cover Modal's primary deployment locations. Future expansion can add `me`, `sa`, `af`, `mx` mapped to nearest available regions.

### Detection Algorithm

On `qmd modal deploy` (when no `--region` specified and no region in config):

1. For each region in `REGION_ENDPOINTS` (in parallel):
   - Ping endpoint 3 times with 10s timeout
   - Calculate median latency
2. Select region with lowest median latency (excluding failures)
3. If all pings fail → default to `us`, log warning
4. Store in config: `modal.region = selected_region`
5. Deploy with: `python modal/serve.py deploy --gpu T4 --region <selected>`

**Config persistence timing:** Save config AFTER successful deploy. If deploy fails, config is not saved, allowing clean retry.

### CLI Interface

```bash
# Auto-detect (first deploy, no config)
qmd modal deploy
# Output: "Detecting fastest region... eu (1.3ms)"

# Manual override
qmd modal deploy --region eu

# Re-detect (force new detection)
qmd modal deploy --detect-region

# View current region
qmd modal status
# Output includes: "Region: eu"

# Reset region (clears from config)
qmd modal deploy --region default
```

### Region Validation

Accepted region values: `us`, `eu`, `ap`, `uk`, `ca`, `me`, `sa`, `af`, `mx`, `default`.

- `default` = clear saved region, use Modal's default (US)
- Invalid values → error with accepted values list

### File Changes

**`src/cli/modal.ts`:**
- Add `--region <value>` flag to deploy command
- Add `--detect-region` flag to force re-detection
- Add `detectRegion(): Promise<string>` function
- Add `pingEndpoint(endpoint: string): Promise<number>` helper (cross-platform)
- Add `isValidRegion(value: string): boolean` validation

**`modal/serve.py`:**
- Add `--region` argument to deploy command
- Precedence: CLI `--region` > env var `QMD_MODAL_REGION` > None
- Pass `region=args.region` to `@app.cls()` decorator

**`src/collections.ts`:**
- Add `region?: string` to `ModalConfig` interface
- Add `region: ""` to `MODAL_DEFAULTS` (empty = not set)
- Document: empty string = not detected, `default` = cleared

### Config Storage

After successful deploy with region:

```yaml
# ~/.config/qmd/index.yml
modal:
  inference: true
  gpu: T4
  scaledown_window: 15
  region: eu  # Detected or manually specified
```

### Error Handling

| Scenario | Behavior |
|----------|----------|
| All pings fail | Default to `us`, log warning: "Region detection failed, using US default" |
| Some pings fail | Exclude failed regions from median calculation |
| No network | Default to `us`, log warning |
| Invalid `--region` | Error: "Invalid region 'xyz'. Valid: us, eu, ap, uk, ca, me, sa, af, mx, default" |
| Deploy fails | Don't save config (preserves ability to retry) |
| Persistent failures | Run 3 ping attempts per region with 10s timeout each |

## Implementation Notes

### Cross-Platform Ping

Use platform-aware ping command:

```typescript
import { platform } from 'os';

function getPingCommand(endpoint: string, count: number): string {
  const isWindows = platform() === 'win32';
  const pingCmd = isWindows ? 'ping' : 'ping';
  const countFlag = isWindows ? '-n' : '-c';
  return `${pingCmd} ${countFlag} ${count} ${endpoint}`;
}

function pingEndpoint(endpoint: string): Promise<number> {
  return new Promise((resolve) => {
    const cmd = getPingCommand(endpoint, 3);
    exec(cmd, { timeout: 30000 }, (error, stdout) => {
      if (error) {
        resolve(Infinity);
        return;
      }
      // Parse RTT from output
      // Linux/Mac: rtt min/avg/max/mdev = 0.5/1.2/2.0/0.5
      // Windows: Minimum = 0ms, Maximum = 2ms, Average = 1ms
      const match = stdout.match(/rtt min\/avg\/max\/mdev = [\d.]+\/([\d.]+)/) ||
                    stdout.match(/Average = (\d+)ms/);
      resolve(match ? parseFloat(match[1]) : Infinity);
    });
  });
}
```

### Parallel Ping Execution

```typescript
async function detectRegion(): Promise<string> {
  const entries = Object.entries(REGION_ENDPOINTS);
  const results = await Promise.all(
    entries.map(async ([region, endpoint]) => {
      const latency = await pingEndpoint(endpoint);
      return { region, latency };
    })
  );
  
  const valid = results.filter(r => isFinite(r.latency));
  if (valid.length === 0) {
    console.warn("Region detection failed, using US default");
    return 'us';
  }
  
  const best = valid.reduce((a, b) => a.latency < b.latency ? a : b);
  console.log(`Detected fastest region: ${best.region} (${best.latency.toFixed(1)}ms)`);
  return best.region;
}
```

### Median Calculation

```typescript
function median(values: number[]): number {
  const finite = values.filter(v => isFinite(v));
  if (finite.length === 0) return Infinity;
  
  const sorted = finite.sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 
    ? sorted[mid] 
    : (sorted[mid - 1] + sorted[mid]) / 2;
}
```

### Modal serve.py Changes

```python
# CLI argument
parser.add_argument(
    "--region",
    type=str,
    default=None,
    choices=["us", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx"],
    help="Region for Modal deployment (auto-detected if not specified)"
)

# Env var takes precedence, then CLI arg
region = os.environ.get("QMD_MODAL_REGION") or args.region

# Pass to decorator
@app.cls(
    gpu=gpu_config,
    region=region,  # None = Modal default (US)
    scaledown_window=idle_timeout,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
```

## Testing

### Unit Tests

- `pingEndpoint()` returns latency on success
- `pingEndpoint()` returns Infinity on failure
- `pingEndpoint()` works on Windows and Unix
- `median([])` returns Infinity
- `median([Infinity, Infinity])` returns Infinity
- `detectRegion()` returns lowest latency region
- `detectRegion()` defaults to 'us' when all fail
- `isValidRegion()` validates correctly

### Integration Tests

- Deploy with `--region eu` saves to config
- Deploy without region (first time) detects and saves
- Deploy without region (second time) uses config
- Deploy with `--detect-region` re-detects
- Deploy with `--region default` clears config
- Config not saved when deploy fails

### Manual Tests

- Run from EU and verify 'eu' is detected
- Run from US and verify 'us' is detected
- Run with no network and verify 'us' default
- Run on Windows and verify ping works

## Rollout

1. Merge to `feat/modal-inference` (current branch)
2. Test with EU-based deployment
3. Include in PR for modal inference feature