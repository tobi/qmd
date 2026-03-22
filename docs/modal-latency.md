# Modal Latency Analysis

## Overview

This document analyzes network latency between a client (Frankfurt, Germany) and Modal's infrastructure, comparing US (default) vs EU regions.

**Benchmark results:** Run `python3 scripts/benchmark_modal_latency.py`

## Server Location

- **Client:** Frankfurt, Germany (EU)
- **Modal Control Plane:** us-east-1 (Virginia, USA) - always fixed
- **Modal Workers:** Default US, configurable to EU/other regions

## Network Measurements (Benchmark Results)

| Endpoint | Location | ICMP Ping |
|----------|----------|-----------|
| api.modal.com | US (control plane) | blocked |
| AWS us-east-1 | Virginia, USA | **96.7ms** |
| AWS eu-central-1 | Frankfurt, Germany | **1.3ms** |

**Potential RTT savings:** ~95ms per Modal call

## Variables

| Symbol | Meaning | US Value | EU Value |
|--------|---------|----------|----------|
| `R` | Worker round-trip time | 97ms | 1ms |
| `C` | Control plane overhead | ~400ms | ~400ms (fixed) |
| `E` | Embed inference time | ~300ms | ~300ms |
| `X` | Expand inference time | ~70ms | ~70ms |
| `K` | Rerank time per chunk | ~10ms | ~10ms |
| `D` | Documents to embed | varies | varies |
| `N` | Chunks to rerank | varies | varies |
| `B` | Embed batch size | 1 | 1 |

## Formulas

### Query Latency (with rerank)

```
L_query = L_local + C + R + X + R + (N × K)

Where:
  L_local = local SQLite + vector search (~300ms)
  C       = control plane overhead (~400ms for session)
  R       = worker round-trip (97ms US, 1ms EU)
  X       = expand inference time
  N       = chunks to rerank
  K       = rerank time per chunk
```

### Network Overhead Percentage

```
Overhead_pct = (modal_calls × R) / L_query × 100

US: (2 × 97) / 478 × 100 = 40.5%
EU: (2 × 1) / 478 × 100 = 0.6%
```

## Comparison Tables

### Query Latency (measured)

| Metric | US | EU | Savings |
|--------|-------|-------|---------|
| Median query (expand + rerank) | 478ms | ~287ms | 191ms |
| Network overhead | 40.5% | 0.6% | 39.9% |
| Per 100 queries | 47.8s | 28.7s | 19.1s |
| Per 1000 queries | 478s | 287s | 191s |

### Per-Operation Breakdown (478ms total)

| Component | Time | % of Total |
|-----------|------|------------|
| Local compute (SQLite+vector) | 300ms | 62.8% |
| Network RTT × 2 (US) | 193ms | 40.5% |
| Expand inference | 71ms | 14.9% |
| Rerank time | 10ms | 2.1% |

**Note:** Percentages don't sum to 100% because components overlap and the network overhead is additive.

## Region Options

| Region Code | Description | Price Multiplier |
|-------------|-------------|------------------|
| `us` | United States | 1.25x |
| `eu` | European Economic Area | 1.25x |
| `ap` | Asia-Pacific | 1.25x |
| `uk` | United Kingdom | 1.25x |
| `ca` | Canada | 2.5x |
| `me` | Middle East | 2.5x |
| `sa` | South America | 2.5x |
| `af` | Africa | 2.5x |
| `mx` | Mexico | 2.5x |

**Note:** Region selection adds a price multiplier on top of base GPU costs. US/EU/UK/AP regions have a 1.25x multiplier.

## Implementation

To use EU region from Frankfurt, Germany:

```python
@app.cls(
    gpu="T4",
    region="eu",  # Add this
    scaledown_window=15,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
```

Or via CLI:

```bash
qmd modal deploy --region eu
```

## Recommendations

- **EU clients:** Use `region="eu"` for ~92ms savings per Modal call
- **US clients:** Default (no region specified) is optimal
- **AP clients:** Use `region="ap"` for similar savings

For bulk operations (embedding many documents), EU region saves significant time:
- 100 docs: ~9 seconds saved
- 500 docs: ~43 seconds saved

For typical queries, savings are modest (~100-200ms per query).

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