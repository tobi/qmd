#!/usr/bin/env python3
"""
Benchmark script to measure Modal latency overhead.

Measures:
- Ping latency to Modal control plane
- Query expand latency
- Full query latency (expand + rerank)

Calculates network overhead as percentage of total operation time.

Known network RTT values (measured from Frankfurt, Germany):
- us-east-1 (Virginia): ~93ms
- eu-central-1 (Frankfurt): ~1ms
"""

import subprocess
import time
import statistics
from dataclasses import dataclass

# Known network RTT values (measured separately)
RTT_US = 93  # ms to us-east-1
RTT_EU = 1  # ms to eu-central-1


@dataclass
class LatencyMeasurement:
    operation: str
    samples: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples)

    @property
    def stdev(self) -> float:
        if len(self.samples) > 1:
            return statistics.stdev(self.samples)
        return 0.0

    @property
    def min(self) -> float:
        return min(self.samples)

    @property
    def max(self) -> float:
        return max(self.samples)


def run_command(
    cmd: list[str], timeout: int = 60, workdir: str | None = None
) -> tuple[float, str]:
    """Run command and return (elapsed_ms, output)."""
    start = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir
    )
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, result.stdout + result.stderr


def measure_modal_status(iterations: int = 5) -> LatencyMeasurement:
    """Measure ping latency to Modal (control plane + worker wake)."""
    samples = []
    print("  Measuring Modal ping latency (status check)...")
    for i in range(iterations):
        elapsed, _ = run_command(["qmd", "modal", "status"])
        samples.append(elapsed)
        print(f"    Sample {i + 1}: {elapsed:.0f}ms")
    return LatencyMeasurement("modal_ping", samples)


def measure_query_expand(iterations: int = 5) -> LatencyMeasurement:
    """Measure query expansion latency (no rerank)."""
    samples = []
    print("  Measuring query expand latency...")

    for i in range(iterations):
        elapsed, _ = run_command(
            ["qmd", "query", "test benchmark query", "--no-rerank", "-n", "1"],
            workdir="/home/ubuntu/repos/keren-kol",
        )
        samples.append(elapsed)
        print(f"    Sample {i + 1}: {elapsed:.0f}ms")

    return LatencyMeasurement("query_expand", samples)


def measure_query_full(iterations: int = 5) -> LatencyMeasurement:
    """Measure full query latency (with rerank)."""
    samples = []
    print("  Measuring full query latency (expand + rerank)...")

    for i in range(iterations):
        elapsed, _ = run_command(
            ["qmd", "query", "communication layer design", "-n", "5"],
            workdir="/home/ubuntu/repos/keren-kol",
        )
        samples.append(elapsed)
        print(f"    Sample {i + 1}: {elapsed:.0f}ms")

    return LatencyMeasurement("query_full", samples)


def get_network_latency_from_ping():
    """Measure RTT to AWS regions using ping."""
    results = {}

    regions = {
        "us_east_1": "ec2.us-east-1.amazonaws.com",
        "eu_central_1": "ec2.eu-central-1.amazonaws.com",
    }

    import re

    for region, host in regions.items():
        try:
            result = subprocess.run(
                ["ping", "-c", "4", host], capture_output=True, text=True, timeout=30
            )
            # Try different ping output formats
            match = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", result.stdout)
            if not match:
                match = re.search(r"time=([\d.]+)\s*ms", result.stdout)
            if match:
                results[region] = float(match.group(1))
        except Exception:
            pass

    return results


def main():
    print("=" * 70)
    print("MODAL LATENCY BENCHMARK")
    print("=" * 70)

    # Warm up Modal (ensure worker is ready)
    print("\n[WARMUP] Warming up Modal worker...")
    for i in range(2):
        run_command(["qmd", "modal", "status"])
        time.sleep(0.5)
    print("  Worker is warm.")

    # Actually measure network latency
    print("\n[NETWORK LATENCY]")
    network = get_network_latency_from_ping()
    if network:
        print(f"  Measured us-east-1:     {network.get('us_east_1', RTT_US):.1f}ms")
        print(f"  Measured eu-central-1: {network.get('eu_central_1', RTT_EU):.1f}ms")
    else:
        print(f"  Using known values (from Frankfurt, Germany):")
        print(f"  us-east-1 (Virginia):   {RTT_US}ms")
        print(f"  eu-central-1 (Frankfurt): {RTT_EU}ms")

    rtt_us = network.get("us_east_1", RTT_US)
    rtt_eu = network.get("eu_central_1", RTT_EU)
    rtt_savings = rtt_us - rtt_eu
    print(f"  Potential RTT savings: {rtt_savings:.1f}ms")

    # Measure Modal operations
    print("\n[MODAL OPERATIONS]")

    ping = measure_modal_status(5)
    expand = measure_query_expand(5)
    full_query = measure_query_full(5)

    # Calculate median times (ignore cold starts)
    ping_median = ping.median
    expand_median = expand.median
    full_median = full_query.median

    # Derived values
    local_estimate = 300  # SQLite + vector search ~300ms
    control_plane_overhead = ping_median - rtt_us  # Ping includes 1 RTT +control plane

    # Calculate inference times
    expand_inference = expand_median - local_estimate - rtt_us - control_plane_overhead
    if expand_inference < 0:
        expand_inference = expand_median - local_estimate - rtt_us

    rerank_time = full_median - expand_median

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[MEDIAN LATENCIES]")
    print(f"  {'Operation':<25} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for m in [ping, expand, full_query]:
        print(
            f"  {m.operation:<25} {m.median:>8.0f}ms {m.mean:>8.0f}ms {m.min:>8.0f}ms {m.max:>8.0f}ms"
        )

    print("\n[LATENCY BREAKDOWN (per query)]")
    print(f"  Local compute (SQLite+vector): ~{local_estimate}ms")
    print(f"  Control plane overhead:          ~{control_plane_overhead:.0f}ms")
    print(f"  Expand inference time:           ~{expand_inference:.0f}ms")
    print(f"  Rerank time (derived):           ~{rerank_time:.0f}ms")
    print(f"  Network RTT (US):               {rtt_us:.0f}ms per call")
    print(f"  Network RTT (EU):               {rtt_eu:.0f}ms per call")

    # Calculate overhead percentages
    print("\n[NETWORK OVERHEAD AS PERCENTAGE]")

    # Full query has 2 Modal calls: expand + rerank
    modal_calls = 2
    network_overhead_us = modal_calls * rtt_us
    network_overhead_eu = modal_calls * rtt_eu

    overhead_pct_us = (network_overhead_us / full_median) * 100
    overhead_pct_eu = (network_overhead_eu / full_median) * 100

    print(f"  Query total time:              {full_median:.0f}ms (median)")
    print(
        f"  Network overhead (US):         {network_overhead_us:.0f}ms = {overhead_pct_us:.1f}%"
    )
    print(
        f"  Network overhead (EU):         {network_overhead_eu:.0f}ms = {overhead_pct_eu:.1f}%"
    )
    print(
        f"  Savings with EU region:        {rtt_savings * modal_calls:.0f}ms = {overhead_pct_us - overhead_pct_eu:.1f}%"
    )

    # Per-component breakdown
    print("\n[TIME BREAKDOWN (% of total)]")
    components = {
        "Local compute": local_estimate,
        f"Network RTT x{modal_calls} (US)": network_overhead_us,
        "Control plane": control_plane_overhead if control_plane_overhead > 0 else 0,
        "Expand inference": expand_inference if expand_inference > 0 else 0,
        "Rerank time": rerank_time if rerank_time > 0 else 0,
    }

    for name, value in components.items():
        pct = (value / full_median) * 100
        print(f"  {name:<30} {value:>6.0f}ms ({pct:>5.1f}%)")

    print("\n[SAVINGS IF SWITCHING TO EU REGION]")
    savings_per_call = rtt_savings
    print(
        f"  Per query ({modal_calls} calls):        {savings_per_call * modal_calls:.0f}ms"
    )
    print(
        f"  Per 100 queries:                {savings_per_call * modal_calls * 100 / 1000:.1f}s"
    )
    print(
        f"  Per 1000 queries:                {savings_per_call * modal_calls * 1000 / 1000:.1f}s"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
