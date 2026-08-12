#!/usr/bin/env python3
"""
QMD Reranker Service — Verification Script
============================================

Runs the three verification tests from the Cowork review task:

5b. Consistency: score the same 10 (query, doc) pairs with batch_size=1
    and batch_size=8; results must match within 1e-3.
5c. Discrimination: an obviously relevant doc must outscore an obviously
    irrelevant one for the same query by a wide margin.

Usage:
    python3 verify-reranker.py [--url http://localhost:8088]

Run this on the Mac Studio after deploying the updated reranker-service.py.
"""

import argparse
import json
import sys
import time

DEFAULT_URL = "http://localhost:8088"


def rerank(url, query, documents, batch_size=None):
    """Call the reranker service and return results."""
    import urllib.request

    payload = {"query": query, "documents": documents}
    if batch_size is not None:
        payload["batch_size"] = batch_size

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/rerank",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result


def test_consistency(url):
    """
    Test 5b: Score 10 (query, doc) pairs with batch_size=1 and batch_size=8.
    Results must match within 1e-3.
    """
    print("\n=== Test 5b: Batch Consistency ===")

    query = "How does QMD memory configuration work with remote Ollama embeddings?"
    documents = [
        "QMD uses BM25 and vector search for hybrid retrieval. Embeddings are routed to Mac Studio Ollama via OLLAMA_EMBED_URL env var.",
        "The Pi 5 runs OpenClaw with cron-driven skills. It has 8GB RAM and no GPU.",
        "Tailscale is a WireGuard-based mesh VPN that creates a secure private network between devices.",
        "The reranker service loads Qwen3-Reranker-0.6B via PyTorch MPS on Apple Silicon for cross-encoder scoring.",
        "Python's Flask framework provides a lightweight HTTP server for serving ML model endpoints.",
        "SQLite is an embedded SQL database engine that stores data in a single file on disk.",
        "The Mac Studio has an M-series chip with Metal GPU acceleration and 64GB of unified memory.",
        "AgentCommons is a shared Google Drive folder connecting multiple AI agents for collaborative work.",
        "Rclone is a command-line program to manage files on cloud storage. It supports bidirectional sync.",
        "The garage temperature monitor uses a BME280 sensor on a Pi Zero to collect readings every 5 minutes.",
    ]

    # Score with batch_size=1 (sequential path)
    print(f"  Scoring {len(documents)} docs with batch_size=1...")
    t0 = time.time()
    result_seq = rerank(url, query, documents, batch_size=1)
    t_seq = time.time() - t0
    # result_seq results are sorted by score; we need to get back to original order
    seq_scores = [0.0] * len(documents)
    for r in result_seq["results"]:
        seq_scores[r["index"]] = r["score"]

    # Score with batch_size=8 (batch path)
    print(f"  Scoring {len(documents)} docs with batch_size=8...")
    t0 = time.time()
    result_batch = rerank(url, query, documents, batch_size=8)
    t_batch = time.time() - t0
    batch_scores = [0.0] * len(documents)
    for r in result_batch["results"]:
        batch_scores[r["index"]] = r["score"]

    # Compare
    max_diff = 0.0
    all_pass = True
    print(f"\n  {'idx':>3}  {'seq_score':>10}  {'batch_score':>10}  {'diff':>10}  {'pass':>5}")
    print(f"  {'---':>3}  {'---':>10}  {'---':>10}  {'---':>10}  {'---':>5}")
    for i in range(len(documents)):
        diff = abs(seq_scores[i] - batch_scores[i])
        passed = diff < 1e-3
        max_diff = max(max_diff, diff)
        if not passed:
            all_pass = False
        print(f"  {i:>3}  {seq_scores[i]:>10.6f}  {batch_scores[i]:>10.6f}  {diff:>10.6f}  {'✅' if passed else '❌':>5}")

    print(f"\n  Max diff: {max_diff:.6f}")
    print(f"  Sequential: {t_seq:.2f}s, Batch: {t_batch:.2f}s")
    print(f"  Result: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    return all_pass


def test_discrimination(url):
    """
    Test 5c: An obviously relevant doc must outscore an obviously irrelevant one
    for the same query by a wide margin.
    """
    print("\n=== Test 5c: Discrimination ===")

    query = "What is the capital of China?"
    relevant_doc = "The capital of China is Beijing, which is located in the northern part of the country and has a population of over 21 million people."
    irrelevant_doc = "Gravity is a fundamental force of nature that attracts two bodies with mass toward each other. It is responsible for the tides and the orbits of planets."

    result = rerank(url, query, [relevant_doc, irrelevant_doc], batch_size=2)

    scores = {r["index"]: r["score"] for r in result["results"]}
    rel_score = scores[0]
    irrel_score = scores[1]
    margin = rel_score - irrel_score

    print(f"  Query: {query}")
    print(f"  Relevant doc score:   {rel_score:.6f}")
    print(f"  Irrelevant doc score: {irrel_score:.6f}")
    print(f"  Margin:              {margin:.6f}")

    # A "wide margin" means at least 0.2 difference
    passed = margin > 0.2
    print(f"  Result: {'PASS ✅' if passed else 'FAIL ❌'} (margin > 0.2 required)")
    return passed


def test_health(url):
    """Check the service is running and reports the v2 format."""
    print("\n=== Health Check ===")
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"  Status: {data.get('status')}")
        print(f"  Model: {data.get('model')}")
        print(f"  Loaded: {data.get('loaded')}")
        print(f"  Device: {data.get('device')}")
        print(f"  Format: {data.get('format')}")
        if data.get("format") != "qwen3-reranker-v2":
            print("  ⚠️  WARNING: Service is not running the v2 format! Deploy the updated reranker-service.py")
            return False
        return data.get("status") == "ok"  # loaded can be False on first call
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify QMD Reranker Service (v2)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Reranker service URL (default: {DEFAULT_URL})")
    args = parser.parse_args()

    print(f"QMD Reranker Service Verification")
    print(f"URL: {args.url}")
    print(f"Format: qwen3-reranker-v2 (official model card)")

    all_pass = True

    if not test_health(args.url):
        print("\n❌ Service not healthy. Deploy the updated reranker-service.py first.")
        sys.exit(1)

    if not test_consistency(args.url):
        all_pass = False

    if not test_discrimination(args.url):
        all_pass = False

    print("\n" + "=" * 50)
    if all_pass:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 50)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()