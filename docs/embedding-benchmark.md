# Embedding Benchmark Results

**Date:** 2026-03-18
**Machine:** Apple M3 Max (36GB RAM, 28GB VRAM), macOS 25.3.0
**Dataset:** 63 chunks sampled from `dig_chat/outcome` (mixed Korean/English ChatGPT conversation logs)
**Chunk size:** ~700 chars (first meaningful text block, YAML frontmatter stripped)
**Script:** `scripts/benchmark-embed.ts`

## Results

| Model | Total (63 chunks) | Per chunk | ~Tokens | ~Cost | Dims |
|-------|:-----------------:|:---------:|--------:|------:|-----:|
| embeddinggemma-300M (local) | 4,536ms | 72ms | 14,479 | $0 (local) | 768 |
| text-embedding-3-small | 1,007ms | 16ms | 14,479 | $0.000290 | 1536 |
| text-embedding-3-large | 829ms | 13ms | 14,479 | $0.001882 | 3072 |
| gemini-embedding-001 | 2,388ms | 38ms | 14,479 | free | 3072 |
| gemini-embedding-2-preview | 2,576ms | 41ms | 14,479 | free | 3072 |

## API Details

- **OpenAI:** `POST /v1/embeddings` — array input, batches of 32
- **Gemini:** `POST /v1beta/models/{model}:batchEmbedContents` — requests array, batches of 32
- **Local:** node-llama-cpp with Metal (MPS) auto-enabled, batches of 8

> `asyncBatchEmbedContent` is a GCS-based async API for large-scale batch jobs and does not support inline content — not usable here.

## Projected: full dig_chat/outcome (7,149 files, ~57,000 chunks estimated)

| Model | Estimated time | Estimated cost |
|-------|:--------------:|:--------------:|
| embeddinggemma-300M (local) | ~45 min | $0 |
| text-embedding-3-small | ~15 min | ~$0.26 |
| text-embedding-3-large | ~12 min | ~$1.70 |
| gemini-embedding-001 | ~36 min | free |
| gemini-embedding-2-preview | ~39 min | free |

Assumes ~8 chunks per file (based on 15KB Korean-text files at 900 tokens/chunk).

## Key Observations

1. **Speed ranking:** text-embedding-3-large ≈ small > gemini-001 ≈ gemini-2-preview > local GGUF.
   OpenAI is 2–3× faster than Gemini — both use batch APIs, so the difference is infrastructure/model throughput, not batching strategy.

2. **Local GGUF:** Metal (MPS) is auto-enabled, but still slower than cloud APIs. The 72ms/chunk figure includes model load overhead — pure inference is ~48ms/chunk. Best for offline or privacy-sensitive workloads.

3. **Cost/performance:**
   - Speed-first → `text-embedding-3-small` (fast + cheap)
   - Free + 3072 dims → `gemini-embedding-001` (solid throughput)
   - Fully local → `embeddinggemma-300M` (slower, $0, private)

4. **RAM note (M3 Max):** With low free RAM, the embedding model loads into VRAM (28GB available), keeping CPU RAM impact minimal. However, the Node.js process itself still consumes CPU RAM.

## Recommended Config

```bash
# Speed + cost balance (recommended)
export QMD_EMBED_API_URL="https://api.openai.com/v1"
export QMD_EMBED_API_KEY="sk-..."
export QMD_EMBED_API_MODEL="text-embedding-3-small"

# Free + high-dimensional
export QMD_EMBED_API_URL="https://generativelanguage.googleapis.com/v1beta"
export QMD_EMBED_API_KEY="AIza..."
export QMD_EMBED_API_MODEL="gemini-embedding-001"

# Fully local (default, no config needed)
unset QMD_EMBED_API_URL
```
