# Remote LLM Mode

QMD can offload embeddings, query expansion, and reranking to remote
[llama-server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
instances instead of loading models locally. This lets you run QMD on a
lightweight machine while a GPU server handles inference.

```
QMD (client)  ──HTTP──→  llama-server (GPU)
  BM25 index               embedding   (:8080)
  SQLite DB                 reranker    (:8081)
  query dispatch             generation  (:8082)
```

## Client setup

Enable remote mode and point QMD at your server(s):

```bash
export QMD_REMOTE_MODE=1
export QMD_REMOTE_EMBED_URL=http://gpu-host:8080
export QMD_REMOTE_RERANK_URL=http://gpu-host:8081
export QMD_REMOTE_GENERATE_URL=http://gpu-host:8082

# Optional: bearer token (if using the auth gateway)
export QMD_REMOTE_API_KEY=your-token-here

# Optional: request timeout (default 30000ms)
export QMD_REMOTE_TIMEOUT_MS=30000

# Optional: override model names sent in API requests
# Useful when routing through LiteLLM or other proxies with custom model aliases
export QMD_REMOTE_EMBED_MODEL=embeddinggemma-300m
export QMD_REMOTE_GENERATE_MODEL=qmd-query-expansion
export QMD_REMOTE_RERANK_MODEL=qwen3-reranker
```

Then use QMD as usual — `qmd embed`, `qmd query`, MCP tools all work
transparently.

If the rerank endpoint is unavailable (404/405), QMD falls back to
retrieval-order scoring so searches still complete.

### Single gateway URL

If you deploy with the included nginx gateway (see below), all three
services share one URL. Point all three env vars at the gateway:

```bash
export QMD_REMOTE_EMBED_URL=http://gpu-host:8080
export QMD_REMOTE_RERANK_URL=http://gpu-host:8080
export QMD_REMOTE_GENERATE_URL=http://gpu-host:8080
export QMD_REMOTE_API_KEY=your-gateway-token
```

## Server deployment

A ready-to-use Docker Compose stack is provided in `deploy/remote-api/`.

### Quick start

```bash
cd deploy/remote-api
cp .env.example .env
# Edit .env — set QMD_GATEWAY_API_KEY to a strong random token

# CPU mode:
docker compose up -d

# NVIDIA GPU mode:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Models are downloaded automatically from HuggingFace on first start
(~300 MB embed + ~1 GB generate + ~600 MB rerank). Progress is visible via
`docker logs -f qmd-llama-embed`.

### What the stack provides

- **nginx gateway** — bearer-token auth + route dispatch on port 8080
- **llama-server (embed)** — `/v1/embeddings` on port 8011
- **llama-server (generate)** — `/v1/chat/completions` on port 8012
- **llama-server (rerank)** — `/rerank` on port 8013

### Smoke test

```bash
# Must return 401 (no token):
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/embeddings

# Must return 200:
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":"hello world"}' | head -c 120
```

### Updating models

Override HuggingFace repos in `.env`:

```env
QMD_EMBED_HF_REPO=ggml-org/embeddinggemma-300M-GGUF
QMD_EMBED_HF_FILE=embeddinggemma-300M-Q4_K_M.gguf
```

Then `docker compose restart llama-embed`.

To clear cached models: `docker compose down -v`.

### Running without Docker

You can also run llama-server directly:

```bash
# Embedding
llama-server --model embed.gguf --host 0.0.0.0 --port 8080 \
  --embedding --n-gpu-layers 99

# Reranker
llama-server --model rerank.gguf --host 0.0.0.0 --port 8081 \
  --reranking --n-gpu-layers 99 --ctx-size 2048

# Query expansion
llama-server --model qexpand.gguf --host 0.0.0.0 --port 8082 \
  --n-gpu-layers 99
```

## Migration notes

- Existing local installs need **no change** — local mode remains the default.
- Remote mode uses a byte-level tokenizer fallback for chunk sizing (slightly
  less precise than the local model tokenizer, but sufficient for search).
- The same CLI commands and MCP tools work in both modes.
