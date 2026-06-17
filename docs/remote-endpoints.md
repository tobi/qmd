# Remote Endpoints

QMD can delegate different LLM roles to remote providers while
keeping local GGUF behavior as the default.

## Roles

QMD treats remote endpoints as four independent roles:

- `embed`: vector indexing and vector-backed retrieval
- `expand`: query expansion
- `rerank`: candidate reranking
- `generate`: general text generation

You can configure one role, all roles, or any mix.

## Activation

QMD stays local unless a remote endpoint URL is configured.

Remote mode is activated by any of:

- environment variables such as `QMD_EMBED_BASE_URL`
- YAML config under `models.*_api_url`
- the backward-compatible embed fallback `OPENAI_BASE_URL`

Config precedence is:

1. environment variables
2. YAML config
3. backward-compatible embed fallbacks

## Formats

Each role has an explicit protocol format.

### Embed

- `auto`
- `openai_v1_embeddings`
- `cohere_v2_embed`
- `ollama_embed`
- `vllm_pooling`

### Expand

- `auto`
- `openai_chat_completions`
- `openai_completions`
- `openai_responses`
- `anthropic_messages`
- `ollama_chat`
- `ollama_generate`

### Rerank

- `auto`
- `cohere_v1_rerank`
- `cohere_v2_rerank`
- `vllm_score`

### Generate

- `auto`
- `openai_chat_completions`
- `openai_completions`
- `openai_responses`
- `anthropic_messages`
- `ollama_chat`
- `ollama_generate`

## Format to Endpoint Mapping

These are the typical endpoint shapes each format expects.

| Role | Format | Typical route(s) |
|------|--------|------------------|
| `embed` | `openai_v1_embeddings` | `/v1/embeddings` |
| `embed` | `cohere_v2_embed` | `/v2/embed` |
| `embed` | `ollama_embed` | `/api/embed`, `/embed` |
| `embed` | `vllm_pooling` | `/pooling`, `/v1/pooling` |
| `expand` / `generate` | `openai_chat_completions` | `/v1/chat/completions` |
| `expand` / `generate` | `openai_completions` | `/v1/completions` |
| `expand` / `generate` | `openai_responses` | `/v1/responses` |
| `expand` / `generate` | `anthropic_messages` | `/v1/messages` |
| `expand` / `generate` | `ollama_chat` | `/api/chat` |
| `expand` / `generate` | `ollama_generate` | `/api/generate` |
| `rerank` | `cohere_v1_rerank` | `/v1/rerank`, `/rerank` |
| `rerank` | `cohere_v2_rerank` | `/rerank`, `/v1/rerank`, `/v2/rerank` |
| `rerank` | `vllm_score` | `/score`, `/v1/score` |

Adapters may normalize paths and retry compatible alternatives when the first
request shape or route is rejected.

## Configuration

You can configure endpoints in YAML:

```yaml
models:
  embed_api_url: http://localhost:8000/v1
  embed_api_model: Qwen/Qwen3-Embedding-0.6B
  embed_api_format: vllm_pooling

  expand_api_url: https://api.openai.com/v1
  expand_api_model: gpt-4.1-mini
  expand_api_format: openai_responses

  rerank_api_url: https://api.cohere.com
  rerank_api_model: rerank-v3.5
  rerank_api_format: cohere_v2_rerank

  generate_api_url: http://localhost:11434
  generate_api_model: llama3.2
  generate_api_format: ollama_generate
```

Or with environment variables:

```sh
export QMD_EMBED_BASE_URL=http://localhost:8000/v1
export QMD_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
export QMD_EMBED_API_FORMAT=vllm_pooling

export QMD_EXPAND_BASE_URL=https://api.openai.com/v1
export QMD_EXPAND_MODEL=gpt-4.1-mini
export QMD_EXPAND_API_FORMAT=openai_responses
export QMD_EXPAND_API_KEY=...
```

## Mixed Provider Example

This is a representative setup where each role is delegated to a different
remote provider:

```yaml
models:
  embed_api_url: http://embed-host:8000
  embed_api_model: Qwen/Qwen3-Embedding-0.6B
  embed_api_format: vllm_pooling

  expand_api_url: https://api.openai.com/v1
  expand_api_model: gpt-4.1-mini
  expand_api_format: openai_responses
  expand_api_key: sk-...

  rerank_api_url: https://api.cohere.com
  rerank_api_model: rerank-v3.5
  rerank_api_format: cohere_v2_rerank
  rerank_api_key: sk-...

  generate_api_url: http://localhost:11434
  generate_api_model: llama3.2
  generate_api_format: ollama_generate
```

Typical workflow after configuring this:

1. run `qmd embed -f` after selecting the final embed model and format
2. validate with `qmd vsearch "test query"`
3. validate the expand + rerank path with `qmd query "test query"`

## Provider Notes

### OpenAI-compatible

- Use `openai_v1_embeddings` for embed.
- Use `openai_chat_completions`, `openai_completions`, or `openai_responses`
  for expand/generate depending on the server contract.

### Anthropic

- Use `anthropic_messages`.
- Applies to expand and generate.
- Uses `x-api-key` and the Anthropic messages payload shape.

### Cohere-compatible and vLLM

- `cohere_v2_embed` normalizes embed requests toward `/v2/embed`.
- `cohere_v1_rerank` and `cohere_v2_rerank` handle common rerank endpoint
  layouts.
- `vllm_pooling` and `vllm_score` support vLLM-style pooling and score routes.

#### Why Prefer a v2-style Embed Endpoint

When a provider offers a Cohere-compatible `/v2/embed` contract, it is often a
better fit than a generic embeddings endpoint because:

- it can distinguish query-oriented and document-oriented embeddings more
  explicitly via `input_type`
- it tends to make the embed contract clearer for retrieval workloads, where
  query/document mode matters for quality
- it maps well onto the adapter behavior here, including path and
  request-shape normalization

That does not mean `/v1/embeddings` is wrong. If your provider only exposes an
OpenAI-compatible embeddings API, `openai_v1_embeddings` remains a valid and
supported choice. The practical advantage of `/v2/embed` is mostly better fit
for providers and models that expose retrieval-specific embedding semantics.

### Ollama

- `ollama_embed` targets embed-style Ollama servers.
- `ollama_chat` and `ollama_generate` support the chat and generate APIs.

## Runtime Behavior

- `auto` keeps the legacy behavior unless a format is explicitly selected.
- Local mode remains the default when no remote URLs are configured.
- Query-aware embedding is forwarded through remote batch embedding paths so
  providers that distinguish query/document embeddings can preserve intent.
- Some adapters normalize endpoint paths and retry alternate protocol shapes
  when a compatible server exposes a slightly different contract.
- Missing expand credentials degrade to passthrough/fallback query variants.
- Missing rerank credentials degrade to uniform rerank scores instead of
  breaking search entirely.
- Generate failures return `null` rather than breaking indexing or retrieval.

## Tokenization

This branch also supports optional remote tokenizer endpoints for more accurate
token-bounded chunking in remote mode.

- default behavior still degrades safely to character-based chunking
- `QMD_REMOTE_TOKENIZER=force` can require remote tokenizer availability
- tokenization and detokenization probe `/tokenize` and `/detokenize`, with
  fallback handling for compatible `/v1/...` layouts
- remote tokenization is mainly useful when remote embedding/search semantics
  need tighter chunk-size control than character-based estimation provides
- if no tokenizer is available, chunking still works; it just uses approximate
  character-based boundaries instead of exact token counts

## Suggested Setups

### Fully local

- configure nothing
- use GGUF models as usual

### Remote embeddings, local everything else

- set only `embed_api_url`, `embed_api_model`, and `embed_api_format`

### Mixed providers

- OpenAI-compatible expand
- Cohere or vLLM rerank
- Ollama generate

### Fully remote

- configure all four roles explicitly
- run `qmd embed` after choosing the final embed model and format

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| embed requests return 404/405 | wrong protocol format or wrong base URL shape | set the matching `embed_api_format` and verify the base URL root |
| semantic search quality looks wrong after switching models | vectors were built with a different embed model or format | rebuild with `qmd embed -f` |
| rerank seems ineffective | rerank endpoint is missing, misconfigured, or falling back to uniform scores | verify `rerank_api_url`, `rerank_api_model`, and `rerank_api_format` |
| expand mostly echoes the input | expand endpoint is missing, format is wrong, or credentials are absent | verify `expand_api_url`, `expand_api_format`, and API key settings |
| chunking is unexpectedly coarse in remote mode | no remote tokenizer is available | leave fallback behavior in place or configure tokenizer support and, if needed, `QMD_REMOTE_TOKENIZER=force` |
