# Embedding Benchmark Results

**Date:** 2026-03-18
**Machine:** Apple M3 Max (36GB RAM, 28GB VRAM), macOS 25.3.0
**Dataset:** 63 chunks sampled from `dig_chat/outcome` (한국어+영어 혼합 ChatGPT 대화록)
**Chunk size:** ~700 chars (첫 번째 의미 블록, YAML frontmatter 제외)
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

- **OpenAI:** `POST /v1/embeddings` — input 배열로 배치 전송 (batch_size=32)
- **Gemini:** `POST /v1beta/models/{model}:batchEmbedContents` — requests 배열로 배치 전송 (batch_size=32)
- **Local:** node-llama-cpp + Metal (MPS) 자동 활성화, batch_size=8

> `asyncBatchEmbedContent`는 GCS 기반 대용량 비동기 API로 inline content 미지원 — 사용 불가

## Projected: dig_chat/outcome 전체 (7,149 파일, ~57,000 청크 추정)

| Model | 예상 시간 | 예상 비용 |
|-------|:---------:|:---------:|
| embeddinggemma-300M (local) | ~45분 | $0 |
| text-embedding-3-small | ~15분 | ~$0.26 |
| text-embedding-3-large | ~12분 | ~$1.70 |
| gemini-embedding-001 | ~36분 | 무료 |
| gemini-embedding-2-preview | ~39분 | 무료 |

청크 수는 파일당 평균 ~8청크(한국어 15KB 기준, 900토큰/청크) 가정.

## Key Observations

1. **속도 순위:** text-embedding-3-large ≒ small > gemini-001 ≒ gemini-2-preview > local GGUF
   OpenAI가 Gemini보다 2~3배 빠른 건 배치 API 방식 차이가 아니라 인프라/모델 추론 속도 차이.

2. **로컬 GGUF:** Metal(MPS) 자동 활성화됨에도 API 대비 느림. 72ms/chunk의 상당 부분은 모델 로딩 오버헤드 — 순수 추론만 약 48ms/chunk 추정. 오프라인/프라이버시 요건 시 적합.

3. **비용 대비 성능:**
   - 속도 우선 → `text-embedding-3-small` (저렴 + 빠름)
   - 무료 + 3072차원 → `gemini-embedding-001` (속도 준수)
   - 완전 로컬 → `embeddinggemma-300M` (느리지만 $0, 프라이버시 보장)

4. **RAM 주의 (M3 Max 기준):** 로컬 embed 실행 시 RAM 여유가 적으면 VRAM(28GB)에 모델이 올라가 CPU RAM 영향은 적지만, Node.js 프로세스 자체 RAM 사용 주의.

## Recommended Config

```bash
# 비용 최소 + 속도 균형 (추천)
export QMD_EMBED_API_URL="https://api.openai.com/v1"
export QMD_EMBED_API_KEY="sk-..."
export QMD_EMBED_API_MODEL="text-embedding-3-small"

# 완전 무료 + 고차원
export QMD_EMBED_API_URL="https://generativelanguage.googleapis.com/v1beta"
export QMD_EMBED_API_KEY="AIza..."
export QMD_EMBED_API_MODEL="gemini-embedding-001"

# 완전 로컬 (기본값, 추가 설정 불필요)
unset QMD_EMBED_API_URL
```
