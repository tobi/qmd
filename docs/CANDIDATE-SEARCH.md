# Grouped candidate retrieval

`QMDStore.searchCandidates()` admits one document per metadata group in each
retrieval leg before reciprocal rank fusion (RRF) and the candidate cutoff.
The existing `search()`, CLI, MCP, and HTTP search paths keep their existing
options, ranking, and result shapes.

## One request through the pipeline

An application stores synthetic Orbit passages with `component: "orbit"` in
`qmd.metadata`. Twenty strong documents belong to two components. Eight
additional documents belong to eight different components. The application
asks for ten components:

```typescript
const response = await store.searchCandidates({
  queries: [{ type: "lex", query: "Orbit" }],
  candidates: {
    rawLimitPerLeg: 28,
    group: { metadataKey: "component", targetGroupsPerLeg: 10 },
  },
  candidateLimit: 20,
  limit: 10,
  rerank: false,
});
```

1. Collection scope and the validated metadata filter select eligible physical
   documents inside the backend query.
2. The backend returns up to `rawLimitPerLeg` ranked documents with metadata
   and hashes. Compact retrieval leaves document bodies in SQLite.
3. Each leg keeps its first document for each group, up to
   `targetGroupsPerLeg`. Group ranks run from `1` through the admitted count.
4. RRF receives group keys and compact group ranks. Each group contributes
   once per leg. The fused list admits up to `candidateLimit` groups.
5. Each admitted group chooses the document with the greatest weighted RRF
   contribution. Input leg order breaks equal contributions. The hydrator
   reads the body belonging to that exact path and expected content hash.
6. Keyword and intent overlap select one chunk per representative. With
   reranking enabled, QMD scores that chunk and blends the reranker score with
   the fused rank using the existing rank weights. `minScore` filters the
   blended score; `limit` cuts the final group list.

The example returns ten groups. Setting `rawLimitPerLeg: 20` observes two
groups and reports a shortfall of eight for that leg. The shortfall describes
the observed retrieval window. It leaves corpus-wide absence unproven.

## Request contract

Supply exactly one of `query` or a nonempty `queries` array. Typed `lex`,
`vec`, and `hyde` queries preserve caller order and vocabulary. The first
nonempty typed leg gets weight `2`; subsequent legs get weight `1`. A plain
`query` runs original lexical and vector legs at weight `2`, then expanded
legs at weight `1`. This candidate method always runs the plain-query
expansion pipeline. Typed queries provide direct control over the legs.

`rawLimitPerLeg`, `targetGroupsPerLeg`, `candidateLimit` (default `40`), and
`limit` (default `10`) accept positive safe integers. `minScore` accepts a
finite value in `[0, 1]`. `rerank` defaults to `true`. `filter` uses QMD's
strict metadata-filter grammar. `collection` and `collections` compose a
collection scope. `intent` and `chunkStrategy` use QMD's existing chunk
selection behavior; the runtime chunk-strategy default is `regex`.

Applications own configurable service ceilings, deadlines, serialized
response limits, and cumulative task budgets. Increasing raw depth increases
backend work. Vector legs run one at a time through the store-selected
embedding model. Each vector leg computes its query embedding. Hydration
processes each admitted source body in turn; the transient body and chunk
allocations follow the source document's size.
Normal file ingestion accepts source files up to `10 MB`.

## Identity, evidence, and coverage

A scalar metadata value groups documents by metadata key, scalar type, and
value. Number `1`, string `"1"`, and Boolean `true` form different groups.
Missing keys and array values fall back to the physical document URI. Group
identity belongs to metadata and paths; content hashes identify source bytes.

Each result carries its representative `file`, full `contentHash`, metadata,
context, group identity, final `score`, `rrfScore`, `rrfRank`, and
`rrfTopRankBonus`. Each `matches` entry carries its own physical path and hash,
query, backend score, raw document rank, compact group rank, weight, and RRF
contribution. The RRF score equals the contribution sum plus the top-rank
bonus. A vector contribution also carries the actual `vectorStartUtf16` and
stored `vectorChunkSeq` when compact retrieval supplies them.
Scores report ranking signals; probability calibration belongs to the caller.

`coverage.legs` reports raw depth, raw count, depth saturation, admitted group
count, and group shortfall. Each leg also records its `queryType`. Vector
legs report their final per-target KNN scan, including the collection name:
requested chunk count, returned chunk count, resolved document count, and
`backendCapReached`. That field becomes true when a scan fills SQLite's
`4096`-chunk ceiling. Raw document-depth saturation and vector chunk saturation
are separate observations. A vector leg with zero scans records an empty scan
list.

Top-level coverage counts fused, admitted, unavailable, and returned groups.
An unavailable group had a representative path/hash whose active source
disappeared before hydration. The returned counts also reflect score filtering
and the final result limit. Context lookup runs for the final returned
representatives. Applications can pin an immutable generation when
search, metadata, and subsequent source reads need one shared generation.

## Validation

From the repository root:

```sh
bun run test:node test/candidate-search.test.ts
bun run test:bun test/candidate-search.test.ts
bun run test:types
bun run lint
```

The fixtures use real FTS5 and seeded sqlite-vec embeddings to verify sibling
capacity, compact-rank fusion, eligibility before grouping, typed identity,
truthful vector-cap coverage, deferred bodies, and default-path compatibility.
