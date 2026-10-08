# Source locations and bounded passages

`searchCandidates({ locations: true, ... })` returns typed source locations
and one bounded passage for each representative. `passage` sets the per-call
UTF-8 byte limit and an optional Unicode-scalar limit. The location mode's
default passage limit is `48,000` UTF-8 bytes. An explicit `passage` request
returns the bounded text even when `locations` is omitted.

## One Orbit passage through retrieval and reading

A synthetic Orbit document begins with `9,001` emoji and stores its nearest
vector chunk at UTF-16 offset `18,002`. The application retrieves that component:

```typescript
const response = await store.searchCandidates({
  queries: [{ type: "vec", query: "Orbit configuration commands" }],
  candidates: {
    rawLimitPerLeg: 80,
    group: { metadataKey: "component", targetGroupsPerLeg: 10 },
  },
  locations: true,
  passage: { maxUtf8Bytes: 48_000, maxUnicodeScalars: 12_000 },
  limit: 10,
});
```

The compact vector result carries the stored chunk sequence and start. Group
fusion chooses a representative contribution and records its
`representativeLeg`. The hydrator reads the body at that physical URI and
expected content hash. The passage builder centers a bounded window around
offset `18,002`, preserving complete Unicode scalars. The reranker scores that
same bounded passage. The result carries `bodyAnchorStatus: "located"`, the
passage's source range, text, byte count, scalar count, and clipping flag.

The vector location carries `kind: "vector_chunk_start"`, the document URI,
full content hash, `startUtf16: 18002`, the stored `chunkSeq`, and
`endUtf16: null`. QMD's vector records contain a start and sequence. The
stored data leaves the vector chunk's exact end unknown.

The application verifies the full document hash before slicing that document
at the returned UTF-16 offsets. A source adapter can then convert the checked
range into source-message offsets and Unicode-scalar coordinates.

## Location origins

| `kind` | Evidence | Range |
|---|---|---|
| `lexical_exact` | A positive literal or quoted phrase exists in the representative body | Exact UTF-16 span and actual `matchedText` |
| `lexical_approximate` | A supported stem or tokenizer-equivalent phrase matches body tokens | Approximate UTF-16 token span, `reason: "stem_or_tokenizer"` |
| `vector_chunk_start` | The vector backend returned this stored chunk | Stored UTF-16 start, stored sequence, unknown end |
| `selection_window` | Keyword and intent overlap chose a chunk | Selected UTF-16 range, `origin: "keyword_intent"` |

Negative lexical clauses contribute zero positive anchors. Compound and CJK
terms follow QMD's lexical query grammar. The approximate helper covers a
small set of token and stem equivalences. FTS5 remains the authority for
retrieval. Title/path hits and unsupported body stems return
`bodyAnchorStatus: "unavailable"` and a labelled selection window.

Each vector contribution retains its own URI, content hash, start, and
sequence. A sibling's vector anchor points into that sibling's source.
The representative's lexical anchor comes from the representative contribution
and body. Every passage belongs to the result's representative `file` and
`contentHash`. The result's `representativeLeg` identifies the contribution
that selected the representative. Its typed location records the anchor or
selection span used to build the passage.

## Limits and clipping

`maxUtf8Bytes` and optional `maxUnicodeScalars` accept nonnegative safe
integers. Budgets validate before retrieval or model work. The builder expands
around the requested anchor, alternating left and right within the available
budget. A full anchor that exceeds the budget keeps its location and returns
a clipped passage with `anchorClipped: true`. A zero-byte budget returns empty
text and exact zero usage. Every returned window preserves scalar boundaries.

With reranking enabled, QMD also measures the passage with the selected
reranker's tokenizer. The available document tokens account for the context
window, template overhead, and the same intent-prefixed query used for
scoring. QMD shrinks the source window around its anchor until a measured
passage fits both budgets. The result reports the fitted source range and
usage. Token fitting preserves the full anchor when it fits both budgets. A
clipped anchor remains labelled `anchorClipped: true`. This path loads the
reranker tokenizer even when scoring can reuse cached scores.

The service adapter owns configurable ceilings and serialized-envelope limits.
These SDK budgets apply to each passage. The SDK hydrates each admitted source
body in turn, so temporary allocation follows the source size. The returned
result contains bounded passages and compact evidence. Generic late lexical
markers remain locatable past QMD's ordinary search-body prefix cap.

Exact locations retain the complete `matchedText` anchor, including an anchor
larger than its clipped passage. Query traces and metadata also carry text
outside `passage.text`. The service applies its request and metadata ceilings
and measures the full serialized envelope before returning a response.

## Validation

From the repository root:

```sh
bun run test:node test/search-locations.test.ts test/candidate-locations.test.ts test/rerank-passage.test.ts
bun run test:bun test/search-locations.test.ts test/candidate-locations.test.ts test/rerank-passage.test.ts
bun run test:types
bun run lint
```

Fixtures verify late lexical and vector starts, astral Unicode, complete-scalar
boundaries, positive/negative query clauses, clipped anchors, unavailable body
anchors, distinct sibling provenance, and equality between the returned
passage and the text passed to the model's scoring context. Token-budget
fixtures cover intent, clipped Unicode anchors, and nonmonotonic token counts.
A short-body regression verifies termination at
both document edges.
