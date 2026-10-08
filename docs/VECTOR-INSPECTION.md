# Deep vector inspection

`QMDStore.inspectVectorIndex({ model? })` checks recorded chunk generations
and the physical partitioned vector index in one SQLite transaction. It
defaults to the store-selected embedding model and computes QMD's embedding
fingerprint from that exact model identifier. `getIndexHealth()` retains its
existing cheap scan and three-field result.

## One missing partition through inspection and repair

A synthetic Orbit document already has embeddings in collection `docs`.
The same content hash becomes active in collection `notes`. Its recorded chunks are
complete, so the ordinary pending-embedding count remains `0`. The new
collection needs a partition row for every stored chunk:

```typescript
const inspection = await store.inspectVectorIndex();
if (inspection.missingRequiredPartitionRows) {
  await store.embed({ model: inspection.model });
}
const repaired = await store.inspectVectorIndex({ model: inspection.model });
```

The first inspection reports missing required partition rows and
`structurallyReady: false`. The existing embedding path copies retained
vectors into the missing collection partition. The second inspection verifies
the physical map/vector peers and complete active coverage. A caller uses
the second result as the structural publication gate.

## Checked facts

The result reports:

- `model` and `embeddingFingerprint`: the selected recorded generation.
- `partitionState`: `absent`, `legacy`, `unreadable`, or `checked`.
  `checked` means the readable partition's peer scan completed; the counters
  determine readiness.
- `activeDocuments` and `needsEmbedding`: active paths and QMD's ordinary
  selected-model pending count.
- `inconsistentChunkLayouts`: active hashes whose stored chunks disagree on
  generation, positive `total_chunks`, the exact sequence `0..N-1`, or recorded
  position shape. Positions require safe nonnegative integers, position `0`
  for sequence `0`, and strict increase by sequence.
- `requiredPartitionRows`: selected recorded chunks multiplied by their
  distinct active content-hash/collection memberships.
- `missingRequiredPartitionRows`: required rows with an absent or inconsistent
  map/vector peer. Legacy or unreadable partitions report `null`.
- `inconsistentPeerRows`: physical rowids with a missing peer, a partition
  mismatch, an unknown collection, or an absent recorded content chunk.
  Legacy or unreadable partitions report `null`.
- `structurallyReady`: the outcome of the precedence table below.

The ordinary pending count and all deep reads share the transaction snapshot.
Exact chunk-layout checks catch excess, gapped, mixed-generation, and null-total
records that the ordinary pending counter can accept. Peer inspection streams
`vector_rows` and the vec0 table `vectors_by_collection` separately, using
indexed rowid probes. Each broken rowid counts once. The JavaScript peer scan uses constant heap space; SQLite
performs the relational aggregation. Coherent inactive cache rows remain valid
because QMD can copy those vectors when the content joins a collection.

The following table reads top-down; the first matching row determines readiness.

| Partition | Pending hashes | Inconsistent layouts | Missing required rows | Inconsistent peers | Active documents | Ready |
|---|---|---|---|---|---|---|
| `legacy` or `unreadable` | any | any | any | any | any | false |
| any | `> 0` | any | any | any | any | false |
| any | `0` | `> 0` | any | any | any | false |
| any | `0` | `0` | positive or `null` | any | any | false |
| any | `0` | `0` | `0` | positive or `null` | any | false |
| `checked` | `0` | `0` | `0` | `0` | any | true |
| `absent` | `0` | `0` | `0` | `0` | `0` | true |
| `absent` | `0` | `0` | `0` | `0` | `> 0` | false |

## Caller-owned provenance checks

QMD's fingerprint describes the model identifier, formatting, and chunking
parameters. The ingesting application records the model-file SHA256 and
verifies source/projection provenance. Structural readiness proves recorded
coverage and physical consistency. Semantic vector/content correctness needs
its own provenance and retrieval checks.

Callers verify source positions against the document's recorded full hash.
Each start must lie within that source body and preserve a Unicode-scalar
boundary. Deep inspection checks the stored position layout using SQL;
callers read source bodies to perform the content and boundary checks.

## Existing repair paths

Sound peers with missing active coverage use `store.embed({ model })`, which
copies retained vectors or generates missing embeddings. Corrupt physical
peers and malformed recorded chunk layouts use
`store.embed({ model, force: true })`, which clears and rebuilds the whole
vector index. Legacy migration runs through QMD's existing index-opening path.
Every repair ends with another explicit inspection. A persistent unreadable
state keeps the publication gate closed.

An existing update copies reusable collection vectors before removing stale
partition rows. That ordering preserves the source vector when content moves
between collections. Deep inspection adds an explicit diagnostic and keeps
those update and repair mechanisms intact.

## Validation

From the repository root:

```sh
bun run test:node test/vector-inspection.test.ts test/sdk-vector-inspection.test.ts
bun run test:bun test/vector-inspection.test.ts test/sdk-vector-inspection.test.ts
bun run test:types
bun run lint
```

Fixtures cover one-snapshot reads, absent/legacy/unreadable storage, selected
model/fingerprint scope, coherent inactive caches, malformed chunk layouts,
both missing-peer directions, partition mismatches, ordinary copy repair,
whole-index repair, and unchanged cheap-health behavior.
