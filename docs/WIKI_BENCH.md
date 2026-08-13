# wiki-bench v0

Retrieval quality benchmark over a small, vendored slice of Tobi Lütke’s
public-appearances wiki (`tobi/wiki`). Ground-truth labels come from
`meta/registry.json` and page YAML frontmatter only — **no LLM-as-judge**.

The fixture exercises exact title/keyword matches, entity aliases, plus
semantic/topical citation queries (the vitest regression suite asserts BM25
floors on `exact` + `alias` only).

## Paths

| Artifact | Path |
|----------|------|
| Fixture | `src/bench/fixtures/wiki-v0.json` |
| Vendored corpus (34 pages) | `test/wiki-bench-docs/**/*.md` |
| BM25 regression test | `test/wiki-bench-bm25.test.ts` |
| One-shot lex bench runner | `scripts/run-wiki-bench.mjs` |
| Corpus re-sync helper | `scripts/sync-wiki-bench-docs.sh` |

Corpus filepaths are relative to the wiki root (`concepts|sources|entities/*.md`)
and are indexed into collection `wiki-bench`.

## Run the BM25 regression suite

Self-contained: indexes the vendored corpus into a temp SQLite DB, runs
`searchFTS`, scores with `scoreResults` from `src/bench/score.ts`.

```bash
npx vitest run test/wiki-bench-bm25.test.ts
# or
bun test test/wiki-bench-bm25.test.ts
```

Asserted floors (measured baseline in the test file header):

- exact: mean recall@3 ≥ 0.85
- alias: mean recall@3 ≥ 0.80
- exact+alias: mean MRR ≥ 0.70

## Full bench harness (`qmd bench`)

`qmd bench` is wired in `src/cli/qmd.ts` and runs the fixture against all
backends (bm25 / vector / hybrid / full) using whatever is already indexed
in your QMD store:

```bash
# After indexing wiki-bench-docs as collection "wiki-bench":
qmd bench src/bench/fixtures/wiki-v0.json -c wiki-bench
qmd bench src/bench/fixtures/wiki-v0.json -c wiki-bench --json
```

For a one-shot **lex/BM25-only** run that indexes the vendored docs into a
temporary DB (no prior `qmd collection add` required):

```bash
node scripts/run-wiki-bench.mjs
# or
bun scripts/run-wiki-bench.mjs
```

## Re-sync corpus from `tobi/wiki`

Requires `gh` authenticated to an account that can read the private
`tobi/wiki` repo:

```bash
./scripts/sync-wiki-bench-docs.sh
```

## Notes

- Eval/benchmark scope only — this does not change search ranking code.
- Labels were derived from registry titles + frontmatter (`title`, `sources`,
  entity names), never from an LLM judge.
