# wiki-bench v0

Retrieval quality benchmark over Tobi Lütke’s public-appearances wiki
(`tobi/wiki`). Ground-truth labels come from `meta/registry.json` and page
YAML frontmatter only — **no LLM-as-judge**.

## Privacy: fixture only in this repo

**This public repository ships the fixture (queries + expected paths/ids)
only.** It does **not** vendor wiki page bodies.

| Artifact | Path | In public repo? |
|----------|------|-----------------|
| Fixture (queries + expected paths) | `src/bench/fixtures/wiki-v0.json` | Yes |
| Fixture schema test | `test/wiki-bench-fixture.test.ts` | Yes |
| Optional BM25 floors (env-gated) | `test/wiki-bench-bm25.test.ts` | Yes (skips without corpus) |
| Wiki markdown corpus | private `tobi/wiki` (or a local checkout) | **No — never commit** |

Do **not** add `test/wiki-bench-docs/**` or any wiki `.md` bodies to this repo.
CI must stay green without access to the private corpus.

Expected file paths in the fixture are **wiki-relative**:
`concepts|sources|entities|syntheses/...md` (collection rooted at the wiki
markdown directory, typically `wiki/` inside `tobi/wiki`).

## Validate the fixture (no corpus required)

```bash
npx vitest run test/wiki-bench-fixture.test.ts
# or
bun test test/wiki-bench-fixture.test.ts
```

This checks version/collection, query shape, allowed `type` values
(`exact|semantic|topical|cross-domain|alias`), unique ids, and that
`expected_files` are safe wiki-relative `.md` paths.

## Index a local wiki checkout

Point QMD at your local clone of the private wiki. Markdown usually lives
under `wiki/`:

```bash
# If markdown is under <wiki-repo>/wiki/{concepts,sources,entities,...}:
qmd collection add /path/to/wiki/wiki --name wiki-bench

# If you already keep a bare wiki-root directory of those folders:
qmd collection add /path/to/wiki-root --name wiki-bench
```

Then update/embed as usual for your install (`qmd update`, etc.).

## Run the full multi-backend bench

`qmd bench` is wired in `src/cli/qmd.ts` and implemented by
`src/bench/bench.ts`. It loads a fixture JSON and scores bm25 / vector /
hybrid / full against an already-indexed collection:

```bash
qmd bench src/bench/fixtures/wiki-v0.json -c wiki-bench
qmd bench src/bench/fixtures/wiki-v0.json -c wiki-bench --json
```

Usage (from CLI): `qmd bench <fixture.json> [--json] [-c collection]`.

## Optional BM25 quality floors (local corpus)

`test/wiki-bench-bm25.test.ts` indexes a **local** corpus into a temp DB and
asserts floors on `exact` + `alias` queries. It **skips** unless one of these
env vars points at an existing directory:

- `QMD_WIKI_BENCH_DOCS` — preferred; directory that contains
  `concepts/`, `sources/`, etc.
- `QMD_WIKI_PATH` — wiki repo root or wiki markdown root; if a nested `wiki/`
  subdirectory contains those folders, that nested path is used.

```bash
export QMD_WIKI_BENCH_DOCS=/path/to/wiki/wiki
npx vitest run test/wiki-bench-bm25.test.ts
```

Without the env var (default CI), the suite skips and stays green.

Asserted floors (measured baseline when corpus is present):

- exact: mean recall@3 ≥ 0.85
- alias: mean recall@3 ≥ 0.80
- exact+alias: mean MRR ≥ 0.70

## Notes

- Eval/benchmark scope only — this does not change search ranking code.
- Labels were derived from registry titles + frontmatter (`title`, `sources`,
  entity names), never from an LLM judge.
- Never commit private wiki `.md` content into `tobi/qmd`.
