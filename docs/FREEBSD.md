# FreeBSD

This is the current source-checkout path for running QMD on FreeBSD.

Current upstream scope:

- Node.js only
- BM25 and sqlite-vec search supported now
- embeddings, query expansion, and reranking follow the upstream `node-llama-cpp` FreeBSD fix
- Bun is not supported on FreeBSD yet

## Verified Prerequisites

For the immediate core path on the current FreeBSD host, the working package set was:

```sh
doas pkg install node24 corepack git python3 gmake bash sqlite3 gettext-runtime
```

If you are validating the follow-up LLM path locally after the `node-llama-cpp` FreeBSD fix lands, also install:

```sh
doas pkg install cmake ninja
```

Notes:

- `node24` is the validated package name on this host. Any Node.js package providing Node `>=22` should be acceptable.
- `gettext-runtime` provides `envsubst`, which `sqlite-vec` uses while generating `sqlite-vec.h`.
- `python3` is needed for native Node addon builds such as `better-sqlite3`.
- `git` is needed to clone `qmd` and `sqlite-vec`. The later `node-llama-cpp` follow-up may also clone `llama.cpp`.

## Build `vec0.so`

QMD's FreeBSD path uses a real SQLite loadable extension, similar in spirit to the macOS Homebrew flow.

Clone `sqlite-vec` next to `qmd`:

```sh
git clone https://github.com/asg017/sqlite-vec ../sqlite-vec
cd ../sqlite-vec
gmake sqlite-vec.h loadable
```

Verify the extension directly:

```sh
sqlite3 :memory: ".load $(pwd)/dist/vec0.so" "select vec_version();"
```

Expected output:

```text
v0.1.10-alpha.3
```

## Install QMD From Source

```sh
cd ../qmd
corepack pnpm install --frozen-lockfile
QMD_SQLITE_VEC_PATH="$(cd ../sqlite-vec && pwd)/dist/vec0.so" corepack pnpm qmd status
```

The install is still usable if `pnpm` reports that the optional `node-llama-cpp` dependency was skipped on FreeBSD.

## Core Smoke Test

Run the immediate FreeBSD verification path from the QMD repo:

```sh
test/freebsd-smoke.sh --quick
```

The script will:

- reuse `QMD_SQLITE_VEC_PATH` if you set it
- otherwise use `../sqlite-vec/dist/vec0.so`
- otherwise build `../sqlite-vec/dist/vec0.so` automatically if `../sqlite-vec` exists

`--quick` validates:

- collection add/index
- BM25 search
- `qmd ls`
- `qmd get`
- direct `sqlite-vec` loading through `sqlite3`

## Full Smoke Follow-up

`test/freebsd-smoke.sh --full` remains in-tree for the later stage where FreeBSD `node-llama-cpp` support is available.

On this immediate upstream path, `--full` requires either the adjacent `node-llama-cpp` FreeBSD fix or another locally working FreeBSD `node-llama-cpp` build.

When that backend is available, `--full` validates:

- collection add/index
- BM25 search
- context add/list/get/remove
- direct `sqlite-vec` loading through `sqlite3`
- `qmd embed`
- `qmd vsearch`
- `qmd query --explain`
- `qmd update` with a collection update hook
- stale-file removal after re-index
- `qmd cleanup` orphan-vector reclamation
- collection rename with preserved context/update settings
- secondary collection add/remove
- optional `qmd status` device probing through `QMD_STATUS_LLM_PROBE=1`

## Runtime Notes

- On FreeBSD, `qmd status` skips the LLM device probe by default to keep status fast and resilient. Use `QMD_STATUS_LLM_PROBE=1` to force the probe.
- On this immediate upstream path, FreeBSD LLM commands use a non-building `node-llama-cpp` policy and fail fast until the upstream FreeBSD `node-llama-cpp` fix lands.
- Set `QMD_LLAMA_GPU=false` to force CPU mode explicitly when the later LLM path is enabled.
- `QMD_SQLITE_VEC_PATH` remains the most deterministic way to point QMD at `vec0.so`, even though QMD also probes known FreeBSD locations.
