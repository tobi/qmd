# Collection processors

A processor converts each matched source file into zero or more Markdown index documents. This is useful for JSON exports, logs, or other formats containing several independent records. Generated documents live only in the index, not on disk. Collections without processors keep their existing Markdown ingestion behavior.

## SDK: explicit callbacks

```ts
import { createStore, type DocumentProcessor } from '@tobilu/qmd'

const processor: DocumentProcessor = {
  fingerprint: 'records-v1',
  async process(input, signal) {
    signal.throwIfAborted()
    const records = JSON.parse(new TextDecoder().decode(input.bytes))
    return records.map((record: { id: string; text: string }) => ({
      key: `${record.id}.md`,
      markdown: `---\nqmd:\n  metadata:\n    source: ${JSON.stringify(input.relativePath)}\n    entry: ${JSON.stringify(record.id)}\n---\n# ${record.id}\n${record.text}\n`,
    }))
  },
}
const store = await createStore({
  dbPath: './index.sqlite',
  config: { collections: { records: { path: '/data/exports', pattern: '**/*.json' } } },
})
const controller = new AbortController()
await store.update({ processors: { records: processor }, signal: controller.signal })
await store.close()
```

`ProcessorInput` contains `version: 1`, `collection`, `relativePath`, SHA-256 `hash`, ISO `modifiedAt`, and a private copy of the original `bytes: Uint8Array`. Even empty files are processed. There are no model, authentication, or index-state fields. The callback owns its fingerprint: a nonempty string of at most 4096 characters that **must change whenever its implementation or configuration changes**. Callbacks may be invoked again after failure or a crash: keep them idempotent and side-effect-free. Callbacks are bounded to 60 seconds; cancellation ignores late completion but cannot interrupt synchronous CPU loops or undo callback side effects.

Each update snapshots the validated fingerprint and original process function before indexing, preserving the callback's original `this` receiver. Replacing descriptor properties during that update does not change the implementation or fingerprint used for later sources. This does not freeze mutable closure or receiver state: callbacks remain trusted code, and callers must keep their implementation/configuration stable during an update.

Reading YAML (including `configPath`) or inline configuration **never authorizes SDK command execution**. A callback explicitly registered for the collection takes precedence over any YAML command, using its own fingerprint. To opt in to a command from an SDK application, register `createCommandProcessor(spec, { cwd: collectionRoot })` explicitly in `update({ processors })`.

## CLI: trusted argv commands

```yaml
collections:
  records:
    path: /data/exports
    pattern: '**/*.json'
    processor:
      command: [node, /data/tools/records.mjs]
      version: records-v1
      timeoutMs: 10000
```

The command is an argv array, not a shell expression. `version` is required, nonempty, and at most 1024 characters. `timeoutMs` is an integer from 1 to 60000, default 10000. Unknown spec fields are rejected. Commands have 1–128 arguments, each at most 8192 characters, with a nonempty executable and no NULs. Normalization copies and freezes the spec and argv. The command fingerprint covers normalized argv, version, and timeout. **Bump version when the script, configuration files it reads, or any transitive dependency changes. Script contents are not automatically hashed or read for approval.**

Project-local `.qmd` commands use the existing `qmd trust` gate. Unapproved processor collections are skipped as a whole, never indexed as raw input. Changing argv, version, or timeout re-arms approval. Global user configuration keeps its existing trusted semantics. Adding an unrelated collection or editing an update hook does not approve previously untrusted processor commands.

Execution uses `node:child_process.spawn`, `shell: false`, and the collection root as cwd. The environment contains only `PATH`, `HOME` (collection root), and `LANG=C.UTF-8`; model credentials and other user environment values are not inherited. **This is not an OS sandbox. A trusted command can read files and use the network with the operating-system user's authority.** It must be idempotent and side-effect-free because retries or crashes can execute it again.

stdin is one JSON object, followed by EOF:

```json
{"version":1,"collection":"records","source":{"path":"export.json","hash":"sha256-of-original-bytes","modifiedAt":"2025-01-01T00:00:00.000Z","contentBase64":"W10="}}
```

stdout must be exactly one valid UTF-8 JSON value with this shape (no unknown fields):

```json
{"version":1,"documents":[{"key":"record-42.md","markdown":"# Record 42\nA searchable record.\n"}]}
```

An example `records.mjs` (using the same metadata shape as the SDK):

```js
let text = ''
for await (const chunk of process.stdin) text += chunk
const input = JSON.parse(text)
const records = JSON.parse(Buffer.from(input.source.contentBase64, 'base64').toString('utf8'))
const documents = records.map(record => ({
  key: `${record.id}.md`,
  markdown: `---\nqmd:\n  metadata:\n    source: ${JSON.stringify(input.source.path)}\n    entry: ${JSON.stringify(record.id)}\n---\n# ${record.id}\n${record.text}\n`,
}))
process.stdout.write(JSON.stringify({ version: 1, documents }))
```

Use stable IDs rather than array positions. Keys must end in `.md`, be safe relative POSIX paths of at most 256 UTF-8 bytes, and contain no empty, `.` or `..` segments, backslashes, colons or control characters. Duplicate keys reject the entire source. Index paths are `sourceRelativePath/key`, such as `export.json/record-42.md`. Collisions with another source's ownership or an ordinary document reject the whole source. Retired derived documents and their ownership rows are deleted together inside the source savepoint, including metadata and FTS entries. A removed source's namespace can be reused. If a file becomes a directory containing a new source with a colliding virtual path, the first scan may report that collision before removing the missing source; the next unchanged scan can publish the new source.

Both keys and Markdown must be well-formed Unicode strings: unpaired surrogates reject the entire source rather than being replaced or normalized, including surrogates encoded as JSON escapes. Markdown is indexed exactly as returned, including its frontmatter and headings. Titles and `qmd.metadata` use the existing extractors. Processors should add semantic source/entry metadata themselves; no automatic headers or position-shifting annotations are inserted.

## Bounds and publication

- Source input: at most 10 MiB, read with a bounded file-descriptor loop even if the file grows after stat. The original snapshot is hashed, then copied for the callback. The file is opened nonblocking and then checked as a regular file, so replacement with a FIFO cannot block the open waiting for a writer. This is not a filesystem snapshot, an OS sandbox, or a deadline for synchronous filesystem/NFS I/O.
- Per source: at most 1000 outputs, 16 MiB aggregate UTF-8 Markdown, and 24 MiB command stdout JSON. Command stderr is discarded, bounded to 64 KiB, and never included in diagnostics. Failure reports use fixed codes, not output snippets or generated keys.
- Timeout, abort, and output overflow stop the owned child and destroy its pipes without waiting forever for inherited stdout. POSIX uses only the child's own process group. Windows stops the direct child only. Escaped/detached descendants are not contained; there is no general process-tree sandbox.

The full result is validated before document writes. The source ledger fingerprints source bytes, runtime processor fingerprint, input/output format versions, metadata extraction version and source identity. mtime-only changes do not cause reruns. A matching successful fingerprint skips processing only when current child count, active membership, hashes and metadata extraction versions remain valid. Empty `documents: []` is successful covered-empty input and removes previous children while retaining coverage.

Each source publishes content, documents, metadata, FTS, child ownership and its last-success fingerprint under one SQLite savepoint. A storage failure rolls the source back and propagates; it is not treated as a script failure. Counters update only after successful publication. Unchanged output hashes keep their existing vectors. Indexed/updated/unchanged/removed counters count documents; progress counts source files.

Read, execution, validation, timeout and abort failures retain the previous complete children and last-success fingerprint. They remain visible through the scan's missing-source cleanup. Missing physical sources retire all their children only after successful root validation and complete enumeration; a missing root or enumeration exception does not remove documents. Enabling a processor on an ordinary collection retires each raw source row only when that source successfully publishes its generated children.

Writer batches are committed before reading or awaiting processor code. Savepoint release is **not** a durability boundary: the outer batch commit is. A crash can lose the current batch, never half of a source; earlier committed batches remain. A source with 1000 outputs is one atomic unit and may exceed the usual batch time target. There is no collection-wide atomicity, exactly-once execution, or stronger power-loss guarantee.

## Removing or changing processors

Changing version/fingerprint reprocesses sources normally. Removing a processor is deliberately different: after a collection enters processor mode, later updates without an explicitly authorized runtime processor refuse **before any collection document writes**, even if the collection originally had zero source files or the SDK opens only the database. Restore the processor, or rebuild by removing and re-adding the collection. There is no automatic raw-input reconciliation. CLI updates preflight every selected collection before hooks or collection document writes; a missing required processor is a hard refusal, not a partial update of earlier collections. Untrusted configured processor collections still skip as a whole. Collection rename preserves mode and ownership; collection removal clears processor specs, mode and ledgers.

Processor tables are derived, idempotently created alongside the existing schema. They do not change `user_version` or bypass a deferred vector migration. This feature does not change providers, embedding policy, or local model behavior.
