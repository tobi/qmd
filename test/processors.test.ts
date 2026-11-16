import { test, expect } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, unlinkSync, renameSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import YAML from "yaml";
import { createStore, createCommandProcessor, type DocumentProcessor, type ProcessorInput, type ProcessorDocument } from "../src/index.js";
import { reindexCollection } from "../src/store.js";
import { copyCommandProcessorSpec, executeProcessor, validateProcessorDocuments } from "../src/processors.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "qmd-processors-"));
  const docs = join(root, "docs");
  mkdirSync(docs);
  return { root, docs, dbPath: join(root, "index.sqlite"), cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
function render(input: ProcessorInput): ProcessorDocument[] {
  const rows = JSON.parse(Buffer.from(input.bytes).toString()) as Array<{ id: string; text: string }>;
  return rows.map(row => ({ key: `${row.id}.md`, markdown: `---\nqmd:\n  metadata:\n    source: ${input.relativePath}\n    entry: ${row.id}\n---\n# ${row.id}\n${row.text}\n` }));
}
const rows = [{ id: "a", text: "orchard" }, { id: "b", text: "meadow" }];

test("SDK source replacement, coverage, metadata, FTS, callback fingerprints and precomputed vectors", async () => {
  const f = fixture();
  writeFileSync(join(f.docs, "data.json"), JSON.stringify(rows));
  const store = await createStore({ dbPath: f.dbPath, config: { collections: { docs: { path: f.docs, pattern: "**/*" } } } });
  let calls = 0;
  const processor: DocumentProcessor = { fingerprint: "v1", process: async input => {
    expect(store.internal.db.inTransaction).toBe(false);
    calls++;
    return render(input);
  } };
  const update = () => store.update({ processors: { docs: processor } });
  try {
    expect((await update()).indexed).toBe(2);
    const db = store.internal.db;
    expect(db.prepare("SELECT path FROM documents WHERE active = 1 ORDER BY path").all()).toEqual([{ path: "data.json/a.md" }, { path: "data.json/b.md" }]);
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    const metadata = db.prepare("SELECT metadata_json FROM document_metadata ORDER BY document_id").all() as { metadata_json: string }[];
    expect(JSON.parse(metadata[0]!.metadata_json)).toEqual({ source: "data.json", entry: "a" });
    const hash = (db.prepare("SELECT hash FROM documents WHERE path = 'data.json/a.md'").get() as { hash: string }).hash;
    store.internal.ensureVecTable(2);
    store.internal.insertEmbedding(hash, 0, 0, new Float32Array([1, 0]), "synthetic", "2025-01-01T00:00:00Z");
    const vectors = db.prepare("SELECT * FROM content_vectors WHERE hash = ?").all(hash);
    expect((await update()).unchanged).toBe(2);
    expect(calls).toBe(1);
    processor.fingerprint = "v2";
    expect((await update()).unchanged).toBe(2);
    expect(calls).toBe(2);
    expect(db.prepare("SELECT * FROM content_vectors WHERE hash = ?").all(hash)).toEqual(vectors);
    const retired = db.prepare("SELECT id FROM documents WHERE path = 'data.json/b.md'").get() as { id: number };
    writeFileSync(join(f.docs, "data.json"), JSON.stringify([rows[0], { id: "c", text: "forest" }]));
    const changed = await update();
    expect([changed.indexed, changed.unchanged, changed.removed]).toEqual([1, 1, 1]);
    expect(db.prepare("SELECT 1 FROM documents WHERE id = ?").get(retired.id)).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM document_metadata WHERE document_id = ?").get(retired.id)).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM document_metadata_values WHERE document_id = ?").get(retired.id)).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM processor_outputs WHERE path = 'data.json/b.md'").get()).toBeFalsy();
    expect(store.internal.searchFTS("meadow")).toHaveLength(0);
    writeFileSync(join(f.docs, "data.json"), JSON.stringify([{ id: "a", text: "river" }, rows[1]]));
    const restored = await update();
    expect([restored.indexed, restored.updated]).toEqual([1, 1]);
    expect(store.internal.searchFTS("orchard")).toHaveLength(0);
    expect(store.internal.searchFTS("river")).toHaveLength(1);
    writeFileSync(join(f.docs, "data.json"), "[]");
    expect((await update()).removed).toBe(2);
    const before = calls;
    expect((await update()).unchanged).toBe(0);
    expect(calls).toBe(before);
    expect(db.prepare("SELECT output_count FROM processor_sources").get()).toEqual({ output_count: 0 });
    unlinkSync(join(f.docs, "data.json"));
    await update();
    expect(db.prepare("SELECT * FROM processor_sources").all()).toEqual([]);
    await expect(store.update()).rejects.toThrow("restore the processor");

    // A removed file may become a directory whose new source owns the same path.
    processor.fingerprint = "v3";
    processor.process = async input => [{ key: input.relativePath === "x.pdf" ? "sub/y.md" : "y.md", markdown: "# Namespace" }];
    writeFileSync(join(f.docs, "x.pdf"), "");
    expect((await update()).indexed).toBe(1);
    const previous = db.prepare("SELECT id FROM documents WHERE path = 'x.pdf/sub/y.md'").get() as { id: number };
    unlinkSync(join(f.docs, "x.pdf"));
    mkdirSync(join(f.docs, "x.pdf"));
    writeFileSync(join(f.docs, "x.pdf", "sub"), "");
    expect((await update()).skipped).toBe(1);
    expect(db.prepare("SELECT 1 FROM processor_outputs WHERE source = 'x.pdf'").get()).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM processor_sources WHERE source = 'x.pdf'").get()).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM documents WHERE id = ?").get(previous.id)).toBeFalsy();
    expect(db.prepare("SELECT 1 FROM document_metadata WHERE document_id = ?").get(previous.id)).toBeFalsy();
    expect(store.internal.searchFTS("Namespace")).toHaveLength(0);
    expect((await update()).indexed).toBe(1);
    expect(db.prepare("SELECT source, path FROM processor_outputs").all()).toEqual([{ source: "x.pdf/sub", path: "x.pdf/sub/y.md" }]);
    expect((await update()).unchanged).toBe(1);
  } finally { await store.close(); f.cleanup(); }
});

test("ill-formed Unicode keys reject the whole SDK source without advancing accepted coverage", async () => {
  const f = fixture();
  writeFileSync(join(f.docs, "a.json"), JSON.stringify(rows));
  writeFileSync(join(f.docs, "z.json"), JSON.stringify([{ id: "other", text: "before" }]));
  const store = await createStore({ dbPath: f.dbPath, config: { collections: { docs: { path: f.docs, pattern: "**/*.json" } } } });
  let invalidCalls = 0;
  const processor: DocumentProcessor = { fingerprint: "v1", process: async input => {
    if (processor.fingerprint === "v2" && input.relativePath === "a.json") {
      invalidCalls++;
      return [{ key: "\uD800.md", markdown: "# First alias" }, { key: "\uDBFF.md", markdown: "# Second alias" }];
    }
    return render(input);
  } };
  const db = store.internal.db;
  const acceptedSource = () => [
    db.prepare("SELECT * FROM documents WHERE path LIKE 'a.json/%' ORDER BY path").all(),
    db.prepare("SELECT c.* FROM content c JOIN documents d ON d.hash = c.hash WHERE d.path LIKE 'a.json/%' ORDER BY d.path").all(),
    db.prepare("SELECT m.* FROM document_metadata m JOIN documents d ON d.id = m.document_id WHERE d.path LIKE 'a.json/%' ORDER BY d.path").all(),
    db.prepare("SELECT m.* FROM document_metadata_values m JOIN documents d ON d.id = m.document_id WHERE d.path LIKE 'a.json/%' ORDER BY d.path, m.key, m.ordinal").all(),
    db.prepare("SELECT * FROM processor_sources WHERE source = 'a.json'").all(),
    db.prepare("SELECT * FROM processor_outputs WHERE source = 'a.json' ORDER BY path").all(),
  ];
  try {
    expect((await store.update({ processors: { docs: processor } })).indexed).toBe(3);
    const before = acceptedSource();
    processor.fingerprint = "v2";
    writeFileSync(join(f.docs, "z.json"), JSON.stringify([{ id: "other", text: "independent" }]));
    const result = await store.update({ processors: { docs: processor } });
    expect(result.skipped).toBe(1);
    expect([result.indexed, result.updated, result.removed]).toEqual([0, 1, 0]);
    expect(acceptedSource()).toEqual(before);
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    expect(store.internal.searchFTS("independent")).toHaveLength(1);
    expect(store.internal.searchFTS("alias")).toHaveLength(0);
    expect((await store.update({ processors: { docs: processor } })).skipped).toBe(1);
    expect(invalidCalls).toBe(2);
    expect(acceptedSource()).toEqual(before);
  } finally { await store.close(); f.cleanup(); }
});

test("failed sources retain their complete children and fingerprint through sweep; SQL faults roll back content, metadata, FTS and ownership", async () => {
  const f = fixture();
  writeFileSync(join(f.docs, "data.json"), JSON.stringify(rows));
  const store = await createStore({ dbPath: f.dbPath, config: { collections: { docs: { path: f.docs, pattern: "**/*.json" } } } });
  const processor: DocumentProcessor = { fingerprint: "v1", process: async input => render(input) };
  const update = () => store.update({ processors: { docs: processor } });
  const db = store.internal.db;
  const snapshot = () => ["documents", "content", "document_metadata", "document_metadata_values", "processor_sources", "processor_outputs"].map(table => db.prepare(`SELECT * FROM ${table}`).all());
  try {
    await update();
    const before = snapshot();
    writeFileSync(join(f.docs, "data.json"), "invalid JSON private value");
    expect((await update()).skipped).toBe(1);
    expect(snapshot()).toEqual(before);
    writeFileSync(join(f.docs, "data.json"), JSON.stringify([{ id: "a", text: "replacement" }, { id: "b", text: "failure" }]));
    db.exec(`CREATE TRIGGER fail_second BEFORE UPDATE ON documents WHEN NEW.path = 'data.json/b.md'
      BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END`);
    await expect(update()).rejects.toThrow("synthetic storage failure");
    expect(snapshot()).toEqual(before);
    expect(store.internal.searchFTS("replacement")).toHaveLength(0);
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    db.exec("DROP TRIGGER fail_second");
    renameSync(f.docs, `${f.docs}-offline`);
    expect((await update()).skipped).toBe(1);
    expect(snapshot()).toEqual(before);
    renameSync(`${f.docs}-offline`, f.docs);
    const abort = new AbortController();
    processor.process = async () => { abort.abort(); return [{ key: "late.md", markdown: "late" }]; };
    expect((await store.update({ processors: { docs: processor }, signal: abort.signal })).skipped).toBe(1);
    expect(snapshot()).toEqual(before);
    const command = createCommandProcessor({ command: [process.execPath, "-e", "setInterval(()=>{},1000)"], version: "v1", timeoutMs: 30 }, { cwd: f.docs });
    const timed = await reindexCollection(store.internal, f.docs, "**/*.json", "docs", { processor: command });
    expect(timed.skippedFiles).toEqual([{ file: "data.json", code: "PROCESSOR_TIMEOUT" }]);
    expect(snapshot()).toEqual(before);
    const bad = createCommandProcessor({ command: [process.execPath, "-e", "process.stdout.write('private malformed output')"], version: "v2" }, { cwd: f.docs });
    expect((await reindexCollection(store.internal, f.docs, "**/*.json", "docs", { processor: bad })).skippedFiles).toEqual([{ file: "data.json", code: "PROCESSOR_OUTPUT_INVALID" }]);
    expect(snapshot()).toEqual(before);
  } finally { await store.close(); f.cleanup(); }
});

test("config does not authorize SDK commands; empty collection mode survives config removal, DB-only open and rename, but remove/re-add clears ownership", async () => {
  const f = fixture();
  const command = { command: [process.execPath, "-e", "require('fs').writeFileSync('executed','yes')"], version: "v1" };
  const configPath = join(f.root, "config.yml");
  writeFileSync(configPath, YAML.stringify({ collections: { docs: { path: f.docs, pattern: "**/*.json", processor: command } } }));
  let store = await createStore({ dbPath: f.dbPath, configPath });
  const callback: DocumentProcessor = { fingerprint: "explicit", process: async () => [] };
  try {
    await expect(store.update()).rejects.toThrow("processor required");
    expect(existsSync(join(f.docs, "executed"))).toBe(false);
    await store.update({ processors: { docs: callback } });
    await store.close();
    writeFileSync(configPath, YAML.stringify({ collections: { docs: { path: f.docs, pattern: "**/*.json" } } }));
    store = await createStore({ dbPath: f.dbPath, configPath });
    await expect(store.update()).rejects.toThrow("rebuild");
    await store.renameCollection("docs", "renamed");
    expect(store.internal.db.prepare("SELECT collection FROM processor_modes").all()).toEqual([{ collection: "renamed" }]);
    await store.close();
    store = await createStore({ dbPath: f.dbPath });
    writeFileSync(join(f.docs, "source.json"), "[]");
    await expect(store.update()).rejects.toThrow("restore");
    expect(store.internal.db.prepare("SELECT * FROM documents").all()).toEqual([]);
    await store.removeCollection("renamed");
    await store.addCollection("renamed", { path: f.docs, pattern: "**/*.json" });
    expect((await store.update()).indexed).toBe(1);
    expect(store.internal.db.prepare("SELECT * FROM processor_modes").all()).toEqual([]);
  } finally { await store.close(); f.cleanup(); }
});

test("processor validation is bounded and rejects whole results; command adapter uses original bytes and a minimal environment", async () => {
  const f = fixture();
  const input: ProcessorInput = { version: 1, collection: "docs", relativePath: "empty.bin", hash: "synthetic", modifiedAt: "2025-01-01T00:00:00Z", bytes: new Uint8Array() };
  try {
    expect(() => copyCommandProcessorSpec({ command: ["node"], version: "" })).toThrow();
    expect(() => copyCommandProcessorSpec({ command: ["node"], version: "v1", extra: true })).toThrow();
    for (const key of ["../a.md", "/a.md", "a\\b.md", "a/./b.md", "a//b.md", "x\na.md", "plain.txt"]) expect(() => validateProcessorDocuments([{ key, markdown: "" }])).toThrow();
    expect(() => validateProcessorDocuments([{ key: "a.md", markdown: "a" }, { key: "a.md", markdown: "b" }])).toThrow();
    expect(() => validateProcessorDocuments(Array.from({ length: 1001 }, (_, n) => ({ key: `${n}.md`, markdown: "" })))).toThrow();
    expect(() => validateProcessorDocuments([{ key: "a.md", markdown: "a".repeat(16 * 1024 * 1024 + 1) }])).toThrow();
    // macOS may inject CoreFoundation metadata into a clean child's environment.
    // Compare with the same runtime launched directly with only the allowed keys,
    // rather than mistaking runtime additions for inherited parent credentials.
    const baseline = spawnSync(process.execPath, ["-e", "console.log(JSON.stringify(Object.keys(process.env).sort()))"], {
      cwd: f.docs, env: { PATH: process.env.PATH, HOME: f.docs, LANG: "C.UTF-8" }, encoding: "utf8", timeout: 10000,
    });
    expect(baseline.error).toBeUndefined();
    expect(baseline.status).toBe(0);
    const environmentNames = JSON.parse(baseline.stdout) as string[];
    const source = `let s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>{const i=JSON.parse(s);process.stdout.write(JSON.stringify({version:1,documents:[{key:'empty.md',markdown:JSON.stringify({environmentNames:Object.keys(process.env).sort(),contentBase64:i.source.contentBase64,path:i.source.path})}]}))})`;
    const processor = createCommandProcessor({ command: [process.execPath, "-e", source], version: "v1" }, { cwd: f.docs });
    const parentOnly = process.env.QMD_TEST_PROCESSOR_PARENT_ONLY;
    process.env.QMD_TEST_PROCESSOR_PARENT_ONLY = "synthetic-not-inherited";
    try {
      const documents = await executeProcessor(processor, input);
      expect(documents).toHaveLength(1);
      expect(documents[0]!.key).toBe("empty.md");
      expect(JSON.parse(documents[0]!.markdown)).toEqual({ environmentNames, contentBase64: "", path: "empty.bin" });
      expect(environmentNames).not.toContain("QMD_TEST_PROCESSOR_PARENT_ONLY");
    } finally {
      if (parentOnly === undefined) delete process.env.QMD_TEST_PROCESSOR_PARENT_ONLY;
      else process.env.QMD_TEST_PROCESSOR_PARENT_ONLY = parentOnly;
    }
    const normalized = copyCommandProcessorSpec({ command: ["node"], version: "v1" });
    expect(Object.isFrozen(normalized.command)).toBe(true);
    expect(normalized.timeoutMs).toBe(10000);
  } finally { f.cleanup(); }
});

const project = fileURLToPath(new URL("..", import.meta.url));
function cli(root: string, args: string[]) {
  const script = join(project, "src", "cli", "qmd.ts");
  const runner = "Bun" in globalThis ? [script] : [join(project, "node_modules", "tsx", "dist", "cli.mjs"), script];
  return new Promise<{ code: number | null; output: string }>((resolve, reject) => {
    const child = spawn(process.execPath, [...runner, ...args], { cwd: root, env: {
      PATH: process.env.PATH, HOME: root, XDG_CONFIG_HOME: join(root, "config"), XDG_CACHE_HOME: join(root, "cache"),
      QMD_CONFIG_DIR: join(root, "config"), CI: "true", NO_COLOR: "1", QMD_DOCTOR_DEVICE_PROBE: "0",
    }, stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    child.stdout.on("data", b => { output += b; });
    child.stderr.on("data", b => { output += b; });
    child.on("error", reject);
    child.on("close", code => resolve({ code, output }));
  });
}

test("CLI command processing is gated on both scan paths; approval changes re-arm trust and generated Markdown is searchable", async () => {
  const f = fixture();
  mkdirSync(join(f.root, ".qmd"));
  const spec = { command: [process.execPath, "-e", `let s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>{const i=JSON.parse(s);const rows=JSON.parse(Buffer.from(i.source.contentBase64,'base64'));require('fs').writeFileSync('executed','yes');process.stdout.write(JSON.stringify({version:1,documents:rows.map(r=>({key:r.id+'.md',markdown:'---\\nqmd:\\n  metadata:\\n    entry: '+r.id+'\\n---\\n# '+r.id+'\\n'+r.text}))}))})`], version: "v1" };
  const configPath = join(f.root, ".qmd", "index.yml");
  const config = { collections: {
    ordinary: { path: f.docs, pattern: "**/*.md" },
    docs: { path: f.docs, pattern: "**/*.json", processor: spec },
  } };
  writeFileSync(configPath, YAML.stringify(config));
  writeFileSync(join(f.docs, "data.json"), JSON.stringify(rows));
  try {
    const denied = await cli(f.root, ["update"]);
    expect(denied.code, denied.output).toBe(0);
    expect(denied.output).toContain("Skipping processor collection");
    expect(existsSync(join(f.docs, "executed"))).toBe(false);
    let store = await createStore({ dbPath: join(f.root, ".qmd", "index.sqlite") });
    expect(store.internal.db.prepare("SELECT * FROM documents").all()).toEqual([]);
    await store.close();
    expect((await cli(f.root, ["trust"])).code).toBe(0);
    const accepted = await cli(f.root, ["update"]);
    expect(accepted.code, accepted.output).toBe(0);
    expect(accepted.output).toContain("Indexed: 2 new");
    expect(existsSync(join(f.docs, "executed"))).toBe(true);
    store = await createStore({ dbPath: join(f.root, ".qmd", "index.sqlite") });
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    expect(store.internal.db.prepare("SELECT path FROM documents WHERE active = 1 ORDER BY path").all()).toEqual([{ path: "data.json/a.md" }, { path: "data.json/b.md" }]);
    await store.close();
    unlinkSync(join(f.docs, "executed"));
    spec.version = "v2";
    writeFileSync(configPath, YAML.stringify(config));
    expect((await cli(f.root, ["update"])).output).toContain("Skipping processor collection");
    expect(existsSync(join(f.docs, "executed"))).toBe(false);
    const add = await cli(f.root, ["collection", "add", f.docs, "--name", "extra", "--mask", "**/*.txt"]);
    expect(add.code, add.output).toBe(0);
    expect((await cli(f.root, ["update"])).output).toContain("Skipping processor collection");
    expect(existsSync(join(f.docs, "executed"))).toBe(false);
    // Hard refusal is preflighted across all collections, before an earlier ordinary write.
    writeFileSync(join(f.docs, "pending.md"), "# Pending ordinary document");
    writeFileSync(configPath, YAML.stringify({ collections: {
      ordinary: { path: f.docs, pattern: "**/*.md" },
      docs: { path: f.docs, pattern: "**/*.json" },
    } }));
    const refused = await cli(f.root, ["update"]);
    expect(refused.code).not.toBe(0);
    expect(refused.output).toContain("restore the processor");
    store = await createStore({ dbPath: join(f.root, ".qmd", "index.sqlite") });
    expect(store.internal.db.prepare("SELECT 1 FROM documents WHERE path = 'pending.md'").get()).toBeFalsy();
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    await store.close();
    // Removing YAML alone must not let collection add's separate scanner index raw input.
    writeFileSync(configPath, YAML.stringify({ collections: {} }));
    const initial = await cli(f.root, ["collection", "add", f.docs, "--name", "docs", "--mask", "**/*.json"]);
    expect(initial.code).not.toBe(0);
    expect(initial.output).toContain("restore the processor");
    store = await createStore({ dbPath: join(f.root, ".qmd", "index.sqlite") });
    expect(store.internal.db.prepare("SELECT 1 FROM documents WHERE path = 'data.json'").get()).toBeFalsy();
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);
    await store.close();
  } finally { f.cleanup(); }
});
