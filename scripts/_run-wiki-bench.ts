/**
 * Internal helper for scripts/run-wiki-bench.mjs.
 * Indexes test/wiki-bench-docs into a temp DB and runs lex/BM25 bench only.
 */
import { mkdtempSync, rmSync, readFileSync, readdirSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { createStore, insertContent, insertDocument } from "../src/store.ts";
import { runBenchmark } from "../src/bench/bench.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const docsDir = join(root, "test", "wiki-bench-docs");
const fixturePath = join(root, "src", "bench", "fixtures", "wiki-v0.json");
const collection = "wiki-bench";

function walkMd(dir: string, base: string = dir): string[] {
  const out: string[] = [];
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walkMd(full, base));
    else if (ent.isFile() && ent.name.endsWith(".md")) out.push(relative(base, full));
  }
  return out.sort();
}

function extractTitle(content: string, fallback: string): string {
  const fm = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (fm) {
    const m = fm[1]!.match(/^title:\s*(.+)$/m);
    if (m) {
      const raw = m[1]!.trim();
      const quoted = raw.match(/^["'](.*)["']$/);
      return (quoted ? quoted[1]! : raw).trim() || fallback;
    }
  }
  const heading = content.match(/^#\s+(.+)$/m);
  return heading?.[1]?.trim() || fallback;
}

const tempDir = mkdtempSync(join(tmpdir(), "qmd-wiki-bench-run-"));
const dbPath = join(tempDir, "wiki-bench.sqlite");
process.env.INDEX_PATH = dbPath;

const store = createStore(dbPath);
const db = store.db;
const files = walkMd(docsDir);
if (files.length === 0) {
  console.error(`No markdown files under ${docsDir}`);
  process.exit(1);
}

for (const rel of files) {
  const content = readFileSync(join(docsDir, rel), "utf-8");
  const title = extractTitle(content, rel);
  const hash = createHash("sha256").update(content).digest("hex").slice(0, 12);
  const now = new Date().toISOString();
  insertContent(db, hash, content, now);
  insertDocument(db, collection, rel, title, hash, now, now);
}
store.close();

console.error(`Indexed ${files.length} docs into ${dbPath}`);
console.error(`Running bm25-only benchmark against ${fixturePath}…`);

try {
  await runBenchmark(fixturePath, {
    collection,
    backends: ["bm25"],
    dbPath,
  });
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
