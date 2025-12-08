#!/usr/bin/env bun
import { Database } from "bun:sqlite";
import { Glob } from "bun";
import { mkdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import * as sqliteVec from "sqlite-vec";

// On macOS, use Homebrew's SQLite which supports extensions
if (process.platform === "darwin") {
  const homebrewSqlitePath = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
  if (existsSync(homebrewSqlitePath)) {
    Database.setCustomSQLite(homebrewSqlitePath);
  }
}

const DEFAULT_EMBED_MODEL = "embeddinggemma";
const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
const DEFAULT_QUERY_MODEL = "qwen3:0.6b";
const DEFAULT_GLOB = "**/*.md";
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Terminal colors (respects NO_COLOR env)
const useColor = !process.env.NO_COLOR && process.stdout.isTTY;
const c = {
  reset: useColor ? "\x1b[0m" : "",
  dim: useColor ? "\x1b[2m" : "",
  bold: useColor ? "\x1b[1m" : "",
  cyan: useColor ? "\x1b[36m" : "",
  yellow: useColor ? "\x1b[33m" : "",
  green: useColor ? "\x1b[32m" : "",
  magenta: useColor ? "\x1b[35m" : "",
  blue: useColor ? "\x1b[34m" : "",
};

// Global state for --index option
let customIndexName: string | null = null;

// Terminal progress bar using OSC 9;4 escape sequence
const progress = {
  set(percent: number) {
    process.stderr.write(`\x1b]9;4;1;${Math.round(percent)}\x07`);
  },
  clear() {
    process.stderr.write(`\x1b]9;4;0\x07`);
  },
  indeterminate() {
    process.stderr.write(`\x1b]9;4;3\x07`);
  },
  error() {
    process.stderr.write(`\x1b]9;4;2\x07`);
  },
};

// Format seconds into human-readable ETA
function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function getDbPath(): string {
  const cacheDir = process.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
  const qmdCacheDir = resolve(cacheDir, "qmd");
  mkdirSync(qmdCacheDir, { recursive: true });
  const dbName = customIndexName || "index";
  return resolve(qmdCacheDir, `${dbName}.sqlite`);
}

function getPwd(): string {
  return process.env.PWD || process.cwd();
}

/*
Schema:

CREATE TABLE collections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pwd TEXT NOT NULL,
  glob_pattern TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(pwd, glob_pattern)
);

CREATE TABLE documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  title TEXT NOT NULL,
  hash TEXT NOT NULL,
  filepath TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  modified_at TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY (collection_id) REFERENCES collections(id)
);

CREATE TABLE content_vectors (
  hash TEXT PRIMARY KEY,
  embedding BLOB NOT NULL,
  model TEXT NOT NULL,
  embedded_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE documents_fts USING fts5(...);
*/

function getDb(): Database {
  const db = new Database(getDbPath());
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");

  // Collections table
  db.exec(`
    CREATE TABLE IF NOT EXISTS collections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pwd TEXT NOT NULL,
      glob_pattern TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(pwd, glob_pattern)
    )
  `);

  // Documents table with collection_id and full filepath
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      filepath TEXT NOT NULL,
      body TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (collection_id) REFERENCES collections(id)
    )
  `);

  // Content vectors keyed by hash (UNIQUE)
  db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT PRIMARY KEY,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL
    )
  `);

  // FTS on documents
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      name, body,
      content='documents',
      content_rowid='id',
      tokenize='porter unicode61'
    )
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
      INSERT INTO documents_fts(rowid, name, body) VALUES (new.id, new.name, new.body);
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
      INSERT INTO documents_fts(documents_fts, rowid, name, body) VALUES('delete', old.id, old.name, old.body);
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
      INSERT INTO documents_fts(documents_fts, rowid, name, body) VALUES('delete', old.id, old.name, old.body);
      INSERT INTO documents_fts(rowid, name, body) VALUES (new.id, new.name, new.body);
    END
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id, active)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath, active)`);

  return db;
}

function ensureVecTable(db: Database, dimensions: number): void {
  const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get() as { sql: string } | null;
  if (tableInfo) {
    const match = tableInfo.sql.match(/float\[(\d+)\]/);
    if (match && parseInt(match[1]) === dimensions) return;
    db.exec("DROP TABLE IF EXISTS vectors_vec");
  }
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash TEXT PRIMARY KEY, embedding float[${dimensions}])`);
}

function getHashesNeedingEmbedding(db: Database): number {
  const result = db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

async function hashContent(content: string): Promise<string> {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(content);
  return hash.digest("hex");
}

// Extract title from first markdown headline, or use filename as fallback
function extractTitle(content: string, filename: string): string {
  const match = content.match(/^##?\s+(.+)$/m);
  if (match) return match[1].trim();
  return filename.replace(/\.md$/, "").split("/").pop() || filename;
}

// Format text for EmbeddingGemma
function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// Auto-pull model if not found
async function ensureModelAvailable(model: string): Promise<void> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    });
    if (response.ok) return;
  } catch {
    // Continue to pull attempt
  }

  console.log(`Model ${model} not found. Pulling...`);
  progress.indeterminate();

  const pullResponse = await fetch(`${OLLAMA_URL}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: model, stream: false }),
  });

  if (!pullResponse.ok) {
    progress.error();
    throw new Error(`Failed to pull model ${model}: ${pullResponse.status} - ${await pullResponse.text()}`);
  }

  progress.clear();
  console.log(`Model ${model} pulled successfully.`);
}

async function getEmbedding(text: string, model: string, isQuery: boolean = false, title?: string, retried: boolean = false): Promise<number[]> {
  const input = isQuery ? formatQueryForEmbedding(text) : formatDocForEmbedding(text, title);

  const response = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return getEmbedding(text, model, isQuery, title, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }
  const data = await response.json() as { embeddings: number[][] };
  return data.embeddings[0];
}

// Qwen3-Reranker prompt format (trained for yes/no relevance classification)
const RERANK_SYSTEM = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`;

function formatRerankPrompt(query: string, title: string, doc: string): string {
  return `<Instruct>: Determine if this document from a Shopify knowledge base is relevant to the search query. The query may reference specific Shopify programs, competitions, features, or named concepts (e.g., "Build a Business" competition, "Shop Pay", "Polaris"). Match documents that discuss the queried topic, even if phrasing differs.
<Query>: ${query}
<Document Title>: ${title}
<Document>: ${doc}`;
}

type LogProb = { token: string; logprob: number };
type RerankResponse = {
  response: string;
  logprobs?: LogProb[];
};

async function rerankSingle(prompt: string, model: string, retried: boolean = false): Promise<number> {
  // Use generate with raw template for qwen3-reranker format
  // Include empty <think> tags as per HuggingFace reference implementation
  const fullPrompt = `<|im_start|>system
${RERANK_SYSTEM}<|im_end|>
<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt: fullPrompt,
      raw: true,
      stream: false,
      logprobs: true,
      options: { num_predict: 1 },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return rerankSingle(prompt, model, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json() as RerankResponse;

  // Extract score from logprobs - required for proper reranking
  if (!data.logprobs || data.logprobs.length === 0) {
    throw new Error("Reranker response missing logprobs - ensure Ollama supports logprobs");
  }

  const firstToken = data.logprobs[0];
  const token = firstToken.token.toLowerCase().trim();
  const confidence = Math.exp(firstToken.logprob); // 0-1, higher = more confident

  if (token === "yes") {
    // Relevant: return confidence (e.g., 0.93 for high confidence yes)
    return confidence;
  }
  if (token === "no") {
    // Not relevant: return low score, scaled by inverse confidence
    // High confidence "no" → very low score
    return (1 - confidence) * 0.3; // Cap at 0.3 for uncertain "no"
  }

  throw new Error(`Unexpected reranker token: "${token}" (expected "yes" or "no")`);
}

async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL): Promise<{ file: string; score: number }[]> {
  const results: { file: string; score: number }[] = [];
  const total = documents.length;
  const PARALLEL = 5;

  process.stderr.write(`Reranking ${total} documents with ${model} (parallel: ${PARALLEL})...\n`);
  progress.indeterminate();

  // Process in parallel batches
  for (let i = 0; i < documents.length; i += PARALLEL) {
    const batch = documents.slice(i, i + PARALLEL);
    const batchResults = await Promise.all(
      batch.map(async (doc) => {
        try {
          // Extract title from filename for reranker context
          const title = doc.file.split('/').pop()?.replace(/\.md$/, '') || doc.file;
          const prompt = formatRerankPrompt(query, title, doc.text.slice(0, 4000));
          const score = await rerankSingle(prompt, model);
          return { file: doc.file, score };
        } catch (err) {
          return { file: doc.file, score: 0 };
        }
      })
    );
    results.push(...batchResults);

    const processed = Math.min(i + PARALLEL, total);
    progress.set((processed / total) * 100);
    process.stderr.write(`\rReranking: ${processed}/${total}`);
  }

  progress.clear();
  process.stderr.write("\n");

  return results.sort((a, b) => b.score - a.score);
}

function getOrCreateCollection(db: Database, pwd: string, globPattern: string): number {
  const now = new Date().toISOString();

  // Use INSERT OR IGNORE to handle race conditions, then SELECT
  db.prepare(`INSERT OR IGNORE INTO collections (pwd, glob_pattern, created_at) VALUES (?, ?, ?)`).run(pwd, globPattern, now);
  const existing = db.prepare(`SELECT id FROM collections WHERE pwd = ? AND glob_pattern = ?`).get(pwd, globPattern) as { id: number };
  return existing.id;
}

function cleanupDuplicateCollections(db: Database): void {
  // Remove duplicate collections keeping the oldest one
  db.exec(`
    DELETE FROM collections WHERE id NOT IN (
      SELECT MIN(id) FROM collections GROUP BY pwd, glob_pattern
    )
  `);
  // Remove bogus "." glob pattern entries (from earlier bug)
  db.exec(`DELETE FROM collections WHERE glob_pattern = '.'`);
}

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function showStatus(): void {
  const dbPath = getDbPath();
  const db = getDb();

  // Cleanup any duplicate collections
  cleanupDuplicateCollections(db);

  // Index size
  let indexSize = 0;
  try {
    const stat = Bun.file(dbPath).size;
    indexSize = stat;
  } catch {}

  // Collections info
  const collections = db.prepare(`
    SELECT c.id, c.pwd, c.glob_pattern, c.created_at,
           COUNT(d.id) as doc_count,
           SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END) as active_count,
           MAX(d.modified_at) as last_modified
    FROM collections c
    LEFT JOIN documents d ON d.collection_id = c.id
    GROUP BY c.id
    ORDER BY c.created_at DESC
  `).all() as { id: number; pwd: string; glob_pattern: string; created_at: string; doc_count: number; active_count: number; last_modified: string | null }[];

  // Overall stats
  const totalDocs = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
  const vectorCount = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
  const needsEmbedding = getHashesNeedingEmbedding(db);

  // Most recent update across all collections
  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null };

  console.log(`${c.bold}QMD Status${c.reset}\n`);
  console.log(`Index: ${dbPath}`);
  console.log(`Size:  ${formatBytes(indexSize)}\n`);

  console.log(`${c.bold}Documents${c.reset}`);
  console.log(`  Total:    ${totalDocs.count} files indexed`);
  console.log(`  Vectors:  ${vectorCount.count} embedded`);
  if (needsEmbedding > 0) {
    console.log(`  ${c.yellow}Pending:  ${needsEmbedding} need embedding${c.reset} (run 'qmd embed')`);
  }
  if (mostRecent.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    console.log(`  Updated:  ${formatTimeAgo(lastUpdate)}`);
  }

  if (collections.length > 0) {
    console.log(`\n${c.bold}Collections${c.reset}`);
    for (const col of collections) {
      const lastMod = col.last_modified ? formatTimeAgo(new Date(col.last_modified)) : "never";
      console.log(`  ${c.cyan}${col.pwd}${c.reset}`);
      console.log(`    ${col.glob_pattern} → ${col.active_count} docs (updated ${lastMod})`);
    }
  } else {
    console.log(`\n${c.dim}No collections. Run 'qmd add .' to index markdown files.${c.reset}`);
  }

  db.close();
}

async function updateAllCollections(): Promise<void> {
  const db = getDb();
  cleanupDuplicateCollections(db);
  const collections = db.prepare(`SELECT id, pwd, glob_pattern FROM collections`).all() as { id: number; pwd: string; glob_pattern: string }[];

  if (collections.length === 0) {
    console.log(`${c.dim}No collections found. Run 'qmd add .' to index markdown files.${c.reset}`);
    db.close();
    return;
  }

  db.close();

  console.log(`${c.bold}Updating ${collections.length} collection(s)...${c.reset}\n`);

  for (let i = 0; i < collections.length; i++) {
    const col = collections[i];
    console.log(`${c.cyan}[${i + 1}/${collections.length}]${c.reset} ${c.bold}${col.pwd}${c.reset}`);
    console.log(`${c.dim}    Pattern: ${col.glob_pattern}${c.reset}`);
    // Temporarily set PWD for indexing
    const originalPwd = process.env.PWD;
    process.env.PWD = col.pwd;
    await indexFiles(col.glob_pattern);
    process.env.PWD = originalPwd;
    console.log("");
  }

  console.log(`${c.green}✓ All collections updated.${c.reset}`);
}

async function dropCollection(globPattern: string): Promise<void> {
  const db = getDb();
  const pwd = getPwd();

  const collection = db.prepare(`SELECT id FROM collections WHERE pwd = ? AND glob_pattern = ?`).get(pwd, globPattern) as { id: number } | null;

  if (!collection) {
    console.log(`No collection found for ${pwd} with pattern ${globPattern}`);
    db.close();
    return;
  }

  // Delete documents in this collection
  const deleted = db.prepare(`DELETE FROM documents WHERE collection_id = ?`).run(collection.id);

  // Delete the collection
  db.prepare(`DELETE FROM collections WHERE id = ?`).run(collection.id);

  console.log(`Dropped collection: ${pwd} (${globPattern})`);
  console.log(`Removed ${deleted.changes} documents`);
  console.log(`(Vectors kept for potential reuse)`);

  db.close();
}

async function indexFiles(globPattern: string = DEFAULT_GLOB): Promise<void> {
  const db = getDb();
  const pwd = getPwd();
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

  // Get or create collection for this (pwd, glob)
  const collectionId = getOrCreateCollection(db, pwd, globPattern);
  console.log(`Collection: ${pwd} (${globPattern})`);

  progress.indeterminate();
  const glob = new Glob(globPattern);
  const files: string[] = [];
  for await (const file of glob.scan({ cwd: pwd, onlyFiles: true, followSymlinks: true })) {
    // Skip node_modules, hidden folders (.*), and other common excludes
    const parts = file.split("/");
    const shouldSkip = parts.some(part =>
      part === "node_modules" ||
      part.startsWith(".") ||
      excludeDirs.includes(part)
    );
    if (!shouldSkip) {
      files.push(file);
    }
  }

  const total = files.length;
  if (total === 0) {
    progress.clear();
    console.log("No files found matching pattern.");
    db.close();
    return;
  }

  const insertStmt = db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)`);
  const deactivateStmt = db.prepare(`UPDATE documents SET active = 0 WHERE collection_id = ? AND filepath = ? AND active = 1`);
  const findActiveStmt = db.prepare(`SELECT id, hash FROM documents WHERE collection_id = ? AND filepath = ? AND active = 1`);

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenFiles = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = resolve(pwd, relativeFile);
    seenFiles.add(filepath);

    const content = await Bun.file(filepath).text();
    const hash = await hashContent(content);
    const name = relativeFile.replace(/\.md$/, "").split("/").pop() || relativeFile;
    const title = extractTitle(content, relativeFile);
    const existing = findActiveStmt.get(collectionId, filepath) as { id: number; hash: string } | null;

    if (existing) {
      if (existing.hash === hash) {
        unchanged++;
      } else {
        deactivateStmt.run(collectionId, filepath);
        updated++;
        const stat = await Bun.file(filepath).stat();
        insertStmt.run(collectionId, name, title, hash, filepath, content, stat ? new Date(stat.birthtime).toISOString() : now, stat ? new Date(stat.mtime).toISOString() : now);
      }
    } else {
      indexed++;
      const stat = await Bun.file(filepath).stat();
      insertStmt.run(collectionId, name, title, hash, filepath, content, stat ? new Date(stat.birthtime).toISOString() : now, stat ? new Date(stat.mtime).toISOString() : now);
    }

    processed++;
    progress.set((processed / total) * 100);
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = processed / elapsed;
    const remaining = (total - processed) / rate;
    const eta = processed > 2 ? ` ETA: ${formatETA(remaining)}` : "";
    process.stderr.write(`\rIndexing: ${processed}/${total}${eta}        `);
  }

  // Deactivate documents in this collection that no longer exist
  const allActive = db.prepare(`SELECT filepath FROM documents WHERE collection_id = ? AND active = 1`).all(collectionId) as { filepath: string }[];
  let removed = 0;
  for (const row of allActive) {
    if (!seenFiles.has(row.filepath)) {
      deactivateStmt.run(collectionId, row.filepath);
      removed++;
    }
  }

  // Check if vector index needs updating
  const needsEmbedding = getHashesNeedingEmbedding(db);

  progress.clear();
  console.log(`\nIndexed: ${indexed} new, ${updated} updated, ${unchanged} unchanged, ${removed} removed`);

  if (needsEmbedding > 0) {
    console.log(`\nRun 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }

  db.close();
}

function renderProgressBar(percent: number, width: number = 30): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);
  return bar;
}

async function vectorIndex(model: string = DEFAULT_EMBED_MODEL, force: boolean = false): Promise<void> {
  const db = getDb();
  const now = new Date().toISOString();

  // If force, clear all vectors
  if (force) {
    console.log(`${c.yellow}Force re-indexing: clearing all vectors...${c.reset}`);
    db.exec(`DELETE FROM content_vectors`);
    db.exec(`DROP TABLE IF EXISTS vectors_vec`);
  }

  // Find unique hashes that need embedding (from active documents)
  const hashesToEmbed = db.prepare(`
    SELECT DISTINCT d.hash, d.title, d.body
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash
    WHERE d.active = 1 AND v.hash IS NULL
  `).all() as { hash: string; title: string; body: string }[];

  if (hashesToEmbed.length === 0) {
    console.log(`${c.green}✓ All content hashes already have embeddings.${c.reset}`);
    db.close();
    return;
  }

  // Calculate total bytes for accurate progress tracking, skip empty files
  const itemsWithSize = hashesToEmbed
    .map(item => ({
      ...item,
      bytes: new TextEncoder().encode(item.body).length
    }))
    .filter(item => item.bytes > 0);  // Skip empty documents

  if (itemsWithSize.length === 0) {
    console.log(`${c.green}✓ No non-empty documents to embed.${c.reset}`);
    db.close();
    return;
  }

  const totalBytes = itemsWithSize.reduce((sum, item) => sum + item.bytes, 0);
  const total = itemsWithSize.length;
  const skipped = hashesToEmbed.length - total;

  console.log(`${c.bold}Embedding ${total} documents${c.reset} ${c.dim}(${formatBytes(totalBytes)})${c.reset}`);
  if (skipped > 0) {
    console.log(`${c.dim}Skipped ${skipped} empty documents${c.reset}`);
  }
  console.log(`${c.dim}Model: ${model}${c.reset}\n`);

  progress.indeterminate();
  const firstEmbedding = await getEmbedding(itemsWithSize[0].body, model, false, itemsWithSize[0].title);
  ensureVecTable(db, firstEmbedding.length);

  const insertVecStmt = db.prepare(`INSERT OR REPLACE INTO vectors_vec (hash, embedding) VALUES (?, ?)`);
  const insertContentVectorStmt = db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, model, embedded_at) VALUES (?, ?, ?)`);

  let embedded = 0, errors = 0, bytesProcessed = 0;
  const startTime = Date.now();

  // Insert first
  insertVecStmt.run(itemsWithSize[0].hash, new Float32Array(firstEmbedding));
  insertContentVectorStmt.run(itemsWithSize[0].hash, model, now);
  embedded++;
  bytesProcessed += itemsWithSize[0].bytes;

  for (let i = 1; i < itemsWithSize.length; i++) {
    const item = itemsWithSize[i];
    try {
      const embedding = await getEmbedding(item.body, model, false, item.title);
      insertVecStmt.run(item.hash, new Float32Array(embedding));
      insertContentVectorStmt.run(item.hash, model, now);
      embedded++;
      bytesProcessed += item.bytes;
    } catch (err) {
      errors++;
      bytesProcessed += item.bytes;
      progress.error();
      console.error(`\n${c.yellow}⚠ Error embedding ${item.hash.slice(0, 8)}...: ${err}${c.reset}`);
    }

    const processed = embedded + errors;
    const percent = (bytesProcessed / totalBytes) * 100;
    progress.set(percent);

    const elapsed = (Date.now() - startTime) / 1000;
    const bytesPerSec = bytesProcessed / elapsed;
    const remainingBytes = totalBytes - bytesProcessed;
    const etaSec = remainingBytes / bytesPerSec;

    const bar = renderProgressBar(percent);
    const percentStr = percent.toFixed(0).padStart(3);
    const throughput = `${formatBytes(bytesPerSec)}/s`;
    const eta = elapsed > 2 ? formatETA(etaSec) : "...";
    const errStr = errors > 0 ? ` ${c.yellow}${errors} err${c.reset}` : "";

    process.stderr.write(`\r${c.cyan}${bar}${c.reset} ${c.bold}${percentStr}%${c.reset} ${c.dim}${embedded}/${total}${c.reset}${errStr} ${c.dim}${throughput} ETA ${eta}${c.reset}   `);
  }

  progress.clear();
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  const avgThroughput = formatBytes(totalBytes / parseFloat(totalTime));

  console.log(`\r${c.green}${renderProgressBar(100)}${c.reset} ${c.bold}100%${c.reset}                                    `);
  console.log(`\n${c.green}✓ Done!${c.reset} Embedded ${c.bold}${embedded}${c.reset} documents in ${c.bold}${totalTime}s${c.reset} ${c.dim}(${avgThroughput}/s)${c.reset}`);
  if (errors > 0) {
    console.log(`${c.yellow}⚠ ${errors} documents failed${c.reset}`);
  }
  db.close();
}

function escapeCSV(value: string): string {
  if (value.includes('"') || value.includes(',') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function extractSnippet(body: string, query: string, maxLen = 500): { line: number; snippet: string } {
  const lines = body.split('\n');
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  let bestLine = 0, bestScore = -1;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = lines[i].toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      if (lineLower.includes(term)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLine = i;
    }
  }

  const startLine = Math.max(0, bestLine - 1);
  const endLine = Math.min(lines.length, bestLine + 2);
  let snippet = lines.slice(startLine, endLine).join('\n');
  if (snippet.length > maxLen) snippet = snippet.substring(0, maxLen - 3) + "...";
  return { line: bestLine + 1, snippet };
}

type SearchResult = { file: string; body: string; score: number; source: "fts" | "vec" };

// Build FTS5 query: phrase-aware with fallback to individual terms
function buildFTS5Query(query: string): string {
  const terms = query
    .split(/\s+/)
    .filter(term => term.length >= 2); // Skip single chars

  if (terms.length === 0) return "";
  if (terms.length === 1) return `"${terms[0].replace(/"/g, '""')}"`;

  // Strategy: exact phrase OR proximity match OR individual terms
  // Exact phrase matches rank highest, then close proximity, then any term
  const phrase = `"${query.replace(/"/g, '""')}"`;
  const quotedTerms = terms.map(t => `"${t.replace(/"/g, '""')}"`);

  // FTS5 NEAR syntax: NEAR(term1 term2, distance)
  const nearPhrase = `NEAR(${quotedTerms.join(' ')}, 10)`;
  const orTerms = quotedTerms.join(' OR ');

  // Exact phrase > proximity > any term
  return `(${phrase}) OR (${nearPhrase}) OR (${orTerms})`;
}

// Normalize BM25 score to 0-1 range using sigmoid
function normalizeBM25(score: number): number {
  // BM25 scores are negative in SQLite (lower = better)
  // Typical range: -15 (excellent) to -2 (weak match)
  // Map to 0-1 where higher is better
  const absScore = Math.abs(score);
  // Sigmoid-ish normalization: maps ~2-15 range to ~0.1-0.95
  return 1 / (1 + Math.exp(-(absScore - 5) / 3));
}

function searchFTS(db: Database, query: string, limit: number = 20): SearchResult[] {
  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  // BM25 weights: name=10, body=1 (title matches ranked higher)
  const stmt = db.prepare(`
    SELECT d.filepath, d.body, bm25(documents_fts, 10.0, 1.0) as score
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    WHERE documents_fts MATCH ? AND d.active = 1
    ORDER BY score
    LIMIT ?
  `);
  const results = stmt.all(ftsQuery, limit) as { filepath: string; body: string; score: number }[];
  return results.map(r => ({
    file: r.filepath,
    body: r.body,
    score: normalizeBM25(r.score),
    source: "fts" as const,
  }));
}

async function searchVec(db: Database, query: string, model: string, limit: number = 20): Promise<SearchResult[]> {
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) return [];

  const queryEmbedding = await getEmbedding(query, model, true);
  const queryVec = new Float32Array(queryEmbedding);

  // Join: documents -> content_vectors -> vectors_vec
  const stmt = db.prepare(`
    SELECT d.filepath, d.body, vec.distance
    FROM vectors_vec vec
    JOIN documents d ON d.hash = vec.hash
    WHERE vec.embedding MATCH ? AND k = ? AND d.active = 1
    ORDER BY vec.distance
  `);
  const results = stmt.all(queryVec, limit) as { filepath: string; body: string; distance: number }[];
  return results.map(r => ({
    file: r.filepath,
    body: r.body,
    score: 1 / (1 + r.distance),
    source: "vec" as const,
  }));
}

function normalizeScores(results: SearchResult[]): SearchResult[] {
  if (results.length === 0) return results;
  const maxScore = Math.max(...results.map(r => r.score));
  const minScore = Math.min(...results.map(r => r.score));
  const range = maxScore - minScore || 1;
  return results.map(r => ({ ...r, score: (r.score - minScore) / range }));
}

// Reciprocal Rank Fusion: combines multiple ranked lists
// RRF score = sum(1 / (k + rank)) across all lists where doc appears
// k=60 is standard, provides good balance between top and lower ranks
type RankedResult = { file: string; body: string; score: number };

function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],  // Weight per result list (default 1.0)
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { score: number; body: string; bestRank: number }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const results = resultLists[listIdx];
    const weight = weights[listIdx] ?? 1.0;
    for (let rank = 0; rank < results.length; rank++) {
      const doc = results[rank];
      const rrfScore = weight / (k + rank + 1);
      const existing = scores.get(doc.file);
      if (existing) {
        existing.score += rrfScore;
        existing.bestRank = Math.min(existing.bestRank, rank);
      } else {
        scores.set(doc.file, { score: rrfScore, body: doc.body, bestRank: rank });
      }
    }
  }

  // Add bonus for best rank: documents that ranked #1-3 in any list get a boost
  // This prevents dilution of exact matches by expansion queries
  return Array.from(scores.entries())
    .map(([file, { score, body, bestRank }]) => {
      let bonus = 0;
      if (bestRank === 0) bonus = 0.05;  // Ranked #1 somewhere
      else if (bestRank <= 2) bonus = 0.02;  // Ranked top-3 somewhere
      return { file, body, score: score + bonus };
    })
    .sort((a, b) => b.score - a.score);
}

type OutputFormat = "cli" | "csv" | "md" | "xml";
type OutputOptions = {
  format: OutputFormat;
  full: boolean;
  limit: number;
  minScore: number;
};

// Extract snippet with more context lines for CLI display
function extractSnippetWithContext(body: string, query: string, contextLines = 3): { line: number; snippet: string; hasMatch: boolean } {
  const lines = body.split('\n');
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  let bestLine = 0, bestScore = -1;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = lines[i].toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      if (lineLower.includes(term)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLine = i;
    }
  }

  // No query match found - return beginning of file
  if (bestScore <= 0) {
    const preview = lines.slice(0, contextLines * 2).join('\n').trim();
    return { line: 1, snippet: preview, hasMatch: false };
  }

  const startLine = Math.max(0, bestLine - contextLines);
  const endLine = Math.min(lines.length, bestLine + contextLines + 1);
  const snippet = lines.slice(startLine, endLine).join('\n').trim();
  return { line: bestLine + 1, snippet, hasMatch: true };
}

// Highlight query terms in text (skip short words < 3 chars)
function highlightTerms(text: string, query: string): string {
  if (!useColor) return text;
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length >= 3);
  let result = text;
  for (const term of terms) {
    const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    result = result.replace(regex, `${c.yellow}${c.bold}$1${c.reset}`);
  }
  return result;
}

// Format score with color based on value
function formatScore(score: number): string {
  const pct = (score * 100).toFixed(0).padStart(3);
  if (!useColor) return `${pct}%`;
  if (score >= 0.7) return `${c.green}${pct}%${c.reset}`;
  if (score >= 0.4) return `${c.yellow}${pct}%${c.reset}`;
  return `${c.dim}${pct}%${c.reset}`;
}

// Shorten filepath for display
function shortPath(filepath: string): string {
  const cwd = getPwd();
  if (filepath.startsWith(cwd)) {
    return filepath.slice(cwd.length + 1);
  }
  // Show last 2 path components
  const parts = filepath.split('/');
  if (parts.length > 2) {
    return '.../' + parts.slice(-2).join('/');
  }
  return filepath;
}

function outputResults(results: { file: string; body: string; score: number }[], query: string, opts: OutputOptions): void {
  const filtered = results.filter(r => r.score >= opts.minScore).slice(0, opts.limit);

  if (filtered.length === 0) {
    console.log("No results found above minimum score threshold.");
    return;
  }

  if (opts.format === "cli") {
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      const { line, snippet, hasMatch } = extractSnippetWithContext(row.body, query, 2);

      // Header: score and filename
      const score = formatScore(row.score);
      const path = shortPath(row.file);
      const lineInfo = hasMatch ? `:${line}` : "";
      console.log(`${c.bold}${score}${c.reset}  ${c.cyan}${path}${c.dim}${lineInfo}${c.reset}`);

      // Snippet with highlighting
      const highlighted = highlightTerms(snippet, query);
      const indented = highlighted.split('\n').map(l => `  ${c.dim}│${c.reset} ${l}`).join('\n');
      console.log(indented);

      if (i < filtered.length - 1) console.log();
    }
  } else if (opts.format === "md") {
    for (const row of filtered) {
      if (opts.full) {
        console.log(`---\n# ${row.file}\n\n${row.body}\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query);
        console.log(`---\n# ${row.file}\n\n${snippet}\n`);
      }
    }
  } else if (opts.format === "xml") {
    for (const row of filtered) {
      if (opts.full) {
        console.log(`<file name="${row.file}">\n${row.body}\n</file>\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query);
        console.log(`<file name="${row.file}">\n${snippet}\n</file>\n`);
      }
    }
  } else {
    // CSV format
    console.log("score,file,line,snippet");
    for (const row of filtered) {
      const { line, snippet } = extractSnippet(row.body, query);
      const content = opts.full ? row.body : snippet;
      console.log(`${row.score.toFixed(4)},${escapeCSV(row.file)},${line},${escapeCSV(content)}`);
    }
  }
}

function search(query: string, opts: OutputOptions): void {
  const db = getDb();
  const results = searchFTS(db, query, 50);
  db.close();

  if (results.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(results, query, opts);
}

async function vectorSearch(query: string, opts: OutputOptions, model: string = DEFAULT_EMBED_MODEL): Promise<void> {
  const db = getDb();

  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) {
    console.error("Vector index not found. Run 'qmd embed' first to create embeddings.");
    db.close();
    return;
  }

  // Expand query to multiple variations
  const queries = await expandQuery(query);
  process.stderr.write(`Searching with ${queries.length} query variations...\n`);

  // Collect results from all query variations
  const allResults = new Map<string, { file: string; body: string; score: number }>();

  for (const q of queries) {
    const vecResults = await searchVec(db, q, model, 20);
    for (const r of vecResults) {
      const existing = allResults.get(r.file);
      if (!existing || r.score > existing.score) {
        allResults.set(r.file, { file: r.file, body: r.body, score: r.score });
      }
    }
  }

  db.close();

  // Sort by max score and limit to requested count
  const results = Array.from(allResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.limit);

  if (results.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(results, query, { ...opts, limit: results.length }); // Already limited
}

async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL): Promise<string[]> {
  process.stderr.write("Generating query variations...\n");

  const prompt = `Generate 3 search query variations to find documents about this topic.

IMPORTANT: Keep multi-word phrases intact if they look like names (e.g., "Build a Business" should stay as "Build a Business", not "create a company").

Query: "${query}"

Output 3 variations, one per line:`;

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      think: false,  // Disable thinking mode for qwen3 models
      options: { num_predict: 150 },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (errorText.includes("not found") || errorText.includes("does not exist")) {
      await ensureModelAvailable(model);
      return expandQuery(query, model);
    }
    // Fall back to original query if expansion fails
    return [query];
  }

  const data = await response.json() as { response: string };
  const lines = data.response.trim().split('\n')
    .map(l => l.replace(/^[\d\.\-\*\"\s]+/, '').replace(/["\s]+$/, '').trim())
    .filter(l => l.length > 0 && !l.startsWith('<'))
    .slice(0, 1);  // Only 1 expanded query to preserve original query signal

  // Original query + expansions (original gets 2x weight in RRF)
  const allQueries = [query, ...lines];
  process.stderr.write(`Queries:\n  - ${allQueries.join('\n  - ')}\n`);
  return allQueries;
}

async function querySearch(query: string, opts: OutputOptions, embedModel: string = DEFAULT_EMBED_MODEL, rerankModel: string = DEFAULT_RERANK_MODEL): Promise<void> {
  const db = getDb();

  // Expand query to multiple variations
  const queries = await expandQuery(query);
  process.stderr.write(`Searching with ${queries.length} query variations...\n`);

  // Collect ranked result lists for RRF fusion
  const rankedLists: RankedResult[][] = [];
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  for (const q of queries) {
    // FTS search - get ranked results
    const ftsResults = searchFTS(db, q, 20);
    if (ftsResults.length > 0) {
      rankedLists.push(ftsResults.map(r => ({ file: r.file, body: r.body, score: r.score })));
    }

    // Vector search - get ranked results
    if (hasVectors) {
      const vecResults = await searchVec(db, q, embedModel, 20);
      if (vecResults.length > 0) {
        rankedLists.push(vecResults.map(r => ({ file: r.file, body: r.body, score: r.score })));
      }
    }
  }

  // Apply Reciprocal Rank Fusion to combine all ranked lists
  // Give 2x weight to original query results (first 2 lists: FTS + vector)
  const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  const candidates = fused.slice(0, 30); // Over-retrieve for reranking

  if (candidates.length === 0) {
    console.log("No results found.");
    db.close();
    return;
  }

  // Rerank with the original query
  const reranked = await rerank(
    query,
    candidates.map(c => ({ file: c.file, text: c.body })),
    rerankModel
  );

  db.close();

  // Blend RRF position score with reranker score using position-aware weights
  // Top retrieval results get more protection from reranker disagreement
  const bodyMap = new Map(candidates.map(c => [c.file, c.body]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1])); // 1-indexed rank

  const finalResults = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || 30;
    // Position-aware blending: top retrieval results preserved more
    // Rank 1-3: 75% RRF, 25% reranker (trust retrieval for exact matches)
    // Rank 4-10: 60% RRF, 40% reranker
    // Rank 11+: 40% RRF, 60% reranker (trust reranker for lower-ranked)
    let rrfWeight: number;
    if (rrfRank <= 3) {
      rrfWeight = 0.75;
    } else if (rrfRank <= 10) {
      rrfWeight = 0.60;
    } else {
      rrfWeight = 0.40;
    }
    const rrfScore = 1 / rrfRank;  // Position-based: 1, 0.5, 0.33...
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;
    return {
      file: r.file,
      body: bodyMap.get(r.file) || "",
      score: blendedScore,
    };
  }).sort((a, b) => b.score - a.score);

  outputResults(finalResults, query, opts);
}

// Parse CLI options
function parseOptions(args: string[], defaultMinScore: number = 0): { opts: OutputOptions; query: string } {
  let format: OutputFormat = "cli";
  let full = false;
  let limit = 5;
  let minScore = defaultMinScore;
  const queryParts: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "-n" && i + 1 < args.length) {
      limit = parseInt(args[++i], 10) || 5;
    } else if (arg === "--min-score" && i + 1 < args.length) {
      minScore = parseFloat(args[++i]) || defaultMinScore;
    } else if (arg === "--full") {
      full = true;
    } else if (arg === "-csv" || arg === "--csv") {
      format = "csv";
    } else if (arg === "-md" || arg === "--md") {
      format = "md";
    } else if (arg === "-xml" || arg === "--xml") {
      format = "xml";
    } else if (!arg.startsWith("-")) {
      queryParts.push(arg);
    }
  }

  return {
    opts: { format, full, limit, minScore },
    query: queryParts.join(" "),
  };
}

// Parse global options and extract remaining args
function parseGlobalOptions(args: string[]): string[] {
  const remaining: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--index" && i + 1 < args.length) {
      customIndexName = args[++i];
    } else {
      remaining.push(args[i]);
    }
  }
  return remaining;
}

// Main CLI
const rawArgs = process.argv.slice(2);
const args = parseGlobalOptions(rawArgs);

if (args.length === 0) {
  console.log("Usage:");
  console.log("  qmd add [--drop] [glob]    - Add/update collection from $PWD (default: **/*.md)");
  console.log("  qmd status                 - Show index status and collections");
  console.log("  qmd update-all             - Re-index all collections");
  console.log("  qmd embed [-f]             - Create vector embeddings for all content");
  console.log("  qmd search <query>         - Full-text search (BM25)");
  console.log("  qmd vsearch <query>        - Vector similarity search");
  console.log("  qmd query <query>          - Combined search with query expansion + reranking");
  console.log("");
  console.log("Global options:");
  console.log("  --index <name>             - Use custom index name (default: index)");
  console.log("");
  console.log("Search options:");
  console.log("  -n <num>                   - Number of results (default: 5)");
  console.log("  --min-score <num>          - Minimum similarity score");
  console.log("  --full                     - Output full document instead of snippet");
  console.log("  -csv                       - CSV output (default is colorized CLI)");
  console.log("  -md                        - Markdown output");
  console.log("  -xml                       - XML output");
  console.log("");
  console.log("Environment:");
  console.log("  OLLAMA_URL                 - Ollama server URL (default: http://localhost:11434)");
  console.log("");
  console.log("Models:");
  console.log(`  Embedding: ${DEFAULT_EMBED_MODEL}`);
  console.log(`  Reranking: ${DEFAULT_RERANK_MODEL}`);
  console.log("");
  console.log(`Index: ${getDbPath()}`);
  process.exit(1);
}

const cmd = args[0];

if (cmd === "add") {
  const addArgs = args.slice(1);
  const drop = addArgs.includes("--drop");
  const globArg = addArgs.find(a => !a.startsWith("-"));
  // Treat "." as "use default glob in current directory"
  const globPattern = (!globArg || globArg === ".") ? DEFAULT_GLOB : globArg;

  if (drop) {
    await dropCollection(globPattern);
  } else {
    await indexFiles(globPattern);
  }
} else if (cmd === "status") {
  showStatus();
} else if (cmd === "update-all") {
  await updateAllCollections();
} else if (cmd === "embed") {
  const embedArgs = args.slice(1);
  const force = embedArgs.includes("-f") || embedArgs.includes("--force");
  await vectorIndex(DEFAULT_EMBED_MODEL, force);
} else if (cmd === "search") {
  const { opts, query } = parseOptions(args.slice(1), 0);
  if (!query) {
    console.error("Usage: qmd search [-n num] [--min-score num] [--full] [-csv|-md|-xml] <query>");
    process.exit(1);
  }
  search(query, opts);
} else if (cmd === "vsearch") {
  const { opts, query } = parseOptions(args.slice(1), 0.3);
  if (!query) {
    console.error("Usage: qmd vsearch [-n num] [--min-score num] [--full] [-csv|-md|-xml] <query>");
    process.exit(1);
  }
  await vectorSearch(query, opts);
} else if (cmd === "query") {
  const { opts, query } = parseOptions(args.slice(1), 0);
  if (!query) {
    console.error("Usage: qmd query [-n num] [--min-score num] [--full] [-csv|-md|-xml] <query>");
    process.exit(1);
  }
  await querySearch(query, opts);
} else {
  console.error(`Unknown command: ${cmd}`);
  console.error("Run 'qmd' without arguments for usage.");
  process.exit(1);
}
