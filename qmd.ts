#!/usr/bin/env bun
import { Database } from "bun:sqlite";
import { Glob, $ } from "bun";
import { parseArgs } from "util";
import * as sqliteVec from "sqlite-vec";

const HOME = Bun.env.HOME || "/tmp";

function homedir(): string {
  return HOME;
}

function resolve(...paths: string[]): string {
  // Simple path resolution
  let result = paths[0].startsWith('/') ? '' : Bun.env.PWD || process.cwd();
  for (const p of paths) {
    if (p.startsWith('/')) {
      result = p;
    } else {
      result = result + '/' + p;
    }
  }
  // Normalize: remove // and resolve . and ..
  const parts = result.split('/').filter(Boolean);
  const normalized: string[] = [];
  for (const part of parts) {
    if (part === '..') normalized.pop();
    else if (part !== '.') normalized.push(part);
  }
  return '/' + normalized.join('/');
}

// On macOS, use Homebrew's SQLite which supports extensions
if (process.platform === "darwin") {
  const homebrewSqlitePath = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
  if (Bun.file(homebrewSqlitePath).size > 0) {
    Database.setCustomSQLite(homebrewSqlitePath);
  }
}

const DEFAULT_EMBED_MODEL = "embeddinggemma";
const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
const DEFAULT_QUERY_MODEL = "qwen3:0.6b";
const DEFAULT_GLOB = "**/*.md";
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Chunking: ~2000 tokens per chunk, ~3 bytes/token = 6KB
const CHUNK_TOKEN_LENGTH = 2000;
const CHUNK_BYTE_SIZE = 6 * 1024;

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

// Terminal cursor control
const cursor = {
  hide() { process.stderr.write('\x1b[?25l'); },
  show() { process.stderr.write('\x1b[?25h'); },
};

// Ensure cursor is restored on exit
process.on('SIGINT', () => { cursor.show(); process.exit(130); });
process.on('SIGTERM', () => { cursor.show(); process.exit(143); });

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
  const cacheDir = Bun.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
  const qmdCacheDir = resolve(cacheDir, "qmd");
  // Ensure cache directory exists
  try { Bun.spawnSync(["mkdir", "-p", qmdCacheDir]); } catch {}
  const dbName = customIndexName || "index";
  return resolve(qmdCacheDir, `${dbName}.sqlite`);
}

function getPwd(): string {
  return process.env.PWD || process.cwd();
}

// Get canonical realpath, falling back to resolved path if file doesn't exist
function getRealPath(path: string): string {
  try {
    const result = Bun.spawnSync(["realpath", path]);
    if (result.success) {
      return result.stdout.toString().trim();
    }
  } catch {}
  return resolve(path);
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
  hash TEXT NOT NULL,
  seq INTEGER NOT NULL DEFAULT 0,  -- chunk sequence (0, 1, 2...)
  pos INTEGER NOT NULL DEFAULT 0,  -- character position in document
  model TEXT NOT NULL,
  embedded_at TEXT NOT NULL,
  PRIMARY KEY (hash, seq)
);

CREATE VIRTUAL TABLE vectors_vec USING vec0(
  hash_seq TEXT PRIMARY KEY,  -- "{hash}_{seq}"
  embedding float[N]
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
      context TEXT,
      UNIQUE(pwd, glob_pattern)
    )
  `);

  // Path-based context (more flexible than collection-level)
  db.exec(`
    CREATE TABLE IF NOT EXISTS path_contexts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path_prefix TEXT NOT NULL UNIQUE,
      context TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_path_contexts_prefix ON path_contexts(path_prefix)`);

  // Cache table for Ollama API calls (not embeddings)
  db.exec(`
    CREATE TABLE IF NOT EXISTS ollama_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
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

  // Content vectors keyed by (hash, seq) for chunked embeddings
  // Migration: check if old schema (no seq column) and recreate
  const cvInfo = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
  const hasSeqColumn = cvInfo.some(col => col.name === 'seq');
  if (cvInfo.length > 0 && !hasSeqColumn) {
    // Old schema without chunking - drop and recreate (embeddings need regenerating anyway)
    db.exec(`DROP TABLE IF EXISTS content_vectors`);
    db.exec(`DROP TABLE IF EXISTS vectors_vec`);
  }
  db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
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
  // Ensure only one active document per filepath
  db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_filepath_active ON documents(filepath) WHERE active = 1`);

  return db;
}

function ensureVecTable(db: Database, dimensions: number): void {
  const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get() as { sql: string } | null;
  if (tableInfo) {
    // Check for correct dimensions and hash_seq key (not old 'hash' key)
    const match = tableInfo.sql.match(/float\[(\d+)\]/);
    const hasHashSeq = tableInfo.sql.includes('hash_seq');
    if (match && parseInt(match[1]) === dimensions && hasHashSeq) return;
    db.exec("DROP TABLE IF EXISTS vectors_vec");
  }
  // Use hash_seq as composite key: "{hash}_{seq}" (e.g., "abc123_0", "abc123_1")
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}])`);
}

function getHashesNeedingEmbedding(db: Database): number {
  // Check for hashes missing the first chunk (seq=0)
  const result = db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

async function hashContent(content: string): Promise<string> {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(content);
  return hash.digest("hex");
}

// Cache helpers for Ollama API calls (not embeddings)
function getCacheKey(url: string, body: object): string {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(url);
  hash.update(JSON.stringify(body));
  return hash.digest("hex");
}

function getCachedResult(db: Database, cacheKey: string): string | null {
  const row = db.prepare(`SELECT result FROM ollama_cache WHERE hash = ?`).get(cacheKey) as { result: string } | null;
  return row?.result || null;
}

function setCachedResult(db: Database, cacheKey: string, result: string): void {
  const now = new Date().toISOString();
  db.prepare(`INSERT OR REPLACE INTO ollama_cache (hash, result, created_at) VALUES (?, ?, ?)`).run(cacheKey, result, now);

  // 1 in 100 chance to truncate to most recent 1000 entries
  if (Math.random() < 0.01) {
    db.exec(`DELETE FROM ollama_cache WHERE hash NOT IN (SELECT hash FROM ollama_cache ORDER BY created_at DESC LIMIT 1000)`);
  }
}

function clearCache(db: Database): void {
  db.exec(`DELETE FROM ollama_cache`);
}

// Extract title from first markdown headline, or use filename as fallback
function extractTitle(content: string, filename: string): string {
  const match = content.match(/^##?\s+(.+)$/m);
  if (match) {
    const title = match[1].trim();
    // Skip generic "📝 Notes" heading, find next ## instead
    if (title === "📝 Notes" || title === "Notes") {
      const nextMatch = content.match(/^##\s+(.+)$/m);
      if (nextMatch) return nextMatch[1].trim();
    }
    return title;
  }
  return filename.replace(/\.md$/, "").split("/").pop() || filename;
}

// Format text for EmbeddingGemma
function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// Chunk document into ~6KB pieces, breaking at word boundaries
function chunkDocument(content: string, maxBytes: number = CHUNK_BYTE_SIZE): { text: string; pos: number }[] {
  const encoder = new TextEncoder();
  const totalBytes = encoder.encode(content).length;

  // Single chunk if small enough
  if (totalBytes <= maxBytes) {
    return [{ text: content, pos: 0 }];
  }

  const chunks: { text: string; pos: number }[] = [];
  let charPos = 0;

  while (charPos < content.length) {
    // Find chunk boundary at ~maxBytes
    let endPos = charPos;
    let byteCount = 0;

    // Advance character by character, counting bytes
    while (endPos < content.length && byteCount < maxBytes) {
      const charBytes = encoder.encode(content[endPos]).length;
      if (byteCount + charBytes > maxBytes) break;
      byteCount += charBytes;
      endPos++;
    }

    // Back up to word boundary (paragraph, newline, or space)
    if (endPos < content.length && endPos > charPos) {
      const slice = content.slice(charPos, endPos);
      // Prefer paragraph break, then sentence end, then newline, then space
      const paragraphBreak = slice.lastIndexOf('\n\n');
      const sentenceEnd = Math.max(
        slice.lastIndexOf('. '),
        slice.lastIndexOf('.\n'),
        slice.lastIndexOf('? '),
        slice.lastIndexOf('?\n'),
        slice.lastIndexOf('! '),
        slice.lastIndexOf('!\n')
      );
      const lineBreak = slice.lastIndexOf('\n');
      const spaceBreak = slice.lastIndexOf(' ');

      let breakPoint = -1;
      if (paragraphBreak > slice.length * 0.5) {
        breakPoint = paragraphBreak + 2; // Include the double newline
      } else if (sentenceEnd > slice.length * 0.5) {
        breakPoint = sentenceEnd + 2; // Include period and space
      } else if (lineBreak > slice.length * 0.3) {
        breakPoint = lineBreak + 1;
      } else if (spaceBreak > slice.length * 0.3) {
        breakPoint = spaceBreak + 1;
      }

      if (breakPoint > 0) {
        endPos = charPos + breakPoint;
      }
    }

    // Ensure we make progress (at least one character)
    if (endPos <= charPos) {
      endPos = charPos + 1;
    }

    chunks.push({ text: content.slice(charPos, endPos), pos: charPos });
    charPos = endPos;
  }

  return chunks;
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

function parseRerankResponse(data: RerankResponse): number {
  if (!data.logprobs || data.logprobs.length === 0) {
    throw new Error("Reranker response missing logprobs");
  }

  const firstToken = data.logprobs[0];
  const token = firstToken.token.toLowerCase().trim();
  const confidence = Math.exp(firstToken.logprob);

  if (token === "yes") {
    return confidence;
  }
  if (token === "no") {
    return (1 - confidence) * 0.3;
  }

  throw new Error(`Unexpected reranker token: "${token}"`);
}

async function rerankSingle(prompt: string, model: string, db?: Database, retried: boolean = false): Promise<number> {
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

  const requestBody = {
    model,
    prompt: fullPrompt,
    raw: true,
    stream: false,
    logprobs: true,
    options: { num_predict: 1 },
  };

  // Check cache
  const cacheKey = db ? getCacheKey(`${OLLAMA_URL}/api/generate`, requestBody) : "";
  if (db) {
    const cached = getCachedResult(db, cacheKey);
    if (cached) {
      const data = JSON.parse(cached) as RerankResponse;
      return parseRerankResponse(data);
    }
  }

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return rerankSingle(prompt, model, db, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json() as RerankResponse;

  // Cache the result
  if (db) {
    setCachedResult(db, cacheKey, JSON.stringify(data));
  }

  return parseRerankResponse(data);
}

async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL, db?: Database): Promise<{ file: string; score: number }[]> {
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
          const score = await rerankSingle(prompt, model, db);
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

  // Get all path contexts
  const pathContexts = db.prepare(`SELECT path_prefix, context FROM path_contexts ORDER BY path_prefix`).all() as { path_prefix: string; context: string }[];

  if (collections.length > 0) {
    console.log(`\n${c.bold}Collections${c.reset}`);
    for (const col of collections) {
      const lastMod = col.last_modified ? formatTimeAgo(new Date(col.last_modified)) : "never";
      console.log(`  ${c.cyan}${col.pwd}${c.reset}`);
      console.log(`    ${col.glob_pattern} → ${col.active_count} docs (updated ${lastMod})`);

      // Show contexts that match this collection's path
      const matchingContexts = pathContexts.filter(ctx =>
        ctx.path_prefix.startsWith(col.pwd) || col.pwd.startsWith(ctx.path_prefix)
      );
      for (const ctx of matchingContexts) {
        const displayPath = shortPath(ctx.path_prefix);
        console.log(`    ${c.dim}context: ${displayPath} → "${ctx.context}"${c.reset}`);
      }
    }
  } else {
    console.log(`\n${c.dim}No collections. Run 'qmd add .' to index markdown files.${c.reset}`);
  }

  db.close();
}

async function updateAllCollections(): Promise<void> {
  const db = getDb();
  cleanupDuplicateCollections(db);

  // Clear Ollama cache on update
  clearCache(db);

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

async function addContext(pathArg: string, contextText: string): Promise<void> {
  const db = getDb();
  const now = new Date().toISOString();

  // Resolve path - could be relative, absolute, or use ~
  let pathPrefix = pathArg;
  if (pathPrefix === '.' || pathPrefix === './') {
    pathPrefix = getPwd();
  } else if (pathPrefix.startsWith('~/')) {
    pathPrefix = homedir() + pathPrefix.slice(1);
  } else if (!pathPrefix.startsWith('/')) {
    pathPrefix = resolve(getPwd(), pathPrefix);
  }

  // Get realpath and normalize: remove trailing slash
  pathPrefix = getRealPath(pathPrefix).replace(/\/$/, '');

  // Insert or update
  db.prepare(`INSERT INTO path_contexts (path_prefix, context, created_at) VALUES (?, ?, ?)
              ON CONFLICT(path_prefix) DO UPDATE SET context = excluded.context`).run(pathPrefix, contextText, now);

  console.log(`${c.green}✓${c.reset} Added context for: ${shortPath(pathPrefix)}`);
  console.log(`${c.dim}Context: ${contextText}${c.reset}`);
  db.close();
}

function getDocument(filename: string): void {
  const db = getDb();

  // Expand ~ to home directory
  let filepath = filename;
  if (filepath.startsWith('~/')) {
    filepath = homedir() + filepath.slice(1);
  }

  // Try exact match first
  let doc = db.prepare(`SELECT body FROM documents WHERE filepath = ? AND active = 1`).get(filepath) as { body: string } | null;

  // Try matching by filename ending (allows partial paths)
  if (!doc) {
    doc = db.prepare(`SELECT body FROM documents WHERE filepath LIKE ? AND active = 1 LIMIT 1`).get(`%${filepath}`) as { body: string } | null;
  }

  if (!doc) {
    console.error(`Document not found: ${filename}`);
    db.close();
    process.exit(1);
  }

  console.log(doc.body);
  db.close();
}

// Get context for a filepath (finds most specific matching path prefix)
function getContextForFile(db: Database, filepath: string): string | null {
  // Find all matching prefixes and return the longest (most specific) one
  const result = db.prepare(`
    SELECT context FROM path_contexts
    WHERE ? LIKE path_prefix || '%'
    ORDER BY LENGTH(path_prefix) DESC
    LIMIT 1
  `).get(filepath) as { context: string } | null;
  return result?.context || null;
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

  // Clear Ollama cache on index
  clearCache(db);

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
  const findActiveStmt = db.prepare(`SELECT id, hash, title FROM documents WHERE collection_id = ? AND filepath = ? AND active = 1`);
  const updateTitleStmt = db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`);

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenFiles = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = getRealPath(resolve(pwd, relativeFile));
    seenFiles.add(filepath);

    const content = await Bun.file(filepath).text();
    const hash = await hashContent(content);
    const name = relativeFile.replace(/\.md$/, "").split("/").pop() || relativeFile;
    const title = extractTitle(content, relativeFile);
    const existing = findActiveStmt.get(collectionId, filepath) as { id: number; hash: string; title: string } | null;

    if (existing) {
      if (existing.hash === hash) {
        // Hash unchanged, but check if title needs updating (e.g., extraction logic improved)
        if (existing.title !== title) {
          updateTitleStmt.run(title, now, existing.id);
          updated++;
        } else {
          unchanged++;
        }
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
  // Use MIN(filepath) to get one representative filepath per hash
  const hashesToEmbed = db.prepare(`
    SELECT d.hash, d.body, MIN(d.filepath) as filepath
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; filepath: string }[];

  if (hashesToEmbed.length === 0) {
    console.log(`${c.green}✓ All content hashes already have embeddings.${c.reset}`);
    db.close();
    return;
  }

  // Prepare documents with chunks
  type ChunkItem = { hash: string; title: string; text: string; seq: number; pos: number; bytes: number; displayName: string };
  const allChunks: ChunkItem[] = [];
  let multiChunkDocs = 0;

  for (const item of hashesToEmbed) {
    const encoder = new TextEncoder();
    const bodyBytes = encoder.encode(item.body).length;
    if (bodyBytes === 0) continue; // Skip empty

    const title = extractTitle(item.body, item.filepath);
    const displayName = shortPath(item.filepath);
    const chunks = chunkDocument(item.body, CHUNK_BYTE_SIZE);

    if (chunks.length > 1) multiChunkDocs++;

    for (let seq = 0; seq < chunks.length; seq++) {
      allChunks.push({
        hash: item.hash,
        title,
        text: chunks[seq].text,
        seq,
        pos: chunks[seq].pos,
        bytes: encoder.encode(chunks[seq].text).length,
        displayName,
      });
    }
  }

  if (allChunks.length === 0) {
    console.log(`${c.green}✓ No non-empty documents to embed.${c.reset}`);
    db.close();
    return;
  }

  const totalBytes = allChunks.reduce((sum, c) => sum + c.bytes, 0);
  const totalChunks = allChunks.length;
  const totalDocs = hashesToEmbed.length;

  console.log(`${c.bold}Embedding ${totalDocs} documents${c.reset} ${c.dim}(${totalChunks} chunks, ${formatBytes(totalBytes)})${c.reset}`);
  if (multiChunkDocs > 0) {
    console.log(`${c.dim}${multiChunkDocs} documents split into multiple chunks${c.reset}`);
  }
  console.log(`${c.dim}Model: ${model}${c.reset}\n`);

  // Hide cursor during embedding
  cursor.hide();

  // Get embedding dimensions from first chunk
  progress.indeterminate();
  const firstEmbedding = await getEmbedding(allChunks[0].text, model, false, allChunks[0].title);
  ensureVecTable(db, firstEmbedding.length);

  const insertVecStmt = db.prepare(`INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`);
  const insertContentVectorStmt = db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)`);

  let chunksEmbedded = 0, errors = 0, bytesProcessed = 0;
  const startTime = Date.now();

  // Insert first chunk
  const firstHashSeq = `${allChunks[0].hash}_${allChunks[0].seq}`;
  insertVecStmt.run(firstHashSeq, new Float32Array(firstEmbedding));
  insertContentVectorStmt.run(allChunks[0].hash, allChunks[0].seq, allChunks[0].pos, model, now);
  chunksEmbedded++;
  bytesProcessed += allChunks[0].bytes;

  for (let i = 1; i < allChunks.length; i++) {
    const chunk = allChunks[i];
    try {
      const embedding = await getEmbedding(chunk.text, model, false, chunk.title);
      const hashSeq = `${chunk.hash}_${chunk.seq}`;
      insertVecStmt.run(hashSeq, new Float32Array(embedding));
      insertContentVectorStmt.run(chunk.hash, chunk.seq, chunk.pos, model, now);
      chunksEmbedded++;
      bytesProcessed += chunk.bytes;
    } catch (err) {
      errors++;
      bytesProcessed += chunk.bytes;
      progress.error();
      console.error(`\n${c.yellow}⚠ Error embedding "${chunk.displayName}" chunk ${chunk.seq}: ${err}${c.reset}`);
    }

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

    process.stderr.write(`\r${c.cyan}${bar}${c.reset} ${c.bold}${percentStr}%${c.reset} ${c.dim}${chunksEmbedded}/${totalChunks}${c.reset}${errStr} ${c.dim}${throughput} ETA ${eta}${c.reset}   `);
  }

  progress.clear();
  cursor.show();
  const totalTimeSec = (Date.now() - startTime) / 1000;
  const avgThroughput = formatBytes(totalBytes / totalTimeSec);

  console.log(`\r${c.green}${renderProgressBar(100)}${c.reset} ${c.bold}100%${c.reset}                                    `);
  console.log(`\n${c.green}✓ Done!${c.reset} Embedded ${c.bold}${chunksEmbedded}${c.reset} chunks from ${c.bold}${totalDocs}${c.reset} documents in ${c.bold}${formatETA(totalTimeSec)}${c.reset} ${c.dim}(${avgThroughput}/s)${c.reset}`);
  if (errors > 0) {
    console.log(`${c.yellow}⚠ ${errors} chunks failed${c.reset}`);
  }
  db.close();
}

function escapeCSV(value: string): string {
  if (value.includes('"') || value.includes(',') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function extractSnippet(body: string, query: string, maxLen = 500, chunkPos?: number): { line: number; snippet: string } {
  // If chunkPos provided, calculate line offset and focus search there
  let lineOffset = 0;
  let searchBody = body;
  if (chunkPos && chunkPos > 0) {
    // Count lines before chunkPos to get line offset
    const beforeChunk = body.slice(0, chunkPos);
    lineOffset = beforeChunk.split('\n').length - 1;
    // Focus search on the chunk area (with some context before)
    const contextStart = Math.max(0, chunkPos - 200);
    searchBody = body.slice(contextStart);
    if (contextStart > 0) {
      lineOffset = body.slice(0, contextStart).split('\n').length - 1;
    }
  }

  const lines = searchBody.split('\n');
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
  return { line: lineOffset + bestLine + 1, snippet };
}

type SearchResult = { file: string; body: string; score: number; source: "fts" | "vec"; chunkPos?: number };

// Sanitize a term for FTS5: remove punctuation except apostrophes
function sanitizeFTS5Term(term: string): string {
  // Remove all non-alphanumeric except apostrophes (for contractions like "don't")
  return term.replace(/[^\w']/g, '').trim();
}

// Build FTS5 query: phrase-aware with fallback to individual terms
function buildFTS5Query(query: string): string {
  // Sanitize the full query for phrase matching
  const sanitizedQuery = query.replace(/[^\w\s']/g, '').trim();

  const terms = query
    .split(/\s+/)
    .map(sanitizeFTS5Term)
    .filter(term => term.length >= 2); // Skip single chars and empty

  if (terms.length === 0) return "";
  if (terms.length === 1) return `"${terms[0].replace(/"/g, '""')}"`;

  // Strategy: exact phrase OR proximity match OR individual terms
  // Exact phrase matches rank highest, then close proximity, then any term
  const phrase = `"${sanitizedQuery.replace(/"/g, '""')}"`;
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

  // Join: vectors_vec -> content_vectors -> documents
  // Over-retrieve to handle multiple chunks per document, then dedupe
  const stmt = db.prepare(`
    SELECT d.filepath, d.body, vec.distance, cv.pos
    FROM vectors_vec vec
    JOIN content_vectors cv ON vec.hash_seq = cv.hash || '_' || cv.seq
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    WHERE vec.embedding MATCH ? AND k = ?
    ORDER BY vec.distance
  `);
  const rawResults = stmt.all(queryVec, limit * 3) as { filepath: string; body: string; distance: number; pos: number }[];

  // Aggregate chunks per document: max score + small bonus for additional matches
  const byFile = new Map<string, { filepath: string; body: string; chunkCount: number; bestPos: number; bestDist: number }>();
  for (const r of rawResults) {
    const existing = byFile.get(r.filepath);
    if (!existing) {
      byFile.set(r.filepath, { filepath: r.filepath, body: r.body, chunkCount: 1, bestPos: r.pos, bestDist: r.distance });
    } else {
      existing.chunkCount++;
      if (r.distance < existing.bestDist) {
        existing.bestDist = r.distance;
        existing.bestPos = r.pos;
      }
    }
  }

  // Score = max chunk score + 0.02 bonus per additional chunk (capped at +0.1)
  return Array.from(byFile.values())
    .map(r => {
      const maxScore = 1 / (1 + r.bestDist);
      const bonusChunks = Math.min(r.chunkCount - 1, 5);
      const bonus = bonusChunks * 0.02;
      return {
        file: r.filepath,
        body: r.body,
        score: maxScore + bonus,
        source: "vec" as const,
        chunkPos: r.bestPos,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
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

type OutputFormat = "cli" | "csv" | "md" | "xml" | "files" | "json";
type OutputOptions = {
  format: OutputFormat;
  full: boolean;
  limit: number;
  minScore: number;
};

// Extract snippet with more context lines for CLI display
function extractSnippetWithContext(body: string, query: string, contextLines = 3, chunkPos?: number): { line: number; snippet: string; hasMatch: boolean } {
  // If chunkPos provided, focus search on that area
  let lineOffset = 0;
  let searchBody = body;
  if (chunkPos && chunkPos > 0) {
    const contextStart = Math.max(0, chunkPos - 200);
    searchBody = body.slice(contextStart);
    if (contextStart > 0) {
      lineOffset = body.slice(0, contextStart).split('\n').length - 1;
    }
  }

  const lines = searchBody.split('\n');
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

  // No query match found - return beginning of chunk area or file
  if (bestScore <= 0) {
    const preview = lines.slice(0, contextLines * 2).join('\n').trim();
    return { line: lineOffset + 1, snippet: preview, hasMatch: false };
  }

  const startLine = Math.max(0, bestLine - contextLines);
  const endLine = Math.min(lines.length, bestLine + contextLines + 1);
  const snippet = lines.slice(startLine, endLine).join('\n').trim();
  return { line: lineOffset + bestLine + 1, snippet, hasMatch: true };
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

// Shorten filepath for display - always relative to $HOME
function shortPath(filepath: string): string {
  const home = homedir();
  if (filepath.startsWith(home)) {
    return '~' + filepath.slice(home.length);
  }
  return filepath;
}

function outputResults(results: { file: string; body: string; score: number; context?: string | null; chunkPos?: number }[], query: string, opts: OutputOptions): void {
  const filtered = results.filter(r => r.score >= opts.minScore).slice(0, opts.limit);

  if (filtered.length === 0) {
    console.log("No results found above minimum score threshold.");
    return;
  }

  if (opts.format === "json") {
    // JSON output for LLM consumption
    const output = filtered.map(row => ({
      score: Math.round(row.score * 100) / 100,
      file: shortPath(row.file),
      ...(row.context && { context: row.context }),
      ...(opts.full && { body: row.body }),
      ...(!opts.full && { snippet: extractSnippet(row.body, query, 300, row.chunkPos).snippet }),
    }));
    console.log(JSON.stringify(output, null, 2));
  } else if (opts.format === "files") {
    // Simple score,filepath,context output
    for (const row of filtered) {
      const path = shortPath(row.file);
      const ctx = row.context ? `,"${row.context.replace(/"/g, '""')}"` : "";
      console.log(`${row.score.toFixed(2)},${path}${ctx}`);
    }
  } else if (opts.format === "cli") {
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      const { line, snippet, hasMatch } = extractSnippetWithContext(row.body, query, 2, row.chunkPos);

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
      const path = shortPath(row.file);
      if (opts.full) {
        console.log(`---\n# ${path}\n\n${row.body}\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
        console.log(`---\n# ${path}\n\n${snippet}\n`);
      }
    }
  } else if (opts.format === "xml") {
    for (const row of filtered) {
      const path = shortPath(row.file);
      if (opts.full) {
        console.log(`<file name="${path}">\n${row.body}\n</file>\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
        console.log(`<file name="${path}">\n${snippet}\n</file>\n`);
      }
    }
  } else {
    // CSV format
    console.log("score,file,line,snippet");
    for (const row of filtered) {
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
      const content = opts.full ? row.body : snippet;
      console.log(`${row.score.toFixed(4)},${escapeCSV(shortPath(row.file))},${line},${escapeCSV(content)}`);
    }
  }
}

function search(query: string, opts: OutputOptions): void {
  const db = getDb();
  const results = searchFTS(db, query, 50);

  // Add context to results
  const resultsWithContext = results.map(r => ({
    ...r,
    context: getContextForFile(db, r.file),
  }));

  db.close();

  if (resultsWithContext.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(resultsWithContext, query, opts);
}

async function vectorSearch(query: string, opts: OutputOptions, model: string = DEFAULT_EMBED_MODEL): Promise<void> {
  const db = getDb();

  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) {
    console.error("Vector index not found. Run 'qmd embed' first to create embeddings.");
    db.close();
    return;
  }

  // Expand query to multiple variations (with caching)
  const queries = await expandQuery(query, DEFAULT_QUERY_MODEL, db);
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

  // Sort by max score and limit to requested count
  const results = Array.from(allResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.limit)
    .map(r => ({ ...r, context: getContextForFile(db, r.file) }));

  db.close();

  if (results.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(results, query, { ...opts, limit: results.length }); // Already limited
}

async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL, db?: Database): Promise<string[]> {
  process.stderr.write("Generating query variations...\n");

  const prompt = `You are a search query expander. Given a search query, generate 2 alternative queries that would help find relevant documents.

Rules:
- Use synonyms and related terminology (e.g., "craft" → "craftsmanship", "quality", "excellence")
- Rephrase to capture different angles (e.g., "engineering culture" → "technical excellence", "developer practices")
- Keep proper nouns and named concepts exactly as written (e.g., "Build a Business", "Stripe", "Shopify")
- Each variation should be 3-8 words, natural search terms
- Do NOT just append words like "search" or "find" or "documents"

Query: "${query}"

Output exactly 2 variations, one per line, no numbering or bullets:`;

  const requestBody = {
    model,
    prompt,
    stream: false,
    think: false,
    options: { num_predict: 150 },
  };

  // Check cache
  const cacheDb = db || getDb();
  const cacheKey = getCacheKey(`${OLLAMA_URL}/api/generate`, requestBody);
  const cached = getCachedResult(cacheDb, cacheKey);

  let responseText: string;
  if (cached) {
    responseText = cached;
  } else {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      if (errorText.includes("not found") || errorText.includes("does not exist")) {
        await ensureModelAvailable(model);
        if (!db) cacheDb.close();
        return expandQuery(query, model, db);
      }
      if (!db) cacheDb.close();
      return [query];
    }

    const data = await response.json() as { response: string };
    responseText = data.response;
    setCachedResult(cacheDb, cacheKey, responseText);
  }

  if (!db) cacheDb.close();

  const lines = responseText.trim().split('\n')
    .map(l => l.replace(/^[\d\.\-\*\"\s]+/, '').replace(/["\s]+$/, '').trim())
    .filter(l => l.length > 2 && l.length < 100 && !l.startsWith('<') && !l.toLowerCase().includes('variation'))
    .slice(0, 2);

  const allQueries = [query, ...lines];
  process.stderr.write(`${c.dim}Queries: ${allQueries.join(' | ')}${c.reset}\n`);
  return allQueries;
}

async function querySearch(query: string, opts: OutputOptions, embedModel: string = DEFAULT_EMBED_MODEL, rerankModel: string = DEFAULT_RERANK_MODEL): Promise<void> {
  const db = getDb();

  // Expand query to multiple variations (with caching)
  const queries = await expandQuery(query, DEFAULT_QUERY_MODEL, db);
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

  // Rerank with the original query (with caching)
  const reranked = await rerank(
    query,
    candidates.map(c => ({ file: c.file, text: c.body })),
    rerankModel,
    db
  );

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
      context: getContextForFile(db, r.file),
    };
  }).sort((a, b) => b.score - a.score);

  db.close();
  outputResults(finalResults, query, opts);
}

// Parse CLI arguments using util.parseArgs
function parseCLI() {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2), // Skip bun and script path
    options: {
      // Global options
      index: { type: "string" },
      help: { type: "boolean", short: "h" },
      // Search options
      n: { type: "string" },
      "min-score": { type: "string" },
      full: { type: "boolean" },
      csv: { type: "boolean" },
      md: { type: "boolean" },
      xml: { type: "boolean" },
      files: { type: "boolean" },
      json: { type: "boolean" },
      // Add options
      drop: { type: "boolean" },
      // Embed options
      force: { type: "boolean", short: "f" },
    },
    allowPositionals: true,
    strict: false, // Allow unknown options to pass through
  });

  // Set global index name
  if (values.index) {
    customIndexName = values.index;
  }

  // Determine output format
  let format: OutputFormat = "cli";
  if (values.csv) format = "csv";
  else if (values.md) format = "md";
  else if (values.xml) format = "xml";
  else if (values.files) format = "files";
  else if (values.json) format = "json";

  // Default limit: 20 for --files/--json, 5 otherwise
  const defaultLimit = (format === "files" || format === "json") ? 20 : 5;

  const opts: OutputOptions = {
    format,
    full: values.full || false,
    limit: values.n ? parseInt(values.n, 10) || defaultLimit : defaultLimit,
    minScore: values["min-score"] ? parseFloat(values["min-score"]) || 0 : 0,
  };

  return {
    command: positionals[0] || "",
    args: positionals.slice(1),
    query: positionals.slice(1).join(" "),
    opts,
    values,
  };
}

function showHelp(): void {
  console.log("Usage:");
  console.log("  qmd add [--drop] [glob]       - Add/update collection from $PWD (default: **/*.md)");
  console.log("  qmd add-context <path> <text> - Add context description for files under path");
  console.log("  qmd get <file>                - Get document body by filepath");
  console.log("  qmd status                    - Show index status and collections");
  console.log("  qmd update-all                - Re-index all collections");
  console.log("  qmd embed [-f]                - Create vector embeddings (chunks ~6KB each)");
  console.log("  qmd cleanup                   - Remove cache and orphaned data, vacuum DB");
  console.log("  qmd search <query>            - Full-text search (BM25)");
  console.log("  qmd vsearch <query>           - Vector similarity search");
  console.log("  qmd query <query>             - Combined search with query expansion + reranking");
  console.log("");
  console.log("Global options:");
  console.log("  --index <name>             - Use custom index name (default: index)");
  console.log("");
  console.log("Search options:");
  console.log("  -n <num>                   - Number of results (default: 5, or 20 for --files)");
  console.log("  --min-score <num>          - Minimum similarity score");
  console.log("  --full                     - Output full document instead of snippet");
  console.log("  --files                    - Output score,filepath,context (default: 20 results)");
  console.log("  --json                     - JSON output with snippets (default: 20 results)");
  console.log("  --csv                      - CSV output with snippets");
  console.log("  --md                       - Markdown output");
  console.log("  --xml                      - XML output");
  console.log("");
  console.log("Environment:");
  console.log("  OLLAMA_URL                 - Ollama server URL (default: http://localhost:11434)");
  console.log("");
  console.log("Models:");
  console.log(`  Embedding: ${DEFAULT_EMBED_MODEL}`);
  console.log(`  Reranking: ${DEFAULT_RERANK_MODEL}`);
  console.log("");
  console.log(`Index: ${getDbPath()}`);
}

// Main CLI
const cli = parseCLI();

if (!cli.command || cli.values.help) {
  showHelp();
  process.exit(cli.values.help ? 0 : 1);
}

switch (cli.command) {
  case "add": {
    const globArg = cli.args[0];
    // Treat "." as "use default glob in current directory"
    const globPattern = (!globArg || globArg === ".") ? DEFAULT_GLOB : globArg;
    if (cli.values.drop) {
      await dropCollection(globPattern);
    } else {
      await indexFiles(globPattern);
    }
    break;
  }

  case "add-context": {
    // qmd add-context <path> <context> OR qmd add-context <context> (uses .)
    if (cli.args.length === 0) {
      console.error("Usage: qmd add-context <path> <context>");
      console.error("       qmd add-context . \"Description of files in current directory\"");
      process.exit(1);
    }
    let pathArg: string;
    let contextText: string;
    if (cli.args.length === 1) {
      // Single arg = context for current directory
      pathArg = ".";
      contextText = cli.args[0];
    } else {
      pathArg = cli.args[0];
      contextText = cli.args.slice(1).join(" ");
    }
    await addContext(pathArg, contextText);
    break;
  }

  case "get": {
    if (!cli.args[0]) {
      console.error("Usage: qmd get <filepath>");
      process.exit(1);
    }
    getDocument(cli.args[0]);
    break;
  }

  case "status":
    showStatus();
    break;

  case "update-all":
    await updateAllCollections();
    break;

  case "embed":
    await vectorIndex(DEFAULT_EMBED_MODEL, cli.values.force || false);
    break;

  case "search":
    if (!cli.query) {
      console.error("Usage: qmd search [options] <query>");
      process.exit(1);
    }
    search(cli.query, cli.opts);
    break;

  case "vsearch":
    if (!cli.query) {
      console.error("Usage: qmd vsearch [options] <query>");
      process.exit(1);
    }
    // Default min-score for vector search is 0.3
    if (!cli.values["min-score"]) {
      cli.opts.minScore = 0.3;
    }
    await vectorSearch(cli.query, cli.opts);
    break;

  case "query":
    if (!cli.query) {
      console.error("Usage: qmd query [options] <query>");
      process.exit(1);
    }
    await querySearch(cli.query, cli.opts);
    break;

  case "cleanup": {
    const db = getDb();

    // 1. Clear ollama_cache
    const cacheCount = db.prepare(`SELECT COUNT(*) as c FROM ollama_cache`).get() as { c: number };
    db.exec(`DELETE FROM ollama_cache`);
    console.log(`${c.green}✓${c.reset} Cleared ${cacheCount.c} cached API responses`);

    // 2. Remove orphaned vectors (no active document with that hash)
    const orphanedVecs = db.prepare(`
      SELECT COUNT(*) as c FROM content_vectors cv
      WHERE NOT EXISTS (
        SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
      )
    `).get() as { c: number };

    if (orphanedVecs.c > 0) {
      db.exec(`
        DELETE FROM vectors_vec WHERE hash_seq IN (
          SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
          WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
          )
        )
      `);
      db.exec(`
        DELETE FROM content_vectors WHERE hash NOT IN (
          SELECT hash FROM documents WHERE active = 1
        )
      `);
      console.log(`${c.green}✓${c.reset} Removed ${orphanedVecs.c} orphaned embedding chunks`);
    } else {
      console.log(`${c.dim}No orphaned embeddings to remove${c.reset}`);
    }

    // 3. Count inactive documents
    const inactiveDocs = db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 0`).get() as { c: number };
    if (inactiveDocs.c > 0) {
      db.exec(`DELETE FROM documents WHERE active = 0`);
      console.log(`${c.green}✓${c.reset} Removed ${inactiveDocs.c} inactive document records`);
    }

    // 4. Vacuum to reclaim space
    db.exec(`VACUUM`);
    console.log(`${c.green}✓${c.reset} Database vacuumed`);

    db.close();
    break;
  }

  default:
    console.error(`Unknown command: ${cli.command}`);
    console.error("Run 'qmd --help' for usage.");
    process.exit(1);
}
