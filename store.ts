/**
 * QMD Store - Core data access and retrieval functions
 *
 * This module provides all database operations, search functions, and document
 * retrieval for QMD. It returns raw data structures that can be formatted by
 * CLI or MCP consumers.
 *
 * Usage:
 *   const store = createStore("/path/to/db.sqlite");
 *   // or use default path:
 *   const store = createStore();
 */

import { Database } from "bun:sqlite";
import { Glob } from "bun";
import * as sqliteVec from "sqlite-vec";
import {
  Ollama,
  getDefaultOllama,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  type RerankDocument,
} from "./llm";

// =============================================================================
// Configuration
// =============================================================================

const HOME = Bun.env.HOME || "/tmp";
export const DEFAULT_EMBED_MODEL = "embeddinggemma";
export const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
export const DEFAULT_QUERY_MODEL = "qwen3:0.6b";
export const DEFAULT_GLOB = "**/*.md";
export const DEFAULT_MULTI_GET_MAX_BYTES = 10 * 1024; // 10KB

// Re-export OLLAMA_URL for backwards compatibility
export const OLLAMA_URL = getDefaultOllama().getBaseUrl();

// Chunking: ~2000 tokens per chunk, ~3 bytes/token = 6KB
const CHUNK_BYTE_SIZE = 6 * 1024;

// =============================================================================
// Path utilities
// =============================================================================

export function homedir(): string {
  return HOME;
}

export function resolve(...paths: string[]): string {
  let result = paths[0].startsWith('/') ? '' : Bun.env.PWD || process.cwd();
  for (const p of paths) {
    if (p.startsWith('/')) {
      result = p;
    } else {
      result = result + '/' + p;
    }
  }
  const parts = result.split('/').filter(Boolean);
  const normalized: string[] = [];
  for (const part of parts) {
    if (part === '..') normalized.pop();
    else if (part !== '.') normalized.push(part);
  }
  return '/' + normalized.join('/');
}

export function getDefaultDbPath(indexName: string = "index"): string {
  // Allow override via INDEX_PATH for testing
  if (Bun.env.INDEX_PATH) {
    return Bun.env.INDEX_PATH;
  }
  const cacheDir = Bun.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
  const qmdCacheDir = resolve(cacheDir, "qmd");
  try { Bun.spawnSync(["mkdir", "-p", qmdCacheDir]); } catch {}
  return resolve(qmdCacheDir, `${indexName}.sqlite`);
}

export function getPwd(): string {
  return process.env.PWD || process.cwd();
}

export function getRealPath(path: string): string {
  try {
    const result = Bun.spawnSync(["realpath", path]);
    if (result.success) {
      return result.stdout.toString().trim();
    }
  } catch {}
  return resolve(path);
}

// =============================================================================
// Virtual Path Utilities (qmd://)
// =============================================================================

export type VirtualPath = {
  collectionName: string;
  path: string;  // relative path within collection
};

/**
 * Parse a virtual path like "qmd://collection-name/path/to/file.md"
 * into its components.
 */
export function parseVirtualPath(virtualPath: string): VirtualPath | null {
  const match = virtualPath.match(/^qmd:\/\/([^\/]+)\/(.+)$/);
  if (!match) return null;
  return {
    collectionName: match[1],
    path: match[2],
  };
}

/**
 * Build a virtual path from collection name and relative path.
 */
export function buildVirtualPath(collectionName: string, path: string): string {
  return `qmd://${collectionName}/${path}`;
}

/**
 * Check if a path is a virtual path (starts with qmd://).
 */
export function isVirtualPath(path: string): boolean {
  return path.startsWith('qmd://');
}

/**
 * Resolve a virtual path to absolute filesystem path.
 */
export function resolveVirtualPath(db: Database, virtualPath: string): string | null {
  const parsed = parseVirtualPath(virtualPath);
  if (!parsed) return null;

  const coll = getCollectionByName(db, parsed.collectionName);
  if (!coll) return null;

  return resolve(coll.pwd, parsed.path);
}

/**
 * Convert an absolute filesystem path to a virtual path.
 * Returns null if the file is not in any indexed collection.
 */
export function toVirtualPath(db: Database, absolutePath: string): string | null {
  const doc = db.prepare(`
    SELECT c.name, d.path
    FROM documents d
    JOIN collections c ON c.id = d.collection_id
    WHERE c.pwd || '/' || d.path = ? AND d.active = 1
    LIMIT 1
  `).get(absolutePath) as { name: string; path: string } | null;

  if (!doc) return null;
  return buildVirtualPath(doc.name, doc.path);
}

// =============================================================================
// Database initialization
// =============================================================================

// On macOS, use Homebrew's SQLite which supports extensions
if (process.platform === "darwin") {
  const homebrewSqlitePath = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
  try {
    if (Bun.file(homebrewSqlitePath).size > 0) {
      Database.setCustomSQLite(homebrewSqlitePath);
    }
  } catch {}
}

function initializeDatabase(db: Database): void {
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");

  // Check if we need to migrate from old schema
  const tables = db.prepare(`SELECT name FROM sqlite_master WHERE type='table'`).all() as { name: string }[];
  const tableNames = tables.map(t => t.name);
  const needsMigration = tableNames.includes('documents') && !tableNames.includes('content');

  if (needsMigration) {
    migrateToContentAddressable(db);
    return; // Migration will call initializeDatabase again
  }

  // Content-addressable storage - the source of truth for document content
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Collections table with name field
  db.exec(`
    CREATE TABLE IF NOT EXISTS collections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      pwd TEXT NOT NULL,
      glob_pattern TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(pwd, glob_pattern)
    )
  `);

  // Documents table - file system layer mapping virtual paths to content hashes
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection_id, path)
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id, active)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active)`);

  // Path-based context (collection-scoped, hierarchical)
  db.exec(`
    CREATE TABLE IF NOT EXISTS path_contexts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      path_prefix TEXT NOT NULL,
      context TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
      UNIQUE(collection_id, path_prefix)
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_path_contexts_collection ON path_contexts(collection_id, path_prefix)`);

  // Cache table for Ollama API calls
  db.exec(`
    CREATE TABLE IF NOT EXISTS ollama_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Content vectors
  const cvInfo = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
  const hasSeqColumn = cvInfo.some(col => col.name === 'seq');
  if (cvInfo.length > 0 && !hasSeqColumn) {
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

  // FTS - index path and content (joined from content table)
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      path, body,
      tokenize='porter unicode61'
    )
  `);

  // Triggers to keep FTS in sync
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
      INSERT INTO documents_fts(rowid, path, body)
      SELECT new.id, new.path, c.doc
      FROM content c
      WHERE c.hash = new.hash;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
      DELETE FROM documents_fts WHERE rowid = old.id;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
      UPDATE documents_fts
      SET path = new.path,
          body = (SELECT doc FROM content WHERE hash = new.hash)
      WHERE rowid = new.id;
    END
  `);
}

function migrateToContentAddressable(db: Database): void {
  console.log("Migrating database to content-addressable schema...");

  // Start transaction
  db.exec("BEGIN TRANSACTION");

  try {
    // Rename old tables
    db.exec("ALTER TABLE documents RENAME TO documents_old");
    db.exec("ALTER TABLE collections RENAME TO collections_old");
    db.exec("ALTER TABLE path_contexts RENAME TO path_contexts_old");
    db.exec("DROP TABLE IF EXISTS documents_fts");
    db.exec("DROP TRIGGER IF EXISTS documents_ai");
    db.exec("DROP TRIGGER IF EXISTS documents_ad");
    db.exec("DROP TRIGGER IF EXISTS documents_au");

    // Create new schema
    db.exec(`
      CREATE TABLE content (
        hash TEXT PRIMARY KEY,
        doc TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    `);

    db.exec(`
      CREATE TABLE collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        pwd TEXT NOT NULL,
        glob_pattern TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(pwd, glob_pattern)
      )
    `);

    db.exec(`
      CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL,
        path TEXT NOT NULL,
        title TEXT NOT NULL,
        hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        modified_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
        FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
        UNIQUE(collection_id, path)
      )
    `);

    db.exec(`
      CREATE TABLE path_contexts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL,
        path_prefix TEXT NOT NULL,
        context TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
        UNIQUE(collection_id, path_prefix)
      )
    `);

    // Migrate data: Extract unique content hashes
    console.log("Migrating content...");
    db.exec(`
      INSERT INTO content (hash, doc, created_at)
      SELECT hash, body, MIN(created_at) as created_at
      FROM documents_old
      WHERE active = 1
      GROUP BY hash
    `);

    // Migrate collections: generate names from pwd basename
    console.log("Migrating collections...");
    db.exec(`
      INSERT INTO collections (id, name, pwd, glob_pattern, created_at, updated_at)
      SELECT
        id,
        CASE
          WHEN INSTR(RTRIM(pwd, '/'), '/') > 0
          THEN SUBSTR(RTRIM(pwd, '/'), INSTR(RTRIM(pwd, '/'), '/') + 1)
          ELSE RTRIM(pwd, '/')
        END as name,
        pwd,
        glob_pattern,
        created_at,
        created_at as updated_at
      FROM collections_old
    `);

    // Handle duplicate collection names by appending collection_id
    const duplicates = db.prepare(`
      SELECT name, COUNT(*) as cnt
      FROM collections
      GROUP BY name
      HAVING cnt > 1
    `).all() as { name: string; cnt: number }[];

    for (const dup of duplicates) {
      const rows = db.prepare(`SELECT id FROM collections WHERE name = ? ORDER BY id`).all(dup.name) as { id: number }[];
      for (let i = 1; i < rows.length; i++) {
        db.prepare(`UPDATE collections SET name = ? WHERE id = ?`).run(`${dup.name}-${rows[i].id}`, rows[i].id);
      }
    }

    // Migrate documents: convert filepath to relative path within collection
    console.log("Migrating documents...");
    const oldDocs = db.prepare(`
      SELECT d.id, d.collection_id, d.filepath, d.title, d.hash, d.created_at, d.modified_at, c.pwd
      FROM documents_old d
      JOIN collections c ON c.id = d.collection_id
      WHERE d.active = 1
    `).all() as Array<{
      id: number;
      collection_id: number;
      filepath: string;
      title: string;
      hash: string;
      created_at: string;
      modified_at: string;
      pwd: string;
    }>;

    const insertDoc = db.prepare(`
      INSERT INTO documents (collection_id, path, title, hash, created_at, modified_at, active)
      VALUES (?, ?, ?, ?, ?, ?, 1)
    `);

    for (const doc of oldDocs) {
      // Convert absolute filepath to relative path within collection
      let path = doc.filepath;
      if (path.startsWith(doc.pwd + '/')) {
        path = path.slice(doc.pwd.length + 1);
      } else if (path.startsWith(doc.pwd)) {
        path = path.slice(doc.pwd.length);
      }
      // Remove leading slash if present
      path = path.replace(/^\/+/, '');

      try {
        insertDoc.run(doc.collection_id, path, doc.title, doc.hash, doc.created_at, doc.modified_at);
      } catch (e) {
        console.warn(`Skipping duplicate path: ${path} in collection ${doc.collection_id}`);
      }
    }

    // Migrate path_contexts: associate with collections based on path prefix
    console.log("Migrating path contexts...");
    const oldContexts = db.prepare(`SELECT * FROM path_contexts_old`).all() as Array<{
      path_prefix: string;
      context: string;
      created_at: string;
    }>;

    const insertContext = db.prepare(`
      INSERT INTO path_contexts (collection_id, path_prefix, context, created_at)
      VALUES (?, ?, ?, ?)
    `);

    const allCollections = db.prepare(`SELECT id, pwd FROM collections`).all() as Array<{ id: number; pwd: string }>;

    for (const ctx of oldContexts) {
      // Find collection(s) that match this path prefix
      for (const coll of allCollections) {
        if (ctx.path_prefix.startsWith(coll.pwd)) {
          // Convert absolute path_prefix to relative within collection
          let relPath = ctx.path_prefix;
          if (relPath.startsWith(coll.pwd + '/')) {
            relPath = relPath.slice(coll.pwd.length + 1);
          } else if (relPath.startsWith(coll.pwd)) {
            relPath = relPath.slice(coll.pwd.length);
          }
          relPath = relPath.replace(/^\/+/, '');

          try {
            insertContext.run(coll.id, relPath, ctx.context, ctx.created_at);
          } catch (e) {
            // Ignore duplicates
          }
        }
      }
    }

    // Drop old tables
    db.exec("DROP TABLE documents_old");
    db.exec("DROP TABLE collections_old");
    db.exec("DROP TABLE path_contexts_old");

    // Recreate FTS and triggers
    db.exec(`
      CREATE VIRTUAL TABLE documents_fts USING fts5(
        path, body,
        tokenize='porter unicode61'
      )
    `);

    db.exec(`
      CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, path, body)
        SELECT new.id, new.path, c.doc
        FROM content c
        WHERE c.hash = new.hash;
      END
    `);

    db.exec(`
      CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
        DELETE FROM documents_fts WHERE rowid = old.id;
      END
    `);

    db.exec(`
      CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
        UPDATE documents_fts
        SET path = new.path,
            body = (SELECT doc FROM content WHERE hash = new.hash)
        WHERE rowid = new.id;
      END
    `);

    // Populate FTS from migrated data
    console.log("Rebuilding full-text search index...");
    db.exec(`
      INSERT INTO documents_fts(rowid, path, body)
      SELECT d.id, d.path, c.doc
      FROM documents d
      JOIN content c ON c.hash = d.hash
      WHERE d.active = 1
    `);

    // Create indexes
    db.exec(`CREATE INDEX idx_documents_collection ON documents(collection_id, active)`);
    db.exec(`CREATE INDEX idx_documents_hash ON documents(hash)`);
    db.exec(`CREATE INDEX idx_documents_path ON documents(path, active)`);
    db.exec(`CREATE INDEX idx_path_contexts_collection ON path_contexts(collection_id, path_prefix)`);

    db.exec("COMMIT");
    console.log("Migration complete!");

  } catch (e) {
    db.exec("ROLLBACK");
    console.error("Migration failed:", e);
    throw e;
  }
}

function ensureVecTableInternal(db: Database, dimensions: number): void {
  const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get() as { sql: string } | null;
  if (tableInfo) {
    const match = tableInfo.sql.match(/float\[(\d+)\]/);
    const hasHashSeq = tableInfo.sql.includes('hash_seq');
    if (match && parseInt(match[1]) === dimensions && hasHashSeq) return;
    db.exec("DROP TABLE IF EXISTS vectors_vec");
  }
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}])`);
}

// =============================================================================
// Store Factory
// =============================================================================

export type Store = {
  db: Database;
  dbPath: string;
  close: () => void;
  ensureVecTable: (dimensions: number) => void;

  // Index health
  getHashesNeedingEmbedding: () => number;
  getIndexHealth: () => IndexHealthInfo;
  getStatus: () => IndexStatus;

  // Caching
  getCacheKey: typeof getCacheKey;
  getCachedResult: (cacheKey: string) => string | null;
  setCachedResult: (cacheKey: string, result: string) => void;
  clearCache: () => void;

  // Context
  getContextForFile: (filepath: string) => string | null;
  getContextForPath: (collectionId: number, path: string) => string | null;
  getCollectionIdByName: (name: string) => number | null;
  getCollectionByName: (name: string) => { id: number; name: string; pwd: string; glob_pattern: string } | null;

  // Virtual paths
  parseVirtualPath: typeof parseVirtualPath;
  buildVirtualPath: typeof buildVirtualPath;
  isVirtualPath: typeof isVirtualPath;
  resolveVirtualPath: (virtualPath: string) => string | null;
  toVirtualPath: (absolutePath: string) => string | null;

  // Search
  searchFTS: (query: string, limit?: number, collectionId?: number) => SearchResult[];
  searchVec: (query: string, model: string, limit?: number, collectionId?: number) => Promise<SearchResult[]>;

  // Query expansion & reranking
  expandQuery: (query: string, model?: string) => Promise<string[]>;
  rerank: (query: string, documents: { file: string; text: string }[], model?: string) => Promise<{ file: string; score: number }[]>;

  // Document retrieval
  findDocument: (filename: string, options?: { includeBody?: boolean }) => DocumentResult | DocumentNotFound;
  getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => string | null;
  findDocuments: (pattern: string, options?: { includeBody?: boolean; maxBytes?: number }) => { docs: MultiGetResult[]; errors: string[] };

  // Legacy compatibility
  getDocument: (filename: string, fromLine?: number, maxLines?: number) => (DocumentResult & { body: string }) | DocumentNotFound;
  getMultipleDocuments: (pattern: string, maxLines?: number, maxBytes?: number) => { files: MultiGetFile[]; errors: string[] };

  // Fuzzy matching
  findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => string[];
  matchFilesByGlob: (pattern: string) => { filepath: string; displayPath: string; bodyLength: number }[];
};

/**
 * Create a new store instance with the given database path.
 * If no path is provided, uses the default path (~/.cache/qmd/index.sqlite).
 *
 * @param dbPath - Path to the SQLite database file
 * @returns Store instance with all methods bound to the database
 */
export function createStore(dbPath?: string): Store {
  const resolvedPath = dbPath || getDefaultDbPath();
  const db = new Database(resolvedPath);
  initializeDatabase(db);

  return {
    db,
    dbPath: resolvedPath,
    close: () => db.close(),
    ensureVecTable: (dimensions: number) => ensureVecTableInternal(db, dimensions),

    // Index health
    getHashesNeedingEmbedding: () => getHashesNeedingEmbedding(db),
    getIndexHealth: () => getIndexHealth(db),
    getStatus: () => getStatus(db),

    // Caching
    getCacheKey,
    getCachedResult: (cacheKey: string) => getCachedResult(db, cacheKey),
    setCachedResult: (cacheKey: string, result: string) => setCachedResult(db, cacheKey, result),
    clearCache: () => clearCache(db),

    // Context
    getContextForFile: (filepath: string) => getContextForFile(db, filepath),
    getContextForPath: (collectionId: number, path: string) => getContextForPath(db, collectionId, path),
    getCollectionIdByName: (name: string) => getCollectionIdByName(db, name),
    getCollectionByName: (name: string) => getCollectionByName(db, name),

    // Virtual paths
    parseVirtualPath,
    buildVirtualPath,
    isVirtualPath,
    resolveVirtualPath: (virtualPath: string) => resolveVirtualPath(db, virtualPath),
    toVirtualPath: (absolutePath: string) => toVirtualPath(db, absolutePath),

    // Search
    searchFTS: (query: string, limit?: number, collectionId?: number) => searchFTS(db, query, limit, collectionId),
    searchVec: (query: string, model: string, limit?: number, collectionId?: number) => searchVec(db, query, model, limit, collectionId),

    // Query expansion & reranking
    expandQuery: (query: string, model?: string) => expandQuery(query, model, db),
    rerank: (query: string, documents: { file: string; text: string }[], model?: string) => rerank(query, documents, model, db),

    // Document retrieval
    findDocument: (filename: string, options?: { includeBody?: boolean }) => findDocument(db, filename, options),
    getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => getDocumentBody(db, doc, fromLine, maxLines),
    findDocuments: (pattern: string, options?: { includeBody?: boolean; maxBytes?: number }) => findDocuments(db, pattern, options),

    // Legacy compatibility
    getDocument: (filename: string, fromLine?: number, maxLines?: number) => getDocument(db, filename, fromLine, maxLines),
    getMultipleDocuments: (pattern: string, maxLines?: number, maxBytes?: number) => getMultipleDocuments(db, pattern, maxLines, maxBytes),

    // Fuzzy matching
    findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => findSimilarFiles(db, query, maxDistance, limit),
    matchFilesByGlob: (pattern: string) => matchFilesByGlob(db, pattern),
  };
}

// =============================================================================
// Legacy compatibility - will be removed
// =============================================================================

let _legacyDb: Database | null = null;
let _legacyDbPath: string | null = null;

/** @deprecated Use createStore() instead */
export function setCustomIndexName(name: string | null): void {
  _legacyDbPath = name ? getDefaultDbPath(name) : null;
  _legacyDb = null; // Reset so next getDb() creates new connection
}

/** @deprecated Use createStore() instead */
export function getDbPath(): string {
  return _legacyDbPath || getDefaultDbPath();
}

/** @deprecated Use createStore() instead */
export function getDb(): Database {
  if (!_legacyDb) {
    _legacyDb = new Database(getDbPath());
    initializeDatabase(_legacyDb);
  }
  return _legacyDb;
}

/** @deprecated Use store.db.close() instead. Closes the legacy db and resets singleton. */
export function closeDb(): void {
  if (_legacyDb) {
    _legacyDb.close();
    _legacyDb = null;
  }
}

/** @deprecated Use store.ensureVecTable() instead */
export function ensureVecTable(db: Database, dimensions: number): void {
  ensureVecTableInternal(db, dimensions);
}

// =============================================================================
// Core Document Type
// =============================================================================

/**
 * Unified document result type with all metadata.
 * Body is optional - use getDocumentBody() to load it separately if needed.
 */
export type DocumentResult = {
  filepath: string;           // Full filesystem path
  displayPath: string;        // Short display path (e.g., "docs/readme.md")
  title: string;              // Document title (from first heading or filename)
  context: string | null;     // Folder context description if configured
  hash: string;               // Content hash for caching/change detection
  collectionId: number;       // Parent collection ID
  modifiedAt: string;         // Last modification timestamp
  bodyLength: number;         // Body length in bytes (useful before loading)
  body?: string;              // Document body (optional, load with getDocumentBody)
};

/**
 * Search result extends DocumentResult with score and source info
 */
export type SearchResult = DocumentResult & {
  score: number;              // Relevance score (0-1)
  source: "fts" | "vec";      // Search source (full-text or vector)
  chunkPos?: number;          // Character position of matching chunk (for vector search)
};

/**
 * Ranked result for RRF fusion (simplified, used internally)
 */
export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

/**
 * Error result when document is not found
 */
export type DocumentNotFound = {
  error: "not_found";
  query: string;
  similarFiles: string[];
};

/**
 * Result from multi-get operations
 */
export type MultiGetResult = {
  doc: DocumentResult;
  skipped: false;
} | {
  doc: Pick<DocumentResult, "filepath" | "displayPath">;
  skipped: true;
  skipReason: string;
};

export type CollectionInfo = {
  id: number;
  path: string;
  pattern: string;
  documents: number;
  lastUpdated: string;
};

export type IndexStatus = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: CollectionInfo[];
};

// =============================================================================
// Index health
// =============================================================================

export function getHashesNeedingEmbedding(db: Database): number {
  const result = db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

export type IndexHealthInfo = {
  needsEmbedding: number;
  totalDocs: number;
  daysStale: number | null;
};

export function getIndexHealth(db: Database): IndexHealthInfo {
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const totalDocs = (db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number }).count;

  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null };
  let daysStale: number | null = null;
  if (mostRecent?.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    daysStale = Math.floor((Date.now() - lastUpdate.getTime()) / (24 * 60 * 60 * 1000));
  }

  return { needsEmbedding, totalDocs, daysStale };
}

// =============================================================================
// Caching
// =============================================================================

export function getCacheKey(url: string, body: object): string {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(url);
  hash.update(JSON.stringify(body));
  return hash.digest("hex");
}

export function getCachedResult(db: Database, cacheKey: string): string | null {
  const row = db.prepare(`SELECT result FROM ollama_cache WHERE hash = ?`).get(cacheKey) as { result: string } | null;
  return row?.result || null;
}

export function setCachedResult(db: Database, cacheKey: string, result: string): void {
  const now = new Date().toISOString();
  db.prepare(`INSERT OR REPLACE INTO ollama_cache (hash, result, created_at) VALUES (?, ?, ?)`).run(cacheKey, result, now);
  if (Math.random() < 0.01) {
    db.exec(`DELETE FROM ollama_cache WHERE hash NOT IN (SELECT hash FROM ollama_cache ORDER BY created_at DESC LIMIT 1000)`);
  }
}

export function clearCache(db: Database): void {
  db.exec(`DELETE FROM ollama_cache`);
}

// =============================================================================
// Document helpers
// =============================================================================

export async function hashContent(content: string): Promise<string> {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(content);
  return hash.digest("hex");
}

export function extractTitle(content: string, filename: string): string {
  const match = content.match(/^##?\s+(.+)$/m);
  if (match) {
    const title = match[1].trim();
    if (title === "📝 Notes" || title === "Notes") {
      const nextMatch = content.match(/^##\s+(.+)$/m);
      if (nextMatch) return nextMatch[1].trim();
    }
    return title;
  }
  return filename.replace(/\.md$/, "").split("/").pop() || filename;
}

// Re-export from llm.ts for backwards compatibility
export { formatQueryForEmbedding, formatDocForEmbedding };

export function chunkDocument(content: string, maxBytes: number = CHUNK_BYTE_SIZE): { text: string; pos: number }[] {
  const encoder = new TextEncoder();
  const totalBytes = encoder.encode(content).length;

  if (totalBytes <= maxBytes) {
    return [{ text: content, pos: 0 }];
  }

  const chunks: { text: string; pos: number }[] = [];
  let charPos = 0;

  while (charPos < content.length) {
    let endPos = charPos;
    let byteCount = 0;

    while (endPos < content.length && byteCount < maxBytes) {
      const charBytes = encoder.encode(content[endPos]).length;
      if (byteCount + charBytes > maxBytes) break;
      byteCount += charBytes;
      endPos++;
    }

    if (endPos < content.length && endPos > charPos) {
      const slice = content.slice(charPos, endPos);
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
        breakPoint = paragraphBreak + 2;
      } else if (sentenceEnd > slice.length * 0.5) {
        breakPoint = sentenceEnd + 2;
      } else if (lineBreak > slice.length * 0.3) {
        breakPoint = lineBreak + 1;
      } else if (spaceBreak > slice.length * 0.3) {
        breakPoint = spaceBreak + 1;
      }

      if (breakPoint > 0) {
        endPos = charPos + breakPoint;
      }
    }

    if (endPos <= charPos) {
      endPos = charPos + 1;
    }

    chunks.push({ text: content.slice(charPos, endPos), pos: charPos });
    charPos = endPos;
  }

  return chunks;
}

// =============================================================================
// Fuzzy matching
// =============================================================================

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp: number[][] = Array.from({ length: m + 1 }, (_, i) => [i]);
  for (let j = 1; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

export function findSimilarFiles(db: Database, query: string, maxDistance: number = 3, limit: number = 5): string[] {
  const allFiles = db.prepare(`SELECT display_path FROM documents WHERE active = 1`).all() as { display_path: string }[];
  const queryLower = query.toLowerCase();
  const scored = allFiles
    .map(f => ({ path: f.display_path, dist: levenshtein(f.display_path.toLowerCase(), queryLower) }))
    .filter(f => f.dist <= maxDistance)
    .sort((a, b) => a.dist - b.dist)
    .slice(0, limit);
  return scored.map(f => f.path);
}

export function matchFilesByGlob(db: Database, pattern: string): { filepath: string; displayPath: string; bodyLength: number }[] {
  const allFiles = db.prepare(`
    SELECT
      'qmd://' || c.name || '/' || d.path as virtual_path,
      LENGTH(content.doc) as body_length,
      d.collection_id,
      d.path
    FROM documents d
    JOIN collections c ON c.id = d.collection_id
    JOIN content ON content.hash = d.hash
    WHERE d.active = 1
  `).all() as { virtual_path: string; body_length: number; collection_id: number; path: string }[];

  const glob = new Glob(pattern);
  return allFiles
    .filter(f => glob.match(f.virtual_path) || glob.match(f.path))
    .map(f => ({
      filepath: f.virtual_path,  // Use virtual path as filepath
      displayPath: f.virtual_path,
      bodyLength: f.body_length
    }));
}

// =============================================================================
// Context
// =============================================================================

/**
 * Get context for a file path using hierarchical inheritance.
 * Contexts are collection-scoped and inherit from parent directories.
 * For example, context at "/talks" applies to "/talks/2024/keynote.md".
 *
 * @param db Database instance
 * @param collectionId Collection ID
 * @param path Relative path within the collection
 * @returns Context string or null if no context is defined
 */
export function getContextForPath(db: Database, collectionId: number, path: string): string | null {
  // Find the most specific (longest) matching path prefix for this collection
  const result = db.prepare(`
    SELECT context FROM path_contexts
    WHERE collection_id = ?
      AND (? LIKE path_prefix || '/%' OR ? = path_prefix OR path_prefix = '')
    ORDER BY LENGTH(path_prefix) DESC
    LIMIT 1
  `).get(collectionId, path, path) as { context: string } | null;
  return result?.context || null;
}

/**
 * Legacy function for backward compatibility - resolves filepath to collection+path first
 */
export function getContextForFile(db: Database, filepath: string): string | null {
  // Try to find the document to get its collection_id and path
  const doc = db.prepare(`
    SELECT d.collection_id, d.path
    FROM documents d
    JOIN collections c ON c.id = d.collection_id
    WHERE c.pwd || '/' || d.path = ? AND d.active = 1
    LIMIT 1
  `).get(filepath) as { collection_id: number; path: string } | null;

  if (!doc) return null;
  return getContextForPath(db, doc.collection_id, doc.path);
}

/**
 * Get collection ID by its name (exact match).
 */
export function getCollectionIdByName(db: Database, name: string): number | null {
  const result = db.prepare(`
    SELECT id FROM collections
    WHERE name = ?
    LIMIT 1
  `).get(name) as { id: number } | null;
  return result?.id || null;
}

/**
 * Get collection by name.
 */
export function getCollectionByName(db: Database, name: string): { id: number; name: string; pwd: string; glob_pattern: string } | null {
  const result = db.prepare(`
    SELECT id, name, pwd, glob_pattern FROM collections
    WHERE name = ?
    LIMIT 1
  `).get(name) as { id: number; name: string; pwd: string; glob_pattern: string } | null;
  return result;
}

// =============================================================================
// FTS Search
// =============================================================================

function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}']/gu, '').toLowerCase();
}

function buildFTS5Query(query: string): string | null {
  const terms = query.split(/\s+/)
    .map(t => sanitizeFTS5Term(t))
    .filter(t => t.length > 0);
  if (terms.length === 0) return null;
  if (terms.length === 1) return `"${terms[0]}"*`;
  return terms.map(t => `"${t}"*`).join(' AND ');
}

export function searchFTS(db: Database, query: string, limit: number = 20, collectionId?: number): SearchResult[] {
  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  let sql = `
    SELECT
      'qmd://' || c.name || '/' || d.path as filepath,
      'qmd://' || c.name || '/' || d.path as display_path,
      d.title,
      content.doc as body,
      bm25(documents_fts, 10.0, 1.0) as score
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    JOIN collections c ON c.id = d.collection_id
    JOIN content ON content.hash = d.hash
    WHERE documents_fts MATCH ? AND d.active = 1
  `;
  const params: (string | number)[] = [ftsQuery];

  if (collectionId !== undefined) {
    sql += ` AND d.collection_id = ?`;
    params.push(collectionId);
  }

  sql += ` ORDER BY score LIMIT ?`;
  params.push(limit);

  const rows = db.prepare(sql).all(...params) as { filepath: string; display_path: string; title: string; body: string; score: number }[];

  const maxScore = rows.length > 0 ? Math.max(...rows.map(r => Math.abs(r.score))) : 1;
  return rows.map(row => ({
    file: row.filepath,
    displayPath: row.display_path,
    title: row.title,
    body: row.body,
    score: Math.abs(row.score) / maxScore,
    source: "fts" as const,
  }));
}

// =============================================================================
// Vector Search
// =============================================================================

export async function searchVec(db: Database, query: string, model: string, limit: number = 20, collectionId?: number): Promise<SearchResult[]> {
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) return [];

  const embedding = await getEmbedding(query, model, true);
  if (!embedding) return [];

  // sqlite-vec requires "k = ?" for KNN queries
  let sql = `
    SELECT
      v.hash_seq,
      v.distance,
      'qmd://' || c.name || '/' || d.path as filepath,
      'qmd://' || c.name || '/' || d.path as display_path,
      d.title,
      content.doc as body,
      cv.pos
    FROM vectors_vec v
    JOIN content_vectors cv ON cv.hash || '_' || cv.seq = v.hash_seq
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    JOIN collections c ON c.id = d.collection_id
    JOIN content ON content.hash = d.hash
    WHERE v.embedding MATCH ? AND k = ?
  `;

  if (collectionId !== undefined) {
    sql += ` AND d.collection_id = ${collectionId}`;
  }

  sql += ` ORDER BY v.distance`;

  const rows = db.prepare(sql).all(new Float32Array(embedding), limit * 3) as { hash_seq: string; distance: number; filepath: string; display_path: string; title: string; body: string; pos: number }[];

  const seen = new Map<string, { row: typeof rows[0]; bestDist: number }>();
  for (const row of rows) {
    const existing = seen.get(row.filepath);
    if (!existing || row.distance < existing.bestDist) {
      seen.set(row.filepath, { row, bestDist: row.distance });
    }
  }

  return Array.from(seen.values())
    .sort((a, b) => a.bestDist - b.bestDist)
    .slice(0, limit)
    .map(({ row }) => ({
      file: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      body: row.body,
      score: 1 / (1 + row.distance),
      source: "vec" as const,
      chunkPos: row.pos,
    }));
}

// =============================================================================
// Embeddings
// =============================================================================

async function getEmbedding(text: string, model: string, isQuery: boolean): Promise<number[] | null> {
  const ollama = getDefaultOllama();
  const result = await ollama.embed(text, { model, isQuery });
  return result?.embedding || null;
}

// =============================================================================
// Query expansion
// =============================================================================

export async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL, db: Database): Promise<string[]> {
  // Check cache first
  const cacheKey = getCacheKey("expandQuery", { query, model });
  const cached = getCachedResult(db, cacheKey);
  if (cached) {
    const lines = cached.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    return [query, ...lines.slice(0, 2)];
  }

  const ollama = getDefaultOllama();
  const results = await ollama.expandQuery(query, model, 2);

  // Cache the expanded queries (excluding original)
  if (results.length > 1) {
    setCachedResult(db, cacheKey, results.slice(1).join('\n'));
  }

  return results;
}

// =============================================================================
// Reranking
// =============================================================================

export async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL, db: Database): Promise<{ file: string; score: number }[]> {
  const cachedResults: Map<string, number> = new Map();
  const uncachedDocs: RerankDocument[] = [];

  // Check cache for each document
  for (const doc of documents) {
    const cacheKey = getCacheKey("rerank", { query, file: doc.file, model });
    const cached = getCachedResult(db, cacheKey);
    if (cached !== null) {
      cachedResults.set(doc.file, parseFloat(cached));
    } else {
      uncachedDocs.push({ file: doc.file, text: doc.text });
    }
  }

  // Rerank uncached documents using Ollama
  if (uncachedDocs.length > 0) {
    const ollama = getDefaultOllama();
    const rerankResult = await ollama.rerank(query, uncachedDocs, { model });

    // Cache results
    for (const result of rerankResult.results) {
      const cacheKey = getCacheKey("rerank", { query, file: result.file, model });
      setCachedResult(db, cacheKey, result.score.toString());
      cachedResults.set(result.file, result.score);
    }
  }

  // Return all results sorted by score
  return documents
    .map(doc => ({ file: doc.file, score: cachedResults.get(doc.file) || 0 }))
    .sort((a, b) => b.score - a.score);
}

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { result: RankedResult; rrfScore: number; topRank: number }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const list = resultLists[listIdx];
    const weight = weights[listIdx] ?? 1.0;

    for (let rank = 0; rank < list.length; rank++) {
      const result = list[rank];
      const rrfContribution = weight / (k + rank + 1);
      const existing = scores.get(result.file);

      if (existing) {
        existing.rrfScore += rrfContribution;
        existing.topRank = Math.min(existing.topRank, rank);
      } else {
        scores.set(result.file, {
          result,
          rrfScore: rrfContribution,
          topRank: rank,
        });
      }
    }
  }

  // Top-rank bonus
  for (const entry of scores.values()) {
    if (entry.topRank === 0) {
      entry.rrfScore += 0.05;
    } else if (entry.topRank <= 2) {
      entry.rrfScore += 0.02;
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(e => ({ ...e.result, score: e.rrfScore }));
}

// =============================================================================
// Document retrieval
// =============================================================================

type DbDocRow = {
  filepath: string;
  display_path: string;
  title: string;
  hash: string;
  collection_id: number;
  modified_at: string;
  body_length: number;
  body?: string;
};

/**
 * Find a document by filename/path (with fuzzy matching)
 * Returns document metadata without body by default
 */
export function findDocument(db: Database, filename: string, options: { includeBody?: boolean } = {}): DocumentResult | DocumentNotFound {
  let filepath = filename;
  const colonMatch = filepath.match(/:(\d+)$/);
  if (colonMatch) {
    filepath = filepath.slice(0, -colonMatch[0].length);
  }

  if (filepath.startsWith('~/')) {
    filepath = homedir() + filepath.slice(1);
  }

  const selectCols = options.includeBody
    ? `filepath, display_path, title, hash, collection_id, modified_at, LENGTH(body) as body_length, body`
    : `filepath, display_path, title, hash, collection_id, modified_at, LENGTH(body) as body_length`;

  // Try various match strategies
  let doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE filepath = ? AND active = 1`).get(filepath) as DbDocRow | null;
  if (!doc) {
    doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE display_path = ? AND active = 1`).get(filepath) as DbDocRow | null;
  }
  if (!doc) {
    doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE filepath LIKE ? AND active = 1 LIMIT 1`).get(`%${filepath}`) as DbDocRow | null;
  }
  if (!doc) {
    doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE display_path LIKE ? AND active = 1 LIMIT 1`).get(`%${filepath}`) as DbDocRow | null;
  }

  if (!doc) {
    const similar = findSimilarFiles(db, filepath, 5, 5);
    return { error: "not_found", query: filename, similarFiles: similar };
  }

  const context = getContextForFile(db, doc.filepath);

  return {
    filepath: doc.filepath,
    displayPath: doc.display_path,
    title: doc.title,
    context,
    hash: doc.hash,
    collectionId: doc.collection_id,
    modifiedAt: doc.modified_at,
    bodyLength: doc.body_length,
    ...(options.includeBody && doc.body !== undefined && { body: doc.body }),
  };
}

/**
 * Get the body content for a document
 * Optionally slice by line range
 */
export function getDocumentBody(db: Database, doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number): string | null {
  const filepath = 'filepath' in doc ? doc.filepath : doc.filepath;
  const row = db.prepare(`SELECT body FROM documents WHERE filepath = ? AND active = 1`).get(filepath) as { body: string } | null;
  if (!row) return null;

  let body = row.body;
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = body.split('\n');
    const start = (fromLine || 1) - 1;
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    body = lines.slice(start, end).join('\n');
  }

  return body;
}

/**
 * Legacy function for backwards compatibility
 * Combines findDocument + getDocumentBody with line slicing
 */
export function getDocument(db: Database, filename: string, fromLine?: number, maxLines?: number): (DocumentResult & { body: string }) | DocumentNotFound {
  // Parse :line suffix
  let parsedFromLine = fromLine;
  let filepath = filename;
  const colonMatch = filepath.match(/:(\d+)$/);
  if (colonMatch && !parsedFromLine) {
    parsedFromLine = parseInt(colonMatch[1], 10);
    filepath = filepath.slice(0, -colonMatch[0].length);
  }

  const result = findDocument(db, filepath, { includeBody: true });
  if ("error" in result) return result;

  let body = result.body || "";
  if (parsedFromLine !== undefined || maxLines !== undefined) {
    const lines = body.split('\n');
    const start = (parsedFromLine || 1) - 1;
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    body = lines.slice(start, end).join('\n');
  }

  return { ...result, body };
}

/**
 * Find multiple documents by glob pattern or comma-separated list
 * Returns documents without body by default (use getDocumentBody to load)
 */
export function findDocuments(
  db: Database,
  pattern: string,
  options: { includeBody?: boolean; maxBytes?: number } = {}
): { docs: MultiGetResult[]; errors: string[] } {
  const isCommaSeparated = pattern.includes(',') && !pattern.includes('*') && !pattern.includes('?');
  const errors: string[] = [];
  const maxBytes = options.maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;

  const selectCols = options.includeBody
    ? `filepath, display_path, title, hash, collection_id, modified_at, LENGTH(body) as body_length, body`
    : `filepath, display_path, title, hash, collection_id, modified_at, LENGTH(body) as body_length`;

  let fileRows: DbDocRow[];

  if (isCommaSeparated) {
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    fileRows = [];
    for (const name of names) {
      let doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE display_path = ? AND active = 1`).get(name) as DbDocRow | null;
      if (!doc) {
        doc = db.prepare(`SELECT ${selectCols} FROM documents WHERE display_path LIKE ? AND active = 1 LIMIT 1`).get(`%${name}`) as DbDocRow | null;
      }
      if (doc) {
        fileRows.push(doc);
      } else {
        const similar = findSimilarFiles(db, name, 5, 3);
        let msg = `File not found: ${name}`;
        if (similar.length > 0) {
          msg += ` (did you mean: ${similar.join(', ')}?)`;
        }
        errors.push(msg);
      }
    }
  } else {
    // Glob pattern match
    const matched = matchFilesByGlob(db, pattern);
    if (matched.length === 0) {
      errors.push(`No files matched pattern: ${pattern}`);
      return { docs: [], errors };
    }
    const filepaths = matched.map(m => m.filepath);
    const placeholders = filepaths.map(() => '?').join(',');
    fileRows = db.prepare(`SELECT ${selectCols} FROM documents WHERE filepath IN (${placeholders}) AND active = 1`).all(...filepaths) as DbDocRow[];
  }

  const results: MultiGetResult[] = [];

  for (const row of fileRows) {
    const context = getContextForFile(db, row.filepath);

    if (row.body_length > maxBytes) {
      results.push({
        doc: { filepath: row.filepath, displayPath: row.display_path },
        skipped: true,
        skipReason: `File too large (${Math.round(row.body_length / 1024)}KB > ${Math.round(maxBytes / 1024)}KB)`,
      });
      continue;
    }

    results.push({
      doc: {
        filepath: row.filepath,
        displayPath: row.display_path,
        title: row.title || row.display_path.split('/').pop() || row.display_path,
        context,
        hash: row.hash,
        collectionId: row.collection_id,
        modifiedAt: row.modified_at,
        bodyLength: row.body_length,
        ...(options.includeBody && row.body !== undefined && { body: row.body }),
      },
      skipped: false,
    });
  }

  return { docs: results, errors };
}

/**
 * Legacy function for backwards compatibility
 */
export function getMultipleDocuments(db: Database, pattern: string, maxLines?: number, maxBytes: number = DEFAULT_MULTI_GET_MAX_BYTES): { files: MultiGetFile[]; errors: string[] } {
  const { docs, errors } = findDocuments(db, pattern, { includeBody: true, maxBytes });

  const files: MultiGetFile[] = docs.map(result => {
    if (result.skipped) {
      return {
        filepath: result.doc.filepath,
        displayPath: result.doc.displayPath,
        title: "",
        body: "",
        context: null,
        skipped: true as const,
        skipReason: result.skipReason,
      };
    }

    let body = result.doc.body || "";
    if (maxLines !== undefined) {
      const lines = body.split('\n');
      body = lines.slice(0, maxLines).join('\n');
      if (lines.length > maxLines) {
        body += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
      }
    }

    return {
      filepath: result.doc.filepath,
      displayPath: result.doc.displayPath,
      title: result.doc.title,
      body,
      context: result.doc.context,
      skipped: false as const,
    };
  });

  return { files, errors };
}

// Keep the old MultiGetFile type for backwards compatibility
export type MultiGetFile = {
  filepath: string;
  displayPath: string;
  title: string;
  body: string;
  context: string | null;
  skipped: false;
} | {
  filepath: string;
  displayPath: string;
  title: string;
  body: string;
  context: string | null;
  skipped: true;
  skipReason: string;
};

// =============================================================================
// Status
// =============================================================================

export function getStatus(db: Database): IndexStatus {
  const collections = db.prepare(`
    SELECT c.id, c.pwd, c.glob_pattern, c.created_at,
           COUNT(d.id) as active_count,
           MAX(d.modified_at) as last_doc_update
    FROM collections c
    LEFT JOIN documents d ON d.collection_id = c.id AND d.active = 1
    GROUP BY c.id
    ORDER BY last_doc_update DESC
  `).all() as { id: number; pwd: string; glob_pattern: string; created_at: string; active_count: number; last_doc_update: string | null }[];

  const totalDocs = (db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 1`).get() as { c: number }).c;
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  return {
    totalDocuments: totalDocs,
    needsEmbedding,
    hasVectorIndex: hasVectors,
    collections: collections.map(col => ({
      id: col.id,
      path: col.pwd,
      pattern: col.glob_pattern,
      documents: col.active_count,
      lastUpdated: col.last_doc_update || col.created_at,
    })),
  };
}

// =============================================================================
// Snippet extraction
// =============================================================================

export type SnippetResult = {
  line: number;           // 1-indexed line number of best match
  snippet: string;        // The snippet text with diff-style header
  linesBefore: number;    // Lines in document before snippet
  linesAfter: number;     // Lines in document after snippet
  snippetLines: number;   // Number of lines in snippet
};

export function extractSnippet(body: string, query: string, maxLen = 500, chunkPos?: number): SnippetResult {
  const totalLines = body.split('\n').length;
  let searchBody = body;
  let lineOffset = 0;

  if (chunkPos && chunkPos > 0) {
    const contextStart = Math.max(0, chunkPos - 100);
    const contextEnd = Math.min(body.length, chunkPos + maxLen + 100);
    searchBody = body.slice(contextStart, contextEnd);
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

  const start = Math.max(0, bestLine - 1);
  const end = Math.min(lines.length, bestLine + 3);
  const snippetLines = lines.slice(start, end);
  let snippetText = snippetLines.join('\n');
  if (snippetText.length > maxLen) snippetText = snippetText.substring(0, maxLen - 3) + "...";

  const absoluteStart = lineOffset + start + 1; // 1-indexed
  const snippetLineCount = snippetLines.length;
  const linesBefore = absoluteStart - 1;
  const linesAfter = totalLines - (absoluteStart + snippetLineCount - 1);

  // Format with diff-style header: @@ -start,count @@ (linesBefore before, linesAfter after)
  const header = `@@ -${absoluteStart},${snippetLineCount} @@ (${linesBefore} before, ${linesAfter} after)`;
  const snippet = `${header}\n${snippetText}`;

  return {
    line: lineOffset + bestLine + 1,
    snippet,
    linesBefore,
    linesAfter,
    snippetLines: snippetLineCount,
  };
}
