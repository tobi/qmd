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
import {
  findContextForPath as collectionsFindContextForPath,
  addContext as collectionsAddContext,
  removeContext as collectionsRemoveContext,
  listAllContexts as collectionsListAllContexts,
  getCollection,
  listCollections as collectionsListCollections,
  addCollection as collectionsAddCollection,
  removeCollection as collectionsRemoveCollection,
  renameCollection as collectionsRenameCollection,
  setGlobalContext,
  type NamedCollection,
} from "./collections";

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
 * Also supports collection root: "qmd://collection-name/" or "qmd://collection-name"
 */
export function parseVirtualPath(virtualPath: string): VirtualPath | null {
  // Match: qmd://collection-name[/optional-path]
  // Allows: qmd://name, qmd://name/, qmd://name/path
  const match = virtualPath.match(/^qmd:\/\/([^\/]+)\/?(.*)$/);
  if (!match) return null;
  return {
    collectionName: match[1],
    path: match[2] || '',  // Empty string for collection root
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
  // Get all collections from YAML config
  const collections = collectionsListCollections();

  // Find which collection this absolute path belongs to
  for (const coll of collections) {
    if (absolutePath.startsWith(coll.path + '/') || absolutePath === coll.path) {
      // Extract relative path
      const relativePath = absolutePath.startsWith(coll.path + '/')
        ? absolutePath.slice(coll.path.length + 1)
        : '';

      // Verify this document exists in the database
      const doc = db.prepare(`
        SELECT d.path
        FROM documents d
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
        LIMIT 1
      `).get(coll.name, relativePath) as { path: string } | null;

      if (doc) {
        return buildVirtualPath(coll.name, relativePath);
      }
    }
  }

  return null;
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

  // Drop legacy tables that are now managed in YAML
  db.exec(`DROP TABLE IF EXISTS path_contexts`);
  db.exec(`DROP TABLE IF EXISTS collections`);

  // Content-addressable storage - the source of truth for document content
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Documents table - file system layer mapping virtual paths to content hashes
  // Collections are now managed in ~/.config/qmd/index.yml
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection TEXT NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection, path)
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active)`);

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

  // FTS - index filepath (collection/path), title, and content
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      filepath, title, body,
      tokenize='porter unicode61'
    )
  `);

  // Triggers to keep FTS in sync
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
    WHEN new.active = 1
    BEGIN
      INSERT INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
      DELETE FROM documents_fts WHERE rowid = old.id;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
    BEGIN
      -- Delete from FTS if no longer active
      DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;

      -- Update FTS if still/newly active
      INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);
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

  // Cleanup and maintenance
  deleteOllamaCache: () => number;
  deleteInactiveDocuments: () => number;
  cleanupOrphanedContent: () => number;
  cleanupOrphanedVectors: () => number;
  cleanupDuplicateCollections: () => number;
  vacuumDatabase: () => void;

  // Context
  getContextForFile: (filepath: string) => string | null;
  getContextForPath: (collectionName: string, path: string) => string | null;
  getCollectionByName: (name: string) => { name: string; pwd: string; glob_pattern: string } | null;
  getCollectionsWithoutContext: () => { name: string; pwd: string; doc_count: number }[];
  getTopLevelPathsWithoutContext: (collectionName: string) => string[];

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

  // Document indexing operations
  insertContent: (hash: string, content: string, createdAt: string) => void;
  insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => void;
  findActiveDocument: (collectionName: string, path: string) => { id: number; hash: string; title: string } | null;
  updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => void;
  updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => void;
  deactivateDocument: (collectionName: string, path: string) => void;
  getActiveDocumentPaths: (collectionName: string) => string[];

  // Vector/embedding operations
  getHashesForEmbedding: () => { hash: string; body: string; path: string }[];
  clearAllEmbeddings: () => void;
  insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => void;
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

    // Cleanup and maintenance
    deleteOllamaCache: () => deleteOllamaCache(db),
    deleteInactiveDocuments: () => deleteInactiveDocuments(db),
    cleanupOrphanedContent: () => cleanupOrphanedContent(db),
    cleanupOrphanedVectors: () => cleanupOrphanedVectors(db),
    cleanupDuplicateCollections: () => cleanupDuplicateCollections(db),
    vacuumDatabase: () => vacuumDatabase(db),

    // Context
    getContextForFile: (filepath: string) => getContextForFile(db, filepath),
    getContextForPath: (collectionName: string, path: string) => getContextForPath(db, collectionName, path),
    getCollectionByName: (name: string) => getCollectionByName(db, name),
    getCollectionsWithoutContext: () => getCollectionsWithoutContext(db),
    getTopLevelPathsWithoutContext: (collectionName: string) => getTopLevelPathsWithoutContext(db, collectionName),

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

    // Document indexing operations
    insertContent: (hash: string, content: string, createdAt: string) => insertContent(db, hash, content, createdAt),
    insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => insertDocument(db, collectionName, path, title, hash, createdAt, modifiedAt),
    findActiveDocument: (collectionName: string, path: string) => findActiveDocument(db, collectionName, path),
    updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => updateDocumentTitle(db, documentId, title, modifiedAt),
    updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => updateDocument(db, documentId, title, hash, modifiedAt),
    deactivateDocument: (collectionName: string, path: string) => deactivateDocument(db, collectionName, path),
    getActiveDocumentPaths: (collectionName: string) => getActiveDocumentPaths(db, collectionName),

    // Vector/embedding operations
    getHashesForEmbedding: () => getHashesForEmbedding(db),
    clearAllEmbeddings: () => clearAllEmbeddings(db),
    insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => insertEmbedding(db, hash, seq, pos, embedding, model, embeddedAt),
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
  collectionName: string;     // Parent collection name
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
  name: string;
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
// Cleanup and maintenance operations
// =============================================================================

/**
 * Delete cached Ollama API responses.
 * Returns the number of cached responses deleted.
 */
export function deleteOllamaCache(db: Database): number {
  const result = db.prepare(`DELETE FROM ollama_cache`).run();
  return result.changes;
}

/**
 * Remove inactive document records (active = 0).
 * Returns the number of inactive documents deleted.
 */
export function deleteInactiveDocuments(db: Database): number {
  const result = db.prepare(`DELETE FROM documents WHERE active = 0`).run();
  return result.changes;
}

/**
 * Remove orphaned content hashes that are not referenced by any active document.
 * Returns the number of orphaned content hashes deleted.
 */
export function cleanupOrphanedContent(db: Database): number {
  const result = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();
  return result.changes;
}

/**
 * Remove orphaned vector embeddings that are not referenced by any active document.
 * Returns the number of orphaned embedding chunks deleted.
 */
export function cleanupOrphanedVectors(db: Database): number {
  // Check if vectors_vec table exists
  const tableExists = db.prepare(`
    SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'
  `).get();

  if (!tableExists) {
    return 0;
  }

  // Count orphaned vectors first
  const countResult = db.prepare(`
    SELECT COUNT(*) as c FROM content_vectors cv
    WHERE NOT EXISTS (
      SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
    )
  `).get() as { c: number };

  if (countResult.c === 0) {
    return 0;
  }

  // Delete from vectors_vec first
  db.exec(`
    DELETE FROM vectors_vec WHERE hash_seq IN (
      SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
      WHERE NOT EXISTS (
        SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
      )
    )
  `);

  // Delete from content_vectors
  db.exec(`
    DELETE FROM content_vectors WHERE hash NOT IN (
      SELECT hash FROM documents WHERE active = 1
    )
  `);

  return countResult.c;
}

/**
 * Remove duplicate collections, keeping the oldest one per (pwd, glob_pattern).
 * NOTE: This function is deprecated since collections are now managed in YAML.
 * Kept for backwards compatibility but returns 0.
 */
export function cleanupDuplicateCollections(db: Database): number {
  // Collections are now managed in YAML, no cleanup needed
  return 0;
}

/**
 * Run VACUUM to reclaim unused space in the database.
 * This operation rebuilds the database file to eliminate fragmentation.
 */
export function vacuumDatabase(db: Database): void {
  db.exec(`VACUUM`);
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

// =============================================================================
// Document indexing operations
// =============================================================================

/**
 * Insert content into the content table (content-addressable storage).
 * Uses INSERT OR IGNORE so duplicate hashes are skipped.
 */
export function insertContent(db: Database, hash: string, content: string, createdAt: string): void {
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
    .run(hash, content, createdAt);
}

/**
 * Insert a new document into the documents table.
 */
export function insertDocument(
  db: Database,
  collectionName: string,
  path: string,
  title: string,
  hash: string,
  createdAt: string,
  modifiedAt: string
): void {
  db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, 1)
  `).run(collectionName, path, title, hash, createdAt, modifiedAt);
}

/**
 * Find an active document by collection name and path.
 */
export function findActiveDocument(
  db: Database,
  collectionName: string,
  path: string
): { id: number; hash: string; title: string } | null {
  return db.prepare(`
    SELECT id, hash, title FROM documents
    WHERE collection = ? AND path = ? AND active = 1
  `).get(collectionName, path) as { id: number; hash: string; title: string } | null;
}

/**
 * Update the title and modified_at timestamp for a document.
 */
export function updateDocumentTitle(
  db: Database,
  documentId: number,
  title: string,
  modifiedAt: string
): void {
  db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`)
    .run(title, modifiedAt, documentId);
}

/**
 * Update an existing document's hash, title, and modified_at timestamp.
 * Used when content changes but the file path stays the same.
 */
export function updateDocument(
  db: Database,
  documentId: number,
  title: string,
  hash: string,
  modifiedAt: string
): void {
  db.prepare(`UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?`)
    .run(title, hash, modifiedAt, documentId);
}

/**
 * Deactivate a document (mark as inactive but don't delete).
 */
export function deactivateDocument(db: Database, collectionName: string, path: string): void {
  db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1`)
    .run(collectionName, path);
}

/**
 * Get all active document paths for a collection.
 */
export function getActiveDocumentPaths(db: Database, collectionName: string): string[] {
  const rows = db.prepare(`
    SELECT path FROM documents WHERE collection = ? AND active = 1
  `).all(collectionName) as { path: string }[];
  return rows.map(r => r.path);
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
  const allFiles = db.prepare(`
    SELECT 'qmd://' || d.collection || '/' || d.path as display_path
    FROM documents d
    WHERE d.active = 1
  `).all() as { display_path: string }[];
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
      'qmd://' || d.collection || '/' || d.path as virtual_path,
      LENGTH(content.doc) as body_length,
      d.path
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE d.active = 1
  `).all() as { virtual_path: string; body_length: number; path: string }[];

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
 * @param db Database instance (unused - kept for compatibility)
 * @param collectionName Collection name
 * @param path Relative path within the collection
 * @returns Context string or null if no context is defined
 */
export function getContextForPath(db: Database, collectionName: string, path: string): string | null {
  const config = collectionsLoadConfig();
  const coll = getCollection(collectionName);

  if (!coll) return null;

  // Collect ALL matching contexts (global + all path prefixes)
  const contexts: string[] = [];

  // Add global context if present
  if (config.global_context) {
    contexts.push(config.global_context);
  }

  // Add all matching path contexts (from most general to most specific)
  if (coll.context) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;

    // Collect all matching prefixes
    const matchingContexts: { prefix: string; context: string }[] = [];
    for (const [prefix, context] of Object.entries(coll.context)) {
      const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
      if (normalizedPath.startsWith(normalizedPrefix)) {
        matchingContexts.push({ prefix: normalizedPrefix, context });
      }
    }

    // Sort by prefix length (shortest/most general first)
    matchingContexts.sort((a, b) => a.prefix.length - b.prefix.length);

    // Add all matching contexts
    for (const match of matchingContexts) {
      contexts.push(match.context);
    }
  }

  // Join all contexts with double newline
  return contexts.length > 0 ? contexts.join('\n\n') : null;
}

/**
 * Legacy function for backward compatibility - resolves filepath to collection+path first
 */
export function getContextForFile(db: Database, filepath: string): string | null {
  // Get all collections from YAML config
  const collections = collectionsListCollections();
  const config = collectionsLoadConfig();

  // Find which collection this absolute path belongs to
  for (const coll of collections) {
    if (filepath.startsWith(coll.path + '/') || filepath === coll.path) {
      // Extract relative path
      const relativePath = filepath.startsWith(coll.path + '/')
        ? filepath.slice(coll.path.length + 1)
        : '';

      // Verify this document exists in the database
      const doc = db.prepare(`
        SELECT d.path
        FROM documents d
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
        LIMIT 1
      `).get(coll.name, relativePath) as { path: string } | null;

      if (doc) {
        // Collect ALL matching contexts (global + all path prefixes)
        const contexts: string[] = [];

        // Add global context if present
        if (config.global_context) {
          contexts.push(config.global_context);
        }

        // Add all matching path contexts (from most general to most specific)
        if (coll.context) {
          const normalizedPath = relativePath.startsWith("/") ? relativePath : `/${relativePath}`;

          // Collect all matching prefixes
          const matchingContexts: { prefix: string; context: string }[] = [];
          for (const [prefix, context] of Object.entries(coll.context)) {
            const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
            if (normalizedPath.startsWith(normalizedPrefix)) {
              matchingContexts.push({ prefix: normalizedPrefix, context });
            }
          }

          // Sort by prefix length (shortest/most general first)
          matchingContexts.sort((a, b) => a.prefix.length - b.prefix.length);

          // Add all matching contexts
          for (const match of matchingContexts) {
            contexts.push(match.context);
          }
        }

        // Join all contexts with double newline
        return contexts.length > 0 ? contexts.join('\n\n') : null;
      }
    }
  }

  return null;
}

/**
 * Get collection by name from YAML config.
 * Returns collection metadata from ~/.config/qmd/index.yml
 */
export function getCollectionByName(db: Database, name: string): { name: string; pwd: string; glob_pattern: string } | null {
  const collection = getCollection(name);
  if (!collection) return null;

  return {
    name: collection.name,
    pwd: collection.path,
    glob_pattern: collection.pattern,
  };
}

/**
 * List all collections with document counts from database.
 * Merges YAML config with database statistics.
 */
export function listCollections(db: Database): { name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null }[] {
  const collections = collectionsListCollections();

  // Get document counts from database for each collection
  const result = collections.map(coll => {
    const stats = db.prepare(`
      SELECT
        COUNT(d.id) as doc_count,
        SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END) as active_count,
        MAX(d.modified_at) as last_modified
      FROM documents d
      WHERE d.collection = ?
    `).get(coll.name) as { doc_count: number; active_count: number; last_modified: string | null } | null;

    return {
      name: coll.name,
      pwd: coll.path,
      glob_pattern: coll.pattern,
      doc_count: stats?.doc_count || 0,
      active_count: stats?.active_count || 0,
      last_modified: stats?.last_modified || null,
    };
  });

  return result;
}

/**
 * Remove a collection and clean up its documents.
 * Uses collections.ts to remove from YAML config and cleans up database.
 */
export function removeCollection(db: Database, collectionName: string): { deletedDocs: number; cleanedHashes: number } {
  // Delete documents from database
  const docResult = db.prepare(`DELETE FROM documents WHERE collection = ?`).run(collectionName);

  // Clean up orphaned content hashes
  const cleanupResult = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();

  // Remove from YAML config (returns true if found and removed)
  collectionsRemoveCollection(collectionName);

  return {
    deletedDocs: docResult.changes,
    cleanedHashes: cleanupResult.changes
  };
}

/**
 * Rename a collection.
 * Updates both YAML config and database documents table.
 */
export function renameCollection(db: Database, oldName: string, newName: string): void {
  // Update all documents with the new collection name in database
  db.prepare(`UPDATE documents SET collection = ? WHERE collection = ?`)
    .run(newName, oldName);

  // Rename in YAML config
  collectionsRenameCollection(oldName, newName);
}

// =============================================================================
// Context Management Operations
// =============================================================================

/**
 * Insert or update a context for a specific collection and path prefix.
 */
export function insertContext(db: Database, collectionId: number, pathPrefix: string, context: string): void {
  // Get collection name from ID
  const coll = db.prepare(`SELECT name FROM collections WHERE id = ?`).get(collectionId) as { name: string } | null;
  if (!coll) {
    throw new Error(`Collection with id ${collectionId} not found`);
  }

  // Use collections.ts to add context
  collectionsAddContext(coll.name, pathPrefix, context);
}

/**
 * Delete a context for a specific collection and path prefix.
 * Returns the number of contexts deleted.
 */
export function deleteContext(db: Database, collectionName: string, pathPrefix: string): number {
  // Use collections.ts to remove context
  const success = collectionsRemoveContext(collectionName, pathPrefix);
  return success ? 1 : 0;
}

/**
 * Delete all global contexts (contexts with empty path_prefix).
 * Returns the number of contexts deleted.
 */
export function deleteGlobalContexts(db: Database): number {
  let deletedCount = 0;

  // Remove global context
  setGlobalContext(undefined);
  deletedCount++;

  // Remove root context (empty string) from all collections
  const collections = collectionsListCollections();
  for (const coll of collections) {
    const success = collectionsRemoveContext(coll.name, '');
    if (success) {
      deletedCount++;
    }
  }

  return deletedCount;
}

/**
 * List all contexts, grouped by collection.
 * Returns contexts ordered by collection name, then by path prefix length (longest first).
 */
export function listPathContexts(db: Database): { collection_name: string; path_prefix: string; context: string }[] {
  const allContexts = collectionsListAllContexts();

  // Convert to expected format and sort
  return allContexts.map(ctx => ({
    collection_name: ctx.collection,
    path_prefix: ctx.path,
    context: ctx.context,
  })).sort((a, b) => {
    // Sort by collection name first
    if (a.collection_name !== b.collection_name) {
      return a.collection_name.localeCompare(b.collection_name);
    }
    // Then by path prefix length (longest first)
    if (a.path_prefix.length !== b.path_prefix.length) {
      return b.path_prefix.length - a.path_prefix.length;
    }
    // Then alphabetically
    return a.path_prefix.localeCompare(b.path_prefix);
  });
}

/**
 * Get all collections (name only - from YAML config).
 */
export function getAllCollections(db: Database): { name: string }[] {
  const collections = collectionsListCollections();
  return collections.map(c => ({ name: c.name }));
}

/**
 * Check which collections don't have any context defined.
 * Returns collections that have no context entries at all (not even root context).
 */
export function getCollectionsWithoutContext(db: Database): { name: string; pwd: string; doc_count: number }[] {
  // Get all collections from YAML config
  const yamlCollections = collectionsListCollections();

  // Filter to those without context
  const collectionsWithoutContext: { name: string; pwd: string; doc_count: number }[] = [];

  for (const coll of yamlCollections) {
    // Check if collection has any context
    if (!coll.context || Object.keys(coll.context).length === 0) {
      // Get doc count from database
      const stats = db.prepare(`
        SELECT COUNT(d.id) as doc_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name) as { doc_count: number } | null;

      collectionsWithoutContext.push({
        name: coll.name,
        pwd: coll.path,
        doc_count: stats?.doc_count || 0,
      });
    }
  }

  return collectionsWithoutContext.sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Get top-level directories in a collection that don't have context.
 * Useful for suggesting where context might be needed.
 */
export function getTopLevelPathsWithoutContext(db: Database, collectionName: string): string[] {
  // Get all paths in the collection from database
  const paths = db.prepare(`
    SELECT DISTINCT path FROM documents
    WHERE collection = ? AND active = 1
  `).all(collectionName) as { path: string }[];

  // Get existing contexts for this collection from YAML
  const yamlColl = getCollection(collectionName);
  if (!yamlColl) return [];

  const contextPrefixes = new Set<string>();
  if (yamlColl.context) {
    for (const prefix of Object.keys(yamlColl.context)) {
      contextPrefixes.add(prefix);
    }
  }

  // Extract top-level directories (first path component)
  const topLevelDirs = new Set<string>();
  for (const { path } of paths) {
    const parts = path.split('/').filter(Boolean);
    if (parts.length > 1) {
      topLevelDirs.add(parts[0]);
    }
  }

  // Filter out directories that already have context (exact or parent)
  const missing: string[] = [];
  for (const dir of topLevelDirs) {
    let hasContext = false;

    // Check if this dir or any parent has context
    for (const prefix of contextPrefixes) {
      if (prefix === '' || prefix === dir || dir.startsWith(prefix + '/')) {
        hasContext = true;
        break;
      }
    }

    if (!hasContext) {
      missing.push(dir);
    }
  }

  return missing.sort();
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
      'qmd://' || d.collection || '/' || d.path as filepath,
      'qmd://' || d.collection || '/' || d.path as display_path,
      d.title,
      content.doc as body,
      bm25(documents_fts, 10.0, 1.0) as score
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    JOIN content ON content.hash = d.hash
    WHERE documents_fts MATCH ? AND d.active = 1
  `;
  const params: (string | number)[] = [ftsQuery];

  if (collectionId !== undefined) {
    // Note: collectionId is a legacy parameter that should be phased out
    // Collections are now managed in YAML. For now, we interpret it as a collection name filter.
    // This code path is likely unused as collection filtering should be done at CLI level.
    sql += ` AND d.collection = ?`;
    params.push(String(collectionId));
  }

  sql += ` ORDER BY score LIMIT ?`;
  params.push(limit);

  const rows = db.prepare(sql).all(...params) as { filepath: string; display_path: string; title: string; body: string; score: number }[];

  const maxScore = rows.length > 0 ? Math.max(...rows.map(r => Math.abs(r.score))) : 1;
  return rows.map(row => ({
    filepath: row.filepath,
    displayPath: row.display_path,
    title: row.title,
    hash: "",  // Not available in FTS query
    collectionName: row.filepath.split('//')[1]?.split('/')[0] || "",  // Extract from virtual path
    modifiedAt: "",  // Not available in FTS query
    bodyLength: row.body.length,
    body: row.body,
    context: null,  // Not loaded in FTS
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
      'qmd://' || d.collection || '/' || d.path as filepath,
      'qmd://' || d.collection || '/' || d.path as display_path,
      d.title,
      content.doc as body,
      cv.pos
    FROM vectors_vec v
    JOIN content_vectors cv ON cv.hash || '_' || cv.seq = v.hash_seq
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    JOIN content ON content.hash = d.hash
    WHERE v.embedding MATCH ? AND k = ?
  `;

  if (collectionId !== undefined) {
    // Convert collectionId to collection name for filtering
    const coll = db.prepare(`SELECT name FROM collections WHERE id = ?`).get(collectionId) as { name: string } | null;
    if (coll) {
      sql += ` AND d.collection = '${coll.name}'`;
    }
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
      filepath: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      hash: "",  // Not available in vec query
      collectionName: row.filepath.split('//')[1]?.split('/')[0] || "",  // Extract from virtual path
      modifiedAt: "",  // Not available in vec query
      bodyLength: row.body.length,
      body: row.body,
      context: null,  // Not loaded in vec
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

/**
 * Get all unique content hashes that need embeddings (from active documents).
 * Returns hash, document body, and a sample path for display purposes.
 */
export function getHashesForEmbedding(db: Database): { hash: string; body: string; path: string }[] {
  return db.prepare(`
    SELECT d.hash, c.doc as body, MIN(d.path) as path
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; path: string }[];
}

/**
 * Clear all embeddings from the database (force re-index).
 * Deletes all rows from content_vectors and drops the vectors_vec table.
 */
export function clearAllEmbeddings(db: Database): void {
  db.exec(`DELETE FROM content_vectors`);
  db.exec(`DROP TABLE IF EXISTS vectors_vec`);
}

/**
 * Insert a single embedding into both content_vectors and vectors_vec tables.
 * The hash_seq key is formatted as "hash_seq" for the vectors_vec table.
 */
export function insertEmbedding(
  db: Database,
  hash: string,
  seq: number,
  pos: number,
  embedding: Float32Array,
  model: string,
  embeddedAt: string
): void {
  const hashSeq = `${hash}_${seq}`;
  const insertVecStmt = db.prepare(`INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`);
  const insertContentVectorStmt = db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)`);

  insertVecStmt.run(hashSeq, embedding);
  insertContentVectorStmt.run(hash, seq, pos, model, embeddedAt);
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
  display_path: string;
  title: string;
  hash: string;
  collection: string;
  path: string;
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

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;

  // Build computed columns for display_path
  // Note: filepath is computed from YAML collections after query
  const selectCols = `
    'qmd://' || d.collection || '/' || d.path as display_path,
    d.title,
    d.hash,
    d.collection,
    d.path,
    d.modified_at,
    LENGTH(content.doc) as body_length
    ${bodyCol}
  `;

  // Try to match by virtual path first
  let doc = db.prepare(`
    SELECT ${selectCols}
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1
  `).get(filepath) as DbDocRow | null;

  // Try fuzzy match by virtual path
  if (!doc) {
    doc = db.prepare(`
      SELECT ${selectCols}
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
      LIMIT 1
    `).get(`%${filepath}`) as DbDocRow | null;
  }

  // Try to match by absolute path (requires looking up collection paths from YAML)
  if (!doc && !filepath.startsWith('qmd://')) {
    const collections = collectionsListCollections();
    for (const coll of collections) {
      let relativePath: string | null = null;

      // If filepath is absolute and starts with collection path, extract relative part
      if (filepath.startsWith(coll.path + '/')) {
        relativePath = filepath.slice(coll.path.length + 1);
      }
      // Otherwise treat filepath as relative to collection
      else if (!filepath.startsWith('/')) {
        relativePath = filepath;
      }

      if (relativePath) {
        doc = db.prepare(`
          SELECT ${selectCols}
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(coll.name, relativePath) as DbDocRow | null;
        if (doc) break;
      }
    }
  }

  if (!doc) {
    const similar = findSimilarFiles(db, filepath, 5, 5);
    return { error: "not_found", query: filename, similarFiles: similar };
  }

  // Compute absolute filepath from collection (in YAML) and relative path
  const coll = getCollection(doc.collection);
  const absoluteFilepath = coll ? `${coll.path}/${doc.path}` : doc.path;
  const context = getContextForFile(db, absoluteFilepath);

  return {
    filepath: absoluteFilepath,
    displayPath: doc.display_path,
    title: doc.title,
    context,
    hash: doc.hash,
    collectionName: doc.collection,
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

  // Try to resolve document by filepath (absolute or virtual)
  let row: { body: string } | null = null;

  // Try virtual path first
  if (filepath.startsWith('qmd://')) {
    row = db.prepare(`
      SELECT content.doc as body
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1
    `).get(filepath) as { body: string } | null;
  }

  // Try absolute path by looking up in YAML collections
  if (!row) {
    const collections = collectionsListCollections();
    for (const coll of collections) {
      if (filepath.startsWith(coll.path + '/')) {
        const relativePath = filepath.slice(coll.path.length + 1);
        row = db.prepare(`
          SELECT content.doc as body
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(coll.name, relativePath) as { body: string } | null;
        if (row) break;
      }
    }
  }

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

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;
  const selectCols = `
    'qmd://' || d.collection || '/' || d.path as display_path,
    d.title,
    d.hash,
    d.collection,
    d.path,
    d.modified_at,
    LENGTH(content.doc) as body_length
    ${bodyCol}
  `;

  let fileRows: DbDocRow[];

  if (isCommaSeparated) {
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    fileRows = [];
    for (const name of names) {
      let doc = db.prepare(`
        SELECT ${selectCols}
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1
      `).get(name) as DbDocRow | null;
      if (!doc) {
        doc = db.prepare(`
          SELECT ${selectCols}
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${name}`) as DbDocRow | null;
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
    const virtualPaths = matched.map(m => m.filepath);
    const placeholders = virtualPaths.map(() => '?').join(',');
    fileRows = db.prepare(`
      SELECT ${selectCols}
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'qmd://' || d.collection || '/' || d.path IN (${placeholders}) AND d.active = 1
    `).all(...virtualPaths) as DbDocRow[];
  }

  const results: MultiGetResult[] = [];

  for (const row of fileRows) {
    // Compute absolute filepath from collection
    const coll = getCollection(row.collection);
    const absoluteFilepath = coll ? `${coll.path}/${row.path}` : row.path;
    const context = getContextForFile(db, absoluteFilepath);

    if (row.body_length > maxBytes) {
      results.push({
        doc: { filepath: absoluteFilepath, displayPath: row.display_path },
        skipped: true,
        skipReason: `File too large (${Math.round(row.body_length / 1024)}KB > ${Math.round(maxBytes / 1024)}KB)`,
      });
      continue;
    }

    results.push({
      doc: {
        filepath: absoluteFilepath,
        displayPath: row.display_path,
        title: row.title || row.display_path.split('/').pop() || row.display_path,
        context,
        hash: row.hash,
        collectionName: row.collection,
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
  // Load collections from YAML
  const yamlCollections = collectionsListCollections();

  // Get document counts and last update times for each collection
  const collections = yamlCollections.map(col => {
    const stats = db.prepare(`
      SELECT
        COUNT(*) as active_count,
        MAX(modified_at) as last_doc_update
      FROM documents
      WHERE collection = ? AND active = 1
    `).get(col.name) as { active_count: number; last_doc_update: string | null };

    return {
      name: col.name,
      path: col.path,
      pattern: col.pattern,
      documents: stats.active_count,
      lastUpdated: stats.last_doc_update || new Date().toISOString(),
    };
  });

  // Sort by last update time (most recent first)
  collections.sort((a, b) => {
    if (!a.lastUpdated) return 1;
    if (!b.lastUpdated) return -1;
    return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
  });

  const totalDocs = (db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 1`).get() as { c: number }).c;
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  return {
    totalDocuments: totalDocs,
    needsEmbedding,
    hasVectorIndex: hasVectors,
    collections,
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
