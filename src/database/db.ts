/**
 * Database connection and schema initialization
 */

import { Database } from 'bun:sqlite';
import * as sqliteVec from 'sqlite-vec';
import { getDbPath } from '../utils/paths.ts';

/**
 * Initialize and return database connection with schema
 * @param indexName - Name of the index (default: "index")
 * @returns Database instance with schema initialized
 */
export function getDb(indexName: string = "index"): Database {
  const db = new Database(getDbPath(indexName));
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");

  initializeSchema(db);
  return db;
}

/**
 * Initialize database schema (tables, indices, triggers)
 * @param db - Database instance
 */
export function initializeSchema(db: Database): void {
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

  // Search history table
  db.exec(`
    CREATE TABLE IF NOT EXISTS search_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      command TEXT NOT NULL CHECK(command IN ('search', 'vsearch', 'query')),
      query TEXT NOT NULL,
      results_count INTEGER NOT NULL,
      index_name TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(timestamp DESC)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_search_history_command ON search_history(command)`);

  // Documents table with collection_id and full filepath
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      filepath TEXT NOT NULL,
      display_path TEXT NOT NULL DEFAULT '',
      body TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      FOREIGN KEY (collection_id) REFERENCES collections(id)
    )
  `);

  // Migration: add display_path column if missing
  const docInfo = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
  const hasDisplayPath = docInfo.some(col => col.name === 'display_path');
  if (!hasDisplayPath) {
    db.exec(`ALTER TABLE documents ADD COLUMN display_path TEXT NOT NULL DEFAULT ''`);
  }

  // Unique index on display_path (only for non-empty values)
  db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_display_path ON documents(display_path) WHERE display_path != '' AND active = 1`);

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
  // Time-based queries (recently modified documents)
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_modified_at ON documents(modified_at DESC) WHERE active = 1`);

  // Additional performance indexes
  db.exec(`CREATE INDEX IF NOT EXISTS idx_collections_context ON collections(context) WHERE context IS NOT NULL`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_content_vectors_model ON content_vectors(model)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_ollama_cache_created_at ON ollama_cache(created_at)`);
}

/**
 * Ensure vector table exists with correct dimensions
 * @param db - Database instance
 * @param dimensions - Number of dimensions for embeddings
 */
export function ensureVecTable(db: Database, dimensions: number): void {
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

/**
 * Get count of documents that need embedding
 * @param db - Database instance
 * @returns Number of documents without embeddings
 */
export function getHashesNeedingEmbedding(db: Database): number {
  // Check for hashes missing the first chunk (seq=0)
  const result = db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

/**
 * Check index health and return warnings
 * @param db - Database instance
 * @returns Health status message or null
 */
export function checkIndexHealth(db: Database): string | null {
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const totalDocs = (db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number }).count;

  // Warn if many docs need embedding
  if (needsEmbedding > 0) {
    const pct = Math.round((needsEmbedding / totalDocs) * 100);
    if (pct >= 10) {
      return `Warning: ${needsEmbedding} documents (${pct}%) need embeddings. Run 'qmd embed' for better results.`;
    }
  }

  return null;
}
