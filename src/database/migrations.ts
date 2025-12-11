/**
 * Database migration system with versioning
 */

import { Database } from 'bun:sqlite';

export interface Migration {
  version: number;
  description: string;
  up: (db: Database) => void;
  down?: (db: Database) => void;  // Optional rollback
}

/**
 * All migrations in chronological order
 */
export const migrations: Migration[] = [
  {
    version: 1,
    description: 'Initial schema with collections, documents, FTS, and supporting tables',
    up: (db) => {
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
          body TEXT NOT NULL,
          created_at TEXT NOT NULL,
          modified_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          FOREIGN KEY (collection_id) REFERENCES collections(id)
        )
      `);

      // Content vectors keyed by (hash, seq) for chunked embeddings
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

      // FTS triggers
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

      // Indexes
      db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id, active)`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath, active)`);
      db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_filepath_active ON documents(filepath) WHERE active = 1`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_modified_at ON documents(modified_at DESC) WHERE active = 1`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_collections_context ON collections(context) WHERE context IS NOT NULL`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_content_vectors_model ON content_vectors(model)`);
      db.exec(`CREATE INDEX IF NOT EXISTS idx_ollama_cache_created_at ON ollama_cache(created_at)`);
    },
  },
  {
    version: 2,
    description: 'Add display_path column to documents table',
    up: (db) => {
      // Check if column already exists (for databases created before migrations)
      const docInfo = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
      const hasDisplayPath = docInfo.some(col => col.name === 'display_path');

      if (!hasDisplayPath) {
        db.exec(`ALTER TABLE documents ADD COLUMN display_path TEXT NOT NULL DEFAULT ''`);
      }

      // Create unique index on display_path
      db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_display_path ON documents(display_path) WHERE display_path != '' AND active = 1`);
    },
  },
  {
    version: 3,
    description: 'Migrate content_vectors to support chunking (seq column)',
    up: (db) => {
      // Check if old schema exists (no seq column) and needs migration
      const cvInfo = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
      const hasSeqColumn = cvInfo.some(col => col.name === 'seq');

      if (cvInfo.length > 0 && !hasSeqColumn) {
        // Old schema without chunking - drop and recreate (embeddings need regenerating anyway)
        db.exec(`DROP TABLE IF EXISTS content_vectors`);
        db.exec(`DROP TABLE IF EXISTS vectors_vec`);

        // Recreate with new schema
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
      }
    },
  },
];

/**
 * Get current schema version from database
 */
function getCurrentVersion(db: Database): number {
  const result = db.prepare(`
    SELECT MAX(version) as version FROM schema_version
  `).get() as { version: number | null };
  return result?.version ?? 0;
}

/**
 * Record applied migration in schema_version table
 */
function setVersion(db: Database, version: number, description: string): void {
  db.prepare(`
    INSERT INTO schema_version (version, description, applied_at)
    VALUES (?, ?, datetime('now'))
  `).run(version, description);
}

/**
 * Apply all pending migrations to database
 */
export function migrate(db: Database): void {
  // Create version tracking table
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER PRIMARY KEY,
      applied_at TEXT NOT NULL,
      description TEXT NOT NULL
    )
  `);

  const currentVersion = getCurrentVersion(db);

  // Apply pending migrations in transaction
  for (const migration of migrations) {
    if (migration.version > currentVersion) {
      db.transaction(() => {
        migration.up(db);
        setVersion(db, migration.version, migration.description);
      })();
    }
  }
}

/**
 * Get migration history from database
 */
export function getMigrationHistory(db: Database): Array<{ version: number; description: string; applied_at: string }> {
  try {
    return db.prepare(`
      SELECT version, description, applied_at
      FROM schema_version
      ORDER BY version ASC
    `).all() as Array<{ version: number; description: string; applied_at: string }>;
  } catch {
    return [];
  }
}
