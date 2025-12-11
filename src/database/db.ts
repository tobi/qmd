/**
 * Database connection and schema initialization
 */

import { Database } from 'bun:sqlite';
import * as sqliteVec from 'sqlite-vec';
import { getDbPath } from '../utils/paths.ts';
import { migrate } from './migrations.ts';

/**
 * Initialize and return database connection with schema
 * @param indexName - Name of the index (default: "index")
 * @returns Database instance with schema initialized
 */
export function getDb(indexName: string = "index"): Database {
  const db = new Database(getDbPath(indexName));
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");

  migrate(db);
  return db;
}

/**
 * @deprecated Use migrate() from migrations.ts instead
 * This function is kept for backward compatibility but is no longer used.
 * The migration system in migrations.ts provides better versioning and control.
 */
export function initializeSchema(db: Database): void {
  migrate(db);
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
