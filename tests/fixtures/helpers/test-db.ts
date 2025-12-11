/**
 * Test database utilities
 * Provides helpers for creating in-memory test databases with schema
 */

import { Database } from 'bun:sqlite';
import * as sqliteVec from 'sqlite-vec';
import { initializeSchema } from '../../../src/database/db.ts';

/**
 * Create an in-memory test database with full schema
 * @returns Database instance with schema initialized
 */
export function createTestDb(): Database {
  const db = new Database(':memory:');

  // Load sqlite-vec extension
  sqliteVec.load(db);

  // Enable WAL mode for better concurrency
  db.exec('PRAGMA journal_mode = WAL');

  // Initialize schema (creates all tables, indices, triggers)
  initializeSchema(db);

  return db;
}

/**
 * Create test database with sample data
 * @returns Database instance with schema and sample data
 */
export function createTestDbWithData(): Database {
  const db = createTestDb();

  // Insert sample collection
  const collectionId = db.prepare(`
    INSERT INTO collections (pwd, glob_pattern, created_at)
    VALUES (?, ?, ?)
  `).run('/test/path', '**/*.md', new Date().toISOString()).lastInsertRowid as number;

  // Insert sample documents
  const now = new Date().toISOString();

  db.prepare(`
    INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
  `).run(
    collectionId,
    'test-doc',
    'Test Document',
    'hash123',
    '/test/path/test-doc.md',
    'test-doc',
    '# Test Document\n\nThis is a test document.',
    now,
    now
  );

  db.prepare(`
    INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
  `).run(
    collectionId,
    'another-doc',
    'Another Document',
    'hash456',
    '/test/path/another-doc.md',
    'another-doc',
    '# Another Document\n\nThis is another test document.',
    now,
    now
  );

  return db;
}

/**
 * Clean up test database
 * @param db - Database instance to close
 */
export function cleanupDb(db: Database): void {
  if (db) {
    db.close();
  }
}

/**
 * Create test database with vectors
 * @param dimensions - Vector dimensions (default 128)
 * @returns Database instance with vectors table
 */
export function createTestDbWithVectors(dimensions: number = 128): Database {
  const db = createTestDbWithData();

  // Create vectors table
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec
    USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}])
  `);

  // Insert sample vector metadata
  const now = new Date().toISOString();
  db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)`).run(
    'hash123',
    0,
    0,
    'test-model',
    now
  );

  // Insert actual vector into vectors_vec table
  const embedding = new Float32Array(dimensions).fill(0.1);
  const embeddingBytes = new Uint8Array(embedding.buffer);

  db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(
    'hash123_0',
    embeddingBytes
  );

  return db;
}

/**
 * Get table names from database
 * @param db - Database instance
 * @returns Array of table names
 */
export function getTableNames(db: Database): string[] {
  const tables = db.prepare(`
    SELECT name FROM sqlite_master
    WHERE type='table'
    ORDER BY name
  `).all() as { name: string }[];

  return tables.map(t => t.name);
}

/**
 * Verify table exists
 * @param db - Database instance
 * @param tableName - Table name to check
 * @returns True if table exists
 */
export function tableExists(db: Database, tableName: string): boolean {
  const result = db.prepare(`
    SELECT name FROM sqlite_master
    WHERE type='table' AND name=?
  `).get(tableName);

  return result !== null;
}

/**
 * Get row count for a table
 * @param db - Database instance
 * @param tableName - Table name
 * @returns Number of rows
 */
export function getRowCount(db: Database, tableName: string): number {
  const result = db.prepare(`SELECT COUNT(*) as count FROM ${tableName}`).get() as { count: number };
  return result.count;
}
