/**
 * Tests for database connection and schema initialization
 * Target coverage: 85%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import {
  getDb,
  initializeSchema,
  ensureVecTable,
  getHashesNeedingEmbedding,
  checkIndexHealth,
} from './db.ts';
import { createTestDb, cleanupDb, getTableNames, tableExists } from '../../tests/fixtures/helpers/test-db.ts';

describe('Database Initialization', () => {
  let db: Database;

  afterEach(() => {
    if (db) {
      cleanupDb(db);
    }
  });

  test('createTestDb creates database with schema', () => {
    db = createTestDb();

    const tables = getTableNames(db);

    expect(tables).toContain('collections');
    expect(tables).toContain('documents');
    expect(tables).toContain('documents_fts');
    expect(tables).toContain('content_vectors');
    expect(tables).toContain('path_contexts');
    expect(tables).toContain('ollama_cache');
  });

  test('initializeSchema creates all required tables', () => {
    db = new Database(':memory:');
    initializeSchema(db);

    expect(tableExists(db, 'collections')).toBe(true);
    expect(tableExists(db, 'documents')).toBe(true);
    expect(tableExists(db, 'documents_fts')).toBe(true);
    expect(tableExists(db, 'content_vectors')).toBe(true);
    expect(tableExists(db, 'path_contexts')).toBe(true);
    expect(tableExists(db, 'ollama_cache')).toBe(true);
  });

  test('initializeSchema is idempotent', () => {
    db = new Database(':memory:');

    // Call multiple times - should not error
    initializeSchema(db);
    initializeSchema(db);
    initializeSchema(db);

    const tables = getTableNames(db);
    expect(tables.length).toBeGreaterThan(0);
  });

  test('collections table has correct schema', () => {
    db = createTestDb();

    const columns = db.prepare(`PRAGMA table_info(collections)`).all() as { name: string; type: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('id');
    expect(columnNames).toContain('pwd');
    expect(columnNames).toContain('glob_pattern');
    expect(columnNames).toContain('created_at');
    expect(columnNames).toContain('context');
  });

  test('documents table has correct schema', () => {
    db = createTestDb();

    const columns = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string; type: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('id');
    expect(columnNames).toContain('collection_id');
    expect(columnNames).toContain('name');
    expect(columnNames).toContain('title');
    expect(columnNames).toContain('hash');
    expect(columnNames).toContain('filepath');
    expect(columnNames).toContain('display_path');
    expect(columnNames).toContain('body');
    expect(columnNames).toContain('active');
    expect(columnNames).toContain('created_at');
    expect(columnNames).toContain('modified_at');
  });

  test('content_vectors table has correct schema', () => {
    db = createTestDb();

    const columns = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('hash');
    expect(columnNames).toContain('seq');
    expect(columnNames).toContain('pos');
    expect(columnNames).toContain('model');
    expect(columnNames).toContain('embedded_at');
  });

  test('documents_fts virtual table exists', () => {
    db = createTestDb();

    expect(tableExists(db, 'documents_fts')).toBe(true);

    // Verify it's an FTS5 table
    const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE name='documents_fts'`).get() as { sql: string } | null;
    expect(tableInfo).not.toBeNull();
    expect(tableInfo?.sql).toContain('fts5');
  });

  test('FTS triggers are created', () => {
    db = createTestDb();

    const triggers = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='trigger'
    `).all() as { name: string }[];

    const triggerNames = triggers.map(t => t.name);

    expect(triggerNames).toContain('documents_ai'); // After insert
    expect(triggerNames).toContain('documents_ad'); // After delete
    expect(triggerNames).toContain('documents_au'); // After update
  });

  test('indices are created correctly', () => {
    db = createTestDb();

    const indices = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'
    `).all() as { name: string }[];

    const indexNames = indices.map(i => i.name);

    expect(indexNames).toContain('idx_documents_collection');
    expect(indexNames).toContain('idx_documents_hash');
    expect(indexNames).toContain('idx_documents_filepath');
    expect(indexNames).toContain('idx_documents_display_path');
    expect(indexNames).toContain('idx_path_contexts_prefix');
  });

  test('display_path migration runs correctly', () => {
    db = new Database(':memory:');

    // First initialization should create display_path
    initializeSchema(db);

    const columns = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
    const hasDisplayPath = columns.some(c => c.name === 'display_path');

    expect(hasDisplayPath).toBe(true);
  });
});

describe('Vector Table Management', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('ensureVecTable creates vectors_vec table', () => {
    ensureVecTable(db, 128);

    expect(tableExists(db, 'vectors_vec')).toBe(true);
  });

  test('ensureVecTable with correct dimensions does not recreate', () => {
    ensureVecTable(db, 128);

    const tableInfo1 = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };

    // Call again with same dimensions - should not recreate
    ensureVecTable(db, 128);

    const tableInfo2 = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };

    expect(tableInfo1.sql).toBe(tableInfo2.sql);
  });

  test('ensureVecTable recreates when dimensions change', () => {
    ensureVecTable(db, 128);

    const tableInfo1 = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };
    expect(tableInfo1.sql).toContain('float[128]');

    // Change dimensions - should recreate
    ensureVecTable(db, 256);

    const tableInfo2 = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };
    expect(tableInfo2.sql).toContain('float[256]');
    expect(tableInfo2.sql).not.toContain('float[128]');
  });

  test('ensureVecTable uses hash_seq as primary key', () => {
    ensureVecTable(db, 128);

    const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };

    expect(tableInfo.sql).toContain('hash_seq');
    expect(tableInfo.sql).toContain('PRIMARY KEY');
  });

  test('ensureVecTable handles different dimension sizes', () => {
    const dimensions = [64, 128, 256, 512, 1024];

    for (const dim of dimensions) {
      ensureVecTable(db, dim);

      const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };
      expect(tableInfo.sql).toContain(`float[${dim}]`);
    }
  });
});

describe('Embedding Status', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();

    // Insert test collection
    db.prepare(`
      INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES (?, ?, ?)
    `).run('/test', '**/*.md', new Date().toISOString());
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('getHashesNeedingEmbedding returns 0 for empty database', () => {
    const count = getHashesNeedingEmbedding(db);
    expect(count).toBe(0);
  });

  test('getHashesNeedingEmbedding counts documents without embeddings', () => {
    const now = new Date().toISOString();

    // Insert documents without embeddings
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc2', 'Doc 2', 'hash2', '/test/doc2.md', 'doc2', 'Content 2', now, now);

    const count = getHashesNeedingEmbedding(db);
    expect(count).toBe(2);
  });

  test('getHashesNeedingEmbedding excludes documents with embeddings', () => {
    const now = new Date().toISOString();

    // Insert document with embedding
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

    db.prepare(`
      INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES (?, 0, 0, ?, ?)
    `).run('hash1', 'test-model', now);

    // Insert document without embedding
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc2', 'Doc 2', 'hash2', '/test/doc2.md', 'doc2', 'Content 2', now, now);

    const count = getHashesNeedingEmbedding(db);
    expect(count).toBe(1); // Only doc2 needs embedding
  });

  test('getHashesNeedingEmbedding only checks seq=0 for first chunk', () => {
    const now = new Date().toISOString();

    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

    // Insert only seq=1 (missing seq=0)
    db.prepare(`
      INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES (?, 1, 1024, ?, ?)
    `).run('hash1', 'test-model', now);

    const count = getHashesNeedingEmbedding(db);
    expect(count).toBe(1); // Still needs embedding (seq=0 missing)
  });

  test('getHashesNeedingEmbedding ignores inactive documents', () => {
    const now = new Date().toISOString();

    // Insert inactive document
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    `).run('doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

    const count = getHashesNeedingEmbedding(db);
    expect(count).toBe(0);
  });
});

describe('Index Health Check', () => {
  let db: Database;

  beforeEach(() => {
    db = createTestDb();

    db.prepare(`
      INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES (?, ?, ?)
    `).run('/test', '**/*.md', new Date().toISOString());
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('checkIndexHealth returns null for healthy index', () => {
    const now = new Date().toISOString();

    // All documents have embeddings
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run('doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

    db.prepare(`
      INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES (?, 0, 0, ?, ?)
    `).run('hash1', 'test-model', now);

    const health = checkIndexHealth(db);
    expect(health).toBeNull();
  });

  test('checkIndexHealth returns warning when many docs need embedding', () => {
    const now = new Date().toISOString();

    // Insert 10 documents without embeddings (100% need embedding)
    for (let i = 0; i < 10; i++) {
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(`doc${i}`, `Doc ${i}`, `hash${i}`, `/test/doc${i}.md`, `doc${i}`, `Content ${i}`, now, now);
    }

    const health = checkIndexHealth(db);

    expect(health).not.toBeNull();
    expect(health).toContain('10 documents');
    expect(health).toContain('100%');
    expect(health).toContain('qmd embed');
  });

  test('checkIndexHealth returns null when few docs need embedding', () => {
    const now = new Date().toISOString();

    // Insert 100 documents with embeddings
    for (let i = 0; i < 100; i++) {
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(`doc${i}`, `Doc ${i}`, `hash${i}`, `/test/doc${i}.md`, `doc${i}`, `Content ${i}`, now, now);

      db.prepare(`
        INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES (?, 0, 0, ?, ?)
      `).run(`hash${i}`, 'test-model', now);
    }

    // Add 5 without embeddings (5% - below 10% threshold)
    for (let i = 100; i < 105; i++) {
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(`doc${i}`, `Doc ${i}`, `hash${i}`, `/test/doc${i}.md`, `doc${i}`, `Content ${i}`, now, now);
    }

    const health = checkIndexHealth(db);
    expect(health).toBeNull(); // Below 10% threshold
  });
});

describe('Schema Migrations', () => {
  let db: Database;

  afterEach(() => {
    if (db) {
      cleanupDb(db);
    }
  });

  test('migration adds display_path column if missing', () => {
    db = new Database(':memory:');

    // Create old schema without display_path
    db.exec(`
      CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        title TEXT NOT NULL,
        hash TEXT NOT NULL,
        filepath TEXT NOT NULL,
        body TEXT NOT NULL,
        created_at TEXT NOT NULL,
        modified_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
      )
    `);

    // Run migration
    initializeSchema(db);

    const columns = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
    const hasDisplayPath = columns.some(c => c.name === 'display_path');

    expect(hasDisplayPath).toBe(true);
  });

  test('migration recreates content_vectors if seq column missing', () => {
    db = new Database(':memory:');

    // Create old schema without seq
    db.exec(`
      CREATE TABLE content_vectors (
        hash TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        embedded_at TEXT NOT NULL
      )
    `);

    // Run migration
    initializeSchema(db);

    const columns = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
    const hasSeq = columns.some(c => c.name === 'seq');

    expect(hasSeq).toBe(true);
  });
});
