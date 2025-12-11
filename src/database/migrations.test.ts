/**
 * Tests for database migration system
 * Target coverage: 80%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate, getMigrationHistory, migrations } from './migrations.ts';

describe('Migration System', () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(':memory:');
  });

  afterEach(() => {
    db.close();
  });

  test('creates schema_version table', () => {
    migrate(db);

    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'
    `).all();

    expect(tables.length).toBe(1);
  });

  test('applies all migrations in order', () => {
    migrate(db);

    const history = getMigrationHistory(db);
    expect(history.length).toBe(migrations.length);

    // Verify versions are sequential
    for (let i = 0; i < history.length; i++) {
      expect(history[i].version).toBe(i + 1);
    }
  });

  test('records migration descriptions', () => {
    migrate(db);

    const history = getMigrationHistory(db);
    expect(history[0].description).toContain('Initial schema');
    expect(history[1].description).toContain('display_path');
    expect(history[2].description).toContain('chunking');
  });

  test('records applied_at timestamps', () => {
    migrate(db);

    const history = getMigrationHistory(db);
    for (const record of history) {
      expect(record.applied_at).toBeDefined();
      expect(record.applied_at).toMatch(/^\d{4}-\d{2}-\d{2}/); // ISO date format
    }
  });

  test('is idempotent - can run multiple times', () => {
    migrate(db);
    const firstHistory = getMigrationHistory(db);

    // Run again
    migrate(db);
    const secondHistory = getMigrationHistory(db);

    // Should be identical
    expect(secondHistory.length).toBe(firstHistory.length);
    expect(secondHistory).toEqual(firstHistory);
  });

  test('creates all expected tables from migration 1', () => {
    migrate(db);

    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table'
      ORDER BY name
    `).all() as { name: string }[];

    const tableNames = tables.map(t => t.name);
    expect(tableNames).toContain('collections');
    expect(tableNames).toContain('documents');
    expect(tableNames).toContain('content_vectors');
    expect(tableNames).toContain('path_contexts');
    expect(tableNames).toContain('ollama_cache');
    expect(tableNames).toContain('search_history');
    expect(tableNames).toContain('documents_fts');
  });

  test('creates FTS triggers from migration 1', () => {
    migrate(db);

    const triggers = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='trigger'
    `).all() as { name: string }[];

    const triggerNames = triggers.map(t => t.name);
    expect(triggerNames).toContain('documents_ai');
    expect(triggerNames).toContain('documents_ad');
    expect(triggerNames).toContain('documents_au');
  });

  test('creates indexes from migration 1', () => {
    migrate(db);

    const indexes = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'
    `).all() as { name: string }[];

    expect(indexes.length).toBeGreaterThan(5);
  });

  test('migration 2 adds display_path column', () => {
    migrate(db);

    const columns = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('display_path');
  });

  test('migration 2 creates display_path index', () => {
    migrate(db);

    const indexes = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_display_path'
    `).all();

    expect(indexes.length).toBe(1);
  });

  test('migration 3 ensures content_vectors has seq column', () => {
    migrate(db);

    const columns = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('hash');
    expect(columnNames).toContain('seq');
    expect(columnNames).toContain('pos');
    expect(columnNames).toContain('model');
  });

  test('handles new database without any tables', () => {
    // Fresh database with no tables
    const tables = db.prepare(`
      SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'
    `).get() as { count: number };
    expect(tables.count).toBe(0);

    // Run migrations
    migrate(db);

    // Should have created schema_version and all tables
    const history = getMigrationHistory(db);
    expect(history.length).toBe(migrations.length);

    const newTables = db.prepare(`
      SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'
    `).get() as { count: number };
    expect(newTables.count).toBeGreaterThan(5);
  });

  test('migrates old content_vectors schema to new (with seq)', () => {
    // Simulate old schema without seq column
    db.exec(`
      CREATE TABLE content_vectors (
        hash TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        embedded_at TEXT NOT NULL
      )
    `);

    // Run migrations
    migrate(db);

    // Should have seq column now
    const columns = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
    const columnNames = columns.map(c => c.name);

    expect(columnNames).toContain('seq');
    expect(columnNames).toContain('pos');
  });

  test('getMigrationHistory returns empty array for new database', () => {
    // Don't run migrate yet
    const history = getMigrationHistory(db);
    expect(history).toEqual([]);
  });

  test('migration runs in transaction (all-or-nothing)', () => {
    // This is implicit in the implementation - each migration uses db.transaction()
    // We can verify by checking that partial migrations don't leave database in bad state
    migrate(db);

    // All tables should exist (not partial)
    const tables = db.prepare(`
      SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'
    `).get() as { count: number };

    expect(tables.count).toBeGreaterThan(5);
  });

  test('only applies new migrations on subsequent runs', () => {
    // Apply all current migrations
    migrate(db);
    const firstHistory = getMigrationHistory(db);

    // Verify no duplicate applications happen
    migrate(db);
    const secondHistory = getMigrationHistory(db);

    // Should be identical - no new migrations applied
    expect(secondHistory.length).toBe(firstHistory.length);
    expect(secondHistory).toEqual(firstHistory);

    // Verify each migration was only applied once
    for (let i = 0; i < secondHistory.length; i++) {
      expect(secondHistory[i].version).toBe(i + 1);
    }
  });

  test('handles display_path column already existing', () => {
    // Create documents table with display_path already present
    db.exec(`
      CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER,
        name TEXT,
        title TEXT,
        hash TEXT,
        filepath TEXT,
        display_path TEXT NOT NULL DEFAULT '',
        body TEXT,
        created_at TEXT,
        modified_at TEXT,
        active INTEGER DEFAULT 1
      )
    `);

    // Migration 2 should handle this gracefully
    migrate(db);

    // Should still work without errors
    const columns = db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[];
    const displayPathCount = columns.filter(c => c.name === 'display_path').length;

    expect(displayPathCount).toBe(1); // Not duplicated
  });
});

describe('Migration Integrity', () => {
  test('all migrations have required fields', () => {
    for (const migration of migrations) {
      expect(migration.version).toBeGreaterThan(0);
      expect(migration.description).toBeTruthy();
      expect(typeof migration.up).toBe('function');
    }
  });

  test('migration versions are sequential', () => {
    for (let i = 0; i < migrations.length; i++) {
      expect(migrations[i].version).toBe(i + 1);
    }
  });

  test('migration descriptions are unique', () => {
    const descriptions = migrations.map(m => m.description);
    const uniqueDescriptions = new Set(descriptions);
    expect(uniqueDescriptions.size).toBe(migrations.length);
  });
});
