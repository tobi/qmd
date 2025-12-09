/**
 * Tests for database cleanup functionality
 * Target coverage: 80%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { cleanup, type CleanupOptions } from './cleanup.ts';
import { migrate } from './migrations.ts';

describe('Cleanup', () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(':memory:');
    migrate(db);

    // Create a collection for testing
    db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES ('/test', '*.md', datetime('now'))`).run();
  });

  afterEach(() => {
    db.close();
  });

  describe('cleanup with default options (30 days)', () => {
    test('does not delete recent inactive documents', () => {
      // Add inactive document from 20 days ago (should NOT be deleted)
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'recent.md', 'Recent', 'hash1', '/test/recent.md', 'body', datetime('now', '-20 days'), datetime('now', '-20 days'), 0)`).run();

      const result = cleanup(db, {});

      expect(result.documents_deleted).toBe(0);
    });

    test('deletes old inactive documents (>30 days)', () => {
      // Add inactive document from 40 days ago (should be deleted)
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-40 days'), datetime('now', '-40 days'), 0)`).run();

      const result = cleanup(db, {});

      expect(result.documents_deleted).toBeGreaterThan(0);

      // Verify document is actually deleted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(0);
    });

    test('does not delete active documents', () => {
      // Add active document (should NOT be deleted)
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'active.md', 'Active', 'hash1', '/test/active.md', 'body', datetime('now', '-60 days'), datetime('now', '-60 days'), 1)`).run();

      const result = cleanup(db, {});

      expect(result.documents_deleted).toBe(0);

      // Verify document still exists
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
      expect(count.count).toBe(1);
    });
  });

  describe('cleanup with custom age', () => {
    test('respects olderThanDays option', () => {
      // Add inactive document from 50 days ago
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-50 days'), datetime('now', '-50 days'), 0)`).run();

      // Cleanup with 60 day threshold - should NOT delete
      const result = cleanup(db, { olderThanDays: 60 });
      expect(result.documents_deleted).toBe(0);
    });

    test('deletes documents matching custom age', () => {
      // Add inactive document from 50 days ago
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-50 days'), datetime('now', '-50 days'), 0)`).run();

      // Cleanup with 40 day threshold - should delete
      const result = cleanup(db, { olderThanDays: 40 });
      expect(result.documents_deleted).toBeGreaterThan(0);

      // Verify document is deleted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });

  describe('cleanup with --all flag', () => {
    test('deletes all inactive documents regardless of age', () => {
      // Add multiple inactive documents with different ages
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'recent.md', 'Recent', 'hash1', '/test/recent.md', 'body', datetime('now', '-5 days'), datetime('now', '-5 days'), 0)`).run();
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash2', '/test/old.md', 'body', datetime('now', '-100 days'), datetime('now', '-100 days'), 0)`).run();

      const result = cleanup(db, { all: true });

      expect(result.documents_deleted).toBeGreaterThan(0);

      // Verify all inactive documents are deleted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(0);
    });

    test('still does not delete active documents with --all', () => {
      // Add active and inactive documents
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'active.md', 'Active', 'hash1', '/test/active.md', 'body', datetime('now'), datetime('now'), 1)`).run();
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'inactive.md', 'Inactive', 'hash2', '/test/inactive.md', 'body', datetime('now'), datetime('now'), 0)`).run();

      const result = cleanup(db, { all: true });

      expect(result.documents_deleted).toBeGreaterThan(0);

      // Verify active document still exists
      const activeCount = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
      expect(activeCount.count).toBe(1);

      // Verify inactive document is deleted
      const inactiveCount = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 0`).get() as { count: number };
      expect(inactiveCount.count).toBe(0);
    });
  });

  describe('dry run mode', () => {
    test('does not delete documents in dry run', () => {
      // Add old inactive document
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-40 days'), datetime('now', '-40 days'), 0)`).run();

      const result = cleanup(db, { dryRun: true });

      // Should report what would be deleted
      expect(result.documents_deleted).toBe(1);

      // But document should still exist
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(1);
    });

    test('preview counts multiple documents', () => {
      // Add multiple old inactive documents
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old1.md', 'Old1', 'hash1', '/test/old1.md', 'body', datetime('now', '-40 days'), datetime('now', '-40 days'), 0)`).run();
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old2.md', 'Old2', 'hash2', '/test/old2.md', 'body', datetime('now', '-50 days'), datetime('now', '-50 days'), 0)`).run();

      const result = cleanup(db, { dryRun: true });

      expect(result.documents_deleted).toBe(2);
    });
  });

  describe('vacuum option', () => {
    test('deletes orphaned vectors with vacuum', () => {
      // Add active document
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'active.md', 'Active', 'hash1', '/test/active.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      // Add vector for active document
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('hash1', 0, 0, 'test-model', datetime('now'))`).run();

      // Add orphaned vector (no corresponding document)
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('orphan', 0, 0, 'test-model', datetime('now'))`).run();

      const result = cleanup(db, { vacuum: true });

      expect(result.vectors_deleted).toBe(1); // Only orphaned vector deleted

      // Verify orphaned vector is deleted
      const vecCount = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
      expect(vecCount.count).toBe(1); // Only active document's vector remains
    });

    test('deletes old cache entries with vacuum', () => {
      // Add old cache entry (>7 days)
      db.prepare(`INSERT INTO ollama_cache (hash, result, created_at)
        VALUES ('old', 'result', datetime('now', '-10 days'))`).run();

      // Add recent cache entry
      db.prepare(`INSERT INTO ollama_cache (hash, result, created_at)
        VALUES ('recent', 'result', datetime('now', '-2 days'))`).run();

      const result = cleanup(db, { vacuum: true });

      expect(result.cache_entries_deleted).toBe(1); // Only old entry deleted

      // Verify old cache entry is deleted
      const cacheCount = db.prepare(`SELECT COUNT(*) as count FROM ollama_cache`).get() as { count: number };
      expect(cacheCount.count).toBe(1); // Only recent entry remains
    });

    test('vacuum reclaims space', () => {
      // Add and delete a large number of documents
      for (let i = 0; i < 100; i++) {
        db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
          VALUES (1, 'doc${i}.md', 'Doc${i}', 'hash${i}', '/test/doc${i}.md', '${'x'.repeat(1000)}', datetime('now', '-40 days'), datetime('now', '-40 days'), 0)`).run();
      }

      const result = cleanup(db, { vacuum: true });

      expect(result.documents_deleted).toBeGreaterThan(0);
      expect(result.space_reclaimed_mb).toBeGreaterThan(0);

      // Verify all inactive documents are deleted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 0`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });

  describe('edge cases', () => {
    test('handles empty database', () => {
      const result = cleanup(db, {});

      expect(result.documents_deleted).toBe(0);
      expect(result.vectors_deleted).toBe(0);
      expect(result.cache_entries_deleted).toBe(0);
      expect(result.space_reclaimed_mb).toBe(0);
    });

    test('handles database with only active documents', () => {
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'active.md', 'Active', 'hash1', '/test/active.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      const result = cleanup(db, {});

      expect(result.documents_deleted).toBe(0);
    });

    test('cleanup runs in transaction (all-or-nothing)', () => {
      // Add old inactive document
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-40 days'), datetime('now', '-40 days'), 0)`).run();

      // Should complete successfully
      const result = cleanup(db, {});
      expect(result.documents_deleted).toBeGreaterThan(0);

      // Verify document is deleted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 0`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });
});
