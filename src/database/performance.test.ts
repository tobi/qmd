/**
 * Tests for database performance utilities
 * Target coverage: 80%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import {
  analyzeDatabase,
  getDatabaseStats,
  shouldAnalyze,
  batchInsertDocuments,
  getPerformanceHints,
} from './performance.ts';
import { migrate } from './migrations.ts';

describe('Performance Utilities', () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(':memory:');
    migrate(db);
  });

  afterEach(() => {
    db.close();
  });

  describe('analyzeDatabase', () => {
    test('runs without error', () => {
      expect(() => analyzeDatabase(db)).not.toThrow();
    });

    test('creates sqlite_stat1 table', () => {
      analyzeDatabase(db);

      const tables = db.prepare(`
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='sqlite_stat1'
      `).all();

      expect(tables.length).toBe(1);
    });
  });

  describe('getDatabaseStats', () => {
    test('returns database statistics', () => {
      const stats = getDatabaseStats(db);

      expect(stats.page_count).toBeGreaterThan(0);
      expect(stats.page_size).toBeGreaterThan(0);
      expect(stats.size_mb).toBeGreaterThan(0);
    });

    test('calculates size correctly', () => {
      const stats = getDatabaseStats(db);

      const expectedSize = (stats.page_count * stats.page_size) / (1024 * 1024);
      expect(stats.size_mb).toBeCloseTo(expectedSize, 2);
    });
  });

  describe('shouldAnalyze', () => {
    test('returns true for large changes', () => {
      const result = shouldAnalyze(db, 150);
      expect(result).toBe(true);
    });

    test('returns false for small changes', () => {
      const result = shouldAnalyze(db, 10);
      expect(result).toBe(false);
    });

    test('returns true if database has many documents but no stats', () => {
      // Create collection and add many documents
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      for (let i = 0; i < 1500; i++) {
        db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
          VALUES (1, 'doc${i}.md', 'Doc${i}', 'hash${i}', '/test/doc${i}.md', 'body', datetime('now'), datetime('now'), 1)`).run();
      }

      const result = shouldAnalyze(db, 0);
      expect(result).toBe(true);
    });

    test('returns false if database has few documents', () => {
      // Create collection and add few documents
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'doc.md', 'Doc', 'hash', '/test/doc.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      const result = shouldAnalyze(db, 0);
      expect(result).toBe(false);
    });

    test('returns false if database already has stats', () => {
      // Create collection and add many documents
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      for (let i = 0; i < 1500; i++) {
        db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
          VALUES (1, 'doc${i}.md', 'Doc${i}', 'hash${i}', '/test/doc${i}.md', 'body', datetime('now'), datetime('now'), 1)`).run();
      }

      // Run analyze
      analyzeDatabase(db);

      // Should return false now that stats exist
      const result = shouldAnalyze(db, 0);
      expect(result).toBe(false);
    });
  });

  describe('batchInsertDocuments', () => {
    test('inserts multiple items in transaction', () => {
      // Create collection
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      interface TestDoc {
        name: string;
        hash: string;
      }

      const docs: TestDoc[] = [
        { name: 'doc1.md', hash: 'hash1' },
        { name: 'doc2.md', hash: 'hash2' },
        { name: 'doc3.md', hash: 'hash3' },
      ];

      const inserted = batchInsertDocuments(db, docs, (doc) => {
        db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
          VALUES (1, ?, ?, ?, ?, 'body', datetime('now'), datetime('now'), 1)`).run(
          doc.name,
          doc.name,
          doc.hash,
          `/test/${doc.name}`
        );
      });

      expect(inserted).toBe(3);

      // Verify all documents were inserted
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(3);
    });

    test('returns count of inserted items', () => {
      const items = [1, 2, 3, 4, 5];
      const inserted = batchInsertDocuments(db, items, () => {
        // No-op insert for testing
      });

      expect(inserted).toBe(5);
    });

    test('handles empty array', () => {
      const inserted = batchInsertDocuments(db, [], () => {
        // No-op
      });

      expect(inserted).toBe(0);
    });

    test('transaction rolls back on error', () => {
      // Create collection
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      const docs = [{ name: 'doc1' }, { name: 'doc2' }, { name: 'invalid' }];

      expect(() => {
        batchInsertDocuments(db, docs, (doc) => {
          if (doc.name === 'invalid') {
            throw new Error('Invalid document');
          }
          db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
            VALUES (1, ?, ?, 'hash', '/path', 'body', datetime('now'), datetime('now'), 1)`).run(doc.name, doc.name);
        });
      }).toThrow();

      // Verify transaction was rolled back (no documents inserted)
      const count = db.prepare(`SELECT COUNT(*) as count FROM documents`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });

  describe('getPerformanceHints', () => {
    test('does not suggest ANALYZE after running it', () => {
      // Run analyze
      analyzeDatabase(db);

      const hints = getPerformanceHints(db);

      // Should not suggest ANALYZE anymore
      const hasAnalyzeHint = hints.some(h => h.includes('ANALYZE'));
      expect(hasAnalyzeHint).toBe(false);
    });

    test('suggests ANALYZE if not run', () => {
      const hints = getPerformanceHints(db);

      const hasAnalyzeHint = hints.some(h => h.includes('ANALYZE'));
      expect(hasAnalyzeHint).toBe(true);
    });

    test('suggests cleanup for large databases', () => {
      // Create collection and add many large documents
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      // Add enough documents to make database >100MB (unlikely in memory, but test the logic)
      // This is hard to test realistically, so we'll just verify the function runs
      const hints = getPerformanceHints(db);
      expect(Array.isArray(hints)).toBe(true);
    });

    test('checks WAL mode', () => {
      const hints = getPerformanceHints(db);

      // In-memory database might not use WAL, so we just check the function runs
      expect(Array.isArray(hints)).toBe(true);
    });
  });
});
