/**
 * Tests for database integrity checks
 * Target coverage: 80%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import {
  checkOrphanedVectors,
  checkPartialEmbeddings,
  checkDisplayPathCollisions,
  checkOrphanedDocuments,
  checkFTSConsistency,
  checkStaleDocuments,
  checkMissingVecTableEntries,
  runAllIntegrityChecks,
  autoFixIssues,
} from './integrity.ts';
import { migrate } from './migrations.ts';

describe('Integrity Checks', () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(':memory:');
    migrate(db);
  });

  afterEach(() => {
    db.close();
  });

  describe('checkOrphanedVectors', () => {
    test('returns null when no orphaned vectors', () => {
      const issue = checkOrphanedVectors(db);
      expect(issue).toBeNull();
    });

    test('detects orphaned vectors', () => {
      // Add orphaned vectors
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('orphan1', 0, 0, 'test-model', datetime('now'))`).run();
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('orphan1', 1, 100, 'test-model', datetime('now'))`).run();

      const issue = checkOrphanedVectors(db);
      expect(issue).not.toBeNull();
      expect(issue?.severity).toBe('warning');
      expect(issue?.type).toBe('orphaned_vectors');
      expect(issue?.fixable).toBe(true);
      expect(issue?.message).toContain('1 orphaned');
    });

    test('fix removes orphaned vectors', () => {
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('orphan1', 0, 0, 'test-model', datetime('now'))`).run();

      const issue = checkOrphanedVectors(db);
      issue?.fix!();

      const count = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });

  describe('checkPartialEmbeddings', () => {
    test('returns null when all embeddings are complete', () => {
      const issue = checkPartialEmbeddings(db);
      expect(issue).toBeNull();
    });

    test('detects partial embeddings (missing seq 0)', () => {
      // Add partial embedding starting at seq 1
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('partial1', 1, 100, 'test-model', datetime('now'))`).run();

      const issue = checkPartialEmbeddings(db);
      expect(issue).not.toBeNull();
      expect(issue?.severity).toBe('error');
      expect(issue?.type).toBe('partial_embeddings');
      expect(issue?.fixable).toBe(true);
    });

    test('detects partial embeddings (gap in sequence)', () => {
      // Add embedding with gap: 0, 1, 3 (missing 2)
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('partial2', 0, 0, 'test-model', datetime('now'))`).run();
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('partial2', 1, 100, 'test-model', datetime('now'))`).run();
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('partial2', 3, 300, 'test-model', datetime('now'))`).run();

      const issue = checkPartialEmbeddings(db);
      expect(issue).not.toBeNull();
      expect(issue?.message).toContain('incomplete chunk sequences');
    });

    test('fix removes partial embeddings', () => {
      db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
        VALUES ('partial1', 1, 100, 'test-model', datetime('now'))`).run();

      const issue = checkPartialEmbeddings(db);
      issue?.fix!();

      const count = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
      expect(count.count).toBe(0);
    });
  });

  describe('checkDisplayPathCollisions', () => {
    test('returns null when no collisions', () => {
      const issue = checkDisplayPathCollisions(db);
      expect(issue).toBeNull();
    });

    test('UNIQUE index prevents display path collisions', () => {
      // Create collection
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      // Add first document
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (1, 'doc1.md', 'Doc1', 'hash1', '/test/doc1.md', 'common', 'body', datetime('now'), datetime('now'), 1)`).run();

      // Try to add second document with same display_path - should fail
      expect(() => {
        db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
          VALUES (1, 'doc2.md', 'Doc2', 'hash2', '/test/doc2.md', 'common', 'body', datetime('now'), datetime('now'), 1)`).run();
      }).toThrow('UNIQUE constraint failed');

      // Since UNIQUE index prevents collisions, check should pass
      const issue = checkDisplayPathCollisions(db);
      expect(issue).toBeNull();
    });
  });

  describe('checkOrphanedDocuments', () => {
    test('returns null when no orphaned documents', () => {
      const issue = checkOrphanedDocuments(db);
      expect(issue).toBeNull();
    });

    test('detects orphaned documents', () => {
      // Add document with non-existent collection_id
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (999, 'orphan.md', 'Orphan', 'hash1', '/test/orphan.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      const issue = checkOrphanedDocuments(db);
      expect(issue).not.toBeNull();
      expect(issue?.severity).toBe('error');
      expect(issue?.type).toBe('orphaned_documents');
      expect(issue?.fixable).toBe(true);
    });

    test('fix deactivates orphaned documents', () => {
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (999, 'orphan.md', 'Orphan', 'hash1', '/test/orphan.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      const issue = checkOrphanedDocuments(db);
      issue?.fix!();

      const doc = db.prepare(`SELECT active FROM documents WHERE collection_id = 999`).get() as { active: number };
      expect(doc.active).toBe(0);
    });
  });

  describe('checkFTSConsistency', () => {
    test('returns null when FTS is consistent', () => {
      // FTS triggers should handle this automatically
      const issue = checkFTSConsistency(db);
      expect(issue).toBeNull();
    });

    test('returns null with documents and FTS in sync', () => {
      // Create collection and document (triggers will keep FTS in sync)
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'doc.md', 'Doc', 'hash1', '/test/doc.md', 'body', datetime('now'), datetime('now'), 1)`).run();

      // Should be consistent thanks to triggers
      const issue = checkFTSConsistency(db);
      expect(issue).toBeNull();
    });

    test('rebuild command is available', () => {
      // Test that FTS rebuild command works
      db.exec(`INSERT INTO documents_fts(documents_fts) VALUES('rebuild')`);

      // Should still be consistent
      const issue = checkFTSConsistency(db);
      expect(issue).toBeNull();
    });
  });

  describe('checkStaleDocuments', () => {
    test('returns null when no stale documents', () => {
      const issue = checkStaleDocuments(db);
      expect(issue).toBeNull();
    });

    test('detects stale documents', () => {
      // Create collection
      db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
        VALUES ('/test', '*.md', datetime('now'))`).run();

      // Add stale inactive document (>90 days old)
      db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
        VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-100 days'), datetime('now', '-100 days'), 0)`).run();

      const issue = checkStaleDocuments(db);
      expect(issue).not.toBeNull();
      expect(issue?.severity).toBe('info');
      expect(issue?.type).toBe('stale_documents');
      expect(issue?.fixable).toBe(false); // Requires explicit cleanup command
    });
  });

  describe('checkMissingVecTableEntries', () => {
    test('returns null when vec table does not exist', () => {
      const issue = checkMissingVecTableEntries(db);
      expect(issue).toBeNull();
    });

    test('returns null when both tables are empty', () => {
      // Both content_vectors and vectors_vec don't exist or are empty
      const issue = checkMissingVecTableEntries(db);
      expect(issue).toBeNull();
    });
  });
});

describe('Integration', () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(':memory:');
    migrate(db);
  });

  afterEach(() => {
    db.close();
  });

  test('runAllIntegrityChecks returns empty array for clean database', () => {
    const issues = runAllIntegrityChecks(db);
    expect(issues).toEqual([]);
  });

  test('runAllIntegrityChecks detects multiple issues', () => {
    // Add orphaned vector
    db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES ('orphan1', 0, 0, 'test-model', datetime('now'))`).run();

    // Add orphaned document
    db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
      VALUES (999, 'orphan.md', 'Orphan', 'hash1', '/test/orphan.md', 'body', datetime('now'), datetime('now'), 1)`).run();

    const issues = runAllIntegrityChecks(db);
    expect(issues.length).toBeGreaterThanOrEqual(2);

    const types = issues.map(i => i.type);
    expect(types).toContain('orphaned_vectors');
    expect(types).toContain('orphaned_documents');
  });

  test('autoFixIssues fixes fixable issues', () => {
    // Add orphaned vector
    db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES ('orphan1', 0, 0, 'test-model', datetime('now'))`).run();

    const issues = runAllIntegrityChecks(db);
    const fixed = autoFixIssues(db, issues);

    expect(fixed).toBeGreaterThan(0);

    // Verify issue is fixed
    const remaining = runAllIntegrityChecks(db);
    expect(remaining.length).toBeLessThan(issues.length);
  });

  test('autoFixIssues skips non-fixable issues', () => {
    // Create collection
    db.prepare(`INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES ('/test', '*.md', datetime('now'))`).run();

    // Add stale document (not fixable)
    db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, body, created_at, modified_at, active)
      VALUES (1, 'old.md', 'Old', 'hash1', '/test/old.md', 'body', datetime('now', '-100 days'), datetime('now', '-100 days'), 0)`).run();

    const issues = runAllIntegrityChecks(db);
    const fixed = autoFixIssues(db, issues);

    expect(fixed).toBe(0); // Stale documents are not auto-fixable
  });

  test('autoFixIssues runs in transaction', () => {
    // Add multiple orphaned vectors
    db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES ('orphan1', 0, 0, 'test-model', datetime('now'))`).run();
    db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES ('orphan2', 0, 0, 'test-model', datetime('now'))`).run();

    const issues = runAllIntegrityChecks(db);
    const fixed = autoFixIssues(db, issues);

    expect(fixed).toBeGreaterThan(0);

    // All should be fixed
    const count = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
    expect(count.count).toBe(0);
  });
});
