/**
 * Tests for PathContext Repository
 * Target coverage: 80%+ with MANDATORY SQL injection tests
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { PathContextRepository } from './path-contexts.ts';
import { createTestDb, cleanupDb } from '../../../tests/fixtures/helpers/test-db.ts';
import { sqlInjectionPayloads } from '../../../tests/fixtures/helpers/fixtures.ts';

describe('PathContextRepository', () => {
  let db: Database;
  let repo: PathContextRepository;

  beforeEach(() => {
    db = createTestDb();
    repo = new PathContextRepository(db);
  });

  afterEach(() => {
    cleanupDb(db);
  });

  describe('findForPath', () => {
    test('finds exact path prefix match', () => {
      repo.upsert('/home/user/docs', 'Documentation directory');

      const context = repo.findForPath('/home/user/docs/readme.md');

      expect(context).not.toBeNull();
      expect(context?.path_prefix).toBe('/home/user/docs');
      expect(context?.context).toBe('Documentation directory');
    });

    test('finds longest matching prefix', () => {
      repo.upsert('/home/user', 'User home');
      repo.upsert('/home/user/docs', 'Documentation');
      repo.upsert('/home/user/docs/api', 'API docs');

      const context = repo.findForPath('/home/user/docs/api/endpoints.md');

      // Should match the longest prefix
      expect(context?.path_prefix).toBe('/home/user/docs/api');
    });

    test('returns null when no match', () => {
      repo.upsert('/home/user/docs', 'Documentation');

      const context = repo.findForPath('/other/path/file.md');

      expect(context).toBeNull();
    });

    test('handles paths with trailing slashes', () => {
      repo.upsert('/home/user/docs/', 'Documentation');

      const context = repo.findForPath('/home/user/docs/file.md');

      expect(context).not.toBeNull();
    });
  });

  describe('findAll', () => {
    test('returns all path contexts', () => {
      repo.upsert('/path1', 'Context 1');
      repo.upsert('/path2', 'Context 2');
      repo.upsert('/path3', 'Context 3');

      const contexts = repo.findAll();

      expect(contexts).toHaveLength(3);
    });

    test('returns empty array when no contexts', () => {
      const contexts = repo.findAll();
      expect(contexts).toHaveLength(0);
    });

    test('orders by path_prefix', () => {
      repo.upsert('/z/path', 'Z');
      repo.upsert('/a/path', 'A');
      repo.upsert('/m/path', 'M');

      const contexts = repo.findAll();

      expect(contexts[0].path_prefix).toBe('/a/path');
      expect(contexts[1].path_prefix).toBe('/m/path');
      expect(contexts[2].path_prefix).toBe('/z/path');
    });
  });

  describe('upsert', () => {
    test('inserts new path context', () => {
      repo.upsert('/test/path', 'Test context');

      const context = repo.findForPath('/test/path/file.md');

      expect(context).not.toBeNull();
      expect(context?.path_prefix).toBe('/test/path');
      expect(context?.context).toBe('Test context');
    });

    test('updates existing path context', () => {
      repo.upsert('/test/path', 'Original context');
      repo.upsert('/test/path', 'Updated context');

      const contexts = repo.findAll();

      expect(contexts).toHaveLength(1);
      expect(contexts[0].context).toBe('Updated context');
    });

    test('handles empty context text', () => {
      repo.upsert('/test/path', '');

      const context = repo.findForPath('/test/path/file.md');

      expect(context).not.toBeNull();
      expect(context?.context).toBe('');
    });
  });

  describe('delete', () => {
    test('deletes path context', () => {
      repo.upsert('/test/path', 'Test context');
      expect(repo.count()).toBe(1);

      repo.delete('/test/path');

      expect(repo.count()).toBe(0);
    });

    test('deleting non-existent path does not error', () => {
      expect(() => {
        repo.delete('/nonexistent');
      }).not.toThrow();
    });
  });

  describe('count', () => {
    test('returns correct count', () => {
      expect(repo.count()).toBe(0);

      repo.upsert('/path1', 'Context 1');
      expect(repo.count()).toBe(1);

      repo.upsert('/path2', 'Context 2');
      expect(repo.count()).toBe(2);
    });

    test('count decreases after delete', () => {
      repo.upsert('/test', 'Test');
      expect(repo.count()).toBe(1);

      repo.delete('/test');
      expect(repo.count()).toBe(0);
    });

    test('upsert does not increase count on update', () => {
      repo.upsert('/test', 'Original');
      expect(repo.count()).toBe(1);

      repo.upsert('/test', 'Updated');
      expect(repo.count()).toBe(1);
    });
  });
});

describe('SQL Injection Prevention', () => {
  let db: Database;
  let repo: PathContextRepository;

  beforeEach(() => {
    db = createTestDb();
    repo = new PathContextRepository(db);

    // Insert test data
    repo.upsert('/safe/path', 'Safe context');
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('findForPath handles malicious input safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.findForPath(payload);
      }).not.toThrow();
    }

    // Verify table still exists
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='path_contexts'
    `).all();
    expect(tables).toHaveLength(1);

    // Verify original data intact
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('upsert handles malicious pathPrefix safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.upsert(payload, 'Test context');
      }).not.toThrow();
    }

    // Verify database integrity
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('upsert handles malicious contextText safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.upsert('/test/path', payload);
      }).not.toThrow();
    }

    // Verify database integrity
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('delete handles malicious input safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.delete(payload);
      }).not.toThrow();
    }

    // Verify original data intact
    const context = repo.findForPath('/safe/path/file.md');
    expect(context).not.toBeNull();
  });

  test('uses prepared statements for all queries', () => {
    const maliciousPath = "'; DROP TABLE path_contexts; --";

    repo.findForPath(maliciousPath);
    repo.upsert(maliciousPath, 'Test');
    repo.delete(maliciousPath);

    // If prepared statements are used, table should still exist
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='path_contexts'
    `).all();

    expect(tables).toHaveLength(1);

    // Original data should be intact
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });
});
