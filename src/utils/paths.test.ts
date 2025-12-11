/**
 * Tests for path utility functions
 * Target coverage: 90%+
 */

import { describe, test, expect, beforeEach } from 'bun:test';
import { getDbPath, getPwd, getRealPath, computeDisplayPath, shortPath } from './paths.ts';
import { homedir } from 'os';
import { resolve } from 'path';

describe('getDbPath', () => {
  const originalXdgCacheHome = process.env.XDG_CACHE_HOME;

  beforeEach(() => {
    // Reset XDG_CACHE_HOME before each test
    if (originalXdgCacheHome) {
      process.env.XDG_CACHE_HOME = originalXdgCacheHome;
    } else {
      delete process.env.XDG_CACHE_HOME;
    }
  });

  test('returns default database path', () => {
    delete process.env.XDG_CACHE_HOME;

    const dbPath = getDbPath();
    const expectedBase = resolve(homedir(), '.cache', 'qmd');

    expect(dbPath).toContain('qmd');
    expect(dbPath).toContain('index.sqlite');
    expect(dbPath).toContain(expectedBase);
  });

  test('respects custom index name', () => {
    const dbPath = getDbPath('custom');

    expect(dbPath).toContain('custom.sqlite');
    expect(dbPath).not.toContain('index.sqlite');
  });

  test('respects XDG_CACHE_HOME environment variable', () => {
    process.env.XDG_CACHE_HOME = '/tmp/custom-cache';

    const dbPath = getDbPath();

    expect(dbPath).toContain('/tmp/custom-cache');
    expect(dbPath).toContain('qmd');
    expect(dbPath).toContain('index.sqlite');
  });

  test('creates cache directory if it does not exist', () => {
    // This test primarily verifies the function doesn't throw
    const dbPath = getDbPath('test-index');

    expect(dbPath).toBeTruthy();
    expect(dbPath).toMatch(/\.sqlite$/);
  });

  test('handles special characters in index name', () => {
    const dbPath = getDbPath('test-123_index');

    expect(dbPath).toContain('test-123_index.sqlite');
  });
});

describe('getPwd', () => {
  const originalPwd = process.env.PWD;

  beforeEach(() => {
    if (originalPwd) {
      process.env.PWD = originalPwd;
    }
  });

  test('returns PWD environment variable if set', () => {
    process.env.PWD = '/test/custom/path';

    const pwd = getPwd();

    expect(pwd).toBe('/test/custom/path');
  });

  test('falls back to process.cwd() if PWD not set', () => {
    delete process.env.PWD;

    const pwd = getPwd();
    const cwd = process.cwd();

    expect(pwd).toBe(cwd);
  });

  test('returns a valid directory path', () => {
    const pwd = getPwd();

    expect(pwd).toBeTruthy();
    expect(pwd).toMatch(/^\//); // Should be absolute path on Unix
  });
});

describe('getRealPath', () => {
  test('returns resolved path for existing files', () => {
    // Use current file as test subject
    const realPath = getRealPath(__filename);

    expect(realPath).toBeTruthy();
    expect(realPath).toMatch(/paths\.test\.ts$/);
  });

  test('returns resolved path for non-existent files', () => {
    const fakePath = '/tmp/nonexistent-file-12345.txt';
    const realPath = getRealPath(fakePath);

    // Should return the resolved version of the path
    expect(realPath).toContain('nonexistent-file-12345.txt');
  });

  test('handles relative paths', () => {
    const realPath = getRealPath('./test-file.md');

    expect(realPath).toBeTruthy();
    expect(realPath).toContain('test-file.md');
  });

  test('handles absolute paths', () => {
    const absolutePath = '/home/user/test.md';
    const realPath = getRealPath(absolutePath);

    expect(realPath).toBeTruthy();
  });

  test('handles paths with ..', () => {
    const realPath = getRealPath('../test.md');

    expect(realPath).toBeTruthy();
    expect(realPath).not.toContain('..');
  });

  test('handles current directory', () => {
    const realPath = getRealPath('.');

    expect(realPath).toBeTruthy();
    expect(realPath).not.toBe('.');
  });
});

describe('computeDisplayPath', () => {
  test('returns minimal unique path', () => {
    const filepath = '/home/user/projects/qmd/docs/readme.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    // Should be minimal path (at least 2 parts: parent + filename)
    expect(displayPath).toContain('readme.md');
    expect(displayPath.split('/').length).toBeGreaterThanOrEqual(2);
  });

  test('includes parent directory when filename conflicts', () => {
    const filepath1 = '/home/user/projects/qmd/docs/readme.md';
    const filepath2 = '/home/user/projects/qmd/src/readme.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath1 = computeDisplayPath(filepath1, collectionPath, existingPaths);
    existingPaths.add(displayPath1);

    const displayPath2 = computeDisplayPath(filepath2, collectionPath, existingPaths);

    expect(displayPath1).not.toBe(displayPath2);
    expect(displayPath1).toContain('readme.md');
    expect(displayPath2).toContain('readme.md');
  });

  test('adds more parent directories until unique', () => {
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set(['qmd/docs/api.md', 'qmd/docs/internal/api.md']);

    const filepath = '/home/user/projects/qmd/docs/public/api.md';
    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).not.toBe('qmd/docs/api.md');
    expect(displayPath).toContain('api.md');
  });

  test('handles files in collection root', () => {
    const filepath = '/home/user/projects/qmd/README.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).toContain('README.md');
  });

  test('handles deeply nested files', () => {
    const filepath = '/home/user/projects/qmd/docs/api/endpoints/search/vector/advanced.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).toContain('advanced.md');
  });

  test('creates relative path from collection', () => {
    const filepath = '/home/user/projects/qmd/docs/readme.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    // Should be relative to collection (e.g., "docs/readme.md")
    expect(displayPath).toContain('readme.md');
    expect(displayPath).not.toContain('/home/user');
  });

  test('handles trailing slash in collection path', () => {
    const filepath = '/home/user/projects/qmd/docs/readme.md';
    const collectionPath = '/home/user/projects/qmd/';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).toBeTruthy();
    expect(displayPath).toContain('readme.md');
  });

  test('handles files outside collection path', () => {
    const filepath = '/other/location/file.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    // Should fall back to full path or significant portion
    expect(displayPath).toContain('file.md');
  });

  test('returns full path when all combinations are taken', () => {
    const filepath = '/home/user/projects/qmd/docs/readme.md';
    const collectionPath = '/home/user/projects/qmd';

    // Simulate all possible display paths already taken (relative paths from collection)
    const existingPaths = new Set([
      'readme.md',
      'docs/readme.md',
      'qmd/docs/readme.md',
    ]);

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    // Should fall back to full filepath when all relative paths are taken
    expect(displayPath).toBe(filepath);
  });

  test('handles empty existing paths set', () => {
    const filepath = '/home/user/projects/qmd/test.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).toBeTruthy();
    expect(displayPath).toContain('test.md');
  });

  test('minimum 2 parts when available', () => {
    const filepath = '/home/user/projects/qmd/docs/api.md';
    const collectionPath = '/home/user/projects/qmd';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    // Should have at least 2 parts (parent + filename)
    const parts = displayPath.split('/');
    expect(parts.length).toBeGreaterThanOrEqual(2);
  });

  test('handles single file name', () => {
    const filepath = '/home/test.md';
    const collectionPath = '/home';
    const existingPaths = new Set<string>();

    const displayPath = computeDisplayPath(filepath, collectionPath, existingPaths);

    expect(displayPath).toContain('test.md');
  });
});

describe('shortPath', () => {
  test('converts home directory to tilde', () => {
    const home = homedir();
    const fullPath = resolve(home, 'projects', 'qmd');

    const short = shortPath(fullPath);

    expect(short).toMatch(/^~/);
    expect(short).toContain('projects/qmd');
    expect(short).not.toContain(home);
  });

  test('leaves non-home paths unchanged', () => {
    const path = '/tmp/test/path';

    const short = shortPath(path);

    expect(short).toBe(path);
  });

  test('handles exact home directory', () => {
    const home = homedir();

    const short = shortPath(home);

    expect(short).toBe('~');
  });

  test('handles home directory with trailing slash', () => {
    const home = homedir() + '/';

    const short = shortPath(home);

    expect(short).toMatch(/^~\/?$/);
  });

  test('handles nested home paths', () => {
    const home = homedir();
    const nested = resolve(home, 'dir1', 'dir2', 'dir3', 'file.txt');

    const short = shortPath(nested);

    expect(short).toMatch(/^~/);
    expect(short).toContain('dir1/dir2/dir3/file.txt');
  });

  test('does not replace partial home directory matches', () => {
    const home = homedir();
    // Create a path that contains home as substring but doesn't start with it
    const notHome = '/other' + home;

    const short = shortPath(notHome);

    expect(short).toBe(notHome);
    expect(short).not.toMatch(/^~/);
  });

  test('handles empty string', () => {
    const short = shortPath('');

    expect(short).toBe('');
  });

  test('handles relative paths', () => {
    const relativePath = './relative/path';

    const short = shortPath(relativePath);

    expect(short).toBe(relativePath);
  });
});

describe('Edge Cases', () => {
  test('getDbPath with empty string', () => {
    const dbPath = getDbPath('');

    expect(dbPath).toContain('.sqlite');
  });

  test('getRealPath with empty string', () => {
    const realPath = getRealPath('');

    expect(realPath).toBeTruthy();
  });

  test('computeDisplayPath with empty strings', () => {
    const displayPath = computeDisplayPath('', '', new Set());

    expect(typeof displayPath).toBe('string');
  });

  test('shortPath handles special characters', () => {
    const home = homedir();
    const withSpecial = resolve(home, 'test (1)', 'file [copy].md');

    const short = shortPath(withSpecial);

    expect(short).toMatch(/^~/);
    expect(short).toContain('test (1)');
  });
});
