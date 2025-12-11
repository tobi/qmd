/**
 * Tests for hash utility functions
 * Target coverage: 95%+
 */

import { describe, test, expect } from 'bun:test';
import { hashContent, getCacheKey } from './hash.ts';
import { edgeCases } from '../../tests/fixtures/helpers/fixtures.ts';

describe('hashContent', () => {
  test('returns consistent hashes for same content', async () => {
    const content = 'test content';
    const hash1 = await hashContent(content);
    const hash2 = await hashContent(content);

    expect(hash1).toBe(hash2);
    expect(hash1).toHaveLength(64); // SHA-256 produces 64 hex characters
  });

  test('produces different hashes for different content', async () => {
    const hash1 = await hashContent('content 1');
    const hash2 = await hashContent('content 2');

    expect(hash1).not.toBe(hash2);
  });

  test('handles empty string', async () => {
    const hash = await hashContent('');

    expect(hash).toBeTruthy();
    expect(hash).toHaveLength(64);
    expect(hash).toMatch(/^[0-9a-f]{64}$/); // Valid hex string
  });

  test('handles whitespace', async () => {
    const hash1 = await hashContent('   ');
    const hash2 = await hashContent('\n\t');

    expect(hash1).not.toBe(hash2); // Different whitespace = different hash
    expect(hash1).toHaveLength(64);
  });

  test('handles unicode characters', async () => {
    const hash = await hashContent('日本語 🎉 café');

    expect(hash).toHaveLength(64);
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });

  test('handles very long strings', async () => {
    const longString = 'x'.repeat(100000);
    const hash = await hashContent(longString);

    expect(hash).toHaveLength(64);
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });

  test('handles newlines and special characters', async () => {
    const content = 'line1\nline2\r\nline3\ttab\0null';
    const hash = await hashContent(content);

    expect(hash).toHaveLength(64);
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });

  test('produces different hashes for content with different newline styles', async () => {
    const hash1 = await hashContent('line1\nline2');
    const hash2 = await hashContent('line1\r\nline2');

    expect(hash1).not.toBe(hash2);
  });

  test('hash is deterministic across multiple calls', async () => {
    const content = 'deterministic test';
    const hashes = await Promise.all([
      hashContent(content),
      hashContent(content),
      hashContent(content),
      hashContent(content),
      hashContent(content),
    ]);

    // All hashes should be identical
    expect(new Set(hashes).size).toBe(1);
  });

  test('handles special markdown characters', async () => {
    const markdown = '# Heading\n\n**bold** *italic* `code` [link](url)';
    const hash = await hashContent(markdown);

    expect(hash).toHaveLength(64);
  });
});

describe('getCacheKey', () => {
  test('returns consistent keys for same URL and body', () => {
    const url = 'http://localhost:11434/api/embed';
    const body = { model: 'test', input: 'test text' };

    const key1 = getCacheKey(url, body);
    const key2 = getCacheKey(url, body);

    expect(key1).toBe(key2);
    expect(key1).toHaveLength(64);
  });

  test('produces different keys for different URLs', () => {
    const body = { model: 'test' };

    const key1 = getCacheKey('http://localhost:11434/api/embed', body);
    const key2 = getCacheKey('http://localhost:11434/api/generate', body);

    expect(key1).not.toBe(key2);
  });

  test('produces different keys for different bodies', () => {
    const url = 'http://localhost:11434/api/embed';

    const key1 = getCacheKey(url, { model: 'model1' });
    const key2 = getCacheKey(url, { model: 'model2' });

    expect(key1).not.toBe(key2);
  });

  test('produces different keys when body properties change', () => {
    const url = 'http://localhost:11434/api/embed';

    const key1 = getCacheKey(url, { input: 'text1' });
    const key2 = getCacheKey(url, { input: 'text2' });

    expect(key1).not.toBe(key2);
  });

  test('handles empty URL', () => {
    const key = getCacheKey('', { test: 'data' });

    expect(key).toHaveLength(64);
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  test('handles empty body', () => {
    const key = getCacheKey('http://localhost:11434/api/test', {});

    expect(key).toHaveLength(64);
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  test('handles nested objects in body', () => {
    const url = 'http://localhost:11434/api/embed';
    const body = {
      model: 'test',
      options: {
        temperature: 0.7,
        top_k: 40,
      },
    };

    const key = getCacheKey(url, body);

    expect(key).toHaveLength(64);
  });

  test('key changes when nested object properties change', () => {
    const url = 'http://localhost:11434/api/embed';

    const key1 = getCacheKey(url, { options: { temp: 0.7 } });
    const key2 = getCacheKey(url, { options: { temp: 0.8 } });

    expect(key1).not.toBe(key2);
  });

  test('handles arrays in body', () => {
    const url = 'http://localhost:11434/api/embed';
    const body = { inputs: ['text1', 'text2', 'text3'] };

    const key = getCacheKey(url, body);

    expect(key).toHaveLength(64);
  });

  test('key changes when array order changes', () => {
    const url = 'http://localhost:11434/api/embed';

    const key1 = getCacheKey(url, { inputs: ['a', 'b'] });
    const key2 = getCacheKey(url, { inputs: ['b', 'a'] });

    expect(key1).not.toBe(key2);
  });

  test('handles unicode in URL and body', () => {
    const url = 'http://localhost/日本語';
    const body = { text: 'café 🎉' };

    const key = getCacheKey(url, body);

    expect(key).toHaveLength(64);
    expect(key).toMatch(/^[0-9a-f]{64}$/);
  });

  test('handles special characters in body', () => {
    const url = 'http://localhost:11434/api/test';
    const body = {
      text: edgeCases.specialChars,
      unicode: edgeCases.unicode,
    };

    const key = getCacheKey(url, body);

    expect(key).toHaveLength(64);
  });

  test('is deterministic across multiple calls', () => {
    const url = 'http://localhost:11434/api/embed';
    const body = { model: 'test', input: 'test' };

    const keys = Array.from({ length: 5 }, () => getCacheKey(url, body));

    // All keys should be identical
    expect(new Set(keys).size).toBe(1);
  });

  test('handles null and undefined in body', () => {
    const url = 'http://localhost:11434/api/test';

    const key1 = getCacheKey(url, { value: null });
    const key2 = getCacheKey(url, { value: undefined });

    // null and undefined should be treated differently by JSON.stringify
    expect(key1).not.toBe(key2);
  });

  test('handles numbers and booleans in body', () => {
    const url = 'http://localhost:11434/api/test';
    const body = {
      count: 42,
      temperature: 0.7,
      enabled: true,
      disabled: false,
    };

    const key = getCacheKey(url, body);

    expect(key).toHaveLength(64);
  });
});

describe('Edge Cases', () => {
  test('hashContent handles all edge case strings', async () => {
    for (const [name, value] of Object.entries(edgeCases)) {
      const hash = await hashContent(value);

      expect(hash).toHaveLength(64);
      expect(hash).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  test('getCacheKey handles edge case URLs and bodies', () => {
    const key1 = getCacheKey(edgeCases.specialChars, {});
    const key2 = getCacheKey('', { text: edgeCases.unicode });
    const key3 = getCacheKey(edgeCases.veryLongString, { data: edgeCases.whitespace });

    expect(key1).toHaveLength(64);
    expect(key2).toHaveLength(64);
    expect(key3).toHaveLength(64);
  });
});
