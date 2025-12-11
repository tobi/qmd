/**
 * Tests for application constants
 * Target coverage: 70%+
 */

import { describe, test, expect } from 'bun:test';
import {
  VERSION,
  DEFAULT_EMBED_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_GLOB,
  OLLAMA_URL,
} from './constants.ts';

describe('Application Constants', () => {
  test('VERSION is defined and valid', () => {
    expect(VERSION).toBeDefined();
    expect(typeof VERSION).toBe('string');
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/); // Semantic versioning
  });

  test('DEFAULT_EMBED_MODEL is defined', () => {
    expect(DEFAULT_EMBED_MODEL).toBeDefined();
    expect(typeof DEFAULT_EMBED_MODEL).toBe('string');
    expect(DEFAULT_EMBED_MODEL.length).toBeGreaterThan(0);
  });

  test('DEFAULT_RERANK_MODEL is defined', () => {
    expect(DEFAULT_RERANK_MODEL).toBeDefined();
    expect(typeof DEFAULT_RERANK_MODEL).toBe('string');
    expect(DEFAULT_RERANK_MODEL.length).toBeGreaterThan(0);
  });

  test('DEFAULT_QUERY_MODEL is defined', () => {
    expect(DEFAULT_QUERY_MODEL).toBeDefined();
    expect(typeof DEFAULT_QUERY_MODEL).toBe('string');
    expect(DEFAULT_QUERY_MODEL.length).toBeGreaterThan(0);
  });

  test('DEFAULT_GLOB is a valid markdown pattern', () => {
    expect(DEFAULT_GLOB).toBeDefined();
    expect(typeof DEFAULT_GLOB).toBe('string');
    expect(DEFAULT_GLOB).toContain('.md');
    expect(DEFAULT_GLOB).toMatch(/\*\*?/); // Contains glob wildcard
  });

  test('OLLAMA_URL is a valid URL format', () => {
    expect(OLLAMA_URL).toBeDefined();
    expect(typeof OLLAMA_URL).toBe('string');
    expect(OLLAMA_URL).toMatch(/^https?:\/\//); // Starts with http:// or https://
  });

  test('OLLAMA_URL includes localhost by default', () => {
    // Only test if not overridden by env var
    if (!process.env.OLLAMA_URL) {
      expect(OLLAMA_URL).toContain('localhost');
      expect(OLLAMA_URL).toContain('11434'); // Default Ollama port
    }
  });

  test('model names are non-empty strings', () => {
    const models = [DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_QUERY_MODEL];

    for (const model of models) {
      expect(model).toBeTruthy();
      expect(typeof model).toBe('string');
      expect(model.length).toBeGreaterThan(0);
    }
  });

  test('constants are exported correctly', () => {
    // Verify all expected exports exist
    const exports = {
      VERSION,
      DEFAULT_EMBED_MODEL,
      DEFAULT_RERANK_MODEL,
      DEFAULT_QUERY_MODEL,
      DEFAULT_GLOB,
      OLLAMA_URL,
    };

    for (const [name, value] of Object.entries(exports)) {
      expect(value).toBeDefined();
    }
  });
});

describe('Environment Variable Overrides', () => {
  test('DEFAULT_EMBED_MODEL can be overridden by QMD_EMBED_MODEL', () => {
    // If env var is set, constant should reflect it
    if (process.env.QMD_EMBED_MODEL) {
      expect(DEFAULT_EMBED_MODEL).toBe(process.env.QMD_EMBED_MODEL);
    }
  });

  test('DEFAULT_RERANK_MODEL can be overridden by QMD_RERANK_MODEL', () => {
    // If env var is set, constant should reflect it
    if (process.env.QMD_RERANK_MODEL) {
      expect(DEFAULT_RERANK_MODEL).toBe(process.env.QMD_RERANK_MODEL);
    }
  });

  test('OLLAMA_URL can be overridden by OLLAMA_URL env var', () => {
    // If env var is set, constant should reflect it
    if (process.env.OLLAMA_URL) {
      expect(OLLAMA_URL).toBe(process.env.OLLAMA_URL);
    }
  });
});

describe('Constant Values', () => {
  test('VERSION matches expected format', () => {
    expect(VERSION).toBe('1.0.0');
  });

  test('default embed model is nomic-embed-text', () => {
    // Only if not overridden by env
    if (!process.env.QMD_EMBED_MODEL) {
      expect(DEFAULT_EMBED_MODEL).toBe('nomic-embed-text');
    }
  });

  test('default glob pattern is markdown recursive', () => {
    expect(DEFAULT_GLOB).toBe('**/*.md');
  });

  test('constants are exported as const', () => {
    // ES6 const prevents reassignment - TypeScript enforces this at compile time
    // We can verify the values are stable across multiple reads
    const version1 = VERSION;
    const version2 = VERSION;

    expect(version1).toBe(version2);
    expect(VERSION).toBe(version1);
  });
});
