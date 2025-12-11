/**
 * Test the test helpers themselves
 * Verifies that our test infrastructure is working correctly
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import {
  createTestDb,
  createTestDbWithData,
  createTestDbWithVectors,
  cleanupDb,
  getTableNames,
  tableExists,
  getRowCount,
} from './test-db';
import {
  mockOllamaEmbed,
  mockOllamaGenerate,
  mockOllamaModelCheck,
  mockOllamaPull,
  mockOllamaComplete,
} from './mock-ollama';
import {
  sampleDocs,
  sampleEmbeddings,
  sqlInjectionPayloads,
  sampleQueries,
  sampleSearchResults,
} from './fixtures';

describe('Test Database Helpers', () => {
  let db: Database;

  afterEach(() => {
    if (db) {
      cleanupDb(db);
    }
  });

  test('createTestDb creates database with schema', () => {
    db = createTestDb();

    const tables = getTableNames(db);

    // Verify essential tables exist
    expect(tables).toContain('collections');
    expect(tables).toContain('documents');
    expect(tables).toContain('documents_fts');
    expect(tables).toContain('content_vectors');
    expect(tables).toContain('path_contexts');
    expect(tables).toContain('ollama_cache');
  });

  test('createTestDbWithData inserts sample data', () => {
    db = createTestDbWithData();

    // Verify collections
    const collectionsCount = getRowCount(db, 'collections');
    expect(collectionsCount).toBe(1);

    // Verify documents
    const documentsCount = getRowCount(db, 'documents');
    expect(documentsCount).toBe(2);
  });

  test('createTestDbWithVectors creates vectors table', () => {
    db = createTestDbWithVectors(128);

    // Verify vectors table exists
    expect(tableExists(db, 'vectors_vec')).toBe(true);

    // Verify vector was inserted
    const vectorsCount = getRowCount(db, 'content_vectors');
    expect(vectorsCount).toBe(1);
  });

  test('tableExists returns correct results', () => {
    db = createTestDb();

    expect(tableExists(db, 'documents')).toBe(true);
    expect(tableExists(db, 'nonexistent_table')).toBe(false);
  });

  test('getRowCount returns correct count', () => {
    db = createTestDbWithData();

    const count = getRowCount(db, 'documents');
    expect(count).toBeGreaterThan(0);
  });
});

describe('Ollama API Mocks', () => {
  afterEach(() => {
    // Restore fetch after each test
    if ((global.fetch as any).mockRestore) {
      (global.fetch as any).mockRestore();
    }
  });

  test('mockOllamaEmbed returns embeddings', async () => {
    const testEmbeddings = [[0.1, 0.2, 0.3]];
    global.fetch = mockOllamaEmbed(testEmbeddings);

    const response = await fetch('http://localhost:11434/api/embed', {
      method: 'POST',
      body: JSON.stringify({ model: 'test', input: 'test text' }),
    });

    expect(response.ok).toBe(true);

    const data = await response.json();
    expect(data.embeddings).toEqual(testEmbeddings);
  });

  test('mockOllamaGenerate returns response', async () => {
    const testResponse = 'yes';
    global.fetch = mockOllamaGenerate(testResponse);

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      body: JSON.stringify({ model: 'test', prompt: 'test' }),
    });

    expect(response.ok).toBe(true);

    const data = await response.json();
    expect(data.response).toBe(testResponse);
    expect(data.done).toBe(true);
  });

  test('mockOllamaModelCheck returns correct status', async () => {
    global.fetch = mockOllamaModelCheck(true);

    const response = await fetch('http://localhost:11434/api/show', {
      method: 'POST',
      body: JSON.stringify({ name: 'test-model' }),
    });

    expect(response.ok).toBe(true);
  });

  test('mockOllamaModelCheck returns 404 when model not found', async () => {
    global.fetch = mockOllamaModelCheck(false);

    const response = await fetch('http://localhost:11434/api/show', {
      method: 'POST',
      body: JSON.stringify({ name: 'nonexistent-model' }),
    });

    expect(response.ok).toBe(false);
    expect(response.status).toBe(404);
  });

  test('mockOllamaPull returns success', async () => {
    global.fetch = mockOllamaPull(true);

    const response = await fetch('http://localhost:11434/api/pull', {
      method: 'POST',
      body: JSON.stringify({ name: 'test-model' }),
    });

    expect(response.ok).toBe(true);
  });

  test('mockOllamaComplete handles all endpoints', async () => {
    global.fetch = mockOllamaComplete({
      embeddings: [[0.1, 0.2]],
      generateResponse: 'yes',
      modelExists: true,
      pullSuccess: true,
    });

    // Test embed endpoint
    const embedResponse = await fetch('http://localhost:11434/api/embed', {
      method: 'POST',
    });
    expect(embedResponse.ok).toBe(true);

    // Test generate endpoint
    const generateResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
    });
    expect(generateResponse.ok).toBe(true);

    // Test show endpoint
    const showResponse = await fetch('http://localhost:11434/api/show', {
      method: 'POST',
    });
    expect(showResponse.ok).toBe(true);
  });
});

describe('Test Fixtures', () => {
  test('sampleDocs contains valid markdown', () => {
    expect(sampleDocs.simple).toContain('# Simple Document');
    expect(sampleDocs.withCode).toContain('```javascript');
    expect(sampleDocs.unicode).toContain('日本語');
    expect(sampleDocs.empty).toBe('');
  });

  test('sampleEmbeddings have correct dimensions', () => {
    expect(sampleEmbeddings.dim128).toHaveLength(128);
    expect(sampleEmbeddings.dim256).toHaveLength(256);
    expect(sampleEmbeddings.dim1024).toHaveLength(1024);
  });

  test('sqlInjectionPayloads contains attack vectors', () => {
    expect(sqlInjectionPayloads.length).toBeGreaterThan(0);
    expect(sqlInjectionPayloads).toContain("'; DROP TABLE documents; --");
    expect(sqlInjectionPayloads).toContain("' OR 1=1 --");
  });

  test('sampleQueries contains various query types', () => {
    expect(sampleQueries.simple).toBe('test query');
    expect(sampleQueries.quoted).toContain('"');
    expect(sampleQueries.empty).toBe('');
  });

  test('sampleSearchResults has correct structure', () => {
    expect(sampleSearchResults).toHaveLength(3);
    expect(sampleSearchResults[0]).toHaveProperty('file');
    expect(sampleSearchResults[0]).toHaveProperty('title');
    expect(sampleSearchResults[0]).toHaveProperty('score');
    expect(sampleSearchResults[0]).toHaveProperty('source');
  });
});
