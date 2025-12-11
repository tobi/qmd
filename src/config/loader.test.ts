/**
 * Tests for configuration loader
 */

import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { writeFileSync, mkdirSync, rmSync, existsSync } from 'fs';
import { resolve } from 'path';
import { loadConfig, getConfigValue, getDefaults, type QmdConfig } from './loader';

describe('Configuration Loader', () => {
  const testDir = resolve(__dirname, '../../.test-config');
  const qmdDir = resolve(testDir, '.qmd');
  const configPath = resolve(qmdDir, 'config.json');

  // Store original env vars
  const originalEnv = {
    QMD_EMBED_MODEL: process.env.QMD_EMBED_MODEL,
    QMD_RERANK_MODEL: process.env.QMD_RERANK_MODEL,
    OLLAMA_URL: process.env.OLLAMA_URL,
    PWD: process.env.PWD,
  };

  beforeEach(() => {
    // Clean up test directory
    if (existsSync(testDir)) {
      rmSync(testDir, { recursive: true, force: true });
    }
    mkdirSync(qmdDir, { recursive: true });

    // Set PWD to test directory so findQmdDir() works
    process.env.PWD = testDir;

    // Clear env vars
    delete process.env.QMD_EMBED_MODEL;
    delete process.env.QMD_RERANK_MODEL;
    delete process.env.OLLAMA_URL;
  });

  afterEach(() => {
    // Restore env vars
    if (originalEnv.QMD_EMBED_MODEL) process.env.QMD_EMBED_MODEL = originalEnv.QMD_EMBED_MODEL;
    else delete process.env.QMD_EMBED_MODEL;

    if (originalEnv.QMD_RERANK_MODEL) process.env.QMD_RERANK_MODEL = originalEnv.QMD_RERANK_MODEL;
    else delete process.env.QMD_RERANK_MODEL;

    if (originalEnv.OLLAMA_URL) process.env.OLLAMA_URL = originalEnv.OLLAMA_URL;
    else delete process.env.OLLAMA_URL;

    if (originalEnv.PWD) process.env.PWD = originalEnv.PWD;

    // Clean up
    if (existsSync(testDir)) {
      rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('getDefaults', () => {
    it('should return default configuration', () => {
      const defaults = getDefaults();
      expect(defaults.embedModel).toBe('nomic-embed-text');
      expect(defaults.rerankModel).toBe('qwen3-reranker:0.6b-q8_0');
      expect(defaults.defaultGlob).toBe('**/*.md');
      expect(defaults.ollamaUrl).toBe('http://localhost:11434');
      expect(defaults.excludeDirs).toEqual(['node_modules', '.git', 'dist', 'build', '.cache']);
    });
  });

  describe('loadConfig - defaults only', () => {
    it('should return defaults when no config file or env vars', () => {
      const config = loadConfig();
      expect(config.embedModel).toBe('nomic-embed-text');
      expect(config.rerankModel).toBe('qwen3-reranker:0.6b-q8_0');
      expect(config.defaultGlob).toBe('**/*.md');
      expect(config.ollamaUrl).toBe('http://localhost:11434');
    });
  });

  describe('loadConfig - file only', () => {
    it('should load from config.json when file exists', () => {
      writeFileSync(configPath, JSON.stringify({
        embedModel: 'custom-embed',
        rerankModel: 'custom-rerank',
        ollamaUrl: 'http://custom:11434',
      }));

      const config = loadConfig();
      expect(config.embedModel).toBe('custom-embed');
      expect(config.rerankModel).toBe('custom-rerank');
      expect(config.ollamaUrl).toBe('http://custom:11434');
      expect(config.defaultGlob).toBe('**/*.md'); // Default
    });

    it('should handle partial config.json', () => {
      writeFileSync(configPath, JSON.stringify({
        embedModel: 'partial-embed',
      }));

      const config = loadConfig();
      expect(config.embedModel).toBe('partial-embed');
      expect(config.rerankModel).toBe('qwen3-reranker:0.6b-q8_0'); // Default
      expect(config.ollamaUrl).toBe('http://localhost:11434'); // Default
    });

    it('should ignore invalid config.json', () => {
      writeFileSync(configPath, 'invalid json{{{');

      const config = loadConfig();
      expect(config.embedModel).toBe('nomic-embed-text'); // Falls back to defaults
    });

    it('should validate field types in config.json', () => {
      writeFileSync(configPath, JSON.stringify({
        embedModel: 123, // Invalid type
        rerankModel: 'valid-rerank',
        excludeDirs: ['valid', 123, 'also-valid'], // Mixed types
      }));

      const config = loadConfig();
      expect(config.embedModel).toBe('nomic-embed-text'); // Ignored invalid, uses default
      expect(config.rerankModel).toBe('valid-rerank');
      expect(config.excludeDirs).toEqual(['valid', 'also-valid']); // Filters out non-strings
    });
  });

  describe('loadConfig - env vars only', () => {
    it('should load from environment variables', () => {
      process.env.QMD_EMBED_MODEL = 'env-embed';
      process.env.QMD_RERANK_MODEL = 'env-rerank';
      process.env.OLLAMA_URL = 'http://env:11434';

      const config = loadConfig();
      expect(config.embedModel).toBe('env-embed');
      expect(config.rerankModel).toBe('env-rerank');
      expect(config.ollamaUrl).toBe('http://env:11434');
    });

    it('should handle partial env vars', () => {
      process.env.QMD_EMBED_MODEL = 'env-embed';

      const config = loadConfig();
      expect(config.embedModel).toBe('env-embed');
      expect(config.rerankModel).toBe('qwen3-reranker:0.6b-q8_0'); // Default
    });
  });

  describe('loadConfig - precedence: env > file', () => {
    it('should prefer env vars over config file', () => {
      // File config
      writeFileSync(configPath, JSON.stringify({
        embedModel: 'file-embed',
        rerankModel: 'file-rerank',
        ollamaUrl: 'http://file:11434',
      }));

      // Env config
      process.env.QMD_EMBED_MODEL = 'env-embed';
      process.env.OLLAMA_URL = 'http://env:11434';

      const config = loadConfig();
      expect(config.embedModel).toBe('env-embed'); // Env wins
      expect(config.rerankModel).toBe('file-rerank'); // From file (no env)
      expect(config.ollamaUrl).toBe('http://env:11434'); // Env wins
    });
  });

  describe('loadConfig - precedence: CLI > env > file', () => {
    it('should prefer CLI overrides over everything', () => {
      // File config
      writeFileSync(configPath, JSON.stringify({
        embedModel: 'file-embed',
        rerankModel: 'file-rerank',
        ollamaUrl: 'http://file:11434',
      }));

      // Env config
      process.env.QMD_EMBED_MODEL = 'env-embed';
      process.env.QMD_RERANK_MODEL = 'env-rerank';

      // CLI overrides
      const config = loadConfig({
        embedModel: 'cli-embed',
        ollamaUrl: 'http://cli:11434',
      });

      expect(config.embedModel).toBe('cli-embed'); // CLI wins
      expect(config.rerankModel).toBe('env-rerank'); // Env wins (no CLI)
      expect(config.ollamaUrl).toBe('http://cli:11434'); // CLI wins
      expect(config.defaultGlob).toBe('**/*.md'); // Default (no override)
    });

    it('should handle all layers: CLI > env > file > default', () => {
      // File config
      writeFileSync(configPath, JSON.stringify({
        embedModel: 'file-embed',
        rerankModel: 'file-rerank',
        defaultGlob: '**/*.markdown',
      }));

      // Env config
      process.env.QMD_RERANK_MODEL = 'env-rerank';
      process.env.OLLAMA_URL = 'http://env:11434';

      // CLI overrides
      const config = loadConfig({
        embedModel: 'cli-embed',
      });

      expect(config.embedModel).toBe('cli-embed'); // CLI
      expect(config.rerankModel).toBe('env-rerank'); // Env
      expect(config.defaultGlob).toBe('**/*.markdown'); // File
      expect(config.ollamaUrl).toBe('http://env:11434'); // Env
      expect(config.excludeDirs).toEqual(['node_modules', '.git', 'dist', 'build', '.cache']); // Default
    });
  });

  describe('getConfigValue', () => {
    it('should get single value with override', () => {
      const value = getConfigValue('embedModel', 'override-model');
      expect(value).toBe('override-model');
    });

    it('should get single value from config when no override', () => {
      process.env.QMD_EMBED_MODEL = 'env-model';
      const value = getConfigValue('embedModel');
      expect(value).toBe('env-model');
    });

    it('should return default when no override or config', () => {
      const value = getConfigValue('embedModel');
      expect(value).toBe('nomic-embed-text');
    });
  });

  describe('no .qmd directory', () => {
    it('should use defaults when .qmd directory not found', () => {
      // Remove .qmd directory
      rmSync(qmdDir, { recursive: true, force: true });

      // Set PWD to somewhere without .qmd
      process.env.PWD = '/tmp';

      const config = loadConfig();
      expect(config.embedModel).toBe('nomic-embed-text');
      expect(config.rerankModel).toBe('qwen3-reranker:0.6b-q8_0');
    });

    it('should still respect env vars when .qmd not found', () => {
      rmSync(qmdDir, { recursive: true, force: true });
      process.env.PWD = '/tmp';
      process.env.QMD_EMBED_MODEL = 'env-model';

      const config = loadConfig();
      expect(config.embedModel).toBe('env-model');
    });
  });
});
