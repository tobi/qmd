/**
 * Configuration loader with unified priority system
 * Priority: CLI flags > Environment variables > Config file > Defaults
 */

import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { findQmdDir } from '../utils/paths';

/**
 * QMD Configuration schema
 */
export interface QmdConfig {
  /** Embedding model for vector search */
  embedModel: string;
  /** Reranking model for hybrid search */
  rerankModel: string;
  /** Default glob pattern for markdown files */
  defaultGlob: string;
  /** Directories to exclude from indexing */
  excludeDirs: string[];
  /** Ollama service URL */
  ollamaUrl: string;
}

/**
 * Partial config for overrides (all fields optional)
 */
export type PartialConfig = Partial<QmdConfig>;

/**
 * Default configuration values
 */
const DEFAULTS: QmdConfig = {
  embedModel: 'nomic-embed-text',
  rerankModel: 'qwen3-reranker:0.6b-q8_0',
  defaultGlob: '**/*.md',
  excludeDirs: ['node_modules', '.git', 'dist', 'build', '.cache'],
  ollamaUrl: 'http://localhost:11434',
};

/**
 * Load configuration with priority: CLI > Env > File > Defaults
 *
 * @param overrides - CLI flag overrides (highest priority)
 * @returns Complete configuration object
 *
 * @example
 * ```typescript
 * // Load with CLI override
 * const config = loadConfig({ embedModel: 'custom-model' });
 *
 * // Load with defaults
 * const config = loadConfig();
 * ```
 */
export function loadConfig(overrides: PartialConfig = {}): QmdConfig {
  // Start with defaults
  let config: QmdConfig = { ...DEFAULTS };

  // Layer 3: Load from .qmd/config.json if exists
  const fileConfig = loadConfigFile();
  if (fileConfig) {
    config = { ...config, ...fileConfig };
  }

  // Layer 2: Override with environment variables
  const envConfig = loadEnvConfig();
  config = { ...config, ...envConfig };

  // Layer 1: Override with CLI flags (highest priority)
  config = { ...config, ...overrides };

  return config;
}

/**
 * Load configuration from .qmd/config.json file
 * @returns Partial config from file, or null if not found/invalid
 */
function loadConfigFile(): PartialConfig | null {
  const qmdDir = findQmdDir();
  if (!qmdDir) return null;

  const configPath = resolve(qmdDir, 'config.json');
  if (!existsSync(configPath)) return null;

  try {
    const content = readFileSync(configPath, 'utf-8');
    const parsed = JSON.parse(content);

    // Validate and extract known fields
    const config: PartialConfig = {};
    if (typeof parsed.embedModel === 'string') config.embedModel = parsed.embedModel;
    if (typeof parsed.rerankModel === 'string') config.rerankModel = parsed.rerankModel;
    if (typeof parsed.defaultGlob === 'string') config.defaultGlob = parsed.defaultGlob;
    if (typeof parsed.ollamaUrl === 'string') config.ollamaUrl = parsed.ollamaUrl;
    if (Array.isArray(parsed.excludeDirs)) {
      config.excludeDirs = parsed.excludeDirs.filter((d: unknown) => typeof d === 'string');
    }

    return Object.keys(config).length > 0 ? config : null;
  } catch (error) {
    // Silently ignore parse errors (config is optional)
    return null;
  }
}

/**
 * Load configuration from environment variables
 * @returns Partial config from env vars
 */
function loadEnvConfig(): PartialConfig {
  const config: PartialConfig = {};

  if (process.env.QMD_EMBED_MODEL) {
    config.embedModel = process.env.QMD_EMBED_MODEL;
  }
  if (process.env.QMD_RERANK_MODEL) {
    config.rerankModel = process.env.QMD_RERANK_MODEL;
  }
  if (process.env.OLLAMA_URL) {
    config.ollamaUrl = process.env.OLLAMA_URL;
  }

  return config;
}

/**
 * Get a specific config value with optional override
 * Convenience function for getting single values
 *
 * @param key - Config key to retrieve
 * @param override - Optional override value (CLI flag)
 * @returns Config value
 *
 * @example
 * ```typescript
 * const model = getConfigValue('embedModel', flags.model);
 * ```
 */
export function getConfigValue<K extends keyof QmdConfig>(
  key: K,
  override?: QmdConfig[K]
): QmdConfig[K] {
  if (override !== undefined) return override;

  const config = loadConfig();
  return config[key];
}

/**
 * Get default config values (without any overrides)
 * Useful for documentation and init command
 */
export function getDefaults(): QmdConfig {
  return { ...DEFAULTS };
}
