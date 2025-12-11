/**
 * Application constants and configuration
 * Uses unified config system: CLI > Env > File > Defaults
 */

import { getConfigValue } from './loader';

/** QMD version */
export const VERSION = "1.0.0";

/** Default query expansion model (not configurable via config system) */
export const DEFAULT_QUERY_MODEL = "qwen3:0.6b";

/**
 * Get default embedding model
 * Priority: CLI flag > QMD_EMBED_MODEL env var > config.json > default
 * @param override - Optional CLI flag override
 */
export function getEmbedModel(override?: string): string {
  return getConfigValue('embedModel', override);
}

/**
 * Get default reranking model
 * Priority: CLI flag > QMD_RERANK_MODEL env var > config.json > default
 * @param override - Optional CLI flag override
 */
export function getRerankModel(override?: string): string {
  return getConfigValue('rerankModel', override);
}

/**
 * Get default glob pattern
 * Priority: CLI flag > config.json > default
 * @param override - Optional CLI flag override
 */
export function getDefaultGlob(override?: string): string {
  return getConfigValue('defaultGlob', override);
}

/**
 * Get Ollama API URL
 * Priority: CLI flag > OLLAMA_URL env var > config.json > default
 * @param override - Optional CLI flag override
 */
export function getOllamaUrl(override?: string): string {
  return getConfigValue('ollamaUrl', override);
}

/**
 * Legacy exports for backward compatibility
 * @deprecated Use getEmbedModel() instead for proper config precedence
 */
export const DEFAULT_EMBED_MODEL = getEmbedModel();

/**
 * Legacy export for backward compatibility
 * @deprecated Use getRerankModel() instead for proper config precedence
 */
export const DEFAULT_RERANK_MODEL = getRerankModel();

/**
 * Legacy export for backward compatibility
 * @deprecated Use getDefaultGlob() instead for proper config precedence
 */
export const DEFAULT_GLOB = getDefaultGlob();

/**
 * Legacy export for backward compatibility
 * @deprecated Use getOllamaUrl() instead for proper config precedence
 */
export const OLLAMA_URL = getOllamaUrl();
