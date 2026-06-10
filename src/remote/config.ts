/**
 * config.ts — Environment variable and YAML config resolution for remote endpoints.
 *
 * Resolves endpoint configuration from three sources in priority order:
 * 1. Environment variables (QMD_{ENDPOINT}_BASE_URL, QMD_{ENDPOINT}_MODEL, QMD_{ENDPOINT}_API_KEY, QMD_{ENDPOINT}_API_FORMAT)
 * 2. YAML models config (embed_api_url, embed_api_format, expand_api_model, etc.)
 * 3. Hardcoded defaults for model names when specific helpers are used
 *
 * ## Design principle
 *
 * Local-first: ALL endpoints default to empty (local LlamaCpp) unless
 * explicitly configured via env vars or YAML. No remote URLs are assumed.
 *
 * @module remote/config
 */

import type { ModelsConfig, RemoteApiFormat } from '../collections.js';
import type { EndpointConfig, EndpointRole, RemoteLLMConfig } from './types.js';

/** Default OpenRouter API base URL. Used when no expand/rerank URL is configured. */
export const OPENROUTER_DEFAULT_URL = 'https://openrouter.ai/api/v1';

const REMOTE_API_FORMATS: readonly RemoteApiFormat[] = [
  'auto',
  'openai_v1_embeddings',
  'cohere_v2_embed',
  'ollama_embed',
  'ollama_chat',
  'ollama_generate',
  'vllm_pooling',
  'openai_chat_completions',
  'openai_completions',
  'openai_responses',
  'anthropic_messages',
  'cohere_v1_rerank',
  'cohere_v2_rerank',
  'vllm_score',
  'openai_audio_transcriptions',
  'openai_audio_translations',
];

const REMOTE_API_FORMAT_ALIASES: Record<string, RemoteApiFormat> = {
  auto: 'auto',
  openai_v1: 'openai_v1_embeddings',
  openai_embeddings: 'openai_v1_embeddings',
  cohere_v2: 'cohere_v2_embed',
  cohere_embed: 'cohere_v2_embed',
  ollama: 'ollama_embed',
  vllm: 'vllm_pooling',
  openai_chat: 'openai_chat_completions',
  chat: 'openai_chat_completions',
  openai_responses: 'openai_responses',
  responses: 'openai_responses',
  openai_completions: 'openai_completions',
  completions: 'openai_completions',
  anthropic: 'anthropic_messages',
  messages: 'anthropic_messages',
  cohere_rerank: 'cohere_v2_rerank',
  rerank: 'cohere_v2_rerank',
};

const ALLOWED_FORMATS_BY_ENDPOINT: Record<EndpointRole, readonly RemoteApiFormat[]> = {
  embed: [
    'auto',
    'openai_v1_embeddings',
    'cohere_v2_embed',
    'ollama_embed',
    'vllm_pooling',
  ],
  expand: [
    'auto',
    'openai_chat_completions',
    'openai_completions',
    'openai_responses',
    'anthropic_messages',
    'ollama_chat',
    'ollama_generate',
  ],
  rerank: [
    'auto',
    'cohere_v1_rerank',
    'cohere_v2_rerank',
    'vllm_score',
  ],
  generate: [
    'auto',
    'openai_chat_completions',
    'openai_completions',
    'openai_responses',
    'anthropic_messages',
    'ollama_chat',
    'ollama_generate',
  ],
};

function normalizeFormat(raw: string): string {
  return raw.trim().toLowerCase().replace(/[-\s]+/g, '_');
}

function toRemoteApiFormat(raw: string): RemoteApiFormat | undefined {
  const normalized = normalizeFormat(raw);
  const aliased = REMOTE_API_FORMAT_ALIASES[normalized];
  if (aliased) return aliased;
  if ((REMOTE_API_FORMATS as readonly string[]).includes(normalized)) {
    return normalized as RemoteApiFormat;
  }
  return undefined;
}

/**
 * Resolve and validate endpoint protocol format from env + YAML.
 *
 * Priority:
 * 1. `QMD_{ENDPOINT}_API_FORMAT`
 * 2. `models.{endpoint}_api_format` in YAML
 * 3. `auto`
 */
export function resolveEndpointFormat(
  endpoint: EndpointRole,
  envSuffix: string,
  yamlFormat?: string,
): RemoteApiFormat {
  const raw = process.env[`QMD_${envSuffix}_API_FORMAT`] || yamlFormat || 'auto';
  const parsed = toRemoteApiFormat(raw);
  if (!parsed) {
    throw new Error(
      `Invalid ${endpoint} API format '${raw}'. ` +
      `Supported formats: ${REMOTE_API_FORMATS.join(', ')}`,
    );
  }

  const allowed = ALLOWED_FORMATS_BY_ENDPOINT[endpoint];
  if (!allowed.includes(parsed)) {
    throw new Error(
      `Invalid ${endpoint} API format '${parsed}'. ` +
      `Allowed formats for ${endpoint}: ${allowed.join(', ')}`,
    );
  }
  return parsed;
}

// =============================================================================
// resolveEndpoint
// =============================================================================

/**
 * Build an EndpointConfig from environment variables with defaults.
 *
 * Follows the pattern:
 *   baseUrl = QMD_{SUFFIX}_BASE_URL || urlDefault
 *   model   = QMD_{SUFFIX}_MODEL || modelDefault
 *   apiKey  = QMD_{SUFFIX}_API_KEY || OPENAI_API_KEY
 *
 * @param _name        - Human-readable endpoint name (for logging, currently unused)
 * @param envSuffix    - Uppercase env var suffix (e.g. "EMBED", "RERANK")
 * @param modelDefault - Fallback model name if no env var is set
 * @param urlDefault   - Fallback base URL if no env var is set
 * @returns Resolved EndpointConfig
 */
export function resolveEndpoint(
  _name: string,
  envSuffix: string,
  modelDefault: string,
  urlDefault: string,
): EndpointConfig {
  const baseUrl =
    (process.env[`QMD_${envSuffix}_BASE_URL`] || urlDefault).replace(/\/+$/, '');
  const model = process.env[`QMD_${envSuffix}_MODEL`] || modelDefault;
  const apiKey = (
    process.env[`QMD_${envSuffix}_API_KEY`] ||
    process.env.OPENAI_API_KEY ||
    ''
  ).trim();
  return { baseUrl, model, apiKey };
}

// =============================================================================
// remoteConfigFromEnv
// =============================================================================

/**
 * Resolve a full RemoteLLMConfig from environment variables and optional
 * YAML models config.
 *
 * ## Priority chain (per endpoint)
 *
 * 1. Environment variables (QMD_{ENDPOINT}_BASE_URL, QMD_{ENDPOINT}_MODEL, QMD_{ENDPOINT}_API_KEY, QMD_{ENDPOINT}_API_FORMAT)
 * 2. YAML models config (embed_api_url, embed_api_format, expand_api_model, etc.)
 * 3. OPENAI_* fallback (only for embed endpoint: OPENAI_BASE_URL, OPENAI_API_KEY)
 * 4. Local-first default: empty baseUrl → falls back to local LlamaCpp
 *
 * @param models - Optional YAML models config from loadConfig()
 * @returns Resolved RemoteLLMConfig with all four endpoints populated
 */
export function remoteConfigFromEnv(models?: ModelsConfig): RemoteLLMConfig {
  // --- Embed ---
  const embed: EndpointConfig = {
    baseUrl: (
      process.env.QMD_EMBED_BASE_URL ||
      models?.embed_api_url ||
      process.env.OPENAI_BASE_URL ||
      ''
    ).replace(/\/+$/, ''),
    model: process.env.QMD_EMBED_MODEL || models?.embed_api_model || '',
    apiKey: (
      process.env.QMD_EMBED_API_KEY ||
      models?.embed_api_key ||
      process.env.OPENAI_API_KEY ||
      ''
    ).trim(),
    format: resolveEndpointFormat('embed', 'EMBED', models?.embed_api_format),
  };

  // --- Expand ---
  const expand: EndpointConfig = {
    baseUrl: (
      process.env.QMD_EXPAND_BASE_URL ||
      models?.expand_api_url ||
      ''
    ).replace(/\/+$/, ''),
    model: process.env.QMD_EXPAND_MODEL || models?.expand_api_model || '',
    apiKey: (
      process.env.QMD_EXPAND_API_KEY ||
      models?.expand_api_key ||
      process.env.OPENAI_API_KEY ||
      ''
    ).trim(),
    format: resolveEndpointFormat('expand', 'EXPAND', models?.expand_api_format),
  };

  // --- Rerank ---
  const rerank: EndpointConfig = {
    baseUrl: (
      process.env.QMD_RERANK_BASE_URL ||
      models?.rerank_api_url ||
      ''
    ).replace(/\/+$/, ''),
    model: process.env.QMD_RERANK_MODEL || models?.rerank_api_model || '',
    apiKey: (
      process.env.QMD_RERANK_API_KEY ||
      models?.rerank_api_key ||
      process.env.OPENAI_API_KEY ||
      ''
    ).trim(),
    format: resolveEndpointFormat('rerank', 'RERANK', models?.rerank_api_format),
  };

  // --- Generate ---
  const generate: EndpointConfig = {
    baseUrl: (
      process.env.QMD_GENERATE_BASE_URL ||
      models?.generate_api_url ||
      ''
    ).replace(/\/+$/, ''),
    model: process.env.QMD_GENERATE_MODEL || models?.generate_api_model || '',
    apiKey: (
      process.env.QMD_GENERATE_API_KEY ||
      models?.generate_api_key ||
      process.env.OPENAI_API_KEY ||
      ''
    ).trim(),
    format: resolveEndpointFormat('generate', 'GENERATE', models?.generate_api_format),
  };

  return { embed, expand, rerank, generate };
}
