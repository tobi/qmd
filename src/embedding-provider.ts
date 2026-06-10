/**
 * embedding-provider.ts — Backward-compat barrel for remote LLM modules.
 *
 * All implementation has moved to src/remote/. This file re-exports
 * the public API unchanged so existing imports continue to work.
 *
 * @module embedding-provider
 */

// Re-export the RemoteLLM class (primary public API)
export { RemoteLLM } from './remote/remote-llm.js';

// Re-export types used by consumers (qmd.ts, index.ts)
export type { EndpointConfig, RemoteLLMConfig } from './remote/types.js';

// Re-export config resolution (used by qmd.ts CLI auto-detection)
export { remoteConfigFromEnv } from './remote/config.js';

// Re-export logger interface and implementations (for tests and custom consumers)
export type { Logger } from './remote/log.js';
export { consoleLogger, silentLogger } from './remote/log.js';
