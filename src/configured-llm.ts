import type { ModelsConfig } from "./collections.js";
import { HybridLLM } from "./hybrid-llm.js";
import { LlamaCpp, type LLM, type LlamaCppConfig } from "./llm.js";
import { RemoteLLM, remoteConfigFromEnv } from "./remote-llm.js";

/**
 * Build the LLM backend implied by config/env.
 *
 * Remote embedding is opt-in via remoteConfigFromEnv(). When configured, a
 * HybridLLM keeps local generation/tokenization/fallback behavior while routing
 * remote-capable operations through the OpenAI-compatible API.
 */
export function createConfiguredLLM(
  models?: ModelsConfig,
  localConfig: LlamaCppConfig = {},
): LLM {
  const remoteConfig = remoteConfigFromEnv(models);
  const local = new LlamaCpp(localConfig);
  if (!remoteConfig) return local;
  return new HybridLLM(new RemoteLLM(remoteConfig), local);
}
