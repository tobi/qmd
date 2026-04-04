/**
 * remote-config.ts - Build RemoteLLMConfig from environment variables.
 *
 * Single source of truth for translating QMD_* env vars into a
 * RemoteLLMConfig that RemoteLLM can consume. Called by:
 *   - src/store.ts  (reranking path)
 *   - src/app/services/llm-service.ts  (LLMPort service)
 *
 * Supported providers:
 *   embed        : siliconflow | openai
 *   query expand : siliconflow | gemini | openai
 *   rerank       : siliconflow | gemini | openai | dashscope | zeroentropy
 *
 * Returns null when no API keys are configured (use local LLM instead).
 */

import type { RemoteLLMConfig } from "./llm.js";

type RerankProvider = "siliconflow" | "gemini" | "openai" | "dashscope" | "zeroentropy";

export function createRemoteConfigFromEnv(): RemoteLLMConfig | null {
  const sfApiKey  = process.env.QMD_SILICONFLOW_API_KEY;
  const gmApiKey  = process.env.QMD_GEMINI_API_KEY;
  const oaApiKey  = process.env.QMD_OPENAI_API_KEY;
  const dsApiKey  = process.env.QMD_DASHSCOPE_API_KEY;
  const zeApiKey  = process.env.QMD_ZEROENTROPY_API_KEY;

  if (!sfApiKey && !gmApiKey && !oaApiKey && !dsApiKey && !zeApiKey) return null;

  const rerankMode = (process.env.QMD_RERANK_MODE as "llm" | "rerank" | undefined) || "llm";
  const sfLlmRerankModel =
    process.env.QMD_SILICONFLOW_LLM_RERANK_MODEL ||
    process.env.QMD_LLM_RERANK_MODEL ||
    "zai-org/GLM-4.5-Air";

  const configuredRerankProvider = process.env.QMD_RERANK_PROVIDER as RerankProvider | undefined;

  let rerankProvider: RerankProvider | undefined;
  if (rerankMode === "rerank") {
    if (configuredRerankProvider === "dashscope" && dsApiKey)        rerankProvider = "dashscope";
    else if (configuredRerankProvider === "zeroentropy" && zeApiKey) rerankProvider = "zeroentropy";
    else if (sfApiKey)                                               rerankProvider = "siliconflow";
    else if (configuredRerankProvider === "gemini" && gmApiKey)      rerankProvider = "gemini";
    else if (configuredRerankProvider === "openai" && oaApiKey)      rerankProvider = "openai";
    else if (dsApiKey)                                               rerankProvider = "dashscope";
    else if (zeApiKey)                                               rerankProvider = "zeroentropy";
    else rerankProvider = gmApiKey ? "gemini" : (oaApiKey ? "openai" : undefined);
  } else {
    if (configuredRerankProvider === "dashscope" && dsApiKey)        rerankProvider = "dashscope";
    else if (configuredRerankProvider === "zeroentropy" && zeApiKey) rerankProvider = "zeroentropy";
    else if (configuredRerankProvider === "gemini" || configuredRerankProvider === "openai") rerankProvider = configuredRerankProvider;
    else if (configuredRerankProvider === "siliconflow")             rerankProvider = sfApiKey ? "siliconflow" : undefined;
    else rerankProvider = dsApiKey ? "dashscope" : (zeApiKey ? "zeroentropy" : (sfApiKey ? "siliconflow" : (gmApiKey ? "gemini" : (oaApiKey ? "openai" : undefined))));
  }

  const embedProvider = (process.env.QMD_EMBED_PROVIDER as "siliconflow" | "openai" | undefined)
    || (sfApiKey ? "siliconflow" : (oaApiKey ? "openai" : undefined));

  const queryExpansionProvider = (process.env.QMD_QUERY_EXPANSION_PROVIDER as "siliconflow" | "gemini" | "openai" | undefined)
    || (sfApiKey ? "siliconflow" : (oaApiKey ? "openai" : (gmApiKey ? "gemini" : undefined)));

  if (!rerankProvider && !embedProvider && !queryExpansionProvider) return null;

  const config: RemoteLLMConfig = {
    rerankProvider: rerankProvider || "siliconflow",
    rerankMode,
    embedProvider,
    queryExpansionProvider,
  };

  if (sfApiKey) {
    config.siliconflow = {
      apiKey: sfApiKey,
      baseUrl: process.env.QMD_SILICONFLOW_BASE_URL,
      model: process.env.QMD_SILICONFLOW_RERANK_MODEL || process.env.QMD_SILICONFLOW_MODEL,
      embedModel: process.env.QMD_SILICONFLOW_EMBED_MODEL,
      queryExpansionModel: process.env.QMD_SILICONFLOW_QUERY_EXPANSION_MODEL,
    };
  }

  if (gmApiKey) {
    config.gemini = {
      apiKey: gmApiKey,
      baseUrl: process.env.QMD_GEMINI_BASE_URL,
      model: process.env.QMD_GEMINI_RERANK_MODEL || process.env.QMD_GEMINI_MODEL,
    };
  }

  if (oaApiKey || (rerankProvider === "openai" && sfApiKey)) {
    config.openai = {
      apiKey: oaApiKey || sfApiKey || "",
      baseUrl: process.env.QMD_OPENAI_BASE_URL || process.env.QMD_SILICONFLOW_BASE_URL,
      model: process.env.QMD_OPENAI_MODEL || (sfApiKey ? sfLlmRerankModel : undefined),
      embedModel: process.env.QMD_OPENAI_EMBED_MODEL,
      embedApiKey:  process.env.QMD_EMBED_OPENAI_API_KEY,
      embedBaseUrl: process.env.QMD_EMBED_OPENAI_BASE_URL,
      rerankApiKey: process.env.QMD_RERANK_OPENAI_API_KEY,
      rerankBaseUrl: process.env.QMD_RERANK_OPENAI_BASE_URL,
      queryApiKey:  process.env.QMD_QUERY_OPENAI_API_KEY,
      queryBaseUrl: process.env.QMD_QUERY_OPENAI_BASE_URL,
    };
  }

  if (dsApiKey || rerankProvider === "dashscope") {
    config.dashscope = {
      apiKey: dsApiKey || "",
      baseUrl: process.env.QMD_DASHSCOPE_BASE_URL,
      model: process.env.QMD_DASHSCOPE_RERANK_MODEL,
    };
  }

  if (zeApiKey || rerankProvider === "zeroentropy") {
    config.zeroentropy = {
      apiKey: zeApiKey || "",
      baseUrl: process.env.QMD_ZEROENTROPY_BASE_URL,
      model: process.env.QMD_ZEROENTROPY_RERANK_MODEL,
    };
  }

  return config;
}
