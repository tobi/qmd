import {
  RemoteLLM,
  type ILLMSession,
  type Queryable,
  type RerankDocument,
} from "../../llm.js";
import type { LLMPort, ExpandOptions } from "../ports/llm.js";
import { createRemoteConfigFromEnv } from "../../remote-config.js";

export function createLLMService(): LLMPort {
  const remoteConfig = createRemoteConfigFromEnv();
  const remote = remoteConfig ? new RemoteLLM(remoteConfig) : null;

  type ProviderName = "siliconflow" | "gemini" | "openai" | "dashscope" | "zeroentropy";
  const providerHealth = new Map<ProviderName, { consecutiveFailures: number; cooldownUntilMs: number }>();
  const FAILURE_THRESHOLD = 3;
  const COOLDOWN_MS = 5 * 60 * 1000;

  const isCoolingDown = (provider: ProviderName): boolean => {
    const state = providerHealth.get(provider);
    if (!state) return false;
    return Date.now() < state.cooldownUntilMs;
  };

  const recordSuccess = (provider: ProviderName): void => {
    providerHealth.delete(provider);
  };

  const recordFailure = (provider: ProviderName): void => {
    const now = Date.now();
    const state = providerHealth.get(provider);
    const consecutiveFailures = (state?.consecutiveFailures ?? 0) + 1;
    const isThreshold = consecutiveFailures >= FAILURE_THRESHOLD;
    const cooldownUntilMs = isThreshold ? now + COOLDOWN_MS : (state?.cooldownUntilMs ?? 0);
    providerHealth.set(provider, { consecutiveFailures: isThreshold ? 0 : consecutiveFailures, cooldownUntilMs });
  };

  const hasRemoteProviderKey = (provider: ProviderName): boolean => {
    if (!remoteConfig) return false;
    if (provider === "siliconflow") return !!remoteConfig.siliconflow?.apiKey;
    if (provider === "gemini") return !!remoteConfig.gemini?.apiKey;
    if (provider === "openai") return !!remoteConfig.openai?.apiKey;
    if (provider === "zeroentropy") return !!remoteConfig.zeroentropy?.apiKey;
    return !!remoteConfig.dashscope?.apiKey;
  };

  const ensureRemote = (): RemoteLLM => {
    if (!remote) {
      throw new Error(
        "No remote LLM configured. Set at least one API key (e.g. QMD_SILICONFLOW_API_KEY / QMD_OPENAI_API_KEY / QMD_GEMINI_API_KEY / QMD_DASHSCOPE_API_KEY / QMD_ZEROENTROPY_API_KEY)."
      );
    }
    return remote;
  };

  return {
    async withSession<T>(fn: (session?: ILLMSession) => Promise<T>, opts?: { maxDuration?: number; name?: string }): Promise<T> {
      void opts;
      return fn(undefined);
    },

    async expandQuery(query: string, options?: ExpandOptions, session?: ILLMSession): Promise<Queryable[]> {
      void session;
      const includeLexical = options?.includeLexical ?? true;
      const context = options?.context;
      const provider = (remoteConfig?.queryExpansionProvider || remoteConfig?.rerankProvider) as ProviderName | undefined;

      const lexicalFallback = (): Queryable[] => (includeLexical ? [{ type: "lex", text: query }] : []);

      if (!remote || !provider || !hasRemoteProviderKey(provider)) {
        return lexicalFallback();
      }
      if (isCoolingDown(provider)) {
        return lexicalFallback();
      }

      try {
        const out = await remote.expandQuery(query, { includeLexical, context });
        recordSuccess(provider);
        return out;
      } catch (err) {
        recordFailure(provider);
        return lexicalFallback();
      }
    },

    async rerank(query: string, documents: RerankDocument[], session?: ILLMSession): Promise<{ file: string; score: number; extract?: string }[]> {
      void session;
      const provider = remoteConfig?.rerankProvider as ProviderName | undefined;
      const llm = ensureRemote();

      if (provider) {
        if (!hasRemoteProviderKey(provider)) {
          throw new Error(`Remote rerank provider "${provider}" is selected but its API key is missing.`);
        }
        if (isCoolingDown(provider)) {
          throw new Error(`Remote provider "${provider}" is cooling down. Please retry later.`);
        }
        try {
          const result = await llm.rerank(query, documents);
          recordSuccess(provider);
          return result.results.map(r => ({ file: r.file, score: r.score, extract: r.extract }));
        } catch (err) {
          recordFailure(provider);
          throw err;
        }
      }

      const result = await llm.rerank(query, documents);
      return result.results.map(r => ({ file: r.file, score: r.score, extract: r.extract }));
    },

    async embed(text: string, options?: { model?: string; isQuery?: boolean }, session?: ILLMSession): Promise<{ embedding: number[] }> {
      void session;
      const provider = remoteConfig?.embedProvider as ProviderName | undefined;
      const llm = ensureRemote();

      if (provider) {
        if (!hasRemoteProviderKey(provider)) {
          throw new Error(`Remote embed provider "${provider}" is selected but its API key is missing.`);
        }
        if (isCoolingDown(provider)) {
          throw new Error(`Remote provider "${provider}" is cooling down. Please retry later.`);
        }
        try {
          const result = await llm.embed(text, options);
          if (!result) throw new Error("Remote embedding returned null");
          recordSuccess(provider);
          return result;
        } catch (err) {
          recordFailure(provider);
          throw err;
        }
      }

      const result = await llm.embed(text, options);
      if (!result) throw new Error("Remote embedding returned null");
      return result;
    },
  };
}
