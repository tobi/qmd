import type { ILLMSession, Queryable, RerankDocument } from "../../llm.js";

export type ExpandOptions = {
  includeLexical?: boolean;
  context?: string;
};

export interface LLMPort {
  withSession<T>(fn: (session?: ILLMSession) => Promise<T>, opts?: { maxDuration?: number; name?: string }): Promise<T>;
  expandQuery(query: string, options?: ExpandOptions, session?: ILLMSession): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], session?: ILLMSession): Promise<{ file: string; score: number; extract?: string }[]>;
  embed(text: string, options?: { model?: string; isQuery?: boolean }, session?: ILLMSession): Promise<{ embedding: number[] }>;
}
