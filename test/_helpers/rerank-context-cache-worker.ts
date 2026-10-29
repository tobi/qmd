import { LlamaCpp } from "../../src/llm.js";
import { createStore } from "../../src/store.js";

const dbPath = process.argv[2];
const contextTokens = Number(process.env.QMD_RERANK_CONTEXT_SIZE);
if (!dbPath || !Number.isSafeInteger(contextTokens) || contextTokens <= 0) {
  throw new Error("An owned database and positive rerank context are required");
}
const store = createStore(dbPath);
const llm = new LlamaCpp({ rerankModel: "hf:example/context-reranker/model.gguf" });
llm.rerank = async (_query, documents) => ({
  results: documents.map((doc, index) => ({ file: doc.file, index, score: contextTokens / 10_000 })),
  model: llm.rerankModelName,
});
store.llm = llm;
try {
  const result = await store.rerank("document meaning", [
    { file: "long.md", text: "Long source material with different possible truncation. ".repeat(10_000) },
  ]);
  console.log(JSON.stringify({ score: result[0]?.score }));
} finally {
  await llm.dispose();
  store.close();
}
