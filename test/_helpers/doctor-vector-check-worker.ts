import { createStore } from "../../src/store.js";
import { checkEmbeddingVectorSamples } from "../../src/cli/qmd.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../../src/llm.js";
import { heapLimitBinds } from "./heap-limit.js";

// Every stored vector is [1, 0] and every re-embedded chunk returns [1, 0], so
// the check passes whenever it can sample, read and chunk the bodies.
class ConstantLlm extends LlamaCpp {
  async tokenize(text: string) { return new Array(Math.ceil(text.length / 16)).fill(1); }
  async embed() { return { embedding: [1, 0], model: "model" }; }
}

const store = createStore(":memory:");
try {
  setDefaultLlamaCpp(new ConstantLlm());
  const body = "word ".repeat(13_107); // About 64 KiB per document.
  store.ensureVecTable(2);
  store.db.transaction(() => {
    for (let doc = 0; doc < 16; doc++) {
      const hash = `document-${doc}`;
      store.insertContent(hash, body, "2026-01-01");
      for (let path = 0; path < 8; path++) {
        store.insertDocument("test", `${hash}-${path}.md`, hash, hash, "2026-01-01", "2026-01-01");
      }
      for (let seq = 0; seq < 32; seq++) {
        store.insertEmbedding(hash, seq, 0, new Float32Array([1, 0]), "model", "2026-01-01", 32, "current");
      }
    }
  })();
  store.db.exec("PRAGMA temp_store = MEMORY");
  store.db.exec("PRAGMA hard_heap_limit = 33554432");
  const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
  console.log(JSON.stringify({ ...result, limitBinds: heapLimitBinds(store.db, 33554432) }));
} finally {
  setDefaultLlamaCpp(null);
  store.close();
}
