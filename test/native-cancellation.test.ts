import { expect, test } from "vitest";
import { LlamaCpp, withLLMSessionForLlm } from "../src/llm.js";

const turn = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r; }); return { promise, resolve }; }
function fixture(contexts: unknown[]) {
  const llm = Object.create(LlamaCpp.prototype) as LlamaCpp;
  Object.assign(llm, { _ciMode: false, embedModelUri: "fixture", touchActivity() {},
    ensureEmbedContexts: async () => contexts, ensureEmbedContext: async () => contexts[0],
    truncateToContextSize: async (text: string) => ({ text, truncated: false, limit: 100 }),
  });
  return llm;
}
test("pre-aborted native embedding admits no shared setup", async () => {
  let setup = 0;
  const llm = fixture([]), controller = new AbortController(); controller.abort();
  Object.assign(llm, { ensureEmbedContext: async () => { setup++; throw new Error("unexpected setup"); },
    ensureEmbedContexts: async () => { setup++; throw new Error("unexpected setup"); } });
  expect(await llm.embed("x", { signal: controller.signal })).toBeNull();
  expect(await llm.embedBatch(["a", "b"], { signal: controller.signal })).toEqual([null, null]);
  expect(setup).toBe(0);
});
test("one cancelled caller cannot cancel shared model setup for another caller", async () => {
  const held = deferred(), controller = new AbortController(); let native = 0;
  const context = { getEmbeddingFor: async () => { native++; return { vector: [1] }; } };
  const llm = fixture([context]);
  Object.assign(llm, { ensureEmbedContexts: async () => { await held.promise; return [context]; } });
  const cancelled = llm.embedBatch(["a"], { signal: controller.signal });
  const unaffected = llm.embedBatch(["b"]);
  await turn(); controller.abort(); held.resolve();
  expect(await cancelled).toEqual([null]);
  expect((await unaffected)[0]?.embedding).toEqual([1]); expect(native).toBe(1);
});
test("abort during token preparation admits no native evaluation", async () => {
  const controller = new AbortController(); let native = 0;
  const llm = fixture([{ getEmbeddingFor: async () => { native++; return { vector: [1] }; } }]);
  Object.assign(llm, { truncateToContextSize: async (text: string) => { controller.abort(); return { text }; } });
  expect(await llm.embed("x", { signal: controller.signal })).toBeNull(); expect(native).toBe(0);
});
test("sequential cancellation awaits admitted evaluation and preserves result positions", async () => {
  const held = deferred(), controller = new AbortController(); let native = 0, settled = false;
  const llm = fixture([{ getEmbeddingFor: async () => { native++; await held.promise; return { vector: [1] }; } }]);
  const pending = llm.embedBatch(["a", "b", "c"], { signal: controller.signal }).then(value => { settled = true; return value; });
  await turn(); controller.abort(); await turn(); expect(settled).toBe(false); expect(native).toBe(1);
  held.resolve(); const value = await pending; expect(value).toHaveLength(3);
  expect(value[0]?.embedding).toEqual([1]); expect(value.slice(1)).toEqual([null, null]); expect(native).toBe(1);
});
test("parallel cancellation stops each next evaluation with stable input ordering", async () => {
  const held = deferred(), controller = new AbortController(); let native = 0;
  const context = () => ({ getEmbeddingFor: async () => { native++; await held.promise; return { vector: [1] }; } });
  const llm = fixture([context(), context()]);
  const pending = llm.embedBatch(["a", "b", "c", "d"], { signal: controller.signal });
  await turn(); expect(native).toBe(2); controller.abort(); held.resolve(); const value = await pending;
  expect(value).toHaveLength(4); expect(value[0]?.embedding).toEqual([1]); expect(value[2]?.embedding).toEqual([1]);
  expect(value[1]).toBeNull(); expect(value[3]).toBeNull(); expect(native).toBe(2);
});
test("LLM session forwards its cancellation signal into native admission", async () => {
  const held = deferred(), controller = new AbortController(); let native = 0;
  const llm = fixture([{ getEmbeddingFor: async () => { native++; await held.promise; return { vector: [1] }; } }]);
  const pending = withLLMSessionForLlm(llm, session => session.embedBatch(["a", "b"]), { signal: controller.signal });
  await turn(); controller.abort(); held.resolve(); const value = await pending;
  expect(native).toBe(1); expect(value[0]?.embedding).toEqual([1]); expect(value[1]).toBeNull();
});
