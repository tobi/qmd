import { afterEach, expect, test, vi } from "vitest";
import { RemoteEmbedder } from "../src/remote-embed.js";

const uri = "http://mock.invalid/v1#synthetic";
afterEach(() => vi.restoreAllMocks());

function provider(maxInputs: number, maxBytes: number) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
    const { model, input } = JSON.parse(String(init?.body)) as { model: string; input: string[] };
    if (input.length > maxInputs || Buffer.byteLength(input.join(""), "utf8") > maxBytes) {
      return new Response("sk-synthetic-secret\nforged log entry", {
        status: 400, statusText: "synthetic-secret-reason",
      });
    }
    return Response.json({
      model,
      data: input.map((text, index) => ({ index, embedding: [Number(text[0])] })).reverse(),
    });
  });
}

test("splits independently on input count and UTF-8 bytes and keeps vector order", async () => {
  const http = provider(2, 6);
  const embedder = new RemoteEmbedder(uri, { maxBatchSize: 2, maxBatchInputBytes: 6 });
  expect(embedder.modelName).toBe(new RemoteEmbedder(uri).modelName);
  expect(await embedder.embedBatch(["1", "2", "3é", "4éé", "5"])).toEqual([[1], [2], [3], [4], [5]]);
  expect(http.mock.calls.map(([, init]) => JSON.parse(String(init?.body)).input))
    .toEqual([["1", "2"], ["3é"], ["4éé", "5"]]);
});

test("rejects an individually oversized input before making any requests", async () => {
  const http = provider(1, 3);
  const embedder = new RemoteEmbedder(uri, { maxBatchSize: 1, maxBatchInputBytes: 3 });
  await expect(embedder.embedBatch(["1", "2éé"])).rejects.toThrow("Input 2 uses 5 UTF-8 bytes");
  expect(http).not.toHaveBeenCalled();
});

test.each([
  [1, 10, ["1", "2"]],
  [10, 3, ["1é", "2"]],
] as const)("reports HTTP status for provider limits %i/%i and suppresses response text", async (count, bytes, inputs) => {
  provider(count, bytes);
  const timeout = vi.spyOn(AbortSignal, "timeout");
  await expect(new RemoteEmbedder(uri).embedBatch([...inputs]))
    .rejects.toThrow(/^Remote model request failed \(HTTP 400\)$/);
  expect(timeout).not.toHaveBeenCalled();
});

test("rejects a dimension mismatch on the second request without dispatching the third", async () => {
  const http = provider(1, 10);
  const vector = (embedding: number[]) => Response.json({ model: "synthetic", data: [{ index: 0, embedding }] });
  http.mockResolvedValueOnce(vector([1])).mockResolvedValueOnce(vector([1, 2]));
  await expect(new RemoteEmbedder(uri, { maxBatchSize: 1 }).embedBatch(["1", "2", "3"]))
    .rejects.toThrow("Invalid remote embedding response");
  expect(http).toHaveBeenCalledTimes(2);
});

test.each(["deadline", "caller"] as const)("aborts an active split request through the %s signal", async source => {
  const caller = new AbortController();
  const deadline = new AbortController();
  const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(deadline.signal);
  const http = provider(1, 10);
  http.mockResolvedValueOnce(Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1] }] }));
  http.mockImplementationOnce(async (_url, init) => {
    expect(init?.signal?.aborted).toBe(false);
    (source === "deadline" ? deadline : caller).abort();
    expect(init?.signal?.aborted).toBe(true);
    return Response.json({ model: "synthetic", data: [{ index: 0, embedding: [2] }] });
  });
  await expect(new RemoteEmbedder(uri, { maxBatchSize: 1 }).embedBatch(["1", "2", "3"], caller.signal))
    .rejects.toThrow("Remote model request aborted");
  expect(timeout).toHaveBeenCalledTimes(1);
  expect(timeout).toHaveBeenCalledWith(30_000);
  expect(http).toHaveBeenCalledTimes(2);
});

test.each([
  { maxBatchSize: 0 }, { maxBatchSize: 65 }, { maxBatchSize: 1.5 },
  { maxBatchInputBytes: -1 }, { maxBatchInputBytes: Infinity },
])("rejects invalid scheduling options %j", options => {
  expect(() => new RemoteEmbedder(uri, options)).toThrow("Invalid remote embedding options");
});
