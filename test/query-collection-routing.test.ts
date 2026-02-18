import { describe, test, expect, vi } from "vitest";
import {
  hybridQuery,
  vectorSearchQuery,
  type Store,
  type SearchResult,
} from "../src/store.js";

describe("multi-collection routing in query pipelines", () => {
  test("hybridQuery passes array collection filters into FTS", async () => {
    const searchFTS = vi.fn().mockReturnValue([] as SearchResult[]);

    const store = {
      db: {
        prepare: () => ({ get: () => undefined }), // no vectors table
      },
      searchFTS,
      expandQuery: vi.fn().mockResolvedValue([]),
    } as unknown as Store;

    await hybridQuery(store, "dominator", {
      collection: ["target-a", "target-b"],
      limit: 10,
    });

    expect(searchFTS).toHaveBeenCalledWith("dominator", 20, ["target-a", "target-b"]);
  });

  test("vectorSearchQuery passes array collection filters into vector search", async () => {
    const vectorResult: SearchResult = {
      filepath: "qmd://target-a/hit-a.md",
      displayPath: "target-a/hit-a.md",
      title: "Hit A",
      hash: "abcdef123456",
      docid: "abcdef",
      collectionName: "target-a",
      modifiedAt: "",
      bodyLength: 10,
      body: "dominator",
      context: null,
      score: 0.9,
      source: "vec",
    };

    const searchVec = vi.fn().mockResolvedValue([vectorResult]);

    const store = {
      db: {
        prepare: () => ({ get: () => ({ name: "vectors_vec" }) }),
      },
      expandQuery: vi.fn().mockResolvedValue([]),
      searchVec,
      getContextForFile: vi.fn().mockReturnValue(null),
    } as unknown as Store;

    await vectorSearchQuery(store, "dominator", {
      collection: ["target-a", "target-b"],
      limit: 7,
      minScore: 0,
    });

    expect(searchVec).toHaveBeenCalledWith(
      "dominator",
      expect.any(String),
      7,
      ["target-a", "target-b"],
    );
  });
});
