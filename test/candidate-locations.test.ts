import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore as createSdkStore,
  type CandidateSearchOptions,
  type QMDStore,
} from "../src/index.js";
import {
  hashContent,
  searchVec,
  type CollectionScope,
  type ExpandedQuery,
  type SearchRetrievalOptions,
  type Store,
} from "../src/store.js";
import { replaceDocumentMetadata } from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION } from "../src/metadata.js";

const MODEL = "candidate-location-test-model";
const GROUP_KEY = "thread";

type StoredDocument = {
  uri: string;
  hash: string;
  body: string;
};

type RetrievalCall = {
  source: "fts" | "vec";
  query: string;
  scope: CollectionScope;
  retrieval: SearchRetrievalOptions | undefined;
};

type RerankCall = {
  query: string;
  intent: string | undefined;
  documents: { file: string; text: string }[];
};

let testDir: string;
let sdk: QMDStore;
let store: Store;
const documents = new Map<string, StoredDocument>();
const retrievalCalls: RetrievalCall[] = [];

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-candidate-locations-"));
  const names = [
    "late-lex",
    "late-vec",
    "unavailable",
    "siblings",
    "rerank",
    "parity",
  ] as const;
  const collections = Object.fromEntries(names.map(name => [
    name,
    { path: join(testDir, name), pattern: "**/*.md" },
  ]));
  await Promise.all(names.map(name => mkdir(join(testDir, name), { recursive: true })));

  sdk = await createSdkStore({
    dbPath: join(testDir, "candidate-locations.sqlite"),
    config: { collections },
  });
  store = sdk.internal;
  store.ensureVecTable(2);

  await seedLateLexicalDocument();
  await seedLateVectorDocument();
  await seedUnavailableDocuments();
  await seedSiblingDocuments();
  await seedRerankDocument();
  await seedParityDocument();
  installFaithfulRetrievalWrappers();
});

afterAll(async () => {
  await sdk.close();
  await rm(testDir, { recursive: true, force: true });
});

async function insertDocument(
  label: string,
  collection: string,
  path: string,
  title: string,
  body: string,
  group: string,
): Promise<StoredDocument> {
  const now = new Date().toISOString();
  const hash = await hashContent(body);
  store.insertContent(hash, body, now);
  const documentId = store.insertDocument(collection, path, title, hash, now, now);
  replaceDocumentMetadata(store.db, documentId, {
    metadata: { [GROUP_KEY]: group },
    extractionVersion: METADATA_EXTRACTION_VERSION,
  });
  const document = { uri: `qmd://${collection}/${path}`, hash, body };
  documents.set(label, document);
  return document;
}

async function seedLateLexicalDocument(): Promise<void> {
  const prefix = `${".".repeat(280_024)} `;
  const body = `${prefix}Orbit command ${"z".repeat(60_000)}`;
  await insertDocument(
    "lateLex",
    "late-lex",
    "orbit.md",
    "Late source anchor",
    body,
    "late-lexical",
  );
}

async function seedLateVectorDocument(): Promise<void> {
  const body = `${"😀".repeat(9_001)}late vector marker`;
  const document = await insertDocument(
    "lateVec",
    "late-vec",
    "vector.md",
    "Late vector source",
    body,
    "late-vector",
  );
  store.insertEmbedding(
    document.hash,
    7,
    body.indexOf("late vector marker"),
    new Float32Array([1, 0]),
    MODEL,
    new Date().toISOString(),
    8,
  );
}

async function seedUnavailableDocuments(): Promise<void> {
  await insertDocument(
    "titleOnly",
    "unavailable",
    "title-only.md",
    "Titlemarker source",
    "Body text chosen by keyword intent fallback.",
    "title-only",
  );
  await insertDocument(
    "pathOnly",
    "unavailable",
    "pathmarker.md",
    "Path source",
    "Another body chosen by keyword intent fallback.",
    "path-only",
  );
  await insertDocument(
    "unsupportedStem",
    "unavailable",
    "stem.md",
    "Porter source",
    "We relate systems safely.",
    "unsupported-stem",
  );
}

async function seedSiblingDocuments(): Promise<void> {
  const lexical = await insertDocument(
    "siblingLex",
    "siblings",
    "lexical.md",
    "Lexical sibling",
    "prefix lexicalmarker suffix",
    "shared-thread",
  );
  const vectorB = await insertDocument(
    "siblingVecB",
    "siblings",
    "vector-b.md",
    "Vector sibling B",
    "B body vector anchor",
    "shared-thread",
  );
  const vectorC = await insertDocument(
    "siblingVecC",
    "siblings",
    "vector-c.md",
    "Vector sibling C",
    "C body vector anchor",
    "shared-thread",
  );
  const now = new Date().toISOString();
  store.insertEmbedding(lexical.hash, 0, 0, new Float32Array([-1, 0]), MODEL, now);
  store.insertEmbedding(
    vectorB.hash,
    2,
    vectorB.body.indexOf("vector"),
    new Float32Array([1, 0]),
    MODEL,
    now,
    3,
  );
  store.insertEmbedding(
    vectorC.hash,
    4,
    vectorC.body.indexOf("vector"),
    new Float32Array([0, 1]),
    MODEL,
    now,
    5,
  );
}

async function seedRerankDocument(): Promise<void> {
  await insertDocument(
    "rerank",
    "rerank",
    "rerank.md",
    "Rerank source",
    `${"😀".repeat(30)} exactmarker ${"😀".repeat(30)}`,
    "rerank",
  );
}

async function seedParityDocument(): Promise<void> {
  await insertDocument(
    "parity",
    "parity",
    "parity.md",
    "Parity source",
    "paritymarker stable ordinary search body",
    "parity",
  );
}

function queryVector(query: string): number[] {
  if (query === "toward-c") return [0, 1];
  return [1, 0];
}

function installFaithfulRetrievalWrappers(): void {
  const db = store.db;
  const actualSearchFts = store.searchFTS;
  store.searchFTS = (query, limit, scope, filter, retrieval) => {
    retrievalCalls.push({ source: "fts", query, scope, retrieval });
    return actualSearchFts(query, limit, scope, filter, retrieval);
  };
  store.searchVec = async (query, model, limit, scope, session, _embedding, filter, retrieval) => {
    retrievalCalls.push({ source: "vec", query, scope, retrieval });
    return searchVec(
      db,
      query,
      model,
      limit,
      scope,
      session,
      queryVector(query),
      undefined,
      filter,
      retrieval,
    );
  };
}

function candidateOptions(
  collection: string,
  queries: readonly ExpandedQuery[],
  extra: Pick<CandidateSearchOptions, "locations" | "passage" | "rerank"> = {},
): CandidateSearchOptions {
  return {
    queries,
    collection,
    candidates: {
      rawLimitPerLeg: 5,
      group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 5 },
    },
    candidateLimit: 5,
    limit: 5,
    rerank: extra.rerank ?? false,
    ...(extra.locations === undefined ? {} : { locations: extra.locations }),
    ...(extra.passage === undefined ? {} : { passage: extra.passage }),
  };
}

function installDeterministicReranker(maxDocumentTokens = 256): {
  budgetCalls: { query: string; intent: string | undefined }[];
  rerankCalls: RerankCall[];
  restore: () => void;
} {
  const budgetCalls: { query: string; intent: string | undefined }[] = [];
  const rerankCalls: RerankCall[] = [];
  const actualGetRerankTokenBudget = store.getRerankTokenBudget;
  const actualRerank = store.rerank;

  store.getRerankTokenBudget = async (query, intent) => {
    budgetCalls.push({ query, intent });
    return {
      maxDocumentTokens,
      countTokens: text => Array.from(text).length,
    };
  };
  store.rerank = async (query, rerankDocuments, _model, intent) => {
    rerankCalls.push({
      query,
      intent,
      documents: rerankDocuments.map(document => ({ ...document })),
    });
    return rerankDocuments.map(document => ({ file: document.file, score: 0.75 }));
  };

  return {
    budgetCalls,
    rerankCalls,
    restore: () => {
      store.getRerankTokenBudget = actualGetRerankTokenBudget;
      store.rerank = actualRerank;
    },
  };
}

function singleResult<T extends { results: unknown[] }>(result: T): T["results"][number] {
  expect(result.results).toHaveLength(1);
  return result.results[0]!;
}

describe("representative source anchors", () => {
  test("locates a positive phrase after UTF-16 offset 280025 and omits the full body", async () => {
    const source = documents.get("lateLex")!;
    const result = await sdk.searchCandidates(candidateOptions(
      "late-lex",
      [{ type: "lex", query: '-"ignored phrase" "Orbit command"' }],
      { locations: true },
    ));
    const hit = singleResult(result);
    const lexical = hit.locations?.find(location => location.kind === "lexical_exact");

    expect(hit.file).toBe(source.uri);
    expect(hit.contentHash).toBe(source.hash);
    expect(hit.representativeLeg).toBe(0);
    expect(hit.bodyAnchorStatus).toBe("located");
    expect(lexical).toEqual({
      kind: "lexical_exact",
      uri: source.uri,
      contentHash: source.hash,
      anchorKind: "phrase",
      matchedText: "Orbit command",
      startUtf16: 280_025,
      endUtf16: 280_025 + "Orbit command".length,
    });
    expect(hit.locations?.filter(location => location.kind.startsWith("lexical"))).toHaveLength(1);
    expect(hit.passage?.text).toContain("Orbit command");
    expect(hit.passage?.utf8Bytes).toBe(48_000);
    expect(hit.passage?.anchorClipped).toBe(false);
    expect(Object.hasOwn(hit, "body")).toBe(false);

    const json = JSON.stringify(result);
    expect(Buffer.byteLength(json, "utf8")).toBeLessThan(55_000);
    expect(json.length).toBeLessThan(source.body.length / 4);
  });

  test("reports the stored vector chunk after 9001 astral scalars", async () => {
    const source = documents.get("lateVec")!;
    const startUtf16 = source.body.indexOf("late vector marker");
    const ordinary = (await store.searchVec("late-vector", MODEL, 1, "late-vec"))[0]!;
    const compact = (await store.searchVec(
      "late-vector",
      MODEL,
      1,
      "late-vec",
      undefined,
      undefined,
      undefined,
      { includeBody: false },
    ))[0]!;
    const result = await sdk.searchCandidates(candidateOptions(
      "late-vec",
      [{ type: "vec", query: "late-vector" }],
      { locations: true, passage: { maxUtf8Bytes: 100 } },
    ));
    const hit = singleResult(result);
    const vector = hit.locations?.find(location => location.kind === "vector_chunk_start");

    expect(startUtf16).toBe(18_002);
    expect(ordinary.body).toBe(source.body);
    expect(ordinary.chunkSeq).toBeUndefined();
    expect(Object.hasOwn(compact, "body")).toBe(false);
    expect(compact.chunkSeq).toBe(7);
    expect(hit.representativeLeg).toBe(0);
    expect(hit.bodyAnchorStatus).toBe("located");
    expect(hit.matches[0]).toMatchObject({
      file: source.uri,
      contentHash: source.hash,
      vectorStartUtf16: 18_002,
      vectorChunkSeq: 7,
    });
    expect(vector).toEqual({
      kind: "vector_chunk_start",
      uri: source.uri,
      contentHash: source.hash,
      startUtf16: 18_002,
      endUtf16: null,
      chunkSeq: 7,
    });
    expect(hit.passage?.text).toContain("late vector marker");
    expect(hit.passage?.utf8Bytes).toBeLessThanOrEqual(100);
  });

  test("identifies the representative leg when one file has lexical and vector anchors", async () => {
    const source = documents.get("lateVec")!;
    const result = await sdk.searchCandidates(candidateOptions(
      "late-vec",
      [
        { type: "lex", query: '"vector marker"' },
        { type: "vec", query: "late-vector" },
      ],
      { locations: true, passage: { maxUtf8Bytes: 100 } },
    ));
    const hit = singleResult(result);
    const lexical = hit.locations?.find(location => location.kind === "lexical_exact");
    const vector = hit.locations?.find(location => location.kind === "vector_chunk_start");

    expect(hit.file).toBe(source.uri);
    expect(hit.matches.map(match => [match.source, match.weight])).toEqual([
      ["fts", 2],
      ["vec", 1],
    ]);
    expect(hit.representativeLeg).toBe(0);
    expect(lexical).toMatchObject({
      uri: source.uri,
      contentHash: source.hash,
      startUtf16: source.body.indexOf("vector marker"),
    });
    expect(vector).toMatchObject({
      uri: source.uri,
      contentHash: source.hash,
      startUtf16: source.body.indexOf("late vector marker"),
      chunkSeq: 7,
    });
    expect(hit.passage?.text).toContain("vector marker");
  });
});

describe("unavailable body anchors", () => {
  for (const [label, query] of [
    ["titleOnly", "titlemarker"],
    ["pathOnly", "pathmarker"],
    ["unsupportedStem", "relational"],
  ] as const) {
    test(`${label} uses a separately labeled keyword-intent selection`, async () => {
      const source = documents.get(label)!;
      const result = await sdk.searchCandidates(candidateOptions(
        "unavailable",
        [{ type: "lex", query }],
        { locations: true, passage: { maxUtf8Bytes: 128 } },
      ));
      const hit = singleResult(result);

      expect(hit.file).toBe(source.uri);
      expect(hit.bodyAnchorStatus).toBe("unavailable");
      expect(hit.locations).toEqual([{
        kind: "selection_window",
        origin: "keyword_intent",
        uri: source.uri,
        contentHash: source.hash,
        startUtf16: 0,
        endUtf16: source.body.length,
      }]);
      expect(hit.passage?.text).toBe(source.body);
    });
  }
});

describe("group provenance", () => {
  test("keeps each vector source and emits lexical evidence for the representative file", async () => {
    const lexicalSource = documents.get("siblingLex")!;
    const vectorB = documents.get("siblingVecB")!;
    const vectorC = documents.get("siblingVecC")!;
    const result = await sdk.searchCandidates({
      ...candidateOptions(
        "siblings",
        [
          { type: "lex", query: "lexicalmarker" },
          { type: "vec", query: "toward-b" },
          { type: "vec", query: "toward-c" },
        ],
        { locations: true, passage: { maxUtf8Bytes: 96 } },
      ),
      candidates: {
        rawLimitPerLeg: 1,
        group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 1 },
      },
      candidateLimit: 1,
      limit: 1,
    });
    const hit = singleResult(result);
    const lexical = hit.locations?.filter(location => location.kind.startsWith("lexical")) ?? [];
    const vectors = hit.locations?.filter(location => location.kind === "vector_chunk_start") ?? [];

    expect(hit.file).toBe(lexicalSource.uri);
    expect(hit.representativeLeg).toBe(0);
    expect(lexical).toEqual([{
      kind: "lexical_exact",
      uri: lexicalSource.uri,
      contentHash: lexicalSource.hash,
      anchorKind: "literal",
      matchedText: "lexicalmarker",
      startUtf16: lexicalSource.body.indexOf("lexicalmarker"),
      endUtf16: lexicalSource.body.indexOf("lexicalmarker") + "lexicalmarker".length,
    }]);
    expect(vectors).toEqual([
      {
        kind: "vector_chunk_start",
        uri: vectorB.uri,
        contentHash: vectorB.hash,
        startUtf16: vectorB.body.indexOf("vector"),
        endUtf16: null,
        chunkSeq: 2,
      },
      {
        kind: "vector_chunk_start",
        uri: vectorC.uri,
        contentHash: vectorC.hash,
        startUtf16: vectorC.body.indexOf("vector"),
        endUtf16: null,
        chunkSeq: 4,
      },
    ]);
    expect(hit.matches.filter(match => match.source === "vec").map(match => ({
      file: match.file,
      contentHash: match.contentHash,
      seq: match.vectorChunkSeq,
    }))).toEqual([
      { file: vectorB.uri, contentHash: vectorB.hash, seq: 2 },
      { file: vectorC.uri, contentHash: vectorC.hash, seq: 4 },
    ]);
    expect(hit.passage?.text).toContain("lexicalmarker");
  });
});

describe("passage budgets and reranking", () => {
  test("uses the returned bounded passage as the reranker input", async () => {
    const reranker = installDeterministicReranker(12);
    try {
      const result = await sdk.searchCandidates(candidateOptions(
        "rerank",
        [{ type: "lex", query: "exactmarker" }],
        {
          locations: true,
          passage: { maxUtf8Bytes: 48, maxUnicodeScalars: 48 },
          rerank: true,
        },
      ));
      const hit = singleResult(result);

      expect(hit.passage?.text).toContain("exactmarker");
      expect(hit.passage?.utf8Bytes).toBeLessThanOrEqual(48);
      expect(hit.passage?.unicodeScalars).toBeLessThanOrEqual(12);
      expect(reranker.budgetCalls).toEqual([{ query: "exactmarker", intent: undefined }]);
      expect(reranker.rerankCalls).toHaveLength(1);
      expect(reranker.rerankCalls[0]!.query).toBe("exactmarker");
      expect(reranker.rerankCalls[0]!.documents).toEqual([{
        file: hit.group.key,
        text: hit.passage!.text,
      }]);
    } finally {
      reranker.restore();
    }
  });

  test("returns a passage without exposing location metadata", async () => {
    const source = documents.get("lateLex")!;
    const result = await sdk.searchCandidates(candidateOptions(
      "late-lex",
      [{ type: "lex", query: '"Orbit command"' }],
      { passage: { maxUtf8Bytes: 32 }, rerank: false },
    ));
    const hit = singleResult(result);

    expect(hit.passage?.text).toContain("Orbit command");
    expect(source.body.slice(hit.passage!.startUtf16, hit.passage!.endUtf16))
      .toBe(hit.passage!.text);
    expect(Object.hasOwn(hit, "locations")).toBe(false);
  });

  test("keeps vector, fallback, and intent passages as exact source slices", async () => {
    const reranker = installDeterministicReranker(18);
    try {
      const vectorResult = await sdk.searchCandidates(candidateOptions(
        "late-vec",
        [{ type: "vec", query: "late-vector" }],
        { passage: { maxUtf8Bytes: 100 }, rerank: true },
      ));
      const fallbackResult = await sdk.searchCandidates(candidateOptions(
        "unavailable",
        [{ type: "lex", query: "titlemarker" }],
        { passage: { maxUtf8Bytes: 128 }, rerank: true },
      ));
      const intentResult = await sdk.searchCandidates({
        ...candidateOptions(
          "unavailable",
          [{ type: "lex", query: "pathmarker" }],
          { passage: { maxUtf8Bytes: 128 }, rerank: true },
        ),
        intent: "keyword intent fallback",
      });

      const passageCases = [
        [vectorResult, documents.get("lateVec")!],
        [fallbackResult, documents.get("titleOnly")!],
        [intentResult, documents.get("pathOnly")!],
      ] as const;
      for (const [index, [result, source]] of passageCases.entries()) {
        const hit = singleResult(result);
        expect(source.body.slice(hit.passage!.startUtf16, hit.passage!.endUtf16))
          .toBe(hit.passage!.text);
        expect(Array.from(hit.passage!.text).length).toBeLessThanOrEqual(18);
        expect(reranker.rerankCalls[index]!.documents[0]!.text).toBe(hit.passage!.text);
      }

      expect(reranker.budgetCalls).toEqual([
        { query: "late-vector", intent: undefined },
        { query: "titlemarker", intent: undefined },
        { query: "pathmarker", intent: "keyword intent fallback" },
      ]);
      expect(reranker.rerankCalls.map(call => ({ query: call.query, intent: call.intent })))
        .toEqual(reranker.budgetCalls);
    } finally {
      reranker.restore();
    }
  });

  test("reuses one token budget across hydrated candidate groups", async () => {
    const reranker = installDeterministicReranker(18);
    try {
      const result = await sdk.searchCandidates(candidateOptions(
        "unavailable",
        [{ type: "lex", query: "body" }],
        { passage: { maxUtf8Bytes: 128 }, rerank: true },
      ));

      expect(result.results).toHaveLength(2);
      expect(reranker.budgetCalls).toEqual([{ query: "body", intent: undefined }]);
      expect(reranker.rerankCalls).toHaveLength(1);
      expect(reranker.rerankCalls[0]!.documents).toHaveLength(2);
    } finally {
      reranker.restore();
    }
  });

  test("defers the token budget until candidate body hydration succeeds", async () => {
    const actualSearchFts = store.searchFTS;
    const compactHit = actualSearchFts(
      "paritymarker",
      5,
      "parity",
      undefined,
      { includeBody: false, includeContext: false },
    )[0]!;
    const unhydratedHit = { ...compactHit, hash: "missing-content-hash" };
    store.searchFTS = () => [unhydratedHit];
    const reranker = installDeterministicReranker(18);

    try {
      const result = await sdk.searchCandidates(candidateOptions(
        "parity",
        [{ type: "lex", query: "paritymarker" }],
        { passage: { maxUtf8Bytes: 128 }, rerank: true },
      ));

      expect(result.results).toEqual([]);
      expect(reranker.budgetCalls).toEqual([]);
      expect(reranker.rerankCalls).toEqual([]);
    } finally {
      reranker.restore();
      store.searchFTS = actualSearchFts;
    }
  });

  test("honors an independent Unicode-scalar override", async () => {
    const result = await sdk.searchCandidates(candidateOptions(
      "rerank",
      [{ type: "lex", query: "exactmarker" }],
      {
        locations: true,
        passage: { maxUtf8Bytes: 1_000, maxUnicodeScalars: 12 },
      },
    ));
    const hit = singleResult(result);

    expect(hit.passage?.text).toContain("exactmarker");
    expect(hit.passage?.unicodeScalars).toBe(12);
    expect(hit.passage?.utf8Bytes).toBeLessThanOrEqual(1_000);
  });

  test("rejects invalid budgets before a retrieval leg runs", async () => {
    const invalid: readonly CandidateSearchOptions["passage"][] = [
      { maxUtf8Bytes: -1 },
      { maxUtf8Bytes: 1.5 },
      { maxUtf8Bytes: 48, maxUnicodeScalars: -1 },
      { maxUtf8Bytes: 48, maxUnicodeScalars: 1.5 },
    ];

    for (const passage of invalid) {
      retrievalCalls.length = 0;
      await expect(sdk.searchCandidates(candidateOptions(
        "rerank",
        [{ type: "lex", query: "exactmarker" }],
        { locations: true, passage },
      ))).rejects.toThrow("must be a non-negative safe integer");
      expect(retrievalCalls).toHaveLength(0);
    }
  });
});

describe("opt-in compatibility", () => {
  test("keeps the default candidate shape and ordinary structured search stable", async () => {
    const options = candidateOptions("parity", [{ type: "lex", query: "paritymarker" }]);
    const defaultCandidates = await sdk.searchCandidates(options);
    const defaultHit = singleResult(defaultCandidates);
    expect(Object.keys(defaultHit).sort()).toEqual([
      "contentHash",
      "context",
      "displayPath",
      "file",
      "group",
      "matches",
      "metadata",
      "rrfRank",
      "rrfScore",
      "rrfTopRankBonus",
      "score",
      "title",
    ]);

    const searchOptions = {
      queries: [{ type: "lex" as const, query: "paritymarker" }],
      collection: "parity",
      rerank: false,
    };
    const before = await sdk.search(searchOptions);
    await sdk.searchCandidates({ ...options, locations: true });
    const after = await sdk.search(searchOptions);

    expect(after).toEqual(before);
    expect(before).toHaveLength(1);
    expect(before[0]!.body).toBe(documents.get("parity")!.body);
  });
});
