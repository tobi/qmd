import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore as createSdkStore,
  type CandidateSearchOptions,
  type QMDStore,
} from "../src/index.js";
import { searchCandidates } from "../src/candidate-search.js";
import {
  hashContent,
  searchVec,
  type CollectionScope,
  type ExpandedQuery,
  type SearchRetrievalOptions,
  type Store,
} from "../src/store.js";
import { replaceDocumentMetadata } from "../src/metadata-store.js";
import {
  METADATA_EXTRACTION_VERSION,
  type DocumentMetadata,
} from "../src/metadata.js";
import type { Database, SQLiteValue, Statement } from "../src/db.js";

const MODEL = "candidate-test-model";
const QUERY_VECTOR = [1, 0] as const;
const GROUP_KEY = "thread";

type RetrievalCall = {
  source: "fts" | "vec";
  query: string;
  limit: number | undefined;
  scope: CollectionScope;
  retrieval: SearchRetrievalOptions | undefined;
  returnedBodies: boolean[];
};

let testDir: string;
let sdk: QMDStore;
let store: Store;
let realDb: Database;
let hydrationReads = 0;
let contextReads = 0;
const retrievalCalls: RetrievalCall[] = [];

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-candidate-search-"));
  const collectionNames = ["corpus", "aliases", "compact", "representative", "cap"] as const;
  const collections = Object.fromEntries(collectionNames.map(name => [
    name,
    { path: join(testDir, name), pattern: "**/*.md" },
  ]));
  await Promise.all(collectionNames.map(name => mkdir(join(testDir, name), { recursive: true })));

  sdk = await createSdkStore({
    dbPath: join(testDir, "candidate.sqlite"),
    config: { collections },
  });
  store = sdk.internal;
  realDb = store.db;
  store.ensureVecTable(QUERY_VECTOR.length);

  await seedGroupedCorpus();
  await seedSameHashAliases();
  await seedCompactDocument();
  await seedRepresentativeDocuments();
  await seedVectorCapDocuments();
  installFaithfulSearchWrappers();
  store.db = observingHydrationDb(realDb);
});

afterAll(async () => {
  await sdk.close();
  await rm(testDir, { recursive: true, force: true });
});

async function insertDocument(
  collection: string,
  path: string,
  title: string,
  body: string,
  metadata: DocumentMetadata,
  vector?: readonly [number, number],
  contentHash?: string,
): Promise<string> {
  const now = new Date().toISOString();
  const hash = contentHash ?? await hashContent(body);
  store.insertContent(hash, body, now);
  const documentId = store.insertDocument(collection, path, title, hash, now, now);
  replaceDocumentMetadata(realDb, documentId, {
    metadata,
    extractionVersion: METADATA_EXTRACTION_VERSION,
  });
  if (vector) {
    store.insertEmbedding(hash, 0, 0, new Float32Array(vector), MODEL, now);
  }
  return hash;
}

async function seedGroupedCorpus(): Promise<void> {
  for (let index = 0; index < 20; index++) {
    const cohort = index < 10 ? "alpha" : "beta";
    const path = `${String(index).padStart(2, "0")}-${cohort}-${index % 10}.md`;
    await insertDocument(
      "corpus",
      path,
      `signal signal signal ${cohort} sibling ${index}`,
      `${"signal ".repeat(20)}sibling ${index}`,
      { [GROUP_KEY]: cohort },
      [1, 0],
    );
  }

  const competitorMetadata: readonly DocumentMetadata[] = [
    { [GROUP_KEY]: 1 },
    { [GROUP_KEY]: "1" },
    {},
    { [GROUP_KEY]: ["array-value"] },
    { [GROUP_KEY]: "competitor-four" },
    { [GROUP_KEY]: "competitor-five" },
    { [GROUP_KEY]: "competitor-six" },
    { [GROUP_KEY]: "competitor-seven" },
  ];
  for (const [index, metadata] of competitorMetadata.entries()) {
    const path = `${20 + index}-competitor-${index}.md`;
    await insertDocument(
      "corpus",
      path,
      `Competitor ${index}`,
      `${"ordinary filler ".repeat(80)}signal competitor ${index}`,
      metadata,
      [0.2, 0.98],
    );
  }
}

async function seedSameHashAliases(): Promise<void> {
  const body = "aliasprobe shared content";
  const hash = await hashContent(body);
  await insertDocument(
    "aliases",
    "root.md",
    "Canonical root aliasprobe",
    body,
    { kind: "root", [GROUP_KEY]: "canonical-root" },
    undefined,
    hash,
  );
  await insertDocument(
    "aliases",
    "workers/worker.md",
    "Eligible worker aliasprobe",
    body,
    { kind: "worker", [GROUP_KEY]: "eligible-worker" },
    undefined,
    hash,
  );
  store.insertEmbedding(hash, 0, 0, new Float32Array(QUERY_VECTOR), MODEL, new Date().toISOString());
}

async function seedCompactDocument(): Promise<void> {
  await insertDocument(
    "compact",
    "unicode.md",
    "Compact probe",
    "compactprobe 😀 café",
    { [GROUP_KEY]: "compact" },
    QUERY_VECTOR,
  );
}

async function seedRepresentativeDocuments(): Promise<void> {
  await insertDocument(
    "representative",
    "target-first.md",
    "Target first",
    "rankone target",
    { [GROUP_KEY]: "target" },
  );
  await insertDocument(
    "representative",
    "target-best.md",
    "ranktwo ranktwo ranktwo target",
    "ranktwo target",
    { [GROUP_KEY]: "target" },
  );
  await insertDocument(
    "representative",
    "target-later.md",
    "Target later",
    "rankthree target",
    { [GROUP_KEY]: "target" },
  );

  for (const [path, query, group] of [
    ["rankone-a.md", "rankone", "rankone-a"],
    ["rankone-b.md", "rankone", "rankone-b"],
    ["rankthree-a.md", "rankthree", "rankthree-a"],
  ] as const) {
    await insertDocument(
      "representative",
      path,
      `${query} ${query} ${query} ${group}`,
      `${query} ${query} ${group}`,
      { [GROUP_KEY]: group },
    );
  }
}

async function seedVectorCapDocuments(): Promise<void> {
  const denseHash = await insertDocument(
    "cap",
    "dense.md",
    "Dense vector document",
    "dense vector body",
    { [GROUP_KEY]: "dense" },
  );
  const fallbackHash = await insertDocument(
    "cap",
    "fallback.md",
    "Explicit lexical fallback",
    "capfallback farther competitor",
    { [GROUP_KEY]: "fallback" },
  );
  const now = new Date().toISOString();
  const close = new Float32Array(QUERY_VECTOR);
  const farther = new Float32Array([0, 1]);
  realDb.transaction(() => {
    for (let seq = 0; seq < 4_097; seq++) {
      store.insertEmbedding(denseHash, seq, seq * 4, close, MODEL, now, 4_097);
    }
    store.insertEmbedding(fallbackHash, 0, 0, farther, MODEL, now);
  }).immediate();
}

function installFaithfulSearchWrappers(): void {
  const actualSearchFts = store.searchFTS;
  const actualGetContextForFile = store.getContextForFile;
  store.getContextForFile = filepath => {
    contextReads++;
    return actualGetContextForFile(filepath);
  };
  store.searchFTS = (query, limit, scope, filter, retrieval) => {
    const results = actualSearchFts(query, limit, scope, filter, retrieval);
    retrievalCalls.push({
      source: "fts",
      query,
      limit,
      scope,
      retrieval,
      returnedBodies: results.map(result => Object.hasOwn(result, "body")),
    });
    return results;
  };

  store.searchVec = async (query, model, limit, scope, session, _embedding, filter, retrieval) => {
    const results = await searchVec(
      realDb,
      query,
      model,
      limit,
      scope,
      session,
      [...QUERY_VECTOR],
      undefined,
      filter,
      retrieval,
    );
    retrievalCalls.push({
      source: "vec",
      query,
      limit,
      scope,
      retrieval,
      returnedBodies: results.map(result => Object.hasOwn(result, "body")),
    });
    return results;
  };
}

function observingHydrationDb(db: Database): Database {
  return {
    get inTransaction() {
      return db.inTransaction;
    },
    exec(sql: string): void {
      db.exec(sql);
    },
    prepare(sql: string): Statement {
      const statement = db.prepare(sql);
      if (!sql.includes("SELECT c.doc AS body")) return statement;
      return {
        run: (...params: SQLiteValue[]) => statement.run(...params),
        get: <T = unknown>(...params: SQLiteValue[]) => {
          hydrationReads++;
          return statement.get<T>(...params);
        },
        all: <T = unknown>(...params: SQLiteValue[]) => statement.all<T>(...params),
        iterate: <T = unknown>(...params: SQLiteValue[]) => statement.iterate<T>(...params),
      };
    },
    loadExtension(path: string): void {
      db.loadExtension(path);
    },
    transaction<T extends (...args: SQLiteValue[]) => unknown>(fn: T): T & { immediate: T } {
      return db.transaction(fn);
    },
    close(): void {
      db.close();
    },
  };
}

function candidates(
  queries: readonly ExpandedQuery[],
  rawLimitPerLeg = 28,
  targetGroupsPerLeg = 10,
): CandidateSearchOptions {
  return {
    queries,
    collection: "corpus",
    candidates: {
      rawLimitPerLeg,
      group: { metadataKey: GROUP_KEY, targetGroupsPerLeg },
    },
    candidateLimit: 20,
    limit: 20,
    rerank: false,
  };
}

function resetObservations(): void {
  retrievalCalls.length = 0;
  hydrationReads = 0;
  contextReads = 0;
}

describe("compact backend retrieval", () => {
  test("omits bodies, reports UTF-8 bytes, and preserves the default JSON shape", async () => {
    resetObservations();
    const expectedBody = "compactprobe 😀 café";
    const compactFts = store.searchFTS(
      "compactprobe",
      1,
      "compact",
      undefined,
      { includeBody: false },
    )[0]!;
    const ordinaryFts = store.searchFTS("compactprobe", 1, "compact")[0]!;
    const compactVec = (await store.searchVec(
      "semantic",
      MODEL,
      1,
      "compact",
      undefined,
      undefined,
      undefined,
      { includeBody: false },
    ))[0]!;
    const ordinaryVec = (await store.searchVec("semantic", MODEL, 1, "compact"))[0]!;

    for (const compact of [compactFts, compactVec]) {
      expect(Object.hasOwn(compact, "body")).toBe(false);
      expect(compact.bodyLength).toBe(Buffer.byteLength(expectedBody, "utf8"));
      expect(Object.hasOwn(JSON.parse(JSON.stringify(compact)), "body")).toBe(false);
    }
    for (const ordinary of [ordinaryFts, ordinaryVec]) {
      expect(ordinary.body).toBe(expectedBody);
      expect(JSON.parse(JSON.stringify(ordinary)).body).toBe(expectedBody);
    }
    expect(Object.keys(ordinaryFts).sort()).toEqual([
      "body", "bodyLength", "collectionName", "context", "displayPath", "docid", "filepath",
      "hash", "metadata", "modifiedAt", "score", "source", "title",
    ].sort());
    expect(Object.keys(ordinaryVec).sort()).toEqual([
      "body", "bodyLength", "chunkPos", "collectionName", "context", "displayPath", "docid",
      "filepath", "hash", "metadata", "modifiedAt", "score", "source", "title",
    ].sort());
  });
});

describe("per-leg group admission", () => {
  for (const source of ["lex", "vec"] as const) {
    test(`${source} reaches ten groups when the raw leg includes all 28 documents`, async () => {
      resetObservations();
      const result = await searchCandidates(
        store,
        candidates([{ type: source, query: source === "lex" ? "signal" : "semantic" }]),
      );

      expect(result.results).toHaveLength(10);
      expect(result.coverage).toEqual({
        legs: [{
          leg: 0,
          query: source === "lex" ? "signal" : "semantic",
          queryType: source,
          source: source === "lex" ? "fts" : "vec",
          rawLimit: 28,
          rawReturned: 28,
          rawLimitReached: true,
          groupsReturned: 10,
          targetGroups: 10,
          groupShortfall: 0,
          vectorScans: source === "lex" ? [] : result.coverage.legs[0]!.vectorScans,
        }],
        fusedGroups: 10,
        admittedGroups: 10,
        unavailableGroups: 0,
        returnedGroups: 10,
      });
      if (source === "vec") {
        expect(result.coverage.legs[0]!.vectorScans).toHaveLength(1);
        expect(result.coverage.legs[0]!.vectorScans[0]).toMatchObject({
          matchedChunks: 28,
          resolvedDocuments: 28,
          backendCapReached: false,
        });
      }
      expect(retrievalCalls).toHaveLength(1);
      expect(retrievalCalls[0]!.retrieval?.includeBody).toBe(false);
      expect(retrievalCalls[0]!.returnedBodies.every(hasBody => !hasBody)).toBe(true);
    });

    test(`${source} reports two groups and an eight-group shortfall at raw limit 20`, async () => {
      resetObservations();
      const result = await sdk.searchCandidates(
        candidates(
          [{ type: source, query: source === "lex" ? "signal" : "semantic" }],
          20,
        ),
      );

      expect(result.results.map(hit => hit.group.value)).toEqual(["alpha", "beta"]);
      expect(result.coverage.legs[0]).toMatchObject({
        rawLimit: 20,
        rawReturned: 20,
        rawLimitReached: true,
        groupsReturned: 2,
        targetGroups: 10,
        groupShortfall: 8,
      });
    });
  }

  test("keeps scalar types distinct and falls back per path for missing and array values", async () => {
    const result = await sdk.searchCandidates(candidates([{ type: "lex", query: "signal" }]));
    const number = result.results.find(hit => hit.group.value === 1)!;
    const string = result.results.find(hit => hit.group.value === "1")!;
    const missing = result.results.find(hit => hit.file.endsWith("22-competitor-2.md"))!;
    const array = result.results.find(hit => hit.file.endsWith("23-competitor-3.md"))!;

    expect(number.group.key).not.toBe(string.group.key);
    expect(number.group).toEqual({
      key: JSON.stringify(["metadata", GROUP_KEY, "number", 1]),
      value: 1,
      fallback: null,
    });
    expect(string.group).toEqual({
      key: JSON.stringify(["metadata", GROUP_KEY, "string", "1"]),
      value: "1",
      fallback: null,
    });
    for (const hit of [missing, array]) {
      expect(hit.group).toEqual({
        key: JSON.stringify(["filepath", hit.file]),
        value: null,
        fallback: "filepath",
      });
    }
    expect(missing.group.key).not.toBe(array.group.key);
  });
});

describe("scope and fusion", () => {
  for (const source of ["lex", "vec"] as const) {
    test(`${source} applies document metadata scope before grouping same-hash aliases`, async () => {
      const result = await sdk.searchCandidates({
        queries: [{ type: source, query: source === "lex" ? "aliasprobe" : "semantic" }],
        collection: "aliases",
        filter: { field: "kind", operator: "eq", value: "worker" },
        candidates: {
          rawLimitPerLeg: 2,
          group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 1 },
        },
        rerank: false,
      });

      expect(result.results).toHaveLength(1);
      expect(result.results[0]!.file).toBe("qmd://aliases/workers/worker.md");
      expect(result.results[0]!.contentHash).toBe(
        store.findActiveDocument("aliases", "root.md")!.hash,
      );
      expect(result.results[0]!.group.value).toBe("eligible-worker");
    });
  }

  test("fuses each group once and exposes exact RRF arithmetic", async () => {
    const result = await sdk.searchCandidates(candidates([
      { type: "lex", query: "signal" },
      { type: "vec", query: "semantic" },
    ]));

    expect(result.results).toHaveLength(10);
    expect(new Set(result.results.map(hit => hit.group.key)).size).toBe(10);
    for (const hit of result.results) {
      expect(hit.matches.map(match => match.source).sort()).toEqual(["fts", "vec"]);
      const contribution = hit.matches.reduce((sum, match) => sum + match.rrfContribution, 0);
      const topRank = Math.min(...hit.matches.map(match => match.groupRank));
      const topRankBonus = topRank === 1 ? 0.05 : topRank <= 3 ? 0.02 : 0;
      expect(hit.rrfTopRankBonus).toBe(topRankBonus);
      expect(hit.rrfScore).toBeCloseTo(contribution + hit.rrfTopRankBonus, 12);
    }
  });

  test("the largest weighted contribution chooses the representative across three legs", async () => {
    const result = await sdk.searchCandidates({
      queries: [
        { type: "lex", query: "rankone" },
        { type: "lex", query: "ranktwo" },
        { type: "lex", query: "rankthree" },
      ],
      collection: "representative",
      candidates: {
        rawLimitPerLeg: 10,
        group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 10 },
      },
      candidateLimit: 10,
      limit: 10,
      rerank: false,
    });
    const target = result.results.find(hit => hit.group.value === "target")!;

    expect(target.matches.map(match => match.groupRank)).toEqual([3, 1, 2]);
    expect(target.matches.map(match => match.weight)).toEqual([2, 1, 1]);
    expect(target.file).toBe("qmd://representative/target-first.md");
  });

  test("uses a vector query as the rerank query when a hyde query appears first", async () => {
    const actualRerank = store.rerank;
    let receivedQuery: string | undefined;
    store.rerank = async (query, documents) => {
      receivedQuery = query;
      return documents.map(document => ({
        file: document.file,
        score: 1,
      }));
    };

    try {
      await sdk.searchCandidates({
        queries: [
          { type: "hyde", query: "hypothetical answer" },
          { type: "vec", query: "semantic question" },
        ],
        collection: "compact",
        candidates: {
          rawLimitPerLeg: 1,
          group: {
            metadataKey: GROUP_KEY,
            targetGroupsPerLeg: 1,
          },
        },
        candidateLimit: 1,
        limit: 1,
      });
    } finally {
      store.rerank = actualRerank;
    }

    expect(receivedQuery).toBe("semantic question");
  });

  test("blends rerank scores with the original RRF positions", async () => {
    const actualRerank = store.rerank;
    store.rerank = async (_query, documents) => documents.map(document => ({
      file: document.file,
      score: 0.8,
    }));

    let result;
    try {
      result = await sdk.searchCandidates({
        ...candidates([{ type: "lex", query: "signal" }]),
        candidateLimit: 10,
        limit: 10,
        rerank: true,
      });
    } finally {
      store.rerank = actualRerank;
    }

    const first = result.results.find(hit => hit.rrfRank === 1)!;
    const fourth = result.results.find(hit => hit.rrfRank === 4)!;
    expect(first.score).toBeCloseTo(0.95, 12);
    expect(fourth.score).toBeCloseTo(0.47, 12);
  });
});

describe("leg weights and backend coverage", () => {
  test("gives weight two to the first nonempty typed leg and weight one to later legs", async () => {
    const result = await sdk.searchCandidates(candidates([
      { type: "lex", query: "absenttoken" },
      { type: "lex", query: "signal" },
      { type: "vec", query: "semantic" },
    ]));

    expect(result.coverage.legs.map(leg => leg.rawReturned)).toEqual([0, 28, 28]);
    expect(new Set(result.results.flatMap(hit => hit.matches)
      .filter(match => match.leg === 1).map(match => match.weight))).toEqual(new Set([2]));
    expect(new Set(result.results.flatMap(hit => hit.matches)
      .filter(match => match.leg === 2).map(match => match.weight))).toEqual(new Set([1]));
  });

  test("gives automatic original legs weight two and an expansion weight one", async () => {
    const actualExpandQuery = store.expandQuery;
    store.expandQuery = async () => [{ type: "lex", query: "competitor" }];
    try {
      const result = await sdk.searchCandidates({
        query: "signal",
        collection: "corpus",
        candidates: {
          rawLimitPerLeg: 28,
          group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 10 },
        },
        candidateLimit: 20,
        limit: 20,
        rerank: false,
      });
      const matches = result.results.flatMap(hit => hit.matches);

      expect(result.coverage.legs.map(leg => [leg.source, leg.query])).toEqual([
        ["fts", "signal"],
        ["vec", "signal"],
        ["fts", "competitor"],
      ]);
      expect(new Set(matches.filter(match => match.queryType === "original")
        .map(match => match.weight))).toEqual(new Set([2]));
      expect(new Set(matches.filter(match => match.query === "competitor")
        .map(match => match.weight))).toEqual(new Set([1]));
    } finally {
      store.expandQuery = actualExpandQuery;
    }
  });

  test("accepts lexical negation in a plain query", async () => {
    const actualExpandQuery = store.expandQuery;
    store.expandQuery = async () => [];
    try {
      const result = await sdk.searchCandidates({
        query: "signal -absenttoken",
        collection: "corpus",
        candidates: {
          rawLimitPerLeg: 28,
          group: {
            metadataKey: GROUP_KEY,
            targetGroupsPerLeg: 10,
          },
        },
        candidateLimit: 20,
        limit: 20,
        rerank: false,
      });

      expect(result.coverage.legs.map(leg => ({
        query: leg.query,
        queryType: leg.queryType,
        source: leg.source,
      }))).toEqual([
        {
          query: "signal -absenttoken",
          queryType: "original",
          source: "fts",
        },
        {
          query: "signal -absenttoken",
          queryType: "original",
          source: "vec",
        },
      ]);
    } finally {
      store.expandQuery = actualExpandQuery;
    }
  });

  test("reports the 4096-row KNN cap and uses the caller's typed lexical fallback", async () => {
    const result = await sdk.searchCandidates({
      queries: [
        { type: "vec", query: "semantic" },
        { type: "lex", query: "capfallback" },
      ],
      collection: "cap",
      candidates: {
        rawLimitPerLeg: 2,
        group: { metadataKey: GROUP_KEY, targetGroupsPerLeg: 2 },
      },
      candidateLimit: 2,
      limit: 2,
      rerank: false,
    });

    expect(result.results.map(hit => hit.group.value).sort()).toEqual(["dense", "fallback"]);
    expect(result.coverage.legs).toHaveLength(2);
    expect(result.coverage.legs[0]).toMatchObject({
      source: "vec",
      rawReturned: 1,
      groupsReturned: 1,
      groupShortfall: 1,
    });
    expect(result.coverage.legs[0]!.vectorScans).toEqual([{
      collectionId: result.coverage.legs[0]!.vectorScans[0]!.collectionId,
      collectionName: "cap",
      requestedK: 4_096,
      matchedChunks: 4_096,
      resolvedDocuments: 1,
      backendCapReached: true,
    }]);
    expect(result.coverage.legs[1]).toMatchObject({
      source: "fts",
      query: "capfallback",
      rawReturned: 1,
      groupsReturned: 1,
    });
  });
});

describe("hydration boundary and validation", () => {
  test("hydrates admitted groups after applying candidateLimit", async () => {
    resetObservations();
    const options = candidates([{ type: "lex", query: "signal" }]);
    options.candidateLimit = 3;
    options.limit = 10;
    const result = await sdk.searchCandidates(options);

    expect(result.coverage).toMatchObject({
      fusedGroups: 10,
      admittedGroups: 3,
      unavailableGroups: 0,
      returnedGroups: 3,
    });
    expect(result.results).toHaveLength(3);
    expect(hydrationReads).toBe(3);
    expect(contextReads).toBe(3);
    expect(retrievalCalls).toHaveLength(1);
    expect(retrievalCalls[0]).toMatchObject({
      source: "fts",
      limit: 28,
      scope: ["corpus"],
      retrieval: {
        includeBody: false,
        includeContext: false,
      },
    });
    expect(retrievalCalls[0]!.returnedBodies.every(hasBody => !hasBody)).toBe(true);
  });

  test("counts an admitted source that disappears before hydration as unavailable", async () => {
    resetObservations();
    const actualSearchFts = store.searchFTS;
    store.searchFTS = (query, limit, scope, filter, retrieval) => {
      const results = actualSearchFts(query, limit, scope, filter, retrieval);
      store.deactivateDocument("compact", "unicode.md");
      return results;
    };

    let result;
    try {
      result = await sdk.searchCandidates({
        queries: [{ type: "lex", query: "compactprobe" }],
        collection: "compact",
        candidates: {
          rawLimitPerLeg: 1,
          group: {
            metadataKey: GROUP_KEY,
            targetGroupsPerLeg: 1,
          },
        },
        candidateLimit: 1,
        limit: 1,
        rerank: false,
      });
    } finally {
      store.searchFTS = actualSearchFts;
      await insertDocument(
        "compact",
        "unicode.md",
        "Compact probe",
        "compactprobe 😀 café",
        { [GROUP_KEY]: "compact" },
      );
    }

    expect(result.results).toEqual([]);
    expect(result.coverage).toMatchObject({
      fusedGroups: 1,
      admittedGroups: 1,
      unavailableGroups: 1,
      returnedGroups: 0,
    });
    expect(contextReads).toBe(0);
  });

  test("applies minScore and limit before reporting returnedGroups", async () => {
    resetObservations();
    const byLimit = await sdk.searchCandidates({
      ...candidates([{ type: "lex", query: "signal" }]),
      candidateLimit: 5,
      limit: 2,
    });
    const limitContextReads = contextReads;
    resetObservations();
    const byScore = await sdk.searchCandidates({
      ...candidates([{ type: "lex", query: "signal" }]),
      candidateLimit: 5,
      limit: 5,
      minScore: 0.4,
    });

    expect(byLimit.results.map(hit => hit.rrfRank)).toEqual([1, 2]);
    expect(byLimit.coverage).toMatchObject({
      admittedGroups: 5,
      returnedGroups: 2,
    });
    expect(limitContextReads).toBe(2);
    expect(byScore.results.map(hit => hit.rrfRank)).toEqual([1, 2]);
    expect(byScore.coverage).toMatchObject({
      admittedGroups: 5,
      returnedGroups: 2,
    });
    expect(contextReads).toBe(2);
  });

  test("skips chunk selection when reranking and evidence are disabled", async () => {
    const options = candidates([{ type: "lex", query: "signal" }]);
    options.candidateLimit = 3;
    let chunkStrategyReads = 0;
    Object.defineProperty(options, "chunkStrategy", {
      configurable: true,
      enumerable: true,
      get: () => {
        chunkStrategyReads++;
        return "auto";
      },
    });

    const result = await sdk.searchCandidates(options);

    expect(result.results).toHaveLength(3);
    expect(chunkStrategyReads).toBe(0);
  });

  test("requires exactly one query form before retrieval", async () => {
    resetObservations();
    const base = candidates([{ type: "lex", query: "signal" }]);
    // @ts-expect-error Exercise the runtime guard for JavaScript callers with no query form.
    await expect(sdk.searchCandidates({
      ...base,
      queries: undefined,
    }))
      .rejects.toThrow("requires exactly one of query or queries");
    // @ts-expect-error Exercise the runtime guard for JavaScript callers with both query forms.
    await expect(sdk.searchCandidates({
      ...base,
      query: "signal",
    }))
      .rejects.toThrow("requires exactly one of query or queries");
    expect(retrievalCalls).toHaveLength(0);
  });

  test("requires positive integer limits before retrieval", async () => {
    const base = candidates([{ type: "lex", query: "signal" }]);
    const cases: readonly [string, CandidateSearchOptions][] = [
      ["rawLimitPerLeg", {
        ...base,
        candidates: { ...base.candidates, rawLimitPerLeg: 0 },
      }],
      ["targetGroupsPerLeg", {
        ...base,
        candidates: {
          ...base.candidates,
          group: { ...base.candidates.group, targetGroupsPerLeg: 1.5 },
        },
      }],
      ["limit", { ...base, limit: 0 }],
      ["candidateLimit", { ...base, candidateLimit: 0 }],
    ];

    for (const [name, options] of cases) {
      resetObservations();
      await expect(sdk.searchCandidates(options)).rejects.toThrow(`${name} must be a positive safe integer`);
      expect(retrievalCalls, name).toHaveLength(0);
    }
  });
});
