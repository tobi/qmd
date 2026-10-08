import {
  chunkDocumentAsync,
  extractIntentTerms,
  primaryQueryFor,
  reciprocalRankFusion,
  RERANK_CANDIDATE_LIMIT,
  rrfContribution,
  rrfPositionWeight,
  rrfTopRankBonus,
  selectBestChunkIndex,
  validateLexQuery,
  validateSemanticQuery,
  type ChunkStrategy,
  type CollectionScope,
  type ExpandedQuery,
  type SearchResult,
  type Store,
  type VectorScanCoverage,
} from "./store.js";
import { getDefaultLlamaCpp, type RerankTokenBudget } from "./llm.js";
import { parseMetadataFilter, type MetadataFilter } from "./metadata-filter.js";
import type { DocumentMetadata, MetadataScalar } from "./metadata.js";
import {
  boundedPassageWindow,
  findLexicalLocation,
  fitPassageWindow,
  isUtf16Boundary,
  selectionWindowLocation,
  validatePassageBudget,
  vectorChunkLocation,
  type PassageBudget,
  type PassageWindow,
  type SearchLocation,
  type Utf16Span,
} from "./search-locations.js";

type CandidateQueryInput =
  | {
    /** Expand one plain query into QMD's standard lexical and semantic retrieval legs. */
    query: string;
    queries?: never;
  }
  | {
    query?: never;
    /** Run caller-supplied lexical, vector, or hypothetical-document retrieval legs. */
    queries: readonly ExpandedQuery[];
  };

export type CandidateSearchOptions = CandidateQueryInput & {
  /** Bound backend work and distinct-group admission independently for each retrieval leg. */
  candidates: {
    rawLimitPerLeg: number;
    group: {
      metadataKey: string;
      targetGroupsPerLeg: number;
    };
  };
  collection?: string;
  collections?: string[];
  filter?: MetadataFilter;
  /** Maximum groups returned after scoring and filtering. Defaults to 10. */
  limit?: number;
  /** Maximum fused groups whose current source bodies are hydrated. */
  candidateLimit?: number;
  /** Inclusive threshold for the final position or blended score. */
  minScore?: number;
  /** Run the reranker after candidate hydration. Defaults to true. */
  rerank?: boolean;
  intent?: string;
  chunkStrategy?: ChunkStrategy;
  /** Include source-location provenance and a bounded passage. */
  locations?: boolean;
  /** Include a bounded source passage using this caller budget. */
  passage?: PassageBudget;
};

export type CandidateGroup = {
  /** Opaque identity that preserves metadata scalar types and path fallbacks. */
  key: string;
  value: MetadataScalar | null;
  fallback: "filepath" | null;
};

export type CandidateMatch = {
  leg: number;
  query: string;
  queryType: "original" | "lex" | "vec" | "hyde";
  source: "fts" | "vec";
  file: string;
  contentHash: string;
  rawRank: number;
  groupRank: number;
  backendScore: number;
  weight: number;
  rrfContribution: number;
  vectorStartUtf16?: number;
  vectorChunkSeq?: number;
};

export type CandidateHit = {
  group: CandidateGroup;
  file: string;
  contentHash: string;
  displayPath: string;
  title: string;
  metadata: DocumentMetadata;
  context: string | null;
  /** Final position score, blended with the reranker score when reranking runs. */
  score: number;
  /** Weighted reciprocal-rank contribution sum plus the top-rank bonus. */
  rrfScore: number;
  /** One-indexed rank after reciprocal-rank fusion. */
  rrfRank: number;
  rrfTopRankBonus: number;
  matches: CandidateMatch[];
  representativeLeg?: number;
  passage?: PassageWindow;
  locations?: SearchLocation[];
  bodyAnchorStatus?: "located" | "unavailable";
};

export type CandidateLegCoverage = {
  /** Zero-indexed position in the executed retrieval legs. */
  leg: number;
  query: string;
  /** Distinguishes automatic original legs from caller-supplied and expanded query types. */
  queryType: CandidateMatch["queryType"];
  source: "fts" | "vec";
  /** Requested backend hit limit for this leg. */
  rawLimit: number;
  /** Backend hits returned before grouping. */
  rawReturned: number;
  /** The backend filled rawLimit, so additional hits may exist. */
  rawLimitReached: boolean;
  /** Distinct groups retained from this leg. */
  groupsReturned: number;
  targetGroups: number;
  /** Additional distinct groups needed to reach targetGroups. */
  groupShortfall: number;
  vectorScans: VectorScanCoverage[];
};

export type CandidateSearchResult = {
  results: CandidateHit[];
  coverage: {
    legs: CandidateLegCoverage[];
    /** Distinct groups produced by reciprocal-rank fusion. */
    fusedGroups: number;
    /** Fused groups selected for exact source hydration. */
    admittedGroups: number;
    /** Admitted groups whose representative hash and path fail exact source hydration. */
    unavailableGroups: number;
    /** Groups remaining after minScore and limit. */
    returnedGroups: number;
  };
};

type RetrievalLeg = ExpandedQuery & {
  queryType: CandidateMatch["queryType"];
  weight: number;
};

type GroupEvidence = {
  group: CandidateGroup;
  representative: SearchResult;
  representativeMatch: CandidateMatch;
  matches: CandidateMatch[];
};

type GroupRankingEntry = {
  /** CandidateGroup.key occupies reciprocalRankFusion's identity field. */
  file: string;
  score: number;
};

type GroupedRetrieval = {
  rankings: GroupRankingEntry[][];
  weights: number[];
  coverage: CandidateLegCoverage[];
  evidence: Map<string, GroupEvidence>;
};

type CandidateDraft = Omit<CandidateHit, "context">;

type HydrationOptions = {
  primaryQuery: string;
  queryTerms: readonly string[];
  intentTerms: readonly string[];
  intent?: string;
  shouldRerank: boolean;
  wantsEvidence: boolean;
  includeLocations: boolean;
  passageBudget: PassageBudget;
  chunkStrategy?: ChunkStrategy;
};

function positiveInteger(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive safe integer`);
  }
}

function groupFor(hit: SearchResult, metadataKey: string): CandidateGroup {
  const value = hit.metadata[metadataKey];
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return {
      key: JSON.stringify(["metadata", metadataKey, typeof value, value]),
      value,
      fallback: null,
    };
  }
  return {
    key: JSON.stringify(["filepath", hit.filepath]),
    value: null,
    fallback: "filepath",
  };
}

function validateQuery(search: ExpandedQuery): void {
  if (!search || !["lex", "vec", "hyde"].includes(search.type) || typeof search.query !== "string" || !search.query.trim()) {
    throw new Error("Each candidate query requires a lex, vec, or hyde type and nonempty query text");
  }
  if (/[\r\n]/.test(search.query)) {
    throw new Error("Candidate queries must be a single line");
  }
  const error = search.type === "lex"
    ? validateLexQuery(search.query)
    : validateSemanticQuery(search.query);
  if (error) throw new Error(error);
}

function validateOptions(options: CandidateSearchOptions): void {
  if ((options.query !== undefined) === (options.queries !== undefined)) {
    throw new Error("searchCandidates() requires exactly one of query or queries");
  }
  if (options.query !== undefined) {
    validateQuery({ type: "lex", query: options.query });
  }
  if (options.queries !== undefined) {
    if (!Array.isArray(options.queries) || options.queries.length === 0) {
      throw new Error("queries must be a nonempty array");
    }
    options.queries.forEach(validateQuery);
  }
  positiveInteger(options.candidates?.rawLimitPerLeg, "rawLimitPerLeg");
  positiveInteger(options.candidates?.group?.targetGroupsPerLeg, "targetGroupsPerLeg");
  parseMetadataFilter({
    field: options.candidates.group.metadataKey,
    operator: "exists",
    value: true,
  });
  if (options.locations !== undefined && typeof options.locations !== "boolean") {
    throw new Error("locations must be a boolean");
  }
  if (options.locations === true || options.passage !== undefined) {
    validatePassageBudget(options.passage ?? { maxUtf8Bytes: 48_000 });
  }
  positiveInteger(options.limit ?? 10, "limit");
  positiveInteger(options.candidateLimit ?? RERANK_CANDIDATE_LIMIT, "candidateLimit");
  const minScore = options.minScore ?? 0;
  if (!Number.isFinite(minScore) || minScore < 0 || minScore > 1) {
    throw new Error("minScore must be between 0 and 1");
  }
}

async function retrievalLegs(store: Store, options: CandidateSearchOptions): Promise<RetrievalLeg[]> {
  if (options.queries !== undefined) {
    return options.queries.map(search => ({
      ...search,
      queryType: search.type,
      weight: 1,
    }));
  }

  const query = options.query;
  const expanded = await store.expandQuery(query);
  expanded.forEach(validateQuery);
  return [
    {
      type: "lex",
      query,
      queryType: "original",
      weight: 2,
    },
    {
      type: "vec",
      query,
      queryType: "original",
      weight: 2,
    },
    ...expanded.map(search => ({
      ...search,
      queryType: search.type,
      weight: 1,
    })),
  ];
}

async function retrieveGroupedCandidates(
  store: Store,
  options: CandidateSearchOptions,
  legs: readonly RetrievalLeg[],
  scope: CollectionScope,
  filter: MetadataFilter | undefined,
): Promise<GroupedRetrieval> {
  const rankings: GroupRankingEntry[][] = [];
  const weights: number[] = [];
  const coverage: CandidateLegCoverage[] = [];
  const evidence = new Map<string, GroupEvidence>();
  const rawLimit = options.candidates.rawLimitPerLeg;
  const targetGroups = options.candidates.group.targetGroupsPerLeg;
  const metadataKey = options.candidates.group.metadataKey;
  let foundTypedResults = false;

  for (const [legIndex, configuredLeg] of legs.entries()) {
    const vectorScans: VectorScanCoverage[] = [];
    const retrieval = {
      includeBody: false,
      includeContext: false,
    };
    const hits = configuredLeg.type === "lex"
      ? store.searchFTS(
        configuredLeg.query,
        rawLimit,
        scope,
        filter,
        retrieval,
      )
      : await store.searchVec(
        configuredLeg.query,
        (store.llm ?? getDefaultLlamaCpp()).embedModelName,
        rawLimit,
        scope,
        undefined,
        undefined,
        filter,
        {
          ...retrieval,
          onVectorScan: scan => vectorScans.push(scan),
        },
      );
    const promotesTypedLeg = options.queries !== undefined && hits.length > 0 && !foundTypedResults;
    const weight = promotesTypedLeg ? 2 : configuredLeg.weight;
    if (promotesTypedLeg) foundTypedResults = true;

    const groups = new Set<string>();
    const ranking: GroupRankingEntry[] = [];
    for (const [rawIndex, hit] of hits.entries()) {
      const group = groupFor(hit, metadataKey);
      if (groups.has(group.key)) continue;
      if (groups.size === targetGroups) break;

      groups.add(group.key);
      const groupRank = groups.size;
      const match: CandidateMatch = {
        leg: legIndex,
        query: configuredLeg.query,
        queryType: configuredLeg.queryType,
        source: hit.source,
        file: hit.filepath,
        contentHash: hit.hash,
        rawRank: rawIndex + 1,
        groupRank,
        backendScore: hit.score,
        weight,
        rrfContribution: rrfContribution(groupRank, weight),
        ...(hit.chunkPos === undefined ? {} : { vectorStartUtf16: hit.chunkPos }),
        ...(hit.chunkSeq === undefined ? {} : { vectorChunkSeq: hit.chunkSeq }),
      };
      const existing = evidence.get(group.key);
      if (existing) {
        existing.matches.push(match);
        // The largest RRF contribution wins; input leg order breaks ties.
        if (match.rrfContribution > existing.representativeMatch.rrfContribution) {
          existing.representative = hit;
          existing.representativeMatch = match;
        }
      } else {
        evidence.set(group.key, {
          group,
          representative: hit,
          representativeMatch: match,
          matches: [match],
        });
      }
      ranking.push({
        file: group.key,
        score: hit.score,
      });
    }

    rankings.push(ranking);
    weights.push(weight);
    coverage.push({
      leg: legIndex,
      query: configuredLeg.query,
      queryType: configuredLeg.queryType,
      source: configuredLeg.type === "lex" ? "fts" : "vec",
      rawLimit,
      rawReturned: hits.length,
      rawLimitReached: hits.length === rawLimit,
      groupsReturned: groups.size,
      targetGroups,
      groupShortfall: Math.max(0, targetGroups - groups.size),
      vectorScans,
    });
  }

  return {
    rankings,
    weights,
    coverage,
    evidence,
  };
}

function passageForAnchor(
  body: string,
  anchor: Utf16Span,
  passageBudget: PassageBudget,
  rerankTokenBudget: RerankTokenBudget | undefined,
): PassageWindow {
  return rerankTokenBudget === undefined
    ? boundedPassageWindow(body, anchor, passageBudget)
    : fitPassageWindow(body, anchor, passageBudget, rerankTokenBudget);
}

async function hydrateCandidates(
  store: Store,
  admittedGroups: readonly GroupRankingEntry[],
  evidence: ReadonlyMap<string, GroupEvidence>,
  options: HydrationOptions,
): Promise<{
  candidates: CandidateDraft[];
  rerankInputs: { file: string; text: string }[];
}> {
  const candidates: CandidateDraft[] = [];
  const rerankInputs: { file: string; text: string }[] = [];
  const bodyOf = store.db.prepare(`
    SELECT c.doc AS body FROM content c JOIN documents d ON d.hash = c.hash
    WHERE d.active = 1 AND c.hash = ? AND 'qmd://' || d.collection || '/' || d.path = ? LIMIT 1
  `);
  let rerankTokenBudget: RerankTokenBudget | undefined;

  for (const [admittedIndex, fusedGroup] of admittedGroups.entries()) {
    const groupEvidence = evidence.get(fusedGroup.file);
    if (groupEvidence === undefined) {
      throw new Error(`Missing retrieval evidence for candidate group ${fusedGroup.file}`);
    }
    const representative = groupEvidence.representative;
    const content = bodyOf.get<{ body: string }>(representative.hash, representative.filepath);
    if (!content) continue;

    const body = content.body;
    if (options.shouldRerank && options.wantsEvidence && rerankTokenBudget === undefined) {
      rerankTokenBudget = await store.getRerankTokenBudget(options.primaryQuery, options.intent);
    }

    const reference = {
      uri: representative.filepath,
      contentHash: representative.hash,
    };
    const locations: SearchLocation[] | undefined = options.includeLocations ? [] : undefined;
    if (locations !== undefined) {
      for (const match of groupEvidence.matches) {
        if (match.vectorStartUtf16 !== undefined) {
          locations.push(vectorChunkLocation(
            {
              uri: match.file,
              contentHash: match.contentHash,
            },
            match.vectorStartUtf16,
            match.vectorChunkSeq ?? null,
          ));
        }
      }
    }

    let passage: PassageWindow | undefined;
    let bodyAnchorStatus: "located" | "unavailable" = "unavailable";
    if (options.wantsEvidence) {
      const representativeMatch = groupEvidence.representativeMatch;
      let anchor: Utf16Span | undefined;
      if (representative.source === "vec") {
        const startUtf16 = representativeMatch.vectorStartUtf16;
        if (startUtf16 !== undefined && startUtf16 <= body.length && isUtf16Boundary(body, startUtf16)) {
          anchor = {
            startUtf16,
            endUtf16: startUtf16,
          };
        }
      } else {
        const lexicalLocation = findLexicalLocation(
          body,
          representativeMatch.query,
          reference,
        );
        if (lexicalLocation !== null) {
          anchor = {
            startUtf16: lexicalLocation.startUtf16,
            endUtf16: lexicalLocation.endUtf16,
          };
          locations?.push(lexicalLocation);
        }
      }

      if (anchor !== undefined) {
        passage = passageForAnchor(body, anchor, options.passageBudget, rerankTokenBudget);
        bodyAnchorStatus = "located";
      }
    }

    let rerankText = passage?.text;
    if (rerankText === undefined && (options.shouldRerank || options.wantsEvidence)) {
      const chunks = await chunkDocumentAsync(
        body,
        undefined,
        undefined,
        undefined,
        representative.filepath,
        options.chunkStrategy,
      );
      const bestChunk = chunks[selectBestChunkIndex(chunks, options.queryTerms, options.intentTerms)];
      rerankText = bestChunk?.text ?? body;
      if (options.wantsEvidence) {
        const selectionSpan = {
          startUtf16: bestChunk?.pos ?? 0,
          endUtf16: (bestChunk?.pos ?? 0) + rerankText.length,
        };
        locations?.push(selectionWindowLocation(reference, selectionSpan));
        passage = passageForAnchor(
          body,
          selectionSpan,
          options.passageBudget,
          rerankTokenBudget,
        );
        rerankText = passage.text;
      }
    }

    if (options.shouldRerank) {
      rerankInputs.push({
        file: fusedGroup.file,
        text: rerankText ?? body,
      });
    }

    const rrfRank = admittedIndex + 1;
    const topGroupRank = Math.min(...groupEvidence.matches.map(match => match.groupRank));
    candidates.push({
      group: groupEvidence.group,
      file: representative.filepath,
      contentHash: representative.hash,
      displayPath: representative.displayPath,
      title: representative.title,
      metadata: representative.metadata,
      score: 1 / rrfRank,
      rrfScore: fusedGroup.score,
      rrfRank,
      rrfTopRankBonus: rrfTopRankBonus(topGroupRank),
      matches: groupEvidence.matches,
      ...(options.wantsEvidence
        ? {
          passage,
          bodyAnchorStatus,
          representativeLeg: groupEvidence.representativeMatch.leg,
        }
        : {}),
      ...(locations === undefined ? {} : { locations }),
    });
  }

  return {
    candidates,
    rerankInputs,
  };
}

async function rerankCandidates(
  store: Store,
  candidates: readonly CandidateDraft[],
  rerankInputs: readonly { file: string; text: string }[],
  primaryQuery: string,
  intent: string | undefined,
): Promise<CandidateDraft[]> {
  if (rerankInputs.length === 0) return [...candidates];

  const reranked = await store.rerank(primaryQuery, [...rerankInputs], undefined, intent);
  const rerankScores = new Map(reranked.map(hit => [hit.file, hit.score]));
  return candidates
    .map(hit => {
      const rrfWeight = rrfPositionWeight(hit.rrfRank);
      return {
        ...hit,
        score: rrfWeight * hit.score + (1 - rrfWeight) * (rerankScores.get(hit.group.key) ?? 0),
      };
    })
    .sort((left, right) => right.score - left.score);
}

/** Group each retrieval leg, fuse groups, hydrate admitted sources, and optionally rerank them. */
export async function searchCandidates(store: Store, options: CandidateSearchOptions): Promise<CandidateSearchResult> {
  validateOptions(options);

  const filter = options.filter === undefined
    ? undefined
    : parseMetadataFilter(options.filter);
  const collections = [
    ...(options.collection ? [options.collection] : []),
    ...(options.collections ?? []),
  ];
  const scope = collections.length > 0 ? collections : undefined;
  const legs = await retrievalLegs(store, options);
  const retrieval = await retrieveGroupedCandidates(store, options, legs, scope, filter);
  const fusedGroups = reciprocalRankFusion(retrieval.rankings, retrieval.weights);
  const candidateLimit = options.candidateLimit ?? RERANK_CANDIDATE_LIMIT;
  const admittedGroups = fusedGroups.slice(0, candidateLimit);
  const primaryQuery = options.query !== undefined
    ? options.query
    : primaryQueryFor(options.queries);
  const shouldRerank = options.rerank !== false;
  const wantsEvidence = options.locations === true || options.passage !== undefined;
  const selectsContent = shouldRerank || wantsEvidence;
  const queryTerms = selectsContent
    ? primaryQuery.toLowerCase().split(/\s+/).filter(term => term.length > 2)
    : [];
  const intentTerms = selectsContent && options.intent ? extractIntentTerms(options.intent) : [];
  const hydrated = await hydrateCandidates(
    store,
    admittedGroups,
    retrieval.evidence,
    {
      primaryQuery,
      queryTerms,
      intentTerms,
      intent: options.intent,
      shouldRerank,
      wantsEvidence,
      includeLocations: options.locations === true,
      passageBudget: options.passage ?? { maxUtf8Bytes: 48_000 },
      chunkStrategy: selectsContent ? options.chunkStrategy : undefined,
    },
  );
  const scoredCandidates = shouldRerank
    ? await rerankCandidates(
      store,
      hydrated.candidates,
      hydrated.rerankInputs,
      primaryQuery,
      options.intent,
    )
    : hydrated.candidates;
  const minScore = options.minScore ?? 0;
  const limit = options.limit ?? 10;
  const results = scoredCandidates
    .filter(hit => hit.score >= minScore)
    .slice(0, limit)
    .map(hit => ({
      ...hit,
      context: store.getContextForFile(hit.file),
    }));

  return {
    results,
    coverage: {
      legs: retrieval.coverage,
      fusedGroups: fusedGroups.length,
      admittedGroups: admittedGroups.length,
      unavailableGroups: admittedGroups.length - hydrated.candidates.length,
      returnedGroups: results.length,
    },
  };
}
