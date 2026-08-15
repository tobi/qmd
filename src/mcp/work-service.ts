import { randomUUID } from "node:crypto";
import type {
  EmbedResult,
  ExpandedQuery,
  HybridQueryResult,
  IndexStatus,
  QMDStore,
  SearchResult,
  CollectionMutation,
  UpdateResult,
} from "../index.js";
import {
  DEFAULT_EMBED_MAX_BATCH_BYTES,
  DEFAULT_EMBED_MAX_DOCS_PER_BATCH,
  type ChunkStrategy,
} from "../store.js";
import { isValidCollectionName } from "../collections.js";

export const INTERACTIVE_QUEUE_LIMIT = 8;
export const INTERACTIVE_QUEUE_TIMEOUT_MS = 2_000;
export const LEXICAL_FALLBACK_CONCURRENCY_LIMIT = 2;
export const MAINTENANCE_QUEUE_LIMIT = 8;
export const API_VERSION = 1;

const MAX_SEARCH_LIMIT = 500;
const MAX_CANDIDATE_LIMIT = 500;
export const DAEMON_FEATURES = [
  "admission",
  "operations",
  "coalesced-update",
  "daemon-embed",
  "collection-ensure",
] as const;

export type DaemonSearchRequest = {
  query?: string;
  searches?: readonly ExpandedQuery[];
  collections?: readonly string[];
  limit?: number;
  minScore?: number;
  candidateLimit?: number;
  intent?: string;
  rerank?: boolean;
  explain?: boolean;
  chunkStrategy?: ChunkStrategy;
};

export type DaemonSearchResult = HybridQueryResult | SearchResult;
export type SearchMode = "semantic" | "lexical";
export type SearchDegradationReason =
  | "queue_full"
  | "queue_timeout"
  | "maintenance_busy"
  | "semantic_error"
  | "store_error"
  | "closed"
  | "malformed";

export type DaemonSearchResponse =
  | {
      status: "ok";
      mode: SearchMode;
      reason?: SearchDegradationReason;
      authoritativeEmpty: boolean;
      indexGeneration: number;
      results: DaemonSearchResult[];
    }
  | {
      status: "unavailable";
      reason: SearchDegradationReason;
      authoritativeEmpty: false;
      indexGeneration: number;
    };

export type AdmissionMetrics = {
  activeHeavy: number;
  queuedInteractive: number;
  maintenanceActive: boolean;
  queuedMaintenance: number;
  deduplicated: number;
  queueFull: number;
  queueTimeout: number;
  degraded: number;
};

export type OperationState = "queued" | "running" | "completed" | "failed";
export type OperationKind = "update" | "embed" | "ensure";

export type OperationStatus = {
  operationId: string;
  kind: OperationKind;
  state: OperationState;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  generation?: number;
  result?: Record<string, number | boolean | string>;
  error?: { reason: "closed" | "maintenance_failed" };
};

export type UpdateScope = { collections?: readonly string[] | null };

export type DaemonEmbedRequest = {
  force?: boolean;
  model?: string;
  collection?: string;
  maxDocsPerBatch?: number;
  maxBatchBytes?: number;
  chunkStrategy?: ChunkStrategy;
};

export type CollectionEnsureItem = {
  name: string;
  path: string;
  pattern?: string;
};

export type CollectionRename = {
  from: string;
  to: string;
};

export type CollectionContext = {
  collection: string;
  path: string;
  context: string;
};

export type CollectionEnsureRequest = {
  adds?: readonly CollectionEnsureItem[];
  updates?: readonly CollectionEnsureItem[];
  renames?: readonly CollectionRename[];
  contexts?: readonly CollectionContext[];
  markDirty?: boolean;
};

export class WorkServiceError extends Error {
  constructor(
    public readonly reason: SearchDegradationReason,
    message?: string,
  ) {
    super(message ?? reason);
    this.name = "WorkServiceError";
  }
}

type NormalizedSearchRequest = {
  query?: string;
  searches?: ExpandedQuery[];
  collections: string[];
  limit: number;
  minScore: number;
  candidateLimit: number;
  intent?: string;
  rerank: boolean;
  explain: boolean;
  chunkStrategy: ChunkStrategy;
  resultMode: SearchMode;
  modelConfigIdentity: string;
};

type SearchWaiter = {
  resolve: (response: DaemonSearchResponse) => void;
  reject: (error: unknown) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
};

type HeavyJob = {
  key: string;
  request: NormalizedSearchRequest;
  state: "queued" | "running" | "fallback";
  waiters: Set<SearchWaiter>;
  timer?: ReturnType<typeof setTimeout>;
};

type MaintenanceJob = {
  operation: OperationStatus;
  run: () => Promise<Record<string, number | boolean | string>>;
  key?: string;
};

type UpdateJob = MaintenanceJob & {
  scope: { all: boolean; collections: Set<string> };
};

type EnsureJob = MaintenanceJob & {
  request: CollectionEnsureRequest;
  reservedUpdate: boolean;
};

const DEFAULT_LIMIT = 10;
const DEFAULT_MIN_SCORE = 0;
const DEFAULT_CANDIDATE_LIMIT = 40;
const DEFAULT_CHUNK_STRATEGY: ChunkStrategy = "regex";

function abortError(): Error {
  const error = new Error("The search request was aborted");
  error.name = "AbortError";
  return error;
}

function raceAbort<T>(
  work: Promise<T>,
  signal: AbortSignal | undefined,
): Promise<T> {
  if (!signal) return work;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener("abort", onAbort);
      reject(abortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
    void work.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function isAbortSignalAborted(signal: AbortSignal | undefined): boolean {
  return signal?.aborted === true;
}

function errorReason(error: unknown): "closed" | "maintenance_failed" {
  return error instanceof WorkServiceError && error.reason === "closed"
    ? "closed"
    : "maintenance_failed";
}

function cloneOperation(operation: OperationStatus): OperationStatus {
  return {
    operationId: operation.operationId,
    kind: operation.kind,
    state: operation.state,
    createdAt: operation.createdAt,
    ...(operation.startedAt ? { startedAt: operation.startedAt } : {}),
    ...(operation.completedAt ? { completedAt: operation.completedAt } : {}),
    ...(operation.generation !== undefined
      ? { generation: operation.generation }
      : {}),
    ...(operation.result ? { result: { ...operation.result } } : {}),
    ...(operation.error ? { error: { ...operation.error } } : {}),
  };
}

function validateOptionalNumber(
  value: number | undefined,
  name: string,
  minimum: number,
  maximum?: number,
): void {
  if (
    value !== undefined &&
    (!Number.isFinite(value) ||
      value < minimum ||
      (maximum !== undefined && value > maximum) ||
      !Number.isInteger(value))
  ) {
    throw new WorkServiceError(
      "malformed",
      `${name} is outside its allowed range`,
    );
  }
}

export function validateDaemonSearchRequest(
  request: DaemonSearchRequest,
): void {
  if (typeof request !== "object" || request === null || Array.isArray(request))
    throw new WorkServiceError("malformed", "request must be an object");
  const hasQuery = request.query !== undefined;
  const hasSearches = request.searches !== undefined;
  if (hasQuery === hasSearches) {
    throw new WorkServiceError(
      "malformed",
      "exactly one of query or searches is required",
    );
  }
  if (
    hasQuery &&
    (typeof request.query !== "string" || request.query.trim().length === 0)
  ) {
    throw new WorkServiceError("malformed", "query must be a non-empty string");
  }
  if (hasSearches) {
    if (
      !Array.isArray(request.searches) ||
      request.searches.length === 0 ||
      request.searches.length > 10
    ) {
      throw new WorkServiceError(
        "malformed",
        "searches must contain between one and ten items",
      );
    }
    for (const search of request.searches) {
      if (
        !search ||
        !["lex", "vec", "hyde"].includes(search.type) ||
        typeof search.query !== "string" ||
        search.query.trim().length === 0
      ) {
        throw new WorkServiceError(
          "malformed",
          "each search must have a supported type and non-empty query",
        );
      }
    }
  }
  if (request.collections !== undefined) {
    if (
      !Array.isArray(request.collections) ||
      request.collections.some(
        (name) => typeof name !== "string" || !isValidCollectionName(name),
      )
    ) {
      throw new WorkServiceError(
        "malformed",
        "collections must be an array of names",
      );
    }
  }
  validateOptionalNumber(request.limit, "limit", 1, MAX_SEARCH_LIMIT);
  validateOptionalNumber(
    request.candidateLimit,
    "candidateLimit",
    1,
    MAX_CANDIDATE_LIMIT,
  );
  if (
    request.minScore !== undefined &&
    (!Number.isFinite(request.minScore) ||
      request.minScore < 0 ||
      request.minScore > 1)
  ) {
    throw new WorkServiceError(
      "malformed",
      "minScore must be between zero and one",
    );
  }
  if (request.intent !== undefined && typeof request.intent !== "string") {
    throw new WorkServiceError("malformed", "intent must be a string");
  }
  if (request.rerank !== undefined && typeof request.rerank !== "boolean") {
    throw new WorkServiceError("malformed", "rerank must be a boolean");
  }
  if (request.explain !== undefined && typeof request.explain !== "boolean") {
    throw new WorkServiceError("malformed", "explain must be a boolean");
  }
  if (
    request.chunkStrategy !== undefined &&
    request.chunkStrategy !== "auto" &&
    request.chunkStrategy !== "regex"
  ) {
    throw new WorkServiceError("malformed", "chunkStrategy is not supported");
  }
}

function isExplicitLexical(request: NormalizedSearchRequest): boolean {
  return (
    request.query === undefined &&
    request.rerank === false &&
    request.searches !== undefined &&
    request.searches.length > 0 &&
    request.searches.every((search) => search.type === "lex")
  );
}

function updateScope(collections: readonly string[] | null | undefined): {
  all: boolean;
  collections: Set<string>;
} {
  if (collections == null) return { all: true, collections: new Set() };
  return { all: false, collections: new Set(collections) };
}

function mergeUpdateScope(
  target: { all: boolean; collections: Set<string> },
  incoming: { all: boolean; collections: Set<string> },
): void {
  if (target.all || incoming.all) {
    target.all = true;
    target.collections.clear();
    return;
  }
  for (const collection of incoming.collections)
    target.collections.add(collection);
}

function validateEnsureRequest(
  request: CollectionEnsureRequest,
): CollectionEnsureRequest {
  if (
    typeof request !== "object" ||
    request === null ||
    Array.isArray(request)
  ) {
    throw new WorkServiceError("malformed", "request must be an object");
  }
  if (
    (request.adds !== undefined && !Array.isArray(request.adds)) ||
    (request.updates !== undefined && !Array.isArray(request.updates)) ||
    (request.renames !== undefined && !Array.isArray(request.renames)) ||
    (request.contexts !== undefined && !Array.isArray(request.contexts))
  ) {
    throw new WorkServiceError(
      "malformed",
      "adds, updates, renames, and contexts must be arrays when provided",
    );
  }
  const adds = [...(request.adds ?? [])];
  const updates = [...(request.updates ?? [])];
  const renames = [...(request.renames ?? [])];
  const contexts = [...(request.contexts ?? [])];
  const names = new Set<string>();
  const normalizeCollections = (
    collections: CollectionEnsureItem[],
  ): CollectionEnsureItem[] => {
    return collections.map((collection) => {
      if (
        typeof collection !== "object" ||
        collection === null ||
        Array.isArray(collection)
      )
        throw new WorkServiceError(
          "malformed",
          "collection lists must contain objects",
        );
      if (
        typeof collection.name !== "string" ||
        !isValidCollectionName(collection.name)
      ) {
        throw new WorkServiceError(
          "malformed",
          "collection names may contain only letters, numbers, hyphens, and underscores",
        );
      }
      if (names.has(collection.name))
        throw new WorkServiceError("malformed", "duplicate collection name");
      names.add(collection.name);
      if (
        typeof collection.path !== "string" ||
        collection.path.trim().length === 0
      ) {
        throw new WorkServiceError(
          "malformed",
          "collection paths must be non-empty strings",
        );
      }
      if (
        collection.pattern !== undefined &&
        (typeof collection.pattern !== "string" ||
          collection.pattern.length === 0)
      ) {
        throw new WorkServiceError(
          "malformed",
          "collection patterns must be non-empty strings",
        );
      }
      return {
        name: collection.name,
        path: collection.path,
        ...(collection.pattern !== undefined
          ? { pattern: collection.pattern }
          : {}),
      };
    });
  };
  const normalizedAdds = normalizeCollections(adds);
  const normalizedUpdates = normalizeCollections(updates);
  const addNames = new Set(normalizedAdds.map((collection) => collection.name));
  const updateNames = new Set(
    normalizedUpdates.map((collection) => collection.name),
  );
  const renameSources = new Set<string>();
  const renameTargets = new Set<string>();
  const normalizedRenames = renames.map((rename) => {
    if (
      typeof rename !== "object" ||
      rename === null ||
      Array.isArray(rename) ||
      typeof rename.from !== "string" ||
      typeof rename.to !== "string" ||
      !isValidCollectionName(rename.from) ||
      !isValidCollectionName(rename.to) ||
      rename.from === rename.to
    ) {
      throw new WorkServiceError(
        "malformed",
        "collection renames must use two different valid names",
      );
    }
    if (renameSources.has(rename.from) || renameTargets.has(rename.to))
      throw new WorkServiceError(
        "malformed",
        "collection renames are ambiguous",
      );
    renameSources.add(rename.from);
    renameTargets.add(rename.to);
    return { ...rename };
  });
  for (const source of renameSources) {
    if (
      renameTargets.has(source) ||
      addNames.has(source) ||
      updateNames.has(source)
    )
      throw new WorkServiceError("malformed", "collection mutations overlap");
  }
  for (const target of renameTargets) {
    if (addNames.has(target))
      throw new WorkServiceError("malformed", "collection mutations overlap");
  }

  const contextTargets = new Set<string>();
  const normalizedContexts = contexts.map((context) => {
    if (
      typeof context !== "object" ||
      context === null ||
      Array.isArray(context) ||
      typeof context.collection !== "string" ||
      !isValidCollectionName(context.collection) ||
      typeof context.path !== "string" ||
      context.path.trim().length === 0 ||
      typeof context.context !== "string"
    ) {
      throw new WorkServiceError(
        "malformed",
        "contexts must contain valid collection, path, and context strings",
      );
    }
    const key = `${context.collection}\0${context.path}`;
    if (contextTargets.has(key) || renameSources.has(context.collection))
      throw new WorkServiceError(
        "malformed",
        "collection contexts are ambiguous",
      );
    contextTargets.add(key);
    return { ...context };
  });
  if (
    request.markDirty !== undefined &&
    typeof request.markDirty !== "boolean"
  ) {
    throw new WorkServiceError("malformed", "markDirty must be a boolean");
  }
  return {
    adds: normalizedAdds,
    updates: normalizedUpdates,
    renames: normalizedRenames,
    contexts: normalizedContexts,
    markDirty: request.markDirty ?? true,
  };
}

export class QMDWorkService {
  private readonly store: QMDStore;
  private readonly modelConfigIdentity: string;
  private readonly heavyQueue: HeavyJob[] = [];
  private readonly heavyJobs = new Map<string, HeavyJob>();
  private readonly fallbackJobs = new Map<
    string,
    Promise<DaemonSearchResponse>
  >();
  private readonly maintenanceQueue: MaintenanceJob[] = [];
  private readonly maintenanceJobs = new Map<string, MaintenanceJob>();
  private readonly operations = new Map<string, OperationStatus>();
  private readonly activeWork = new Set<Promise<unknown>>();
  private readonly metricsState: Omit<
    AdmissionMetrics,
    "queuedInteractive" | "queuedMaintenance"
  > = {
    activeHeavy: 0,
    maintenanceActive: false,
    deduplicated: 0,
    queueFull: 0,
    queueTimeout: 0,
    degraded: 0,
  };
  private pendingUpdate: UpdateJob | undefined;
  private activeMaintenanceJob: MaintenanceJob | undefined;
  private activeFallbacks = 0;
  private reservedMaintenanceSlots = 0;
  private nextGeneration = 0;
  private indexGeneration = 0;
  private pumping = false;
  private closed = false;
  private closePromise: Promise<void> | undefined;

  constructor(store: QMDStore, options: { modelConfigIdentity?: string } = {}) {
    this.store = store;
    this.modelConfigIdentity = options.modelConfigIdentity ?? "default";
  }

  get metrics(): AdmissionMetrics {
    return {
      ...this.metricsState,
      queuedInteractive: this.heavyQueue.length,
      queuedMaintenance: this.maintenanceQueue.length,
    };
  }

  async health(): Promise<{
    admission: AdmissionMetrics;
    indexGeneration: number;
  }> {
    return { admission: this.metrics, indexGeneration: this.indexGeneration };
  }

  async search(
    request: DaemonSearchRequest,
    signal?: AbortSignal,
  ): Promise<DaemonSearchResponse> {
    if (this.closed)
      throw new WorkServiceError("closed", "the QMD work service is closed");
    if (isAbortSignalAborted(signal)) throw abortError();

    let normalized: NormalizedSearchRequest;
    try {
      normalized = await this.normalizeSearchRequest(request);
    } catch (error) {
      if (error instanceof WorkServiceError) {
        if (error.reason === "closed" || error.reason === "malformed")
          throw error;
        return this.unavailable(error.reason);
      }
      return this.unavailable("store_error");
    }
    if (isAbortSignalAborted(signal)) throw abortError();
    this.assertOpen();

    if (normalized.collections.length === 0) {
      return {
        status: "ok",
        mode: normalized.resultMode,
        authoritativeEmpty: true,
        indexGeneration: this.indexGeneration,
        results: [],
      };
    }
    if (isExplicitLexical(normalized))
      return this.runLexical(normalized, undefined, signal);
    return this.enqueueHeavy(normalized, signal);
  }

  private read<T>(operation: () => Promise<T>): Promise<T> {
    return this.trackStoreOperation(operation).catch((error: unknown) => {
      if (error instanceof WorkServiceError) throw error;
      throw new WorkServiceError("store_error", "QMD store unavailable");
    });
  }

  get(pathOrDocid: string, options?: { includeBody?: boolean }) {
    return this.read(() => this.store.get(pathOrDocid, options));
  }

  getDocumentBody(
    pathOrDocid: string,
    options?: { fromLine?: number; maxLines?: number },
  ) {
    return this.read(() => this.store.getDocumentBody(pathOrDocid, options));
  }

  multiGet(
    pattern: string,
    options?: { includeBody?: boolean; maxBytes?: number },
  ) {
    return this.read(() => this.store.multiGet(pattern, options));
  }

  getStatus(): Promise<IndexStatus> {
    return this.read(() => this.store.getStatus());
  }

  getGlobalContext(): Promise<string | undefined> {
    return this.read(() => this.store.getGlobalContext());
  }

  listCollections() {
    return this.read(async () => {
      const collections = await this.store.listCollections();
      return collections.map((collection) => ({
        name: collection.name,
        path: collection.pwd,
        pattern: collection.glob_pattern,
        documents: collection.doc_count,
        activeDocuments: collection.active_count,
        lastModified: collection.last_modified,
        includeByDefault: collection.includeByDefault,
      }));
    });
  }

  scheduleUpdate(scope: UpdateScope = {}): {
    operationId: string;
    state: OperationState;
    generation: number;
    coalesced: boolean;
  } {
    this.assertOpen();
    if (
      typeof scope !== "object" ||
      scope === null ||
      Array.isArray(scope) ||
      (scope.collections !== undefined &&
        scope.collections !== null &&
        (!Array.isArray(scope.collections) ||
          scope.collections.some(
            (name) => typeof name !== "string" || !isValidCollectionName(name),
          )))
    )
      throw new WorkServiceError(
        "malformed",
        "collections must be an array of names or null",
      );
    const incoming = updateScope(scope.collections);
    if (this.pendingUpdate) {
      mergeUpdateScope(this.pendingUpdate.scope, incoming);
      return {
        ...this.operationResponse(this.pendingUpdate.operation, true),
        generation: this.pendingUpdate.operation.generation!,
      };
    }

    this.assertMaintenanceCapacity();
    const operation = this.createOperation("update", ++this.nextGeneration);
    const job: UpdateJob = {
      operation,
      scope: incoming,
      run: async () => {
        const result = await this.store.update({
          collections: job.scope.all ? undefined : [...job.scope.collections],
        });
        this.indexGeneration = Math.max(
          this.indexGeneration,
          operation.generation ?? 0,
        );
        return aggregateUpdateResult(result);
      },
    };
    this.pendingUpdate = job;
    this.enqueueMaintenance(job);
    this.pump();
    return {
      ...this.operationResponse(operation, false),
      generation: operation.generation!,
    };
  }

  scheduleEmbed(request: DaemonEmbedRequest = {}): {
    operationId: string;
    state: OperationState;
    coalesced: boolean;
  } {
    this.assertOpen();
    validateEmbedRequest(request);
    const key = embedJobKey(request);
    const existing = this.maintenanceJobs.get(key);
    if (
      existing &&
      !(existing === this.activeMaintenanceJob && this.maintenanceQueue.length > 0)
    )
      return this.operationResponse(existing.operation, true);
    this.assertMaintenanceCapacity();
    const operation = this.createOperation("embed");
    const job: MaintenanceJob = {
      operation,
      key,
      run: async () => {
        const result = await this.store.embed({ ...request });
        return aggregateEmbedResult(result);
      },
    };
    this.enqueueMaintenance(job);
    this.pump();
    return this.operationResponse(operation, false);
  }

  scheduleEnsure(request: CollectionEnsureRequest): {
    operationId: string;
    state: OperationState;
    coalesced: boolean;
  } {
    this.assertOpen();
    const validated = validateEnsureRequest(request);
    const key = ensureJobKey(validated);
    const existing = this.maintenanceJobs.get(key);
    if (existing) return this.operationResponse(existing.operation, true);
    const hasMutations =
      (validated.adds?.length ?? 0) +
        (validated.updates?.length ?? 0) +
        (validated.renames?.length ?? 0) +
        (validated.contexts?.length ?? 0) >
      0;
    const reservedUpdate =
      validated.markDirty !== false && hasMutations && !this.pendingUpdate;
    this.assertMaintenanceCapacity(reservedUpdate ? 2 : 1);
    const operation = this.createOperation("ensure");
    const job: EnsureJob = {
      operation,
      key,
      request: validated,
      reservedUpdate,
      run: () => this.runEnsure(job),
    };
    if (reservedUpdate) this.reservedMaintenanceSlots += 1;
    this.enqueueMaintenance(job, true);
    this.pump();
    return this.operationResponse(operation, false);
  }

  getOperation(id: string): OperationStatus | undefined {
    const operation = this.operations.get(id);
    return operation ? cloneOperation(operation) : undefined;
  }

  async close(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.closed = true;
    this.rejectQueuedSearches(
      new WorkServiceError("closed", "the QMD work service is closed"),
    );
    this.failQueuedOperations();
    this.closePromise = (async () => {
      await Promise.allSettled([...this.activeWork]);
      await this.store.close();
    })();
    return this.closePromise;
  }

  private assertOpen(): void {
    if (this.closed)
      throw new WorkServiceError("closed", "the QMD work service is closed");
  }

  private async normalizeSearchRequest(
    request: DaemonSearchRequest,
  ): Promise<NormalizedSearchRequest> {
    validateDaemonSearchRequest(request);
    const collections =
      request.collections === undefined
        ? await this.read(() => this.store.getDefaultCollectionNames())
        : [...request.collections];
    const searches =
      request.searches === undefined
        ? undefined
        : request.searches.map((search) => ({ ...search }));
    const rerank = request.rerank ?? true;
    return {
      ...(request.query !== undefined ? { query: request.query } : {}),
      ...(searches ? { searches } : {}),
      collections,
      limit: request.limit ?? DEFAULT_LIMIT,
      minScore: request.minScore ?? DEFAULT_MIN_SCORE,
      candidateLimit: request.candidateLimit ?? DEFAULT_CANDIDATE_LIMIT,
      ...(request.intent !== undefined ? { intent: request.intent } : {}),
      rerank,
      explain: request.explain ?? false,
      chunkStrategy: request.chunkStrategy ?? DEFAULT_CHUNK_STRATEGY,
      resultMode:
        searches &&
        rerank === false &&
        searches.every((search) => search.type === "lex")
          ? "lexical"
          : "semantic",
      modelConfigIdentity: this.modelConfigIdentity,
    };
  }

  private searchKey(request: NormalizedSearchRequest): string {
    return JSON.stringify({
      query: request.query ?? null,
      searches:
        request.searches?.map((search) => ({
          type: search.type,
          query: search.query,
          line: search.line ?? null,
        })) ?? null,
      collections: request.collections,
      limit: request.limit,
      minScore: request.minScore,
      candidateLimit: request.candidateLimit,
      intent: request.intent ?? null,
      rerank: request.rerank,
      explain: request.explain,
      chunkStrategy: request.chunkStrategy,
      resultMode: request.resultMode,
      modelConfigIdentity: request.modelConfigIdentity,
    });
  }

  private enqueueHeavy(
    request: NormalizedSearchRequest,
    signal: AbortSignal | undefined,
  ): Promise<DaemonSearchResponse> {
    const key = this.searchKey(request);
    const existing = this.heavyJobs.get(key);
    if (existing) {
      this.metricsState.deduplicated += 1;
      return this.addWaiter(existing, signal);
    }
    const fallback = this.fallbackJobs.get(key);
    if (fallback) {
      this.metricsState.deduplicated += 1;
      return raceAbort(fallback, signal);
    }
    if (this.heavyQueue.length >= INTERACTIVE_QUEUE_LIMIT) {
      this.metricsState.queueFull += 1;
      return this.degrade(request, "queue_full", signal);
    }

    const job: HeavyJob = { key, request, state: "queued", waiters: new Set() };
    job.timer = setTimeout(
      () => this.timeoutHeavyJob(job),
      INTERACTIVE_QUEUE_TIMEOUT_MS,
    );
    job.timer.unref?.();
    this.heavyJobs.set(key, job);
    this.heavyQueue.push(job);
    const waiter = this.addWaiter(job, signal);
    this.pump();
    return waiter;
  }

  private addWaiter(
    job: HeavyJob,
    signal: AbortSignal | undefined,
  ): Promise<DaemonSearchResponse> {
    if (isAbortSignalAborted(signal)) return Promise.reject(abortError());
    return new Promise<DaemonSearchResponse>((resolve, reject) => {
      const waiter: SearchWaiter = { resolve, reject, signal };
      const onAbort = () => {
        if (!job.waiters.delete(waiter)) return;
        reject(abortError());
        if (job.state === "queued" && job.waiters.size === 0)
          this.removeQueuedJob(job);
      };
      waiter.onAbort = onAbort;
      job.waiters.add(waiter);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted) onAbort();
    });
  }

  private detachQueuedJob(job: HeavyJob): void {
    const index = this.heavyQueue.indexOf(job);
    if (index >= 0) this.heavyQueue.splice(index, 1);
    if (job.timer) clearTimeout(job.timer);
    this.heavyJobs.delete(job.key);
  }

  private removeQueuedJob(job: HeavyJob): void {
    if (job.state !== "queued") return;
    this.detachQueuedJob(job);
    this.pump();
  }

  private timeoutHeavyJob(job: HeavyJob): void {
    if (job.state !== "queued") return;
    this.detachQueuedJob(job);
    job.state = "fallback";
    this.metricsState.queueTimeout += 1;
    this.trackWork(this.degradeJob(job, "queue_timeout"));
    this.pump();
  }

  private async degradeJob(
    job: HeavyJob,
    reason: SearchDegradationReason,
  ): Promise<void> {
    const response = await this.degrade(job.request, reason);
    this.settleJob(job, response);
  }

  private async degrade(
    request: NormalizedSearchRequest,
    reason: SearchDegradationReason,
    signal?: AbortSignal,
  ): Promise<DaemonSearchResponse> {
    this.metricsState.degraded += 1;
    const key = this.searchKey(request);
    const existing = this.fallbackJobs.get(key);
    if (existing) {
      this.metricsState.deduplicated += 1;
      return raceAbort(existing, signal);
    }
    if (this.activeFallbacks >= LEXICAL_FALLBACK_CONCURRENCY_LIMIT) {
      return this.unavailable(reason);
    }

    this.activeFallbacks += 1;
    const work = this.runLexical(request, reason).finally(() => {
      this.activeFallbacks -= 1;
      if (this.fallbackJobs.get(key) === work) this.fallbackJobs.delete(key);
    });
    this.fallbackJobs.set(key, work);
    return raceAbort(work, signal);
  }

  private async runLexical(
    request: NormalizedSearchRequest,
    degradedReason: SearchDegradationReason | undefined,
    signal?: AbortSignal,
  ): Promise<DaemonSearchResponse> {
    const work = this.trackStoreOperation(async () => {
      const results = await this.executeLexical(request);
      return {
        status: "ok" as const,
        mode: "lexical" as const,
        ...(degradedReason ? { reason: degradedReason } : {}),
        authoritativeEmpty: degradedReason === undefined && results.length === 0,
        indexGeneration: this.indexGeneration,
        results,
      };
    });
    try {
      return await raceAbort(work, signal);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") throw error;
      return this.unavailable(
        error instanceof WorkServiceError ? error.reason : "store_error",
      );
    }
  }

  private async executeLexical(
    request: NormalizedSearchRequest,
  ): Promise<SearchResult[]> {
    const queries =
      request.searches?.map((search) => search.query) ??
      (request.query ? [request.query] : []);
    if (queries.length === 0) throw new WorkServiceError("malformed");
    const byFile = new Map<string, SearchResult>();
    for (const query of queries) {
      const results = await this.store.searchLex(query, {
        limit: request.limit,
        collection:
          request.collections.length > 0 ? request.collections : undefined,
      });
      for (const result of results) {
        if (result.score < request.minScore) continue;
        const key = result.filepath || result.displayPath;
        const previous = byFile.get(key);
        if (!previous || result.score > previous.score) byFile.set(key, result);
      }
    }
    return [...byFile.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, request.limit);
  }

  private async runHeavy(job: HeavyJob): Promise<void> {
    this.metricsState.activeHeavy += 1;
    try {
      let response: DaemonSearchResponse;
      try {
        const results = await this.store.search({
          ...(job.request.query !== undefined
            ? { query: job.request.query }
            : {}),
          ...(job.request.searches !== undefined
            ? { queries: job.request.searches }
            : {}),
          collections:
            job.request.collections.length > 0
              ? job.request.collections
              : undefined,
          limit: job.request.limit,
          minScore: job.request.minScore,
          candidateLimit: job.request.candidateLimit,
          ...(job.request.intent !== undefined
            ? { intent: job.request.intent }
            : {}),
          rerank: job.request.rerank,
          explain: job.request.explain,
          chunkStrategy: job.request.chunkStrategy,
        });
        if (!Array.isArray(results))
          throw new Error("search returned a non-array result");
        response = {
          status: "ok",
          mode: "semantic",
          authoritativeEmpty: results.length === 0,
          indexGeneration: this.indexGeneration,
          results,
        };
      } catch {
        response = await this.degrade(job.request, "semantic_error");
      }
      this.settleJob(job, response);
    } catch (error) {
      this.settleJob(
        job,
        this.unavailable(
          error instanceof WorkServiceError ? error.reason : "store_error",
        ),
      );
    } finally {
      if (this.heavyJobs.get(job.key) === job) this.heavyJobs.delete(job.key);
      this.metricsState.activeHeavy -= 1;
    }
  }

  private settleJob(job: HeavyJob, response: DaemonSearchResponse): void {
    for (const waiter of job.waiters) {
      waiter.signal?.removeEventListener("abort", waiter.onAbort!);
      waiter.resolve(response);
    }
    job.waiters.clear();
  }

  private unavailable(reason: SearchDegradationReason): DaemonSearchResponse {
    return {
      status: "unavailable",
      reason,
      authoritativeEmpty: false,
      indexGeneration: this.indexGeneration,
    };
  }

  private trackStoreOperation<T>(operation: () => Promise<T>): Promise<T> {
    return this.trackWork(
      (async () => {
        if (this.closed)
          throw new WorkServiceError(
            "closed",
            "the QMD work service is closed",
          );
        if (this.metricsState.maintenanceActive)
          throw new WorkServiceError(
            "maintenance_busy",
            "maintenance is active",
          );
        this.activeStoreOperationStarted();
        try {
          return await operation();
        } finally {
          this.activeStoreOperationFinished();
        }
      })(),
    );
  }

  private trackWork<T>(work: Promise<T>): Promise<T> {
    this.activeWork.add(work);
    void work.then(
      () => this.finishTrackedWork(work),
      () => this.finishTrackedWork(work),
    );
    return work;
  }

  private activeStoreOperationStarted(): void {
    this.storeOperationCount += 1;
  }

  private activeStoreOperationFinished(): void {
    this.storeOperationCount -= 1;
    this.pump();
  }

  private storeOperationCount = 0;

  private finishTrackedWork(work: Promise<unknown>): void {
    this.activeWork.delete(work);
    this.pump();
  }

  private pump(): void {
    if (this.pumping || this.closed || this.metricsState.maintenanceActive)
      return;
    this.pumping = true;
    try {
      if (this.metricsState.activeHeavy === 0 && this.heavyQueue.length > 0) {
        const job = this.heavyQueue.shift()!;
        if (job.state === "queued") {
          if (job.timer) clearTimeout(job.timer);
          job.state = "running";
          this.trackWork(this.runHeavy(job));
        }
        return;
      }
      if (
        this.storeOperationCount === 0 &&
        this.metricsState.activeHeavy === 0 &&
        this.maintenanceQueue.length > 0
      ) {
        const job = this.maintenanceQueue.shift()!;
        this.startMaintenance(job);
      }
    } finally {
      this.pumping = false;
    }
  }

  private startMaintenance(job: MaintenanceJob): void {
    this.metricsState.maintenanceActive = true;
    this.activeMaintenanceJob = job;
    this.storeOperationCount += 1;
    job.operation.state = "running";
    job.operation.startedAt = new Date().toISOString();
    if (
      job.operation.kind === "update" &&
      this.pendingUpdate?.operation.operationId === job.operation.operationId
    )
      this.pendingUpdate = undefined;
    const work = (async () => {
      try {
        const result = await job.run();
        job.operation.state = "completed";
        job.operation.result = result;
      } catch (error) {
        job.operation.state = "failed";
        job.operation.error = { reason: errorReason(error) };
      } finally {
        job.operation.completedAt = new Date().toISOString();
        this.metricsState.maintenanceActive = false;
        if (this.activeMaintenanceJob === job)
          this.activeMaintenanceJob = undefined;
        if (job.key && this.maintenanceJobs.get(job.key) === job)
          this.maintenanceJobs.delete(job.key);
        this.storeOperationCount -= 1;
        this.pump();
      }
    })();
    this.trackWork(work);
  }

  private createOperation(
    kind: OperationKind,
    generation?: number,
  ): OperationStatus {
    const operation: OperationStatus = {
      operationId: `op_${randomUUID()}`,
      kind,
      state: "queued",
      createdAt: new Date().toISOString(),
      ...(generation !== undefined ? { generation } : {}),
    };
    this.operations.set(operation.operationId, operation);
    if (this.operations.size > 128) {
      for (const [id, candidate] of this.operations) {
        if (candidate.state === "completed" || candidate.state === "failed") {
          this.operations.delete(id);
          break;
        }
      }
    }
    return operation;
  }

  private operationResponse(
    operation: OperationStatus,
    coalesced: boolean,
  ): {
    operationId: string;
    state: OperationState;
    generation?: number;
    coalesced: boolean;
  } {
    return {
      operationId: operation.operationId,
      state: operation.state,
      ...(operation.generation !== undefined
        ? { generation: operation.generation }
        : {}),
      coalesced,
    };
  }

  private assertMaintenanceCapacity(slots = 1): void {
    if (
      this.maintenanceQueue.length + this.reservedMaintenanceSlots + slots >
      MAINTENANCE_QUEUE_LIMIT
    ) {
      throw new WorkServiceError(
        "maintenance_busy",
        "the maintenance queue is full",
      );
    }
  }

  private enqueueMaintenance(
    job: MaintenanceJob,
    beforePendingUpdate = false,
  ): void {
    this.assertMaintenanceCapacity();
    if (job.key) this.maintenanceJobs.set(job.key, job);
    if (beforePendingUpdate && this.pendingUpdate) {
      const index = this.maintenanceQueue.indexOf(this.pendingUpdate);
      this.maintenanceQueue.splice(Math.max(0, index), 0, job);
      return;
    }
    this.insertInteractiveMaintenance(job);
  }

  private insertInteractiveMaintenance(job: MaintenanceJob): void {
    const firstEmbed = this.maintenanceQueue.findIndex(
      (queued) => queued.operation.kind === "embed",
    );
    if (firstEmbed >= 0) this.maintenanceQueue.splice(firstEmbed, 0, job);
    else this.maintenanceQueue.push(job);
  }

  private async runEnsure(
    job: EnsureJob,
  ): Promise<Record<string, number | boolean | string>> {
    const request = job.request;
    const affected = new Set<string>();
    try {
      await this.preflightEnsure(request);
      const mutations: CollectionMutation[] = [
        ...(request.renames ?? []).map((rename) => ({
          kind: "rename" as const,
          from: rename.from,
          to: rename.to,
        })),
        ...[...(request.adds ?? []), ...(request.updates ?? [])].map(
          (collection) => ({
            kind: "upsert" as const,
            name: collection.name,
            path: collection.path,
            ...(collection.pattern !== undefined
              ? { pattern: collection.pattern }
              : {}),
          }),
        ),
        ...(request.contexts ?? []).map((context) => ({
          kind: "context" as const,
          collection: context.collection,
          path: context.path,
          context: context.context,
        })),
      ];
      await this.store.applyCollectionMutations(mutations);
      for (const rename of request.renames ?? []) affected.add(rename.to);
      for (const collection of [
        ...(request.adds ?? []),
        ...(request.updates ?? []),
      ]) {
        affected.add(collection.name);
      }
      for (const context of request.contexts ?? []) {
        affected.add(context.collection);
      }
      let updateOperationId: string | undefined;
      if (request.markDirty !== false && affected.size > 0) {
        this.releaseEnsureUpdateReservation(job);
        updateOperationId = this.scheduleUpdate({
          collections: [...affected],
        }).operationId;
      }
      return {
        adds: (request.adds ?? []).length,
        updates: (request.updates ?? []).length,
        renames: (request.renames ?? []).length,
        contexts: (request.contexts ?? []).length,
        ...(updateOperationId ? { updateOperationId } : {}),
      };
    } finally {
      this.releaseEnsureUpdateReservation(job);
    }
  }

  private async preflightEnsure(
    request: CollectionEnsureRequest,
  ): Promise<void> {
    const existing = new Set(
      (await this.store.listCollections()).map((collection) => collection.name),
    );
    const finalNames = new Set(existing);
    for (const rename of request.renames ?? []) {
      if (!existing.has(rename.from) || existing.has(rename.to))
        throw new Error("collection rename preflight failed");
      finalNames.delete(rename.from);
      finalNames.add(rename.to);
    }
    for (const collection of request.adds ?? []) {
      if (finalNames.has(collection.name))
        throw new Error("collection add preflight failed");
      finalNames.add(collection.name);
    }
    for (const collection of request.updates ?? []) {
      if (!finalNames.has(collection.name))
        throw new Error("collection update preflight failed");
    }
    for (const context of request.contexts ?? []) {
      if (!finalNames.has(context.collection))
        throw new Error("collection context preflight failed");
    }
  }

  private releaseEnsureUpdateReservation(job: EnsureJob): void {
    if (!job.reservedUpdate) return;
    job.reservedUpdate = false;
    this.reservedMaintenanceSlots -= 1;
  }

  private rejectQueuedSearches(error: WorkServiceError): void {
    for (const job of this.heavyQueue.splice(0)) {
      this.detachQueuedJob(job);
      for (const waiter of job.waiters) {
        waiter.signal?.removeEventListener("abort", waiter.onAbort!);
        waiter.reject(error);
      }
      job.waiters.clear();
    }
  }

  private failQueuedOperations(): void {
    for (const job of this.maintenanceQueue.splice(0)) {
      if (job.key && this.maintenanceJobs.get(job.key) === job)
        this.maintenanceJobs.delete(job.key);
      if (job.operation.kind === "ensure")
        this.releaseEnsureUpdateReservation(job as EnsureJob);
      job.operation.state = "failed";
      job.operation.error = { reason: "closed" };
      job.operation.completedAt = new Date().toISOString();
    }
    this.pendingUpdate = undefined;
  }
}

function aggregateUpdateResult(
  result: UpdateResult,
): Record<string, number | boolean | string> {
  return {
    collections: result.collections,
    indexed: result.indexed,
    updated: result.updated,
    unchanged: result.unchanged,
    removed: result.removed,
    skipped: result.skipped,
    needsEmbedding: result.needsEmbedding,
  };
}

function aggregateEmbedResult(
  result: EmbedResult,
): Record<string, number | boolean | string> {
  return {
    docsProcessed: result.docsProcessed,
    chunksEmbedded: result.chunksEmbedded,
    errors: result.errors,
    durationMs: result.durationMs,
  };
}

function validateEmbedRequest(request: DaemonEmbedRequest): void {
  if (typeof request !== "object" || request === null || Array.isArray(request))
    throw new WorkServiceError("malformed", "request must be an object");
  if (request.force !== undefined && typeof request.force !== "boolean")
    throw new WorkServiceError("malformed", "force must be a boolean");
  if (
    request.model !== undefined &&
    (typeof request.model !== "string" || request.model.length === 0)
  )
    throw new WorkServiceError("malformed", "model must be a non-empty string");
  if (
    request.collection !== undefined &&
    (typeof request.collection !== "string" ||
      !isValidCollectionName(request.collection))
  )
    throw new WorkServiceError(
      "malformed",
      "collection must be a non-empty string",
    );
  validateOptionalNumber(request.maxDocsPerBatch, "maxDocsPerBatch", 1);
  validateOptionalNumber(request.maxBatchBytes, "maxBatchBytes", 1);
  if (
    request.chunkStrategy !== undefined &&
    request.chunkStrategy !== "auto" &&
    request.chunkStrategy !== "regex"
  )
    throw new WorkServiceError("malformed", "chunkStrategy is not supported");
}

function embedJobKey(request: DaemonEmbedRequest): string {
  return JSON.stringify({
    force: request.force ?? false,
    model: request.model ?? null,
    collection: request.collection ?? null,
    maxDocsPerBatch:
      request.maxDocsPerBatch ?? DEFAULT_EMBED_MAX_DOCS_PER_BATCH,
    maxBatchBytes: request.maxBatchBytes ?? DEFAULT_EMBED_MAX_BATCH_BYTES,
    chunkStrategy: request.chunkStrategy ?? "regex",
  });
}

function ensureJobKey(request: CollectionEnsureRequest): string {
  return JSON.stringify({
    adds: request.adds ?? [],
    updates: request.updates ?? [],
    renames: request.renames ?? [],
    contexts: request.contexts ?? [],
    markDirty: request.markDirty !== false,
  });
}
