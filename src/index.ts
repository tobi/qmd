/**
 * QMD SDK - Library mode for programmatic access to QMD search and indexing.
 *
 * Usage:
 *   import { createStore } from '@tobilu/qmd'
 *
 *   const store = await createStore({
 *     dbPath: './my-index.sqlite',
 *     config: {
 *       collections: {
 *         docs: { path: '/path/to/docs', pattern: '**\/*.md' }
 *       }
 *     }
 *   })
 *
 *   const results = await store.search({ query: "how does auth work?" })
 *   await store.close()
 */
import { AsyncLocalStorage } from "node:async_hooks";

import {
  createStore as createStoreInternal,
  hybridQuery,
  structuredSearch,
  extractSnippet,
  addLineNumbers,
  DEFAULT_MULTI_GET_MAX_BYTES,
  reindexCollection,
  EmbedLockBusyError,
  generateEmbeddings,
  listCollections as storeListCollections,
  syncConfigToDb,
  invalidateConfigHash,
  getStoreCollections,
  getStoreCollection,
  getStoreGlobalContext,
  getStoreContexts,
  upsertStoreCollection,
  deleteStoreCollection,
  renameStoreCollection,
  removeCollection as removeIndexedCollection,
  renameCollection as renameIndexedCollection,
  reconcilePublishedRename,
  assertRenameTargetAvailable,
  hasCollectionIndexState,
  updateStoreContext,
  removeStoreContext,
  setStoreGlobalContext,
  vacuumDatabase,
  cleanupOrphanedContent,
  cleanupOrphanedVectors,
  copyVectorsToNewCollections,
  deleteLLMCache,
  deleteInactiveDocuments,
  clearAllEmbeddings,
  type Store as InternalStore,
  type DocumentResult,
  type DocumentNotFound,
  type DocumentExcludedByIgnore,
  type DocumentLookupError,
  type SearchResult,
  type HybridQueryResult,
  type HybridQueryOptions,
  type HybridQueryExplain,
  type ExpandedQuery,
  type StructuredSearchOptions,
  type MultiGetResult,
  type IndexStatus,
  type IndexHealthInfo,
  type SearchHooks,
  type ReindexProgress,
  type ReindexResult,
  type EmbedProgress,
  type EmbedResult,
  type ChunkStrategy,
} from "./store.js";
import {
  LlamaCpp,
  withLLMSessionForLlm,
} from "./llm.js";
export { RemoteEmbedder } from "./remote-embed.js";
export type { RemoteEmbedOptions } from "./remote-embed.js";
export { JevReranker } from "./jev.js";
export type { JevRerankOptions } from "./jev.js";
export { createCommandProcessor } from "./processors.js";
export type { DocumentProcessor, ProcessorInput, ProcessorDocument, CommandProcessorSpec } from "./processors.js";
import { snapshotProcessor, type DocumentProcessor } from "./processors.js";
import { requireProcessor, hasProcessorMode } from "./processor-store.js";
import type {
  DocumentMetadata,
  MetadataScalar,
  MetadataScalarArray,
  MetadataValue,
  MetadataValueType,
} from "./metadata.js";
import {
  parseMetadataFilter,
  parseMetadataMatch,
  MetadataFilterError,
  type MetadataFilter,
  type MetadataFilterGroup,
  type MetadataFilterNegation,
  type MetadataCondition,
  type MetadataPredicate,
  type MetadataPredicateGroup,
  type MetadataPredicateNegation,
  type MetadataMatch,
  type MetadataEntryCondition,
  type MetadataEntryField,
} from "./metadata-filter.js";
import {
  listMetadata as storeListMetadata,
  MetadataBindingBudgetError,
  MetadataOptionError,
  type ListMetadataOptions,
  type ListMetadataResult,
  type MetadataKeySummary,
  type MetadataKeyTypeSummary,
  type MetadataValueCount,
  type MetadataKeyOverview,
} from "./metadata-store.js";
import {
  CollectionConfigScope,
  ConfigScopeClosedError,
  type ConfigMirror,
  type ConfigSourceOptions,
  type Collection,
  type CollectionConfig,
  type NamedCollection,
  type ContextMap,
} from "./collections.js";

// Re-export types for SDK consumers
export type {
  DocumentResult,
  DocumentNotFound,
  DocumentExcludedByIgnore,
  DocumentLookupError,
  SearchResult,
  HybridQueryResult,
  HybridQueryOptions,
  HybridQueryExplain,
  ExpandedQuery,
  StructuredSearchOptions,
  MultiGetResult,
  IndexStatus,
  IndexHealthInfo,
  SearchHooks,
  ReindexProgress,
  ReindexResult,
  EmbedProgress,
  EmbedResult,
  Collection,
  CollectionConfig,
  NamedCollection,
  ContextMap,
};

// Re-export metadata and metadata-filter types shared by every search surface
export type {
  DocumentMetadata,
  MetadataScalar,
  MetadataScalarArray,
  MetadataValue,
  MetadataValueType,
  MetadataFilter,
  MetadataFilterGroup,
  MetadataFilterNegation,
  MetadataCondition,
  MetadataPredicate,
  MetadataPredicateGroup,
  MetadataPredicateNegation,
  MetadataMatch,
  MetadataEntryCondition,
  MetadataEntryField,
};
export { parseMetadataFilter, parseMetadataMatch, MetadataFilterError };

// Re-export metadata discovery types (listMetadata() and status metadata keys)
export type {
  ListMetadataOptions,
  ListMetadataResult,
  MetadataKeySummary,
  MetadataKeyTypeSummary,
  MetadataValueCount,
  MetadataKeyOverview,
};
export { MetadataBindingBudgetError, MetadataOptionError };

// Re-export the internal Store type for advanced consumers
export type { InternalStore };

// Re-export utility functions and types used by frontends
export { extractSnippet, addLineNumbers, DEFAULT_MULTI_GET_MAX_BYTES };

// Re-export the embed busy error so SDK callers can recognize a run that lost
// the index's embed lock to another process instead of retrying blindly.
export { EmbedLockBusyError };
export type { ChunkStrategy } from "./store.js";

// Re-export the per-store collection config ownership API. Each store binds one
// scope; consumers can build scopes directly to mirror the CLI's behavior.
export { CollectionConfigScope, ConfigScopeClosedError };
export { PendingRenameError, CorruptPendingRenameError, ConfigLockTimeoutError, NestedConfigMutationError } from "./collections.js";
export type { ConfigSourceOptions, ConfigMirror };

// Re-export getDefaultDbPath for CLI/MCP that need the default database location
export { getDefaultDbPath } from "./store.js";

// Re-export Maintenance class for CLI housekeeping operations
export { Maintenance } from "./maintenance.js";

/**
 * Progress info emitted during update() for each file processed.
 */
export type UpdateProgress = {
  collection: string;
  file: string;
  current: number;
  total: number;
};

/**
 * Aggregated result from update() across all collections.
 */
export type UpdateResult = {
  collections: number;
  indexed: number;
  updated: number;
  unchanged: number;
  removed: number;
  skipped: number;
  /** Vector rows removed because their (hash, collection) no longer has an active document. */
  staleVectorsRemoved: number;
  /** Vector rows copied into the partition of a collection that gained an already-embedded hash. */
  vectorsCopied: number;
  needsEmbedding: number;
};

/**
 * Options for the unified search() method.
 */
export interface SearchOptions {
  /** Cancels remote embedding/rerank stages; local generation may finish before cancellation is observed. */
  signal?: AbortSignal;
  /** Simple query string — will be auto-expanded via LLM */
  query?: string;
  /** Pre-expanded queries (from expandQuery) — skips auto-expansion */
  queries?: ExpandedQuery[];
  /** Domain intent hint — steers reranking and snippet/chunk selection */
  intent?: string;
  /** Rerank results using LLM (default: true) */
  rerank?: boolean;
  /** Filter to a specific collection */
  collection?: string;
  /** Filter to specific collections */
  collections?: string[];
  /** Metadata filter — every returned result satisfies it */
  filter?: MetadataFilter;
  /** Max results (default: 10) */
  limit?: number;
  /** Max candidates to rerank (default: 40) */
  candidateLimit?: number;
  /** Minimum score threshold */
  minScore?: number;
  /** Include explain traces */
  explain?: boolean;
  /** Chunk strategy: "auto" (default, uses AST for code files) or "regex" (legacy) */
  chunkStrategy?: ChunkStrategy;
}

/**
 * Options for searchLex() — BM25 keyword search.
 */
export interface LexSearchOptions {
  limit?: number;
  collection?: string | string[];
  /** Metadata filter — every returned result satisfies it */
  filter?: MetadataFilter;
}

/**
 * Options for searchVector() — vector similarity search.
 */
export interface VectorSearchOptions {
  signal?: AbortSignal;
  limit?: number;
  collection?: string | string[];
  /** Metadata filter — every returned result satisfies it */
  filter?: MetadataFilter;
}

/**
 * Options for expandQuery() — manual query expansion.
 */
export interface ExpandQueryOptions {
  /**
   * @deprecated Ignored. Intent no longer feeds the expansion model — caller
   * intent is meta-language the model reproduced verbatim as sub-queries.
   * Pass intent via SearchOptions instead, where it shapes reranking and
   * snippet selection.
   */
  intent?: string;
}

/**
 * Options for creating a QMD store.
 *
 * Provide `dbPath` and optionally `configPath` (YAML file) or `config` (inline).
 * If neither configPath nor config is provided, the store reads from existing
 * DB state (useful for reopening a previously-configured store).
 *
 * A given `configPath` or `config` is owned by the store that was created with
 * it: collection and context mutations write through to it, and to nothing
 * else, for as long as that store is open. Multiple stores may be open at once;
 * closing one never rebinds another.
 */
export interface StoreOptions {
  /** Path to the SQLite database file */
  dbPath: string;
  /** Path to a YAML config file owned by this store (mutually exclusive with `config`) */
  configPath?: string;
  /** Inline collection config owned by this store (mutually exclusive with `configPath`) */
  config?: CollectionConfig;
}

/**
 * The QMD SDK store — provides search, retrieval, collection management,
 * context management, and indexing operations.
 *
 * All methods are async. The store manages its own LlamaCpp instance
 * (lazy-loaded, auto-unloaded after inactivity) — no global singletons.
 */
export interface QMDStore {
  /**
   * The underlying internal store (for advanced use).
   *
   * It bypasses the SDK operation lifecycle: calls made through it, or closing
   * it directly, are neither tracked nor drained by close(), so they can race
   * an in-flight operation of this store.
   */
  readonly internal: InternalStore;
  /** Path to the SQLite database */
  readonly dbPath: string;

  // ── Search ──────────────────────────────────────────────────────────

  /** Full search: query expansion + multi-signal retrieval + LLM reranking */
  search(options: SearchOptions): Promise<HybridQueryResult[]>;

  /** BM25 keyword search (fast, no LLM) */
  searchLex(query: string, options?: LexSearchOptions): Promise<SearchResult[]>;

  /** Vector similarity search (embedding model, no reranking) */
  searchVector(query: string, options?: VectorSearchOptions): Promise<SearchResult[]>;

  /** Expand a query into typed sub-searches (lex/vec/hyde) for manual control */
  expandQuery(query: string, options?: ExpandQueryOptions): Promise<ExpandedQuery[]>;

  /** Find documents semantically similar to a given document using embedding cosine similarity */
  findSimilarByEmbedding(pathOrDocid: string, limit?: number, collection?: string): Promise<SearchResult[]>;

  // ── Document Retrieval ──────────────────────────────────────────────

  /** Get a single document by path or docid */
  get(pathOrDocid: string, options?: { includeBody?: boolean }): Promise<DocumentResult | DocumentLookupError>;

  /** Get the body content of a document, optionally sliced by line range */
  getDocumentBody(pathOrDocid: string, opts?: { fromLine?: number; maxLines?: number }): Promise<string | null>;

  /** Get multiple documents by glob pattern or comma-separated list */
  multiGet(pattern: string, options?: { includeBody?: boolean; maxBytes?: number }): Promise<{ docs: MultiGetResult[]; errors: string[] }>;

  // ── Collection Management ───────────────────────────────────────────

  /**
   * Add or update a collection.
   *
   * For a store bound to a configuration, the change is published there first
   * and mirrored into SQLite under the same config lock, so a refused config
   * write leaves no database-only edit behind.
   */
  addCollection(name: string, opts: { path: string; pattern?: string; ignore?: string[] }): Promise<void>;

  /**
   * Remove a collection. Returns whether anything was removed.
   *
   * Repeating it finishes a removal whose index half failed or was interrupted:
   * documents, vectors and processor state still under the name are removed even
   * if a later config sync already dropped its `store_collections` row.
   */
  removeCollection(name: string): Promise<boolean>;

  /**
   * Rename a collection. Returns whether anything was renamed.
   *
   * The configuration is published first, preceded by a marker for this
   * old→new pair, and the index half runs against that evidence. A retry after
   * a failed or interrupted index move finishes it, including when a config
   * sync has already written the target row. Without such evidence the call
   * changes nothing and returns false: a name the configuration no longer lists
   * is not moved into a collection that merely happens to exist there.
   */
  renameCollection(oldName: string, newName: string): Promise<boolean>;

  /** List all collections with document stats */
  listCollections(): Promise<{ name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null; includeByDefault: boolean }[]>;

  /**
   * Discover metadata keys, types, and value counts across the documents in
   * scope. `filter` selects which documents are counted. `match` selects
   * which of their metadata entries are reported, with the filter grammar
   * evaluated against each entry (a condition's `field` is the entry's `key` or
   * `value`). Keys and values are windowed by `keyLimit`/`keyOffset` and
   * `valueLimit`/`valueOffset`, and the result carries the totals and
   * remainders needed to page. Every value reported is one an `eq` filter can
   * match under the same scope, and the whole result comes from one database
   * snapshot. Throws MetadataFilterError for an invalid predicate,
   * MetadataOptionError for an option outside its domain, and
   * MetadataBindingBudgetError when `filter` and `match` together bind more
   * SQL parameters than one statement allows.
   */
  listMetadata(options?: ListMetadataOptions): Promise<ListMetadataResult>;

  /** Get names of collections included by default in queries */
  getDefaultCollectionNames(): Promise<string[]>;

  // ── Context Management ──────────────────────────────────────────────

  /** Add context for a path within a collection */
  addContext(collectionName: string, pathPrefix: string, contextText: string): Promise<boolean>;

  /** Remove context from a collection path */
  removeContext(collectionName: string, pathPrefix: string): Promise<boolean>;

  /** Set global context (applies to all collections) */
  setGlobalContext(context: string | undefined): Promise<void>;

  /** Get global context */
  getGlobalContext(): Promise<string | undefined>;

  /** List all contexts across all collections */
  listContexts(): Promise<Array<{ collection: string; path: string; context: string }>>;

  // ── Indexing ────────────────────────────────────────────────────────

  /** Re-index collections by scanning the filesystem */
  update(options?: {
    processors?: Record<string, DocumentProcessor>;
    signal?: AbortSignal;
    collections?: string[];
    onProgress?: (info: UpdateProgress) => void;
  }): Promise<UpdateResult>;

  /**
   * Generate vector embeddings for documents that need them.
   *
   * Embedding is exclusive per index: this rejects with EmbedLockBusyError
   * while another CLI/SDK/direct run holds this index's embed lock, rather than
   * interleaving partial-chunk cleanup and vector writes with it. The lock is
   * released on success, error and cancellation.
   */
  embed(options?: {
    signal?: AbortSignal;
    force?: boolean;
    model?: string;
    /** Restrict embedding to documents in one collection. */
    collection?: string;
    maxDocsPerBatch?: number;
    maxBatchBytes?: number;
    chunkStrategy?: ChunkStrategy;
    onProgress?: (info: EmbedProgress) => void;
  }): Promise<EmbedResult>;

  // ── Index Health ────────────────────────────────────────────────────

  /** Get index status (document counts, collections, embedding state) */
  getStatus(): Promise<IndexStatus>;

  /** Get index health info (stale embeddings, etc.) */
  getIndexHealth(): Promise<IndexHealthInfo>;

  // ── Lifecycle ───────────────────────────────────────────────────────

  /**
   * Close the store and release all resources (LLM models, DB connection).
   *
   * The transition is synchronous: operations started after this call reject
   * with StoreClosedError. Operations already admitted run to completion before
   * the LLM instance is disposed and SQLite is closed, so none of them observes
   * a torn-down model or a closed connection mid-flight. Concurrent and
   * repeated calls share one completion: they all resolve once that teardown
   * finished.
   *
   * Do not await close() from inside an operation of this store — a processor,
   * a progress callback, or anything they await. Such a call waits for the very
   * operation it runs in and can only deadlock, so it rejects with
   * StoreCloseDeadlockError instead.
   *
   * The drain is deliberately unbounded: an operation that never settles keeps
   * close() — and any shutdown awaiting it — pending, because closing SQLite or
   * disposing models underneath it is the race this lifecycle exists to remove.
   * Bound the operation itself (AbortSignal, processor timeout), not the close.
   */
  close(): Promise<void>;
}

// =============================================================================
// Store operation lifecycle
// =============================================================================
//
// A store's SQLite connection and its LlamaCpp instance are shared by every
// method it exposes, while the LLM layer leases resources per model call
// (embed/rerank/expand). An SDK operation such as search() or update() keeps
// using SQLite after its last LLM call returns, so closing must drain whole SDK
// operations — not merely their LLM leases — before it touches either resource:
//
//   open     → operations are admitted immediately.
//   closing  → new operations are rejected; admitted ones run to completion.
//   closed   → terminal: resources released, nothing can be admitted.
//
// close() makes the transition synchronously, waits for the admitted count to
// drop to zero, and only then disposes the LLM and closes SQLite. Every caller
// shares one completion promise, so concurrent closes neither tear the store
// down twice nor return before the drain finished.
//
// The drain is unbounded on purpose. A process-wide shutdown that wants a
// deadline must bound the operations it admits (AbortSignal, processor
// timeouts); closing under a running operation would trade a hang for a
// corrupt read or a native crash.
//
// Admission is tracked through AsyncLocalStorage so close() can tell when it is
// called from inside one of this store's own operations. That call waits for
// the operation it belongs to, which waits for the close: it can never settle,
// so it fails fast instead of deadlocking.

/** Thrown when an operation is attempted on a store that is closing or closed. */
export class StoreClosedError extends Error {
  constructor(message = "QMD store is closed") {
    super(message);
    this.name = "StoreClosedError";
  }
}

/** Thrown when close() is awaited from inside an operation of the same store. */
export class StoreCloseDeadlockError extends Error {
  constructor() {
    super(
      "close() was called from inside an operation of the same store; " +
      "that operation's completion is what close() is waiting for, so awaiting it can only deadlock",
    );
    this.name = "StoreCloseDeadlockError";
  }
}

type StoreLifecycleState = "open" | "closing" | "closed";

/**
 * One node per admitted operation. Nested operations keep their parent, so an
 * operation admitted by another store remains distinguishable.
 *
 * `active` is cleared when the operation settles. Async context outlives the
 * call that created it (a timer or abort listener set up inside an operation
 * still runs under its frame), so a frame that is no longer active must not
 * make a later, legitimate close() look like a deadlock.
 */
type OperationFrame = {
  store: StoreOperationLifecycle;
  active: boolean;
  parent: OperationFrame | undefined;
};

const operationContext = new AsyncLocalStorage<OperationFrame>();

class StoreOperationLifecycle {
  private _state: StoreLifecycleState = "open";
  private admitted = 0;
  private settleDrain: (() => void) | null = null;
  private completion: Promise<void> | null = null;

  /**
   * Admit one SDK operation and run it inside this store's operation context.
   *
   * The state check and the count happen in one synchronous block: close()
   * flips the state synchronously too, so an operation either takes its slot
   * before close() begins (and is drained) or observes "closing" and is
   * rejected. The count is released in `finally`, so a throwing or aborting
   * operation cannot strand the drain.
   */
  async run<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    if (this._state !== "open") {
      throw new StoreClosedError(`QMD store is ${this._state}: ${operation}() cannot start`);
    }
    this.admitted += 1;
    const frame: OperationFrame = { store: this, active: true, parent: operationContext.getStore() };
    try {
      return await operationContext.run(frame, fn);
    } finally {
      frame.active = false;
      this.admitted -= 1;
      if (this.admitted === 0) {
        const settle = this.settleDrain;
        this.settleDrain = null;
        settle?.();
      }
    }
  }

  /**
   * Enter "closing" and return the shared completion of the teardown.
   *
   * `teardown` runs only after every admitted operation finished. It is invoked
   * exactly once, because only the call that observed "open" starts it; later
   * calls share its completion.
   */
  close(teardown: () => Promise<void>): Promise<void> {
    // Checked before the state so a close from inside a callback is reported as
    // the deadlock it is, instead of joining a completion that cannot settle.
    if (this.insideOwnOperation()) {
      return Promise.reject(new StoreCloseDeadlockError());
    }
    if (this.completion) return this.completion;

    this._state = "closing";
    // Nothing can be admitted past the synchronous transition above, so the
    // admitted count only decreases from here and this drain settles once.
    const { promise: drained, resolve: settleDrain } = Promise.withResolvers<void>();
    if (this.admitted === 0) settleDrain();
    else this.settleDrain = settleDrain;
    const completion = (async () => {
      try {
        await drained;
        await teardown();
      } finally {
        this._state = "closed";
      }
    })();
    this.completion = completion;
    return completion;
  }

  /** True while an active operation of this store is in the async context. */
  private insideOwnOperation(): boolean {
    for (let frame = operationContext.getStore(); frame; frame = frame.parent) {
      if (frame.active && frame.store === this) return true;
    }
    return false;
  }
}

/**
 * Create a QMD store for programmatic access to search and indexing.
 *
 * @example
 * ```typescript
 * // With a YAML config file
 * const store = await createStore({
 *   dbPath: './index.sqlite',
 *   configPath: './qmd.yml',
 * })
 *
 * // With inline config (no files needed besides the DB)
 * const store = await createStore({
 *   dbPath: './index.sqlite',
 *   config: {
 *     collections: {
 *       docs: { path: '/path/to/docs', pattern: '**\/*.md' }
 *     }
 *   }
 * })
 *
 * const results = await store.search({ query: "authentication flow" })
 * await store.close()
 * ```
 */
export async function createStore(options: StoreOptions): Promise<QMDStore> {
  if (!options.dbPath) {
    throw new Error("dbPath is required");
  }
  if (options.configPath && options.config) {
    throw new Error("Provide either configPath or config, not both");
  }

  // Each store owns its own configuration scope. A YAML or inline source is
  // bound to this store for its lifetime, so another store created in the same
  // process can neither redirect nor reset this store's write-through target.
  // DB-only mode has no external config and uses existing store_collections.
  const configScope = options.configPath
    ? new CollectionConfigScope({ configPath: options.configPath })
    : options.config
      ? new CollectionConfigScope({ config: options.config })
      : undefined;

  // Resolve model selection from this store's config before opening SQLite, so
  // an invalid selection fails without creating a database.
  const configuredConfig: CollectionConfig | undefined = configScope?.load();

  // Create a per-store LlamaCpp instance — lazy-loads models on first use,
  // auto-unloads after 5 min inactivity to free VRAM.
  const llm = new LlamaCpp({
    embedModel: configuredConfig?.models?.embed,
    embedOptions: configuredConfig?.models?.embedOptions,
    generateModel: configuredConfig?.models?.generate,
    rerankModel: configuredConfig?.models?.rerank,
    rerankOptions: configuredConfig?.models?.rerankOptions,
    inactivityTimeoutMs: 5 * 60 * 1000,
    disposeModelsOnInactivity: true,
  });
  const internal = createStoreInternal(options.dbPath);
  const db = internal.db;

  // Seed SQLite from the configuration that is authoritative now, read under
  // the config lock. A lock-free read and a later sync would delete a
  // collection another process published in between, and stamp the stale
  // document's hash over the result. An inline snapshot is process-local: it
  // has no file to serialize on and is applied as given.
  configScope?.withLockedConfig((authoritative) => syncConfigToDb(db, authoritative));
  internal.llm = llm;

  // Every operation this store exposes is admitted here before it touches the
  // database or the LLM, so close() can drain them instead of racing them.
  const lifecycle = new StoreOperationLifecycle();
  const admit = <T>(operation: string, fn: () => Promise<T>): Promise<T> => lifecycle.run(operation, fn);

  // Mirror half of a configuration mutation. It runs after this store's config
  // scope published the change and while that scope still holds the config
  // lock, so the configuration stays authoritative and no concurrent sync can
  // interleave between the publish and the mirror. A direct row write also
  // invalidates the stored sync hash: the rows and the document the hash
  // certifies must not disagree.
  const mirrorToDb = (apply: () => boolean | void): ConfigMirror => () => {
    const changed = apply() === true;
    invalidateConfigHash(db);
    return changed;
  };

  // Teardown runs only after the drain. It attempts every step so one failure
  // cannot leak the others: native models and contexts first, then the SQLite
  // connection, then this store's config target.
  const teardown = async (): Promise<void> => {
    const failures: unknown[] = [];
    try {
      await llm.dispose();
    } catch (error) {
      failures.push(error);
    }
    try {
      internal.close();
    } catch (error) {
      failures.push(error);
    }
    // Release this store's config target even if resources above could not be
    // released. Later mutations fail with ConfigScopeClosedError instead of
    // writing to another store's config.
    configScope?.close();
    if (failures.length === 1) throw failures[0];
    if (failures.length > 1) throw new AggregateError(failures, "QMD store teardown failed");
  };

  const store: QMDStore = {
    internal,
    dbPath: internal.dbPath,

    // Search
    search: (opts) => admit("search", async () => {
      opts.signal?.throwIfAborted();
      if (!opts.query && !opts.queries) {
        throw new Error("search() requires either 'query' or 'queries'");
      }
      // Normalize collection/collections
      const collections = [
        ...(opts.collection ? [opts.collection] : []),
        ...(opts.collections ?? []),
      ];
      const skipRerank = opts.rerank === false;
      // The SDK is also a JavaScript boundary: TypeScript declarations do not
      // protect plain-JS callers or deserialized input. Apply the same bounded,
      // strict validation used by CLI, MCP, and HTTP before compiling SQL.
      const filter = opts.filter === undefined ? undefined : parseMetadataFilter(opts.filter);

      if (opts.queries) {
        // Pre-expanded queries — use structuredSearch
        return structuredSearch(internal, opts.queries, {
          signal: opts.signal,
          collections: collections.length > 0 ? collections : undefined,
          filter,
          limit: opts.limit,
          minScore: opts.minScore,
          explain: opts.explain,
          intent: opts.intent,
          candidateLimit: opts.candidateLimit,
          skipRerank,
          chunkStrategy: opts.chunkStrategy,
        });
      }

      // Simple query string — use hybridQuery (expand + search + rerank)
      return hybridQuery(internal, opts.query!, {
        signal: opts.signal,
        collection: collections.length > 0 ? collections : undefined,
        filter,
        limit: opts.limit,
        minScore: opts.minScore,
        explain: opts.explain,
        intent: opts.intent,
        candidateLimit: opts.candidateLimit,
        skipRerank,
        chunkStrategy: opts.chunkStrategy,
      });
    }),
    searchLex: (q, opts) => admit("searchLex", async () => {
      const filter = opts?.filter === undefined ? undefined : parseMetadataFilter(opts.filter);
      return internal.searchFTS(q, opts?.limit, opts?.collection, filter);
    }),
    searchVector: (q, opts) => admit("searchVector", async () => {
      const filter = opts?.filter === undefined ? undefined : parseMetadataFilter(opts.filter);
      return withLLMSessionForLlm(llm, session =>
        internal.searchVec(q, llm.embedModelName, opts?.limit, opts?.collection, session, undefined, filter),
        { signal: opts?.signal, name: "searchVector" });
    }),
    expandQuery: (q) => admit("expandQuery", async () => internal.expandQuery(q)),
    findSimilarByEmbedding: (pathOrDocid, limit, collection) => admit("findSimilarByEmbedding", async () =>
      internal.findSimilarByEmbedding(pathOrDocid, limit, collection)),
    get: (pathOrDocid, opts) => admit("get", async () => internal.findDocument(pathOrDocid, opts)),
    getDocumentBody: (pathOrDocid, opts) => admit("getDocumentBody", async () => {
      const result = internal.findDocument(pathOrDocid, { includeBody: false });
      if ("error" in result) return null;
      return internal.getDocumentBody(result, opts?.fromLine, opts?.maxLines);
    }),
    multiGet: (pattern, opts) => admit("multiGet", async () => internal.findDocuments(pattern, opts)),

    // Collection Management — the bound configuration is authoritative: each
    // mutation publishes it first and mirrors the result into SQLite under the
    // same config lock. The old order (SQLite first) left a database-only edit
    // behind whenever the config write failed.
    addCollection: (name, opts) => admit("addCollection", async () => {
      const upsert = (): void => {
        db.transaction(() => {
          upsertStoreCollection(db, name, { path: opts.path, pattern: opts.pattern, ignore: opts.ignore });
        }).immediate();
      };
      // Write-through goes to this store's config scope only.
      if (configScope) configScope.addCollection(name, opts.path, opts.pattern, mirrorToDb(upsert));
      else upsert();
    }),
    removeCollection: (name) => admit("removeCollection", async () => {
      const removeFromDb = (): boolean => {
        // Full removal whenever the index still holds anything under this name:
        // a config sync may already have dropped the row from an already
        // published file while documents, sync rows, processor state or a
        // vector partition remain from an interrupted removal.
        if (!hasCollectionIndexState(db, name)) return deleteStoreCollection(db, name);
        removeIndexedCollection(db, name);
        return true;
      };
      if (!configScope) return removeFromDb();
      return configScope.removeCollection(name, mirrorToDb(removeFromDb));
    }),
    renameCollection: (oldName, newName) => admit("renameCollection", async () => {
      if (!configScope) {
        const processed = hasProcessorMode(db, oldName) || !!getStoreCollection(db, oldName)?.processor;
        if (!processed) return renameStoreCollection(db, oldName, newName);
        renameIndexedCollection(db, oldName, newName);
        return true;
      }

      // An ordinary index-side refusal happens before the file is published, so
      // it cannot leave the configuration renamed ahead of the index. A target
      // row a sync wrote from an already published file has no documents and is
      // finished by the mirror instead.
      assertRenameTargetAvailable(db, newName);
      return configScope.renameCollection(oldName, newName, mirrorToDb(() =>
        reconcilePublishedRename(db, oldName, newName)));
    }),
    listCollections: () => admit("listCollections", async () => storeListCollections(db)),
    listMetadata: (opts) => admit("listMetadata", async () => storeListMetadata(db, {
      ...opts,
      match: opts?.match === undefined ? undefined : parseMetadataMatch(opts.match),
      filter: opts?.filter === undefined ? undefined : parseMetadataFilter(opts.filter),
    })),
    getDefaultCollectionNames: () => admit("getDefaultCollectionNames", async () => {
      const collections = storeListCollections(db);
      return collections.filter(c => c.includeByDefault).map(c => c.name);
    }),

    // Context Management — same order as collections: the configuration is
    // authoritative, SQLite mirrors it under the config lock.
    addContext: (collectionName, pathPrefix, contextText) => admit("addContext", async () => {
      if (!configScope) return updateStoreContext(db, collectionName, pathPrefix, contextText);
      return configScope.addContext(collectionName, pathPrefix, contextText, mirrorToDb(() =>
        updateStoreContext(db, collectionName, pathPrefix, contextText)));
    }),
    removeContext: (collectionName, pathPrefix) => admit("removeContext", async () => {
      if (!configScope) return removeStoreContext(db, collectionName, pathPrefix);
      return configScope.removeContext(collectionName, pathPrefix, mirrorToDb(() =>
        removeStoreContext(db, collectionName, pathPrefix)));
    }),
    setGlobalContext: (context) => admit("setGlobalContext", async () => {
      const apply = (): void => setStoreGlobalContext(db, context);
      if (configScope) configScope.setGlobalContext(context, mirrorToDb(apply));
      else apply();
    }),
    getGlobalContext: () => admit("getGlobalContext", async () => getStoreGlobalContext(db)),
    listContexts: () => admit("listContexts", async () => getStoreContexts(db)),

    // Indexing — reads collections from SQLite
    update: (updateOpts) => admit("update", async () => {
      const collections = getStoreCollections(db);
      const filtered = updateOpts?.collections
        ? collections.filter(c => updateOpts.collections!.includes(c.name))
        : collections;

      const prepared = filtered.map(col => {
        const registered = updateOpts?.processors?.[col.name];
        const processor = registered ? snapshotProcessor(registered) : undefined;
        requireProcessor(db, col.name, !!col.processor, processor);
        return { col, processor };
      });
      internal.clearCache();

      let totalIndexed = 0, totalUpdated = 0, totalUnchanged = 0, totalRemoved = 0, totalSkipped = 0;

      for (const { col, processor } of prepared) {
        const result = await reindexCollection(internal, col.path, col.pattern || "**/*.md", col.name, {
          ignorePatterns: col.ignore,
          processor,
          signal: updateOpts?.signal,
          onProgress: updateOpts?.onProgress
            ? (info) => updateOpts.onProgress!({ collection: col.name, ...info })
            : undefined,
        });
        totalIndexed += result.indexed;
        totalUpdated += result.updated;
        totalUnchanged += result.unchanged;
        totalRemoved += result.removed;
        totalSkipped += result.skipped;
      }

      // A changed file rewrites its document's hash in place and strands the
      // old hash's partition rows, which take k slots from scoped searches;
      // a hash that joined a collection while embedded elsewhere stays
      // unsearchable there until its rows are copied. The copy runs first:
      // the cleanup deletes the partition rows it copies from, so a document
      // moved between collections would otherwise need a fresh embed.
      const vectorsCopied = copyVectorsToNewCollections(db).copied;
      const staleVectorsRemoved = cleanupOrphanedVectors(db);

      return {
        collections: filtered.length,
        indexed: totalIndexed,
        updated: totalUpdated,
        unchanged: totalUnchanged,
        removed: totalRemoved,
        skipped: totalSkipped,
        staleVectorsRemoved,
        vectorsCopied,
        needsEmbedding: internal.getHashesNeedingEmbedding(),
      };
    }),

    embed: (embedOpts) => admit("embed", async () => generateEmbeddings(internal, {
      signal: embedOpts?.signal,
      force: embedOpts?.force,
      model: embedOpts?.model,
      collection: embedOpts?.collection,
      maxDocsPerBatch: embedOpts?.maxDocsPerBatch,
      maxBatchBytes: embedOpts?.maxBatchBytes,
      chunkStrategy: embedOpts?.chunkStrategy,
      onProgress: embedOpts?.onProgress,
    })),

    // Index Health
    getStatus: () => admit("getStatus", async () => internal.getStatus()),
    getIndexHealth: () => admit("getIndexHealth", async () => internal.getIndexHealth()),

    // Lifecycle
    close: () => lifecycle.close(teardown),
  };

  return store;
}
