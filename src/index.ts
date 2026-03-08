/**
 * QMD SDK - Library mode for programmatic access to QMD search and indexing.
 *
 * Usage:
 *   import { createStore } from '@tobilu/qmd'
 *
 *   const store = createStore({
 *     dbPath: './my-index.sqlite',
 *     config: {
 *       collections: {
 *         docs: { path: '/path/to/docs', pattern: '**\/*.md' }
 *       }
 *     }
 *   })
 *
 *   const results = await store.query("how does auth work?")
 *   store.close()
 */

import {
  createStore as createStoreInternal,
  hybridQuery,
  structuredSearch,
  listCollections as storeListCollections,
  type Store as InternalStore,
  type DocumentResult,
  type DocumentNotFound,
  type SearchResult,
  type HybridQueryResult,
  type HybridQueryOptions,
  type HybridQueryExplain,
  type StructuredSubSearch,
  type StructuredSearchOptions,
  type MultiGetResult,
  type IndexStatus,
  type IndexHealthInfo,
  type ExpandedQuery,
  type SearchHooks,
} from "./store.js";
import {
  setConfigSource,
  loadConfig,
  addCollection as collectionsAddCollection,
  removeCollection as collectionsRemoveCollection,
  renameCollection as collectionsRenameCollection,
  listCollections as collectionsListCollections,
  addContext as collectionsAddContext,
  removeContext as collectionsRemoveContext,
  setGlobalContext as collectionsSetGlobalContext,
  getGlobalContext as collectionsGetGlobalContext,
  listAllContexts as collectionsListAllContexts,
  type Collection,
  type CollectionConfig,
  type NamedCollection,
  type ContextMap,
} from "./collections.js";

// Re-export types for SDK consumers
export type {
  DocumentResult,
  DocumentNotFound,
  SearchResult,
  HybridQueryResult,
  HybridQueryOptions,
  HybridQueryExplain,
  StructuredSubSearch,
  StructuredSearchOptions,
  MultiGetResult,
  IndexStatus,
  IndexHealthInfo,
  ExpandedQuery,
  SearchHooks,
  Collection,
  CollectionConfig,
  NamedCollection,
  ContextMap,
};

/**
 * Options for creating a QMD store.
 * You must provide `dbPath` and either `configPath` (YAML file) or `config` (inline).
 */
export interface StoreOptions {
  /** Path to the SQLite database file */
  dbPath: string;
  /** Path to a YAML config file (mutually exclusive with `config`) */
  configPath?: string;
  /** Inline collection config (mutually exclusive with `configPath`) */
  config?: CollectionConfig;
}

/**
 * The QMD SDK store — provides search, retrieval, collection management,
 * context management, and indexing operations.
 */
export interface QMDStore {
  /** The underlying internal store (for advanced use) */
  readonly internal: InternalStore;
  /** Path to the SQLite database */
  readonly dbPath: string;

  // ── Search & Retrieval ──────────────────────────────────────────────

  /** Hybrid search: BM25 + vector + query expansion + LLM reranking */
  query(query: string, options?: HybridQueryOptions): Promise<HybridQueryResult[]>;

  /** BM25 full-text keyword search (fast, no LLM) */
  search(query: string, options?: { limit?: number; collection?: string }): SearchResult[];

  /** Structured search with pre-expanded queries (for LLM callers) */
  structuredSearch(searches: StructuredSubSearch[], options?: StructuredSearchOptions): Promise<HybridQueryResult[]>;

  /** Get a single document by path or docid */
  get(pathOrDocid: string, options?: { includeBody?: boolean }): DocumentResult | DocumentNotFound;

  /** Get multiple documents by glob pattern or comma-separated list */
  multiGet(pattern: string, options?: { includeBody?: boolean; maxBytes?: number }): { docs: MultiGetResult[]; errors: string[] };

  // ── Collection Management ───────────────────────────────────────────

  /** Add or update a collection */
  addCollection(name: string, opts: { path: string; pattern?: string; ignore?: string[] }): void;

  /** Remove a collection */
  removeCollection(name: string): boolean;

  /** Rename a collection */
  renameCollection(oldName: string, newName: string): boolean;

  /** List all collections with document stats */
  listCollections(): { name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null }[];

  // ── Context Management ──────────────────────────────────────────────

  /** Add context for a path within a collection */
  addContext(collectionName: string, pathPrefix: string, contextText: string): boolean;

  /** Remove context from a collection path */
  removeContext(collectionName: string, pathPrefix: string): boolean;

  /** Set global context (applies to all collections) */
  setGlobalContext(context: string | undefined): void;

  /** Get global context */
  getGlobalContext(): string | undefined;

  /** List all contexts across all collections */
  listContexts(): Array<{ collection: string; path: string; context: string }>;

  // ── Index Health ────────────────────────────────────────────────────

  /** Get index status (document counts, collections, embedding state) */
  getStatus(): IndexStatus;

  /** Get index health info (stale embeddings, etc.) */
  getIndexHealth(): IndexHealthInfo;

  // ── Lifecycle ───────────────────────────────────────────────────────

  /** Close the database connection */
  close(): void;
}

/**
 * Create a QMD store for programmatic access to search and indexing.
 *
 * @example
 * ```typescript
 * // With a YAML config file
 * const store = createStore({
 *   dbPath: './index.sqlite',
 *   configPath: './qmd.yml',
 * })
 *
 * // With inline config (no files needed besides the DB)
 * const store = createStore({
 *   dbPath: './index.sqlite',
 *   config: {
 *     collections: {
 *       docs: { path: '/path/to/docs', pattern: '**\/*.md' }
 *     }
 *   }
 * })
 *
 * const results = await store.query("authentication flow")
 * store.close()
 * ```
 */
export function createStore(options: StoreOptions): QMDStore {
  if (!options.dbPath) {
    throw new Error("dbPath is required");
  }
  if (!options.configPath && !options.config) {
    throw new Error("Either configPath or config is required");
  }
  if (options.configPath && options.config) {
    throw new Error("Provide either configPath or config, not both");
  }

  // Inject config source into collections module
  setConfigSource({
    configPath: options.configPath,
    config: options.config,
  });

  // Create the internal store
  const internal = createStoreInternal(options.dbPath);

  const store: QMDStore = {
    internal,
    dbPath: internal.dbPath,

    // Search & Retrieval
    query: (q, opts) => hybridQuery(internal, q, opts),
    search: (q, opts) => internal.searchFTS(q, opts?.limit, opts?.collection),
    structuredSearch: (searches, opts) => structuredSearch(internal, searches, opts),
    get: (pathOrDocid, opts) => internal.findDocument(pathOrDocid, opts),
    multiGet: (pattern, opts) => internal.findDocuments(pattern, opts),

    // Collection Management
    addCollection: (name, opts) => {
      collectionsAddCollection(name, opts.path, opts.pattern);
    },
    removeCollection: (name) => collectionsRemoveCollection(name),
    renameCollection: (oldName, newName) => collectionsRenameCollection(oldName, newName),
    listCollections: () => storeListCollections(internal.db),

    // Context Management
    addContext: (collectionName, pathPrefix, contextText) =>
      collectionsAddContext(collectionName, pathPrefix, contextText),
    removeContext: (collectionName, pathPrefix) =>
      collectionsRemoveContext(collectionName, pathPrefix),
    setGlobalContext: (context) => collectionsSetGlobalContext(context),
    getGlobalContext: () => collectionsGetGlobalContext(),
    listContexts: () => collectionsListAllContexts(),

    // Index Health
    getStatus: () => internal.getStatus(),
    getIndexHealth: () => internal.getIndexHealth(),

    // Lifecycle
    close: () => {
      internal.close();
      setConfigSource(undefined); // Reset config source
    },
  };

  return store;
}
