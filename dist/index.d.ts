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
import { extractSnippet, addLineNumbers, DEFAULT_MULTI_GET_MAX_BYTES, type Store as InternalStore, type DocumentResult, type DocumentNotFound, type SearchResult, type HybridQueryResult, type HybridQueryOptions, type HybridQueryExplain, type ExpandedQuery, type StructuredSearchOptions, type MultiGetResult, type IndexStatus, type IndexHealthInfo, type SearchHooks, type ReindexProgress, type ReindexResult, type EmbedProgress, type EmbedResult, type ChunkStrategy } from "./store.js";
import { type Collection, type CollectionConfig, type NamedCollection, type ContextMap } from "./collections.js";
export type { DocumentResult, DocumentNotFound, SearchResult, HybridQueryResult, HybridQueryOptions, HybridQueryExplain, ExpandedQuery, StructuredSearchOptions, MultiGetResult, IndexStatus, IndexHealthInfo, SearchHooks, ReindexProgress, ReindexResult, EmbedProgress, EmbedResult, Collection, CollectionConfig, NamedCollection, ContextMap, };
export type { InternalStore };
export { extractSnippet, addLineNumbers, DEFAULT_MULTI_GET_MAX_BYTES };
export type { ChunkStrategy } from "./store.js";
export { getDefaultDbPath } from "./store.js";
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
    needsEmbedding: number;
};
/**
 * Options for the unified search() method.
 */
export interface SearchOptions {
    /** Simple query string — will be auto-expanded via LLM */
    query?: string;
    /** Pre-expanded queries (from expandQuery) — skips auto-expansion */
    queries?: ExpandedQuery[];
    /** Domain intent hint — steers expansion and reranking */
    intent?: string;
    /** Rerank results using LLM (default: true) */
    rerank?: boolean;
    /** Filter to a specific collection */
    collection?: string;
    /** Filter to specific collections */
    collections?: string[];
    /** Max results (default: 10) */
    limit?: number;
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
    collection?: string;
}
/**
 * Options for searchVector() — vector similarity search.
 */
export interface VectorSearchOptions {
    limit?: number;
    collection?: string;
}
/**
 * Options for expandQuery() — manual query expansion.
 */
export interface ExpandQueryOptions {
    intent?: string;
}
/**
 * Options for creating a QMD store.
 *
 * Provide `dbPath` and optionally `configPath` (YAML file) or `config` (inline).
 * If neither configPath nor config is provided, the store reads from existing
 * DB state (useful for reopening a previously-configured store).
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
 *
 * All methods are async. The store manages its own LlamaCpp instance
 * (lazy-loaded, auto-unloaded after inactivity) — no global singletons.
 */
export interface QMDStore {
    /** The underlying internal store (for advanced use) */
    readonly internal: InternalStore;
    /** Path to the SQLite database */
    readonly dbPath: string;
    /** Full search: query expansion + multi-signal retrieval + LLM reranking */
    search(options: SearchOptions): Promise<HybridQueryResult[]>;
    /** BM25 keyword search (fast, no LLM) */
    searchLex(query: string, options?: LexSearchOptions): Promise<SearchResult[]>;
    /** Vector similarity search (embedding model, no reranking) */
    searchVector(query: string, options?: VectorSearchOptions): Promise<SearchResult[]>;
    /** Expand a query into typed sub-searches (lex/vec/hyde) for manual control */
    expandQuery(query: string, options?: ExpandQueryOptions): Promise<ExpandedQuery[]>;
    /** Get a single document by path or docid */
    get(pathOrDocid: string, options?: {
        includeBody?: boolean;
    }): Promise<DocumentResult | DocumentNotFound>;
    /** Get the body content of a document, optionally sliced by line range */
    getDocumentBody(pathOrDocid: string, opts?: {
        fromLine?: number;
        maxLines?: number;
    }): Promise<string | null>;
    /** Get multiple documents by glob pattern or comma-separated list */
    multiGet(pattern: string, options?: {
        includeBody?: boolean;
        maxBytes?: number;
    }): Promise<{
        docs: MultiGetResult[];
        errors: string[];
    }>;
    /** Add or update a collection */
    addCollection(name: string, opts: {
        path: string;
        pattern?: string;
        ignore?: string[];
    }): Promise<void>;
    /** Remove a collection */
    removeCollection(name: string): Promise<boolean>;
    /** Rename a collection */
    renameCollection(oldName: string, newName: string): Promise<boolean>;
    /** List all collections with document stats */
    listCollections(): Promise<{
        name: string;
        pwd: string;
        glob_pattern: string;
        doc_count: number;
        active_count: number;
        last_modified: string | null;
        includeByDefault: boolean;
    }[]>;
    /** Get names of collections included by default in queries */
    getDefaultCollectionNames(): Promise<string[]>;
    /** Add context for a path within a collection */
    addContext(collectionName: string, pathPrefix: string, contextText: string): Promise<boolean>;
    /** Remove context from a collection path */
    removeContext(collectionName: string, pathPrefix: string): Promise<boolean>;
    /** Set global context (applies to all collections) */
    setGlobalContext(context: string | undefined): Promise<void>;
    /** Get global context */
    getGlobalContext(): Promise<string | undefined>;
    /** List all contexts across all collections */
    listContexts(): Promise<Array<{
        collection: string;
        path: string;
        context: string;
    }>>;
    /** Re-index collections by scanning the filesystem */
    update(options?: {
        collections?: string[];
        onProgress?: (info: UpdateProgress) => void;
    }): Promise<UpdateResult>;
    /** Generate vector embeddings for documents that need them */
    embed(options?: {
        force?: boolean;
        model?: string;
        maxDocsPerBatch?: number;
        maxBatchBytes?: number;
        chunkStrategy?: ChunkStrategy;
        onProgress?: (info: EmbedProgress) => void;
    }): Promise<EmbedResult>;
    /** Get index status (document counts, collections, embedding state) */
    getStatus(): Promise<IndexStatus>;
    /** Get index health info (stale embeddings, etc.) */
    getIndexHealth(): Promise<IndexHealthInfo>;
    /** Close the store and release all resources (LLM models, DB connection) */
    close(): Promise<void>;
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
export declare function createStore(options: StoreOptions): Promise<QMDStore>;
