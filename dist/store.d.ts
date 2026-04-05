/**
 * QMD Store - Core data access and retrieval functions
 *
 * This module provides all database operations, search functions, and document
 * retrieval for QMD. It returns raw data structures that can be formatted by
 * CLI or MCP consumers.
 *
 * Usage:
 *   const store = createStore("/path/to/db.sqlite");
 *   // or use default path:
 *   const store = createStore();
 */
import type { Database } from "./db.js";
import { LlamaCpp, formatQueryForEmbedding, formatDocForEmbedding, type ILLMSession } from "./llm.js";
import type { NamedCollection, Collection, CollectionConfig } from "./collections.js";
export declare const DEFAULT_EMBED_MODEL: string;
export declare const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
export declare const DEFAULT_QUERY_MODEL = "Qwen/Qwen3-1.7B";
export declare const DEFAULT_GLOB = "**/*.md";
export declare const DEFAULT_MULTI_GET_MAX_BYTES: number;
export declare const DEFAULT_EMBED_MAX_DOCS_PER_BATCH = 64;
export declare const DEFAULT_EMBED_MAX_BATCH_BYTES: number;
export declare const CHUNK_SIZE_TOKENS = 900;
export declare const CHUNK_OVERLAP_TOKENS: number;
export declare const CHUNK_SIZE_CHARS: number;
export declare const CHUNK_OVERLAP_CHARS: number;
export declare const CHUNK_WINDOW_TOKENS = 200;
export declare const CHUNK_WINDOW_CHARS: number;
/**
 * A potential break point in the document with a base score indicating quality.
 */
export interface BreakPoint {
    pos: number;
    score: number;
    type: string;
}
/**
 * A region where a code fence exists (between ``` markers).
 * We should never split inside a code fence.
 */
export interface CodeFenceRegion {
    start: number;
    end: number;
}
/**
 * Patterns for detecting break points in markdown documents.
 * Higher scores indicate better places to split.
 * Scores are spread wide so headings decisively beat lower-quality breaks.
 * Order matters for scoring - more specific patterns first.
 */
export declare const BREAK_PATTERNS: [RegExp, number, string][];
/**
 * Scan text for all potential break points.
 * Returns sorted array of break points with higher-scoring patterns taking precedence
 * when multiple patterns match the same position.
 */
export declare function scanBreakPoints(text: string): BreakPoint[];
/**
 * Find all code fence regions in the text.
 * Code fences are delimited by ``` and we should never split inside them.
 */
export declare function findCodeFences(text: string): CodeFenceRegion[];
/**
 * Check if a position is inside a code fence region.
 * Uses binary search since fences are sorted by start position.
 */
export declare function isInsideCodeFence(pos: number, fences: CodeFenceRegion[]): boolean;
/**
 * Find the best cut position using scored break points with distance decay.
 *
 * Uses squared distance for gentler early decay - headings far back still win
 * over low-quality breaks near the target.
 *
 * @param breakPoints - Pre-scanned break points from scanBreakPoints()
 * @param targetCharPos - The ideal cut position (e.g., maxChars boundary)
 * @param windowChars - How far back to search for break points (default ~200 tokens)
 * @param decayFactor - How much to penalize distance (0.7 = 30% score at window edge)
 * @param codeFences - Code fence regions to avoid splitting inside
 * @returns The best position to cut at
 */
export declare function findBestCutoff(breakPoints: BreakPoint[], targetCharPos: number, windowChars?: number, decayFactor?: number, codeFences?: CodeFenceRegion[]): number;
export type ChunkStrategy = "auto" | "regex";
/**
 * Merge two sets of break points (e.g. regex + AST), keeping the highest
 * score at each position. Result is sorted by position.
 */
export declare function mergeBreakPoints(a: BreakPoint[], b: BreakPoint[]): BreakPoint[];
/**
 * Core chunk algorithm that operates on precomputed break points and code fences.
 * This is the shared implementation used by both regex-only and AST-aware chunking.
 */
export declare function chunkDocumentWithBreakPoints(content: string, breakPoints: BreakPoint[], codeFences: CodeFenceRegion[], maxChars?: number, overlapChars?: number, windowChars?: number): {
    text: string;
    pos: number;
}[];
export declare const STRONG_SIGNAL_MIN_SCORE = 0.85;
export declare const STRONG_SIGNAL_MIN_GAP = 0.15;
export declare const RERANK_CANDIDATE_LIMIT = 40;
/**
 * A typed query expansion result. Decoupled from llm.ts internal Queryable —
 * same shape, but store.ts owns its own public API type.
 *
 * - lex: keyword variant → routes to FTS only
 * - vec: semantic variant → routes to vector only
 * - hyde: hypothetical document → routes to vector only
 */
export type ExpandedQuery = {
    type: 'lex' | 'vec' | 'hyde';
    query: string;
    /** Optional line number for error reporting (CLI parser) */
    line?: number;
};
export declare function homedir(): string;
/**
 * Check if a path is absolute.
 * Supports:
 * - Unix paths: /path/to/file
 * - Windows native: C:\path or C:/path
 * - Git Bash: /c/path or /C/path (C-Z drives, excluding A/B floppy drives)
 *
 * Note: /c without trailing slash is treated as Unix path (directory named "c"),
 * while /c/ or /c/path are treated as Git Bash paths (C: drive).
 */
export declare function isAbsolutePath(path: string): boolean;
/**
 * Normalize path separators to forward slashes.
 * Converts Windows backslashes to forward slashes.
 */
export declare function normalizePathSeparators(path: string): string;
/**
 * Get the relative path from a prefix.
 * Returns null if path is not under prefix.
 * Returns empty string if path equals prefix.
 */
export declare function getRelativePathFromPrefix(path: string, prefix: string): string | null;
export declare function resolve(...paths: string[]): string;
export declare function enableProductionMode(): void;
export declare function getDefaultDbPath(indexName?: string): string;
export declare function getPwd(): string;
export declare function getRealPath(path: string): string;
export type VirtualPath = {
    collectionName: string;
    path: string;
};
/**
 * Normalize explicit virtual path formats to standard qmd:// format.
 * Only handles paths that are already explicitly virtual:
 * - qmd://collection/path.md (already normalized)
 * - qmd:////collection/path.md (extra slashes - normalize)
 * - //collection/path.md (missing qmd: prefix - add it)
 *
 * Does NOT handle:
 * - collection/path.md (bare paths - could be filesystem relative)
 * - :linenum suffix (should be parsed separately before calling this)
 */
export declare function normalizeVirtualPath(input: string): string;
/**
 * Parse a virtual path like "qmd://collection-name/path/to/file.md"
 * into its components.
 * Also supports collection root: "qmd://collection-name/" or "qmd://collection-name"
 */
export declare function parseVirtualPath(virtualPath: string): VirtualPath | null;
/**
 * Build a virtual path from collection name and relative path.
 */
export declare function buildVirtualPath(collectionName: string, path: string): string;
/**
 * Check if a path is explicitly a virtual path.
 * Only recognizes explicit virtual path formats:
 * - qmd://collection/path.md
 * - //collection/path.md
 *
 * Does NOT consider bare collection/path.md as virtual - that should be
 * handled separately by checking if the first component is a collection name.
 */
export declare function isVirtualPath(path: string): boolean;
/**
 * Resolve a virtual path to absolute filesystem path.
 */
export declare function resolveVirtualPath(db: Database, virtualPath: string): string | null;
/**
 * Convert an absolute filesystem path to a virtual path.
 * Returns null if the file is not in any indexed collection.
 */
export declare function toVirtualPath(db: Database, absolutePath: string): string | null;
export declare function verifySqliteVecLoaded(db: Database): void;
export declare function getStoreCollections(db: Database): NamedCollection[];
export declare function getStoreCollection(db: Database, name: string): NamedCollection | null;
export declare function getStoreGlobalContext(db: Database): string | undefined;
export declare function getStoreContexts(db: Database): Array<{
    collection: string;
    path: string;
    context: string;
}>;
export declare function upsertStoreCollection(db: Database, name: string, collection: Omit<Collection, 'pattern'> & {
    pattern?: string;
}): void;
export declare function deleteStoreCollection(db: Database, name: string): boolean;
export declare function renameStoreCollection(db: Database, oldName: string, newName: string): boolean;
export declare function updateStoreContext(db: Database, collectionName: string, path: string, text: string): boolean;
export declare function removeStoreContext(db: Database, collectionName: string, path: string): boolean;
export declare function setStoreGlobalContext(db: Database, value: string | undefined): void;
/**
 * Sync external config (YAML/inline) into SQLite store_collections.
 * External config always wins. Skips sync if config hash hasn't changed.
 */
export declare function syncConfigToDb(db: Database, config: CollectionConfig): void;
export declare function isSqliteVecAvailable(): boolean;
export type Store = {
    db: Database;
    dbPath: string;
    /** Optional LlamaCpp instance for this store (overrides the global singleton) */
    llm?: LlamaCpp;
    close: () => void;
    ensureVecTable: (dimensions: number) => void;
    getHashesNeedingEmbedding: () => number;
    getIndexHealth: () => IndexHealthInfo;
    getStatus: () => IndexStatus;
    getCacheKey: typeof getCacheKey;
    getCachedResult: (cacheKey: string) => string | null;
    setCachedResult: (cacheKey: string, result: string) => void;
    clearCache: () => void;
    deleteLLMCache: () => number;
    deleteInactiveDocuments: () => number;
    cleanupOrphanedContent: () => number;
    cleanupOrphanedVectors: () => number;
    vacuumDatabase: () => void;
    getContextForFile: (filepath: string) => string | null;
    getContextForPath: (collectionName: string, path: string) => string | null;
    getCollectionByName: (name: string) => {
        name: string;
        pwd: string;
        glob_pattern: string;
    } | null;
    getCollectionsWithoutContext: () => {
        name: string;
        pwd: string;
        doc_count: number;
    }[];
    getTopLevelPathsWithoutContext: (collectionName: string) => string[];
    parseVirtualPath: typeof parseVirtualPath;
    buildVirtualPath: typeof buildVirtualPath;
    isVirtualPath: typeof isVirtualPath;
    resolveVirtualPath: (virtualPath: string) => string | null;
    toVirtualPath: (absolutePath: string) => string | null;
    searchFTS: (query: string, limit?: number, collectionName?: string) => SearchResult[];
    searchVec: (query: string, model: string, limit?: number, collectionName?: string, session?: ILLMSession, precomputedEmbedding?: number[]) => Promise<SearchResult[]>;
    expandQuery: (query: string, model?: string, intent?: string) => Promise<ExpandedQuery[]>;
    rerank: (query: string, documents: {
        file: string;
        text: string;
    }[], model?: string, intent?: string) => Promise<{
        file: string;
        score: number;
    }[]>;
    findDocument: (filename: string, options?: {
        includeBody?: boolean;
    }) => DocumentResult | DocumentNotFound;
    getDocumentBody: (doc: DocumentResult | {
        filepath: string;
    }, fromLine?: number, maxLines?: number) => string | null;
    findDocuments: (pattern: string, options?: {
        includeBody?: boolean;
        maxBytes?: number;
    }) => {
        docs: MultiGetResult[];
        errors: string[];
    };
    findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => string[];
    matchFilesByGlob: (pattern: string) => {
        filepath: string;
        displayPath: string;
        bodyLength: number;
    }[];
    findDocumentByDocid: (docid: string) => {
        filepath: string;
        hash: string;
    } | null;
    insertContent: (hash: string, content: string, createdAt: string) => void;
    insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => void;
    findActiveDocument: (collectionName: string, path: string) => {
        id: number;
        hash: string;
        title: string;
    } | null;
    updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => void;
    updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => void;
    deactivateDocument: (collectionName: string, path: string) => void;
    getActiveDocumentPaths: (collectionName: string) => string[];
    getHashesForEmbedding: () => {
        hash: string;
        body: string;
        path: string;
    }[];
    clearAllEmbeddings: () => void;
    insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => void;
};
export type ReindexProgress = {
    file: string;
    current: number;
    total: number;
};
export type ReindexResult = {
    indexed: number;
    updated: number;
    unchanged: number;
    removed: number;
    orphanedCleaned: number;
};
/**
 * Re-index a single collection by scanning the filesystem and updating the database.
 * Pure function — no console output, no db lifecycle management.
 */
export declare function reindexCollection(store: Store, collectionPath: string, globPattern: string, collectionName: string, options?: {
    ignorePatterns?: string[];
    onProgress?: (info: ReindexProgress) => void;
}): Promise<ReindexResult>;
export type EmbedProgress = {
    chunksEmbedded: number;
    totalChunks: number;
    bytesProcessed: number;
    totalBytes: number;
    errors: number;
};
export type EmbedResult = {
    docsProcessed: number;
    chunksEmbedded: number;
    errors: number;
    durationMs: number;
};
export type EmbedOptions = {
    force?: boolean;
    model?: string;
    maxDocsPerBatch?: number;
    maxBatchBytes?: number;
    chunkStrategy?: ChunkStrategy;
    onProgress?: (info: EmbedProgress) => void;
};
/**
 * Generate vector embeddings for documents that need them.
 * Pure function — no console output, no db lifecycle management.
 * Uses the store's LlamaCpp instance if set, otherwise the global singleton.
 */
export declare function generateEmbeddings(store: Store, options?: EmbedOptions): Promise<EmbedResult>;
/**
 * Create a new store instance with the given database path.
 * If no path is provided, uses the default path (~/.cache/qmd/index.sqlite).
 *
 * @param dbPath - Path to the SQLite database file
 * @returns Store instance with all methods bound to the database
 */
export declare function createStore(dbPath?: string): Store;
/**
 * Unified document result type with all metadata.
 * Body is optional - use getDocumentBody() to load it separately if needed.
 */
export type DocumentResult = {
    filepath: string;
    displayPath: string;
    title: string;
    context: string | null;
    hash: string;
    docid: string;
    collectionName: string;
    modifiedAt: string;
    bodyLength: number;
    body?: string;
};
/**
 * Extract short docid from a full hash (first 6 characters).
 */
export declare function getDocid(hash: string): string;
export declare function handelize(path: string): string;
/**
 * Search result extends DocumentResult with score and source info
 */
export type SearchResult = DocumentResult & {
    score: number;
    source: "fts" | "vec";
    chunkPos?: number;
};
/**
 * Ranked result for RRF fusion (simplified, used internally)
 */
export type RankedResult = {
    file: string;
    displayPath: string;
    title: string;
    body: string;
    score: number;
};
export type RRFContributionTrace = {
    listIndex: number;
    source: "fts" | "vec";
    queryType: "original" | "lex" | "vec" | "hyde";
    query: string;
    rank: number;
    weight: number;
    backendScore: number;
    rrfContribution: number;
};
export type RRFScoreTrace = {
    contributions: RRFContributionTrace[];
    baseScore: number;
    topRank: number;
    topRankBonus: number;
    totalScore: number;
};
export type HybridQueryExplain = {
    ftsScores: number[];
    vectorScores: number[];
    rrf: {
        rank: number;
        positionScore: number;
        weight: number;
        baseScore: number;
        topRankBonus: number;
        totalScore: number;
        contributions: RRFContributionTrace[];
    };
    rerankScore: number;
    blendedScore: number;
};
/**
 * Error result when document is not found
 */
export type DocumentNotFound = {
    error: "not_found";
    query: string;
    similarFiles: string[];
};
/**
 * Result from multi-get operations
 */
export type MultiGetResult = {
    doc: DocumentResult;
    skipped: false;
} | {
    doc: Pick<DocumentResult, "filepath" | "displayPath">;
    skipped: true;
    skipReason: string;
};
export type CollectionInfo = {
    name: string;
    path: string | null;
    pattern: string | null;
    documents: number;
    lastUpdated: string;
};
export type IndexStatus = {
    totalDocuments: number;
    needsEmbedding: number;
    hasVectorIndex: boolean;
    collections: CollectionInfo[];
};
export declare function getHashesNeedingEmbedding(db: Database): number;
export type IndexHealthInfo = {
    needsEmbedding: number;
    totalDocs: number;
    daysStale: number | null;
};
export declare function getIndexHealth(db: Database): IndexHealthInfo;
export declare function getCacheKey(url: string, body: object): string;
export declare function getCachedResult(db: Database, cacheKey: string): string | null;
export declare function setCachedResult(db: Database, cacheKey: string, result: string): void;
export declare function clearCache(db: Database): void;
/**
 * Delete cached LLM API responses.
 * Returns the number of cached responses deleted.
 */
export declare function deleteLLMCache(db: Database): number;
/**
 * Remove inactive document records (active = 0).
 * Returns the number of inactive documents deleted.
 */
export declare function deleteInactiveDocuments(db: Database): number;
/**
 * Remove orphaned content hashes that are not referenced by any active document.
 * Returns the number of orphaned content hashes deleted.
 */
export declare function cleanupOrphanedContent(db: Database): number;
/**
 * Remove orphaned vector embeddings that are not referenced by any active document.
 * Returns the number of orphaned embedding chunks deleted.
 */
export declare function cleanupOrphanedVectors(db: Database): number;
/**
 * Run VACUUM to reclaim unused space in the database.
 * This operation rebuilds the database file to eliminate fragmentation.
 */
export declare function vacuumDatabase(db: Database): void;
export declare function hashContent(content: string): Promise<string>;
export declare function extractTitle(content: string, filename: string): string;
/**
 * Insert content into the content table (content-addressable storage).
 * Uses INSERT OR IGNORE so duplicate hashes are skipped.
 */
export declare function insertContent(db: Database, hash: string, content: string, createdAt: string): void;
/**
 * Insert a new document into the documents table.
 */
export declare function insertDocument(db: Database, collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string): void;
/**
 * Find an active document by collection name and path.
 */
export declare function findActiveDocument(db: Database, collectionName: string, path: string): {
    id: number;
    hash: string;
    title: string;
} | null;
/**
 * Update the title and modified_at timestamp for a document.
 */
export declare function updateDocumentTitle(db: Database, documentId: number, title: string, modifiedAt: string): void;
/**
 * Update an existing document's hash, title, and modified_at timestamp.
 * Used when content changes but the file path stays the same.
 */
export declare function updateDocument(db: Database, documentId: number, title: string, hash: string, modifiedAt: string): void;
/**
 * Deactivate a document (mark as inactive but don't delete).
 */
export declare function deactivateDocument(db: Database, collectionName: string, path: string): void;
/**
 * Get all active document paths for a collection.
 */
export declare function getActiveDocumentPaths(db: Database, collectionName: string): string[];
export { formatQueryForEmbedding, formatDocForEmbedding };
/**
 * Chunk a document using regex-only break point detection.
 * This is the sync, backward-compatible API used by tests and legacy callers.
 */
export declare function chunkDocument(content: string, maxChars?: number, overlapChars?: number, windowChars?: number): {
    text: string;
    pos: number;
}[];
/**
 * Async AST-aware chunking. Detects language from filepath, computes AST
 * break points for supported code files, merges with regex break points,
 * and delegates to the shared chunk algorithm.
 *
 * Falls back to regex-only when strategy is "regex", filepath is absent,
 * or language is unsupported.
 */
export declare function chunkDocumentAsync(content: string, maxChars?: number, overlapChars?: number, windowChars?: number, filepath?: string, chunkStrategy?: ChunkStrategy): Promise<{
    text: string;
    pos: number;
}[]>;
/**
 * Chunk a document by actual token count using the LLM tokenizer.
 * More accurate than character-based chunking but requires async.
 *
 * When filepath and chunkStrategy are provided, uses AST-aware break points
 * for supported code files.
 */
export declare function chunkDocumentByTokens(content: string, maxTokens?: number, overlapTokens?: number, windowTokens?: number, filepath?: string, chunkStrategy?: ChunkStrategy, signal?: AbortSignal): Promise<{
    text: string;
    pos: number;
    tokens: number;
}[]>;
/**
 * Normalize a docid input by stripping surrounding quotes and leading #.
 * Handles: "#abc123", 'abc123', "abc123", #abc123, abc123
 * Returns the bare hex string.
 */
export declare function normalizeDocid(docid: string): string;
/**
 * Check if a string looks like a docid reference.
 * Accepts: #abc123, abc123, "#abc123", "abc123", '#abc123', 'abc123'
 * Returns true if the normalized form is a valid hex string of 6+ chars.
 */
export declare function isDocid(input: string): boolean;
/**
 * Find a document by its short docid (first 6 characters of hash).
 * Returns the document's virtual path if found, null otherwise.
 * If multiple documents match the same short hash (collision), returns the first one.
 *
 * Accepts lenient input: #abc123, abc123, "#abc123", "abc123"
 */
export declare function findDocumentByDocid(db: Database, docid: string): {
    filepath: string;
    hash: string;
} | null;
export declare function findSimilarFiles(db: Database, query: string, maxDistance?: number, limit?: number): string[];
export declare function matchFilesByGlob(db: Database, pattern: string): {
    filepath: string;
    displayPath: string;
    bodyLength: number;
}[];
/**
 * Get context for a file path using hierarchical inheritance.
 * Contexts are collection-scoped and inherit from parent directories.
 * For example, context at "/talks" applies to "/talks/2024/keynote.md".
 *
 * @param db Database instance (unused - kept for compatibility)
 * @param collectionName Collection name
 * @param path Relative path within the collection
 * @returns Context string or null if no context is defined
 */
export declare function getContextForPath(db: Database, collectionName: string, path: string): string | null;
/**
 * Get context for a file path (virtual or filesystem).
 * Resolves the collection and relative path from the DB store_collections table.
 */
export declare function getContextForFile(db: Database, filepath: string): string | null;
/**
 * Get collection by name from DB store_collections table.
 */
export declare function getCollectionByName(db: Database, name: string): {
    name: string;
    pwd: string;
    glob_pattern: string;
} | null;
/**
 * List all collections with document counts from database.
 * Merges store_collections config with database statistics.
 */
export declare function listCollections(db: Database): {
    name: string;
    pwd: string;
    glob_pattern: string;
    doc_count: number;
    active_count: number;
    last_modified: string | null;
    includeByDefault: boolean;
}[];
/**
 * Remove a collection and clean up its documents.
 * Uses collections.ts to remove from YAML config and cleans up database.
 */
export declare function removeCollection(db: Database, collectionName: string): {
    deletedDocs: number;
    cleanedHashes: number;
};
/**
 * Rename a collection.
 * Updates both YAML config and database documents table.
 */
export declare function renameCollection(db: Database, oldName: string, newName: string): void;
/**
 * Insert or update a context for a specific collection and path prefix.
 */
export declare function insertContext(db: Database, collectionId: number, pathPrefix: string, context: string): void;
/**
 * Delete a context for a specific collection and path prefix.
 * Returns the number of contexts deleted.
 */
export declare function deleteContext(db: Database, collectionName: string, pathPrefix: string): number;
/**
 * Delete all global contexts (contexts with empty path_prefix).
 * Returns the number of contexts deleted.
 */
export declare function deleteGlobalContexts(db: Database): number;
/**
 * List all contexts, grouped by collection.
 * Returns contexts ordered by collection name, then by path prefix length (longest first).
 */
export declare function listPathContexts(db: Database): {
    collection_name: string;
    path_prefix: string;
    context: string;
}[];
/**
 * Get all collections (name only - from YAML config).
 */
export declare function getAllCollections(db: Database): {
    name: string;
}[];
/**
 * Check which collections don't have any context defined.
 * Returns collections that have no context entries at all (not even root context).
 */
export declare function getCollectionsWithoutContext(db: Database): {
    name: string;
    pwd: string;
    doc_count: number;
}[];
/**
 * Get top-level directories in a collection that don't have context.
 * Useful for suggesting where context might be needed.
 */
export declare function getTopLevelPathsWithoutContext(db: Database, collectionName: string): string[];
/**
 * Validate that a vec/hyde query doesn't use lex-only syntax.
 * Returns error message if invalid, null if valid.
 */
export declare function validateSemanticQuery(query: string): string | null;
export declare function validateLexQuery(query: string): string | null;
export declare function searchFTS(db: Database, query: string, limit?: number, collectionName?: string): SearchResult[];
export declare function searchVec(db: Database, query: string, model: string, limit?: number, collectionName?: string, session?: ILLMSession, precomputedEmbedding?: number[]): Promise<SearchResult[]>;
/**
 * Get all unique content hashes that need embeddings (from active documents).
 * Returns hash, document body, and a sample path for display purposes.
 */
export declare function getHashesForEmbedding(db: Database): {
    hash: string;
    body: string;
    path: string;
}[];
/**
 * Clear all embeddings from the database (force re-index).
 * Deletes all rows from content_vectors and drops the vectors_vec table.
 */
export declare function clearAllEmbeddings(db: Database): void;
/**
 * Insert a single embedding into both content_vectors and vectors_vec tables.
 * The hash_seq key is formatted as "hash_seq" for the vectors_vec table.
 *
 * content_vectors is inserted first so that getHashesForEmbedding (which checks
 * only content_vectors) won't re-select the hash on a crash between the two inserts.
 *
 * vectors_vec uses DELETE + INSERT instead of INSERT OR REPLACE because sqlite-vec's
 * vec0 virtual tables silently ignore the OR REPLACE conflict clause.
 */
export declare function insertEmbedding(db: Database, hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string): void;
export declare function expandQuery(query: string, model: string | undefined, db: Database, intent?: string, llmOverride?: LlamaCpp): Promise<ExpandedQuery[]>;
export declare function rerank(query: string, documents: {
    file: string;
    text: string;
}[], model: string | undefined, db: Database, intent?: string, llmOverride?: LlamaCpp): Promise<{
    file: string;
    score: number;
}[]>;
export declare function reciprocalRankFusion(resultLists: RankedResult[][], weights?: number[], k?: number): RankedResult[];
/**
 * Build per-document RRF contribution traces for explain/debug output.
 */
export declare function buildRrfTrace(resultLists: RankedResult[][], weights?: number[], listMeta?: RankedListMeta[], k?: number): Map<string, RRFScoreTrace>;
/**
 * Find a document by filename/path, docid (#hash), or with fuzzy matching.
 * Returns document metadata without body by default.
 *
 * Supports:
 * - Virtual paths: qmd://collection/path/to/file.md
 * - Absolute paths: /path/to/file.md
 * - Relative paths: path/to/file.md
 * - Short docid: #abc123 (first 6 chars of hash)
 */
export declare function findDocument(db: Database, filename: string, options?: {
    includeBody?: boolean;
}): DocumentResult | DocumentNotFound;
/**
 * Get the body content for a document
 * Optionally slice by line range
 */
export declare function getDocumentBody(db: Database, doc: DocumentResult | {
    filepath: string;
}, fromLine?: number, maxLines?: number): string | null;
/**
 * Find multiple documents by glob pattern or comma-separated list
 * Returns documents without body by default (use getDocumentBody to load)
 */
export declare function findDocuments(db: Database, pattern: string, options?: {
    includeBody?: boolean;
    maxBytes?: number;
}): {
    docs: MultiGetResult[];
    errors: string[];
};
export declare function getStatus(db: Database): IndexStatus;
export type SnippetResult = {
    line: number;
    snippet: string;
    linesBefore: number;
    linesAfter: number;
    snippetLines: number;
};
/** Weight for intent terms relative to query terms (1.0) in snippet scoring */
export declare const INTENT_WEIGHT_SNIPPET = 0.3;
/** Weight for intent terms relative to query terms (1.0) in chunk selection */
export declare const INTENT_WEIGHT_CHUNK = 0.5;
/**
 * Extract meaningful terms from an intent string, filtering stop words and punctuation.
 * Uses Unicode-aware punctuation stripping so domain terms like "API" survive.
 * Returns lowercase terms suitable for text matching.
 */
export declare function extractIntentTerms(intent: string): string[];
export declare function extractSnippet(body: string, query: string, maxLen?: number, chunkPos?: number, chunkLen?: number, intent?: string): SnippetResult;
/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 */
export declare function addLineNumbers(text: string, startLine?: number): string;
/**
 * Optional progress hooks for search orchestration.
 * CLI wires these to stderr for user feedback; MCP leaves them unset.
 */
export interface SearchHooks {
    /** BM25 probe found strong signal — expansion will be skipped */
    onStrongSignal?: (topScore: number) => void;
    /** Query expansion starting */
    onExpandStart?: () => void;
    /** Query expansion complete. Empty array = strong signal skip. elapsedMs = time taken. */
    onExpand?: (original: string, expanded: ExpandedQuery[], elapsedMs: number) => void;
    /** Embedding starting (vec/hyde queries) */
    onEmbedStart?: (count: number) => void;
    /** Embedding complete */
    onEmbedDone?: (elapsedMs: number) => void;
    /** Reranking is about to start */
    onRerankStart?: (chunkCount: number) => void;
    /** Reranking finished */
    onRerankDone?: (elapsedMs: number) => void;
}
export interface HybridQueryOptions {
    collection?: string;
    limit?: number;
    minScore?: number;
    candidateLimit?: number;
    explain?: boolean;
    intent?: string;
    skipRerank?: boolean;
    chunkStrategy?: ChunkStrategy;
    hooks?: SearchHooks;
}
export interface HybridQueryResult {
    file: string;
    displayPath: string;
    title: string;
    body: string;
    bestChunk: string;
    bestChunkPos: number;
    score: number;
    context: string | null;
    docid: string;
    explain?: HybridQueryExplain;
}
export type RankedListMeta = {
    source: "fts" | "vec";
    queryType: "original" | "lex" | "vec" | "hyde";
    query: string;
};
/**
 * Hybrid search: BM25 + vector + query expansion + RRF + chunked reranking.
 *
 * Pipeline:
 * 1. BM25 probe → skip expansion if strong signal
 * 2. expandQuery() → typed query variants (lex/vec/hyde)
 * 3. Type-routed search: original→vector, lex→FTS, vec/hyde→vector
 * 4. RRF fusion → slice to candidateLimit
 * 5. chunkDocument() + keyword-best-chunk selection
 * 6. rerank on chunks (NOT full bodies — O(tokens) trap)
 * 7. Position-aware score blending (RRF rank × reranker score)
 * 8. Dedup by file, filter by minScore, slice to limit
 */
export declare function hybridQuery(store: Store, query: string, options?: HybridQueryOptions): Promise<HybridQueryResult[]>;
export interface VectorSearchOptions {
    collection?: string;
    limit?: number;
    minScore?: number;
    intent?: string;
    hooks?: Pick<SearchHooks, 'onExpand'>;
}
export interface VectorSearchResult {
    file: string;
    displayPath: string;
    title: string;
    body: string;
    score: number;
    context: string | null;
    docid: string;
}
/**
 * Vector-only semantic search with query expansion.
 *
 * Pipeline:
 * 1. expandQuery() → typed variants, filter to vec/hyde only (lex irrelevant here)
 * 2. searchVec() for original + vec/hyde variants (sequential — node-llama-cpp embed limitation)
 * 3. Dedup by filepath (keep max score)
 * 4. Sort by score descending, filter by minScore, slice to limit
 */
export declare function vectorSearchQuery(store: Store, query: string, options?: VectorSearchOptions): Promise<VectorSearchResult[]>;
/**
 * A single sub-search in a structured search request.
 * Matches the format used in QMD training data.
 */
export interface StructuredSearchOptions {
    collections?: string[];
    limit?: number;
    minScore?: number;
    candidateLimit?: number;
    explain?: boolean;
    /** Domain intent hint for disambiguation — steers reranking and chunk selection */
    intent?: string;
    /** Skip LLM reranking, use only RRF scores */
    skipRerank?: boolean;
    chunkStrategy?: ChunkStrategy;
    hooks?: SearchHooks;
}
/**
 * Structured search: execute pre-expanded queries without LLM query expansion.
 *
 * Designed for LLM callers (MCP/HTTP) that generate their own query expansions.
 * Skips the internal expandQuery() step — goes directly to:
 *
 * Pipeline:
 * 1. Route searches: lex→FTS, vec/hyde→vector (batch embed)
 * 2. RRF fusion across all result lists
 * 3. Chunk documents + keyword-best-chunk selection
 * 4. Rerank on chunks
 * 5. Position-aware score blending
 * 6. Dedup, filter, slice
 *
 * This is the recommended endpoint for capable LLMs — they can generate
 * better query variations than our small local model, especially for
 * domain-specific or nuanced queries.
 */
export declare function structuredSearch(store: Store, searches: ExpandedQuery[], options?: StructuredSearchOptions): Promise<HybridQueryResult[]>;
