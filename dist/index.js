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
import { createStore as createStoreInternal, hybridQuery, structuredSearch, extractSnippet, addLineNumbers, DEFAULT_EMBED_MODEL, DEFAULT_MULTI_GET_MAX_BYTES, reindexCollection, generateEmbeddings, listCollections as storeListCollections, syncConfigToDb, getStoreCollections, getStoreCollection, getStoreGlobalContext, getStoreContexts, upsertStoreCollection, deleteStoreCollection, renameStoreCollection, updateStoreContext, removeStoreContext, setStoreGlobalContext, vacuumDatabase, cleanupOrphanedContent, cleanupOrphanedVectors, deleteLLMCache, deleteInactiveDocuments, clearAllEmbeddings, } from "./store.js";
import { LlamaCpp, } from "./llm.js";
import { setConfigSource, loadConfig, addCollection as collectionsAddCollection, removeCollection as collectionsRemoveCollection, renameCollection as collectionsRenameCollection, addContext as collectionsAddContext, removeContext as collectionsRemoveContext, setGlobalContext as collectionsSetGlobalContext, } from "./collections.js";
// Re-export utility functions and types used by frontends
export { extractSnippet, addLineNumbers, DEFAULT_MULTI_GET_MAX_BYTES };
// Re-export getDefaultDbPath for CLI/MCP that need the default database location
export { getDefaultDbPath } from "./store.js";
// Re-export Maintenance class for CLI housekeeping operations
export { Maintenance } from "./maintenance.js";
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
export async function createStore(options) {
    if (!options.dbPath) {
        throw new Error("dbPath is required");
    }
    if (options.configPath && options.config) {
        throw new Error("Provide either configPath or config, not both");
    }
    // Create the internal store (opens DB, creates tables)
    const internal = createStoreInternal(options.dbPath);
    const db = internal.db;
    // Track whether we have a YAML config path for write-through
    const hasYamlConfig = !!options.configPath;
    // Sync config into SQLite store_collections
    if (options.configPath) {
        // YAML mode: inject config source for write-through, sync to DB
        setConfigSource({ configPath: options.configPath });
        const config = loadConfig();
        syncConfigToDb(db, config);
    }
    else if (options.config) {
        // Inline config mode: inject config source for mutations, sync to DB
        setConfigSource({ config: options.config });
        syncConfigToDb(db, options.config);
    }
    // else: DB-only mode — no external config, use existing store_collections
    // Create a per-store LlamaCpp instance — lazy-loads models on first use,
    // auto-unloads after inactivity to free VRAM.
    //
    // QMD_MODEL_KEEP_ALIVE=1  — disable inactivity timer entirely (models + contexts stay in RAM)
    // QMD_MODEL_TTL=<seconds> — override inactivity timeout (default: 300s / 5 min)
    const keepAlive = process.env.QMD_MODEL_KEEP_ALIVE === '1';
    const ttlSec = process.env.QMD_MODEL_TTL !== undefined
        ? Number(process.env.QMD_MODEL_TTL)
        : 300;
    const llm = new LlamaCpp({
        inactivityTimeoutMs: keepAlive ? 0 : ttlSec * 1000,
        disposeModelsOnInactivity: !keepAlive,
    });
    internal.llm = llm;
    const store = {
        internal,
        dbPath: internal.dbPath,
        // Search
        search: async (opts) => {
            if (!opts.query && !opts.queries) {
                throw new Error("search() requires either 'query' or 'queries'");
            }
            // Normalize collection/collections
            const collections = [
                ...(opts.collection ? [opts.collection] : []),
                ...(opts.collections ?? []),
            ];
            const skipRerank = opts.rerank === false;
            if (opts.queries) {
                // Pre-expanded queries — use structuredSearch
                return structuredSearch(internal, opts.queries, {
                    collections: collections.length > 0 ? collections : undefined,
                    limit: opts.limit,
                    minScore: opts.minScore,
                    explain: opts.explain,
                    intent: opts.intent,
                    skipRerank,
                    chunkStrategy: opts.chunkStrategy,
                });
            }
            // Simple query string — use hybridQuery (expand + search + rerank)
            return hybridQuery(internal, opts.query, {
                collection: collections[0],
                limit: opts.limit,
                minScore: opts.minScore,
                explain: opts.explain,
                intent: opts.intent,
                skipRerank,
                chunkStrategy: opts.chunkStrategy,
            });
        },
        searchLex: async (q, opts) => internal.searchFTS(q, opts?.limit, opts?.collection),
        searchVector: async (q, opts) => internal.searchVec(q, DEFAULT_EMBED_MODEL, opts?.limit, opts?.collection),
        expandQuery: async (q, opts) => internal.expandQuery(q, undefined, opts?.intent),
        get: async (pathOrDocid, opts) => internal.findDocument(pathOrDocid, opts),
        getDocumentBody: async (pathOrDocid, opts) => {
            const result = internal.findDocument(pathOrDocid, { includeBody: false });
            if ("error" in result)
                return null;
            return internal.getDocumentBody(result, opts?.fromLine, opts?.maxLines);
        },
        multiGet: async (pattern, opts) => internal.findDocuments(pattern, opts),
        // Collection Management — write to SQLite + write-through to YAML/inline if configured
        addCollection: async (name, opts) => {
            upsertStoreCollection(db, name, { path: opts.path, pattern: opts.pattern, ignore: opts.ignore });
            if (hasYamlConfig || options.config) {
                collectionsAddCollection(name, opts.path, opts.pattern);
            }
        },
        removeCollection: async (name) => {
            const result = deleteStoreCollection(db, name);
            if (hasYamlConfig || options.config) {
                collectionsRemoveCollection(name);
            }
            return result;
        },
        renameCollection: async (oldName, newName) => {
            const result = renameStoreCollection(db, oldName, newName);
            if (hasYamlConfig || options.config) {
                collectionsRenameCollection(oldName, newName);
            }
            return result;
        },
        listCollections: async () => storeListCollections(db),
        getDefaultCollectionNames: async () => {
            const collections = storeListCollections(db);
            return collections.filter(c => c.includeByDefault).map(c => c.name);
        },
        // Context Management — write to SQLite + write-through to YAML/inline if configured
        addContext: async (collectionName, pathPrefix, contextText) => {
            const result = updateStoreContext(db, collectionName, pathPrefix, contextText);
            if (hasYamlConfig || options.config) {
                collectionsAddContext(collectionName, pathPrefix, contextText);
            }
            return result;
        },
        removeContext: async (collectionName, pathPrefix) => {
            const result = removeStoreContext(db, collectionName, pathPrefix);
            if (hasYamlConfig || options.config) {
                collectionsRemoveContext(collectionName, pathPrefix);
            }
            return result;
        },
        setGlobalContext: async (context) => {
            setStoreGlobalContext(db, context);
            if (hasYamlConfig || options.config) {
                collectionsSetGlobalContext(context);
            }
        },
        getGlobalContext: async () => getStoreGlobalContext(db),
        listContexts: async () => getStoreContexts(db),
        // Indexing — reads collections from SQLite
        update: async (updateOpts) => {
            const collections = getStoreCollections(db);
            const filtered = updateOpts?.collections
                ? collections.filter(c => updateOpts.collections.includes(c.name))
                : collections;
            internal.clearCache();
            let totalIndexed = 0, totalUpdated = 0, totalUnchanged = 0, totalRemoved = 0;
            for (const col of filtered) {
                const result = await reindexCollection(internal, col.path, col.pattern || "**/*.md", col.name, {
                    ignorePatterns: col.ignore,
                    onProgress: updateOpts?.onProgress
                        ? (info) => updateOpts.onProgress({ collection: col.name, ...info })
                        : undefined,
                });
                totalIndexed += result.indexed;
                totalUpdated += result.updated;
                totalUnchanged += result.unchanged;
                totalRemoved += result.removed;
            }
            return {
                collections: filtered.length,
                indexed: totalIndexed,
                updated: totalUpdated,
                unchanged: totalUnchanged,
                removed: totalRemoved,
                needsEmbedding: internal.getHashesNeedingEmbedding(),
            };
        },
        embed: async (embedOpts) => {
            return generateEmbeddings(internal, {
                force: embedOpts?.force,
                model: embedOpts?.model,
                maxDocsPerBatch: embedOpts?.maxDocsPerBatch,
                maxBatchBytes: embedOpts?.maxBatchBytes,
                chunkStrategy: embedOpts?.chunkStrategy,
                onProgress: embedOpts?.onProgress,
            });
        },
        // Index Health
        getStatus: async () => internal.getStatus(),
        getIndexHealth: async () => internal.getIndexHealth(),
        // Lifecycle
        close: async () => {
            await llm.dispose();
            internal.close();
            if (hasYamlConfig || options.config) {
                setConfigSource(undefined); // Reset config source
            }
        },
    };
    return store;
}
