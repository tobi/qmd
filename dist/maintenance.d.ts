/**
 * Maintenance - Database cleanup operations for QMD.
 *
 * Wraps low-level store operations that the CLI needs for housekeeping.
 * Takes an internal Store in the constructor — allowed to access DB directly.
 */
import type { Store } from "./store.js";
export declare class Maintenance {
    private store;
    constructor(store: Store);
    /** Run VACUUM on the SQLite database to reclaim space */
    vacuum(): void;
    /** Remove content rows that are no longer referenced by any document */
    cleanupOrphanedContent(): number;
    /** Remove vector embeddings for content that no longer exists */
    cleanupOrphanedVectors(): number;
    /** Clear the LLM response cache (query expansion, reranking) */
    clearLLMCache(): number;
    /** Delete documents marked as inactive (removed from filesystem) */
    deleteInactiveDocs(): number;
    /** Clear all vector embeddings (forces re-embedding) */
    clearEmbeddings(): void;
}
