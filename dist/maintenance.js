/**
 * Maintenance - Database cleanup operations for QMD.
 *
 * Wraps low-level store operations that the CLI needs for housekeeping.
 * Takes an internal Store in the constructor — allowed to access DB directly.
 */
import { vacuumDatabase, cleanupOrphanedContent, cleanupOrphanedVectors, deleteLLMCache, deleteInactiveDocuments, clearAllEmbeddings, } from "./store.js";
export class Maintenance {
    store;
    constructor(store) {
        this.store = store;
    }
    /** Run VACUUM on the SQLite database to reclaim space */
    vacuum() {
        vacuumDatabase(this.store.db);
    }
    /** Remove content rows that are no longer referenced by any document */
    cleanupOrphanedContent() {
        return cleanupOrphanedContent(this.store.db);
    }
    /** Remove vector embeddings for content that no longer exists */
    cleanupOrphanedVectors() {
        return cleanupOrphanedVectors(this.store.db);
    }
    /** Clear the LLM response cache (query expansion, reranking) */
    clearLLMCache() {
        return deleteLLMCache(this.store.db);
    }
    /** Delete documents marked as inactive (removed from filesystem) */
    deleteInactiveDocs() {
        return deleteInactiveDocuments(this.store.db);
    }
    /** Clear all vector embeddings (forces re-embedding) */
    clearEmbeddings() {
        clearAllEmbeddings(this.store.db);
    }
}
