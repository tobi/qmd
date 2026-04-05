/**
 * Collections configuration management
 *
 * This module manages the YAML-based collection configuration at ~/.config/qmd/index.yml.
 * Collections define which directories to index and their associated contexts.
 */
/**
 * Context definitions for a collection
 * Key is path prefix (e.g., "/", "/2024", "/Board of Directors")
 * Value is the context description
 */
export type ContextMap = Record<string, string>;
/**
 * A single collection configuration
 */
export interface Collection {
    path: string;
    pattern: string;
    ignore?: string[];
    context?: ContextMap;
    update?: string;
    includeByDefault?: boolean;
}
/**
 * The complete configuration file structure
 */
export interface CollectionConfig {
    global_context?: string;
    collections: Record<string, Collection>;
}
/**
 * Collection with its name (for return values)
 */
export interface NamedCollection extends Collection {
    name: string;
}
/**
 * Set the config source for SDK mode.
 * - File path: load/save from a specific YAML file
 * - Inline config: use an in-memory CollectionConfig (saveConfig updates in place, no file I/O)
 * - undefined: reset to default file-based config
 */
export declare function setConfigSource(source?: {
    configPath?: string;
    config?: CollectionConfig;
}): void;
/**
 * Set the current index name for config file lookup
 * Config file will be ~/.config/qmd/{indexName}.yml
 */
export declare function setConfigIndexName(name: string): void;
/**
 * Load configuration from the configured source.
 * - Inline config: returns the in-memory object directly
 * - File-based: reads from YAML file (default ~/.config/qmd/index.yml)
 * Returns empty config if file doesn't exist
 */
export declare function loadConfig(): CollectionConfig;
/**
 * Save configuration to the configured source.
 * - Inline config: updates the in-memory object (no file I/O)
 * - File-based: writes to YAML file (default ~/.config/qmd/index.yml)
 */
export declare function saveConfig(config: CollectionConfig): void;
/**
 * Get a specific collection by name
 * Returns null if not found
 */
export declare function getCollection(name: string): NamedCollection | null;
/**
 * List all collections
 */
export declare function listCollections(): NamedCollection[];
/**
 * Get collections that are included by default in queries
 */
export declare function getDefaultCollections(): NamedCollection[];
/**
 * Get collection names that are included by default
 */
export declare function getDefaultCollectionNames(): string[];
/**
 * Update a collection's settings
 */
export declare function updateCollectionSettings(name: string, settings: {
    update?: string | null;
    includeByDefault?: boolean;
}): boolean;
/**
 * Add or update a collection
 */
export declare function addCollection(name: string, path: string, pattern?: string): void;
/**
 * Remove a collection
 */
export declare function removeCollection(name: string): boolean;
/**
 * Rename a collection
 */
export declare function renameCollection(oldName: string, newName: string): boolean;
/**
 * Get global context
 */
export declare function getGlobalContext(): string | undefined;
/**
 * Set global context
 */
export declare function setGlobalContext(context: string | undefined): void;
/**
 * Get all contexts for a collection
 */
export declare function getContexts(collectionName: string): ContextMap | undefined;
/**
 * Add or update a context for a specific path in a collection
 */
export declare function addContext(collectionName: string, pathPrefix: string, contextText: string): boolean;
/**
 * Remove a context from a collection
 */
export declare function removeContext(collectionName: string, pathPrefix: string): boolean;
/**
 * List all contexts across all collections
 */
export declare function listAllContexts(): Array<{
    collection: string;
    path: string;
    context: string;
}>;
/**
 * Find best matching context for a given collection and path
 * Returns the most specific matching context (longest path prefix match)
 */
export declare function findContextForPath(collectionName: string, filePath: string): string | undefined;
/**
 * Get the config file path (useful for error messages)
 */
export declare function getConfigPath(): string;
/**
 * Check if config file exists
 */
export declare function configExists(): boolean;
/**
 * Validate a collection name
 * Collection names must be valid and not contain special characters
 */
export declare function isValidCollectionName(name: string): boolean;
