/**
 * Collections configuration management
 *
 * This module manages the YAML-based collection configuration at ~/.config/qmd/index.yml.
 * Collections define which directories to index and their associated contexts.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";
import YAML from "yaml";
// ============================================================================
// Configuration paths
// ============================================================================
// Current index name (default: "index")
let currentIndexName = "index";
// SDK mode: optional in-memory config or custom config path
let configSource = { type: 'file' };
/**
 * Set the config source for SDK mode.
 * - File path: load/save from a specific YAML file
 * - Inline config: use an in-memory CollectionConfig (saveConfig updates in place, no file I/O)
 * - undefined: reset to default file-based config
 */
export function setConfigSource(source) {
    if (!source) {
        configSource = { type: 'file' };
        return;
    }
    if (source.config) {
        // Ensure collections object exists
        if (!source.config.collections) {
            source.config.collections = {};
        }
        configSource = { type: 'inline', config: source.config };
    }
    else if (source.configPath) {
        configSource = { type: 'file', path: source.configPath };
    }
    else {
        configSource = { type: 'file' };
    }
}
/**
 * Set the current index name for config file lookup
 * Config file will be ~/.config/qmd/{indexName}.yml
 */
export function setConfigIndexName(name) {
    // Resolve relative paths to absolute paths and sanitize for use as filename
    if (name.includes('/')) {
        const { resolve } = require('path');
        const { cwd } = require('process');
        const absolutePath = resolve(cwd(), name);
        // Replace path separators with underscores to create a valid filename
        currentIndexName = absolutePath.replace(/\//g, '_').replace(/^_/, '');
    }
    else {
        currentIndexName = name;
    }
}
function getConfigDir() {
    // Allow override via QMD_CONFIG_DIR for testing
    if (process.env.QMD_CONFIG_DIR) {
        return process.env.QMD_CONFIG_DIR;
    }
    // Respect XDG Base Directory specification (consistent with store.ts)
    if (process.env.XDG_CONFIG_HOME) {
        return join(process.env.XDG_CONFIG_HOME, "qmd");
    }
    return join(homedir(), ".config", "qmd");
}
function getConfigFilePath() {
    return join(getConfigDir(), `${currentIndexName}.yml`);
}
/**
 * Ensure config directory exists
 */
function ensureConfigDir() {
    const configDir = getConfigDir();
    if (!existsSync(configDir)) {
        mkdirSync(configDir, { recursive: true });
    }
}
// ============================================================================
// Core functions
// ============================================================================
/**
 * Load configuration from the configured source.
 * - Inline config: returns the in-memory object directly
 * - File-based: reads from YAML file (default ~/.config/qmd/index.yml)
 * Returns empty config if file doesn't exist
 */
export function loadConfig() {
    // SDK inline config mode
    if (configSource.type === 'inline') {
        return configSource.config;
    }
    // File-based config (SDK custom path or default)
    const configPath = configSource.path || getConfigFilePath();
    if (!existsSync(configPath)) {
        return { collections: {} };
    }
    try {
        const content = readFileSync(configPath, "utf-8");
        const config = YAML.parse(content);
        // Ensure collections object exists
        if (!config.collections) {
            config.collections = {};
        }
        return config;
    }
    catch (error) {
        throw new Error(`Failed to parse ${configPath}: ${error}`);
    }
}
/**
 * Save configuration to the configured source.
 * - Inline config: updates the in-memory object (no file I/O)
 * - File-based: writes to YAML file (default ~/.config/qmd/index.yml)
 */
export function saveConfig(config) {
    // SDK inline config mode: update in place, no file I/O
    if (configSource.type === 'inline') {
        configSource.config = config;
        return;
    }
    const configPath = configSource.path || getConfigFilePath();
    const configDir = dirname(configPath);
    if (!existsSync(configDir)) {
        mkdirSync(configDir, { recursive: true });
    }
    try {
        const yaml = YAML.stringify(config, {
            indent: 2,
            lineWidth: 0, // Don't wrap lines
        });
        writeFileSync(configPath, yaml, "utf-8");
    }
    catch (error) {
        throw new Error(`Failed to write ${configPath}: ${error}`);
    }
}
/**
 * Get a specific collection by name
 * Returns null if not found
 */
export function getCollection(name) {
    const config = loadConfig();
    const collection = config.collections[name];
    if (!collection) {
        return null;
    }
    return { name, ...collection };
}
/**
 * List all collections
 */
export function listCollections() {
    const config = loadConfig();
    return Object.entries(config.collections).map(([name, collection]) => ({
        name,
        ...collection,
    }));
}
/**
 * Get collections that are included by default in queries
 */
export function getDefaultCollections() {
    return listCollections().filter(c => c.includeByDefault !== false);
}
/**
 * Get collection names that are included by default
 */
export function getDefaultCollectionNames() {
    return getDefaultCollections().map(c => c.name);
}
/**
 * Update a collection's settings
 */
export function updateCollectionSettings(name, settings) {
    const config = loadConfig();
    const collection = config.collections[name];
    if (!collection)
        return false;
    if (settings.update !== undefined) {
        if (settings.update === null) {
            delete collection.update;
        }
        else {
            collection.update = settings.update;
        }
    }
    if (settings.includeByDefault !== undefined) {
        if (settings.includeByDefault === true) {
            // true is default, remove the field
            delete collection.includeByDefault;
        }
        else {
            collection.includeByDefault = settings.includeByDefault;
        }
    }
    saveConfig(config);
    return true;
}
/**
 * Add or update a collection
 */
export function addCollection(name, path, pattern = "**/*.md") {
    const config = loadConfig();
    config.collections[name] = {
        path,
        pattern,
        context: config.collections[name]?.context, // Preserve existing context
    };
    saveConfig(config);
}
/**
 * Remove a collection
 */
export function removeCollection(name) {
    const config = loadConfig();
    if (!config.collections[name]) {
        return false;
    }
    delete config.collections[name];
    saveConfig(config);
    return true;
}
/**
 * Rename a collection
 */
export function renameCollection(oldName, newName) {
    const config = loadConfig();
    if (!config.collections[oldName]) {
        return false;
    }
    if (config.collections[newName]) {
        throw new Error(`Collection '${newName}' already exists`);
    }
    config.collections[newName] = config.collections[oldName];
    delete config.collections[oldName];
    saveConfig(config);
    return true;
}
// ============================================================================
// Context management
// ============================================================================
/**
 * Get global context
 */
export function getGlobalContext() {
    const config = loadConfig();
    return config.global_context;
}
/**
 * Set global context
 */
export function setGlobalContext(context) {
    const config = loadConfig();
    config.global_context = context;
    saveConfig(config);
}
/**
 * Get all contexts for a collection
 */
export function getContexts(collectionName) {
    const collection = getCollection(collectionName);
    return collection?.context;
}
/**
 * Add or update a context for a specific path in a collection
 */
export function addContext(collectionName, pathPrefix, contextText) {
    const config = loadConfig();
    const collection = config.collections[collectionName];
    if (!collection) {
        return false;
    }
    if (!collection.context) {
        collection.context = {};
    }
    collection.context[pathPrefix] = contextText;
    saveConfig(config);
    return true;
}
/**
 * Remove a context from a collection
 */
export function removeContext(collectionName, pathPrefix) {
    const config = loadConfig();
    const collection = config.collections[collectionName];
    if (!collection?.context?.[pathPrefix]) {
        return false;
    }
    delete collection.context[pathPrefix];
    // Remove empty context object
    if (Object.keys(collection.context).length === 0) {
        delete collection.context;
    }
    saveConfig(config);
    return true;
}
/**
 * List all contexts across all collections
 */
export function listAllContexts() {
    const config = loadConfig();
    const results = [];
    // Add global context if present
    if (config.global_context) {
        results.push({
            collection: "*",
            path: "/",
            context: config.global_context,
        });
    }
    // Add collection contexts
    for (const [name, collection] of Object.entries(config.collections)) {
        if (collection.context) {
            for (const [path, context] of Object.entries(collection.context)) {
                results.push({
                    collection: name,
                    path,
                    context,
                });
            }
        }
    }
    return results;
}
/**
 * Find best matching context for a given collection and path
 * Returns the most specific matching context (longest path prefix match)
 */
export function findContextForPath(collectionName, filePath) {
    const config = loadConfig();
    const collection = config.collections[collectionName];
    if (!collection?.context) {
        return config.global_context;
    }
    // Find all matching prefixes
    const matches = [];
    for (const [prefix, context] of Object.entries(collection.context)) {
        // Normalize paths for comparison
        const normalizedPath = filePath.startsWith("/") ? filePath : `/${filePath}`;
        const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
        if (normalizedPath.startsWith(normalizedPrefix)) {
            matches.push({ prefix: normalizedPrefix, context });
        }
    }
    // Return most specific match (longest prefix)
    if (matches.length > 0) {
        matches.sort((a, b) => b.prefix.length - a.prefix.length);
        return matches[0].context;
    }
    // Fallback to global context
    return config.global_context;
}
// ============================================================================
// Utility functions
// ============================================================================
/**
 * Get the config file path (useful for error messages)
 */
export function getConfigPath() {
    if (configSource.type === 'inline')
        return '<inline>';
    return configSource.path || getConfigFilePath();
}
/**
 * Check if config file exists
 */
export function configExists() {
    if (configSource.type === 'inline')
        return true;
    const path = configSource.path || getConfigFilePath();
    return existsSync(path);
}
/**
 * Validate a collection name
 * Collection names must be valid and not contain special characters
 */
export function isValidCollectionName(name) {
    // Allow alphanumeric, hyphens, underscores
    return /^[a-zA-Z0-9_-]+$/.test(name);
}
