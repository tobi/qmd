/**
 * Collections configuration management
 *
 * This module manages the YAML-based collection configuration at ~/.config/qmd/index.yml.
 * Collections define which directories to index and their associated contexts.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import YAML from "yaml";

// ============================================================================
// Types
// ============================================================================

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
  path: string;           // Absolute path to index
  pattern: string;        // Glob pattern (e.g., "**/*.md")
  context?: ContextMap;   // Optional context definitions
  update?: string;        // Optional bash command to run during qmd update
}

/**
 * The complete configuration file structure
 */
export interface CollectionConfig {
  global_context?: string;                    // Context applied to all collections
  collections: Record<string, Collection>;    // Collection name -> config
}

/**
 * Collection with its name (for return values)
 */
export interface NamedCollection extends Collection {
  name: string;
}

// ============================================================================
// Configuration paths
// ============================================================================

// Current index name (default: "index")
let currentIndexName: string = "index";

/**
 * Set the current index name for config file lookup
 * Config file will be ~/.config/qmd/{indexName}.yml
 */
export function setConfigIndexName(name: string): void {
  currentIndexName = name;
}

function getConfigDir(): string {
  // Allow override via QMD_CONFIG_DIR for testing
  if (process.env.QMD_CONFIG_DIR) {
    return process.env.QMD_CONFIG_DIR;
  }
  return join(homedir(), ".config", "qmd");
}

function getConfigFilePath(): string {
  return join(getConfigDir(), `${currentIndexName}.yml`);
}

/**
 * Ensure config directory exists
 */
function ensureConfigDir(): void {
  const configDir = getConfigDir();
  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }
}

// ============================================================================
// Core functions
// ============================================================================

/**
 * Load configuration from ~/.config/qmd/index.yml
 * Returns empty config if file doesn't exist
 */
export function loadConfig(): CollectionConfig {
  const configPath = getConfigFilePath();
  if (!existsSync(configPath)) {
    return { collections: {} };
  }

  try {
    const content = readFileSync(configPath, "utf-8");
    const config = YAML.parse(content) as CollectionConfig;

    // Ensure collections object exists
    if (!config.collections) {
      config.collections = {};
    }

    return config;
  } catch (error) {
    throw new Error(`Failed to parse ${configPath}: ${error}`);
  }
}

/**
 * Save configuration to ~/.config/qmd/index.yml
 */
export function saveConfig(config: CollectionConfig): void {
  ensureConfigDir();
  const configPath = getConfigFilePath();

  try {
    const yaml = YAML.stringify(config, {
      indent: 2,
      lineWidth: 0,  // Don't wrap lines
    });
    writeFileSync(configPath, yaml, "utf-8");
  } catch (error) {
    throw new Error(`Failed to write ${configPath}: ${error}`);
  }
}

/**
 * Get a specific collection by name
 * Returns null if not found
 */
export function getCollection(name: string): NamedCollection | null {
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
export function listCollections(): NamedCollection[] {
  const config = loadConfig();
  return Object.entries(config.collections).map(([name, collection]) => ({
    name,
    ...collection,
  }));
}

/**
 * Add or update a collection
 */
export function addCollection(
  name: string,
  path: string,
  pattern: string = "**/*.md"
): void {
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
export function removeCollection(name: string): boolean {
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
export function renameCollection(oldName: string, newName: string): boolean {
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
export function getGlobalContext(): string | undefined {
  const config = loadConfig();
  return config.global_context;
}

/**
 * Set global context
 */
export function setGlobalContext(context: string | undefined): void {
  const config = loadConfig();
  config.global_context = context;
  saveConfig(config);
}

/**
 * Get all contexts for a collection
 */
export function getContexts(collectionName: string): ContextMap | undefined {
  const collection = getCollection(collectionName);
  return collection?.context;
}

/**
 * Add or update a context for a specific path in a collection
 */
export function addContext(
  collectionName: string,
  pathPrefix: string,
  contextText: string
): boolean {
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
export function removeContext(
  collectionName: string,
  pathPrefix: string
): boolean {
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
export function listAllContexts(): Array<{
  collection: string;
  path: string;
  context: string;
}> {
  const config = loadConfig();
  const results: Array<{ collection: string; path: string; context: string }> = [];

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
export function findContextForPath(
  collectionName: string,
  filePath: string
): string | undefined {
  const config = loadConfig();
  const collection = config.collections[collectionName];

  if (!collection?.context) {
    return config.global_context;
  }

  // Find all matching prefixes
  const matches: Array<{ prefix: string; context: string }> = [];

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
    return matches[0]!.context;
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
export function getConfigPath(): string {
  return getConfigFilePath();
}

/**
 * Check if config file exists
 */
export function configExists(): boolean {
  return existsSync(getConfigFilePath());
}

/**
 * Validate a collection name
 * Collection names must be valid and not contain special characters
 */
export function isValidCollectionName(name: string): boolean {
  // Allow alphanumeric, hyphens, underscores
  return /^[a-zA-Z0-9_-]+$/.test(name);
}
