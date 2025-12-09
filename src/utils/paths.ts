/**
 * Path utility functions for file path handling and display
 */

import { resolve } from 'path';
import { homedir } from 'os';
import { existsSync } from 'fs';

/**
 * Find .qmd/ directory by walking up from current directory
 * @param startDir - Starting directory (defaults to PWD)
 * @returns Path to .qmd/ directory or null if not found
 */
export function findQmdDir(startDir?: string): string | null {
  let dir = startDir || getPwd();
  const root = resolve('/');

  // Walk up directory tree
  while (dir !== root) {
    const qmdDir = resolve(dir, '.qmd');
    if (existsSync(qmdDir)) {
      return qmdDir;
    }
    const parent = resolve(dir, '..');
    if (parent === dir) break; // Safety check for root
    dir = parent;
  }

  return null;
}

/**
 * Get the database path for a given index name
 * Priority: 1) .qmd/ directory, 2) QMD_CACHE_DIR env var, 3) XDG_CACHE_HOME/qmd/
 * @param indexName - Name of the index (default: "index")
 * @returns Full path to the SQLite database file
 */
export function getDbPath(indexName: string = "index"): string {
  let qmdCacheDir: string;

  // Priority 1: Check for .qmd/ directory in current project
  const projectQmdDir = findQmdDir();
  if (projectQmdDir) {
    qmdCacheDir = projectQmdDir;
  }
  // Priority 2: Check QMD_CACHE_DIR environment variable
  else if (process.env.QMD_CACHE_DIR) {
    qmdCacheDir = resolve(process.env.QMD_CACHE_DIR);
  }
  // Priority 3: Use XDG_CACHE_HOME or ~/.cache/qmd (default)
  else {
    const cacheDir = process.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
    qmdCacheDir = resolve(cacheDir, "qmd");
  }

  // Ensure cache directory exists
  try {
    Bun.spawnSync(["mkdir", "-p", qmdCacheDir]);
  } catch {}

  return resolve(qmdCacheDir, `${indexName}.sqlite`);
}

/**
 * Get the current working directory (PWD)
 * @returns Current working directory path
 */
export function getPwd(): string {
  return process.env.PWD || process.cwd();
}

/**
 * Get canonical real path, falling back to resolved path if file doesn't exist
 * @param path - File path to resolve
 * @returns Canonical real path
 */
export function getRealPath(path: string): string {
  try {
    const result = Bun.spawnSync(["realpath", path]);
    if (result.success) {
      return result.stdout.toString().trim();
    }
  } catch {}
  return resolve(path);
}

/**
 * Compute a unique display path for a file
 * Uses minimal path components needed for uniqueness (e.g., "project/docs/readme.md")
 *
 * @param filepath - Full file path
 * @param collectionPath - Base collection directory path
 * @param existingPaths - Set of already-used display paths (for uniqueness check)
 * @returns Minimal unique display path
 */
export function computeDisplayPath(
  filepath: string,
  collectionPath: string,
  existingPaths: Set<string>
): string {
  // Get path relative to collection (include collection dir name)
  const collectionDir = collectionPath.replace(/\/$/, '');
  const collectionName = collectionDir.split('/').pop() || '';

  let relativePath: string;
  if (filepath.startsWith(collectionDir + '/')) {
    // filepath is under collection: use collection name + relative path
    relativePath = collectionName + filepath.slice(collectionDir.length);
  } else {
    // Fallback: just use the filepath
    relativePath = filepath;
  }

  const parts = relativePath.split('/').filter(p => p.length > 0);

  // Always include at least parent folder + filename (minimum 2 parts if available)
  // Then add more parent dirs until unique
  const minParts = Math.min(2, parts.length);
  for (let i = parts.length - minParts; i >= 0; i--) {
    const candidate = parts.slice(i).join('/');
    if (!existingPaths.has(candidate)) {
      return candidate;
    }
  }

  // Absolute fallback: use full path (should be unique)
  return filepath;
}

/**
 * Convert absolute path to tilde notation if in home directory
 * @param dirpath - Directory path
 * @returns Path with ~ for home directory, or original path
 */
export function shortPath(dirpath: string): string {
  const home = homedir();
  if (dirpath.startsWith(home)) {
    return '~' + dirpath.slice(home.length);
  }
  return dirpath;
}
