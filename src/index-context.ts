import { createHash } from "node:crypto";
import { join } from "node:path";
import { realpathSync, statSync } from "node:fs";
import { findLocalConfigPath, getConfigDir, getLocalDbPath, normalizeConfigIndexName } from "./collections.js";
import { resolveDefaultDbPath } from "./store.js";

export interface IndexContext {
  dbPath: string;
  configPath?: string;
}
export interface SelectedIndexContext extends IndexContext {
  local: boolean;
  configPath: string;
}


/** The same selection as CLI startup, without opening or creating anything. */
export function resolveIndexContext(indexName?: string): SelectedIndexContext {
  const local = indexName ? undefined : findLocalConfigPath();
  const name = indexName ? normalizeConfigIndexName(indexName) : "index";
  return {
    local: local !== undefined,
    dbPath: local ? getLocalDbPath(local) : resolveDefaultDbPath(name),
    configPath: local ?? join(getConfigDir(), `${name}.yml`),
  };
}

// Version 1 metadata inputs, not an export of the process environment.
export const HEALTH_ENVIRONMENT_NAMES = [
  "HOME", "USERPROFILE", "PATH", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
  "QMD_CONFIG_DIR", "INDEX_PATH", "QMD_EMBED_MODEL", "QMD_GENERATE_MODEL",
  "QMD_RERANK_MODEL", "QMD_FORCE_CPU", "QMD_LLAMA_GPU", "QMD_EMBED_PARALLELISM",
  "QMD_EMBED_CONTEXT_SIZE", "QMD_EXPAND_CONTEXT_SIZE", "QMD_RERANK_CONTEXT_SIZE", "QMD_SOURCE_MODE",
  "PWD", "WSL_DISTRO_NAME", "WSL_INTEROP",
] as const;

export function metadataIdentity(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

export function environmentIdentity(): string {
  return metadataIdentity([
    process.cwd(),
    HEALTH_ENVIRONMENT_NAMES.map(name => [name, process.env[name] ?? null]),
  ]);
}

/** Bind the open store to its actual file, not just a reusable filename. */
export function indexIdentity(context: IndexContext): string {
  const file = statSync(context.dbPath, { bigint: true });
  if (!file.isFile()) throw new Error("Index is not a regular file");
  return metadataIdentity([
    realpathSync(context.dbPath), file.dev.toString(), file.ino.toString(),
    context.configPath === undefined ? null : realpathSync(context.configPath),
  ]);
}
