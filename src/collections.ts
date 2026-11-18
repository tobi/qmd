/**
 * Collections configuration management
 *
 * This module manages the YAML-based collection configuration at ~/.config/qmd/index.yml.
 * Collections define which directories to index and their associated contexts.
 *
 * Several processes share that file (CLI commands, the MCP daemon, SDK stores),
 * so every mutation is a read-modify-write under an exclusive per-config lock
 * and every publish is an atomic rename: no update is lost to a concurrent
 * writer and no reader ever observes a partially written YAML document.
 */

import {
  closeSync,
  existsSync,
  fchmodSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  renameSync,
  statSync,
  unlinkSync,
  writeSync,
  type Stats,
} from "fs";
import { basename, dirname, isAbsolute, join, resolve } from "path";
import { randomBytes } from "crypto";
import { qmdHomedir } from "./paths.js";
import { isBusyError, openDatabase } from "./db.js";
import YAML from "yaml";
import type { RemoteEmbedOptions } from "./remote-embed.js";
import type { JevRerankOptions } from "./jev.js";
import type { CommandProcessorSpec } from "./processors.js";

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
  path: string;              // Absolute path to index
  pattern: string;           // Glob pattern (e.g., "**/*.md")
  ignore?: string[];         // Glob patterns to exclude (e.g., ["Sessions/**"])
  context?: ContextMap;      // Optional context definitions
  processor?: CommandProcessorSpec; // Optional file-to-Markdown command (CLI trust required)
  update?: string;           // Optional bash command to run during qmd update
  includeByDefault?: boolean; // Include in queries by default (default: true)
}

/**
 * Model configuration for embedding, reranking, and generation
 */
export interface ModelsConfig {
  embed?: string;
  embedOptions?: RemoteEmbedOptions;
  rerank?: string;
  rerankOptions?: JevRerankOptions;
  generate?: string;
}

/**
 * The complete configuration file structure
 */
export interface CollectionConfig {
  global_context?: string;                    // Context applied to all collections
  editor_uri?: string;                        // Editor URI template for terminal hyperlinks
  editor_uri_template?: string;               // Alias for editor_uri
  editorUri?: string;                         // camelCase alias for editor_uri
  "editor-uri"?: string;                      // kebab-case alias for editor_uri
  collections: Record<string, Collection>;    // Collection name -> config
  models?: ModelsConfig;
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

// Current index name (default: "index") used by the default (CLI) scope.
let currentIndexName: string = "index";

/**
 * Set the current index name for config file lookup
 * Config file will be ~/.config/qmd/{indexName}.yml
 */
export function setConfigIndexName(name: string): void {
  // Resolve relative paths to absolute paths and sanitize for use as filename.
  // Windows absolute paths (C:\..., \\server\share) are already absolute --
  // skip resolve() for those (POSIX path.resolve doesn't recognize a drive
  // letter as absolute and would wrongly prefix it with cwd) and sanitize
  // `:` and `\` alongside `/` so the derived filename is valid on Windows.
  const isWindowsAbsolute = /^[a-zA-Z]:[\\/]/.test(name) || /^\\/.test(name);
  if (isWindowsAbsolute || /[\\/]/.test(name)) {
    const absolutePath = isWindowsAbsolute ? name : resolve(process.cwd(), name);
    // Replace path separators with underscores to create a valid filename
    currentIndexName = absolutePath.replace(/[:\\/]+/g, '_').replace(/^_+/, '');
  } else {
    currentIndexName = name;
  }
}

export function getConfigDir(): string {
  // Allow override via QMD_CONFIG_DIR for testing
  if (process.env.QMD_CONFIG_DIR) {
    return process.env.QMD_CONFIG_DIR;
  }
  // Respect XDG Base Directory specification (consistent with store.ts)
  if (process.env.XDG_CONFIG_HOME) {
    return join(process.env.XDG_CONFIG_HOME, "qmd");
  }
  return join(qmdHomedir(), ".config", "qmd");
}

function getConfigFilePath(): string {
  return join(getConfigDir(), `${currentIndexName}.yml`);
}

/**
 * Find a project-local QMD config by walking upward from startDir.
 * The local config lives at .qmd/index.yaml or .qmd/index.yml and,
 * when used by the CLI, keeps both config and index DB writes inside
 * the project instead of the global ~/.config / ~/.cache locations.
 */
export function findLocalConfigPath(startDir: string = process.cwd()): string | undefined {
  let dir = resolve(startDir);

  while (true) {
    const qmdDir = join(dir, ".qmd");
    const yamlPath = join(qmdDir, "index.yaml");
    if (existsSync(yamlPath)) return yamlPath;

    const ymlPath = join(qmdDir, "index.yml");
    if (existsSync(ymlPath)) return ymlPath;

    const parent = dirname(dir);
    if (parent === dir) return undefined;
    dir = parent;
  }
}

/** Return the local SQLite index path paired with a local .qmd/index.yaml file. */
export function getLocalDbPath(configPath: string): string {
  return join(dirname(configPath), "index.sqlite");
}

// ============================================================================
// Configuration source ownership
// ============================================================================

/**
 * Where one owner's collection configuration is read from and written to.
 *
 * - `file` with a path: that YAML file (SDK custom path, local `.qmd` config)
 * - `file` without a path: the default file for the current index name
 *   (`~/.config/qmd/{indexName}.yml`), resolved lazily on every access so the
 *   CLI can switch indexes without rebuilding the scope
 * - `inline`: a caller-owned in-memory config; saves update it in place and
 *   never touch disk
 * - `closed`: the owner was released; reads and writes throw instead of
 *   silently falling back to some other owner's target
 */
type FileConfigSource = { readonly kind: "file"; readonly path: string | undefined };
type InlineConfigSource = { readonly kind: "inline"; config: CollectionConfig };
type ClosedConfigSource = { readonly kind: "closed" };
type ConfigSourceState = FileConfigSource | InlineConfigSource | ClosedConfigSource;

/** How a scope should be bound to a configuration target. */
export interface ConfigSourceOptions {
  /** Path to a YAML config file (takes effect when `config` is absent) */
  configPath?: string;
  /** Inline config to own; wins over `configPath` when both are given */
  config?: CollectionConfig;
}

/** Thrown when a closed configuration scope is read or written. */
export class ConfigScopeClosedError extends Error {
  constructor() {
    super("Collection configuration scope is closed; create a new scope or store");
    this.name = "ConfigScopeClosedError";
  }
}

function sourceStateFrom(options?: ConfigSourceOptions): ConfigSourceState {
  if (!options) {
    return { kind: "file", path: undefined };
  }
  if (options.config) {
    // Ensure collections object exists on the caller's object: inline mode
    // mutates it in place, so callers observe every save.
    if (!options.config.collections) {
      options.config.collections = {};
    }
    return { kind: "inline", config: options.config };
  }
  if (options.configPath !== undefined) {
    if (options.configPath.trim() === "") {
      throw new Error("configPath must be a non-empty path");
    }
    return { kind: "file", path: options.configPath };
  }
  return { kind: "file", path: undefined };
}

// ============================================================================
// Atomic configuration publication and cross-process locking
// ============================================================================

/**
 * A configuration file is shared by every qmd process on the machine (the CLI,
 * the MCP daemon, and any SDK store bound to the same path), so a mutation is
 * a read-modify-write of a contended file:
 *
 *   - every mutation and every whole-file save holds an exclusive write
 *     transaction on a sidecar database beside the config (the same mechanism
 *     `src/trust.ts` and `qmd embed` use), so two processes cannot both read
 *     the same base version and then overwrite each other's change;
 *   - a save publishes by writing a sibling temp file, fsyncing it, and
 *     renaming it over the config, so readers (lock-free `load`, an editor, a
 *     human with `cat`) always observe a complete YAML document — never the
 *     truncate-then-write window of an in-place write.
 *
 * SQLite takes the write lock inside `BEGIN EXCLUSIVE`, and the kernel drops it
 * when the process exits or dies: a crashed writer cannot leave a lock a live
 * one would have to reclaim, there is no PID to recycle, and no age at which a
 * *live* writer looks abandoned. The sidecar is never unlinked — unlinking a
 * lock by pathname is what makes pidfile locks racy. A writer that cannot take
 * the lock within its budget throws instead of writing unlocked.
 */

/** Suffix of the sidecar database that serialises writers for one config. */
const LOCK_SUFFIX = ".lock.sqlite";
/** Budget for taking the write lock; the critical section is one file publish. */
const DEFAULT_LOCK_TIMEOUT_MS = 15_000;
const MAX_SYMLINK_HOPS = 16;

/**
 * Suffix of the sidecar that records a collection rename whose index half has
 * not finished yet.
 *
 * A rename is two resources: the configuration file (published first, so a
 * refused publish cannot lose index data) and the index state keyed by the old
 * name. The publication is preceded by this marker, so a later run has durable
 * evidence of the specific old→new mapping. Without it, "the file no longer
 * lists the old name and lists another name" is not evidence of a rename: that
 * other name may be an unrelated collection, and moving documents into it would
 * corrupt its ownership. A marker that does not match is never acted on.
 */
const PENDING_RENAME_SUFFIX = ".pending-rename.json";

/** The old→new mapping a published rename still owes its index half. */
export type PendingRename = { readonly from: string; readonly to: string };

/**
 * The pending-rename marker for a config, or undefined when there is none.
 *
 * Only a missing file (ENOENT) proves there is no journal. Anything else —
 * unreadable, not JSON, or not a valid name pair — means an existing journal
 * cannot be ruled out, and that journal is the only record of a rename whose
 * index move may not have finished, so it fails closed instead of being read as
 * absence and letting a mutation clobber or ignore the evidence.
 */
function readPendingRename(configPath: string): PendingRename | undefined {
  const markerPath = configSidecarPath(configPath, PENDING_RENAME_SUFFIX);
  let raw: string;
  try {
    raw = readFileSync(markerPath, "utf-8");
  } catch (error) {
    if (errnoCode(error) === "ENOENT") return undefined;
    throw new CorruptPendingRenameError(
      markerPath,
      `unreadable: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new CorruptPendingRenameError(
      markerPath,
      `not JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new CorruptPendingRenameError(markerPath, "expected an object with 'from' and 'to'");
  }
  if (!("from" in parsed) || !("to" in parsed)) {
    throw new CorruptPendingRenameError(markerPath, "expected 'from' and 'to'");
  }

  const { from, to } = parsed;
  if (typeof from !== "string" || typeof to !== "string") {
    throw new CorruptPendingRenameError(markerPath, "expected string 'from' and 'to'");
  }
  if (!isValidCollectionName(from) || !isValidCollectionName(to)) {
    throw new CorruptPendingRenameError(markerPath, `invalid collection name '${from}' -> '${to}'`);
  }
  if (from === to) {
    throw new CorruptPendingRenameError(markerPath, "a rename must change the name");
  }
  return { from, to };
}

/** Record the rename this config publication is about to make. */
function writePendingRename(configPath: string, pending: PendingRename): void {
  writeFileAtomically(configSidecarPath(configPath, PENDING_RENAME_SUFFIX), JSON.stringify(pending));
}

/** Drop the marker once the index half of the rename finished. */
function clearPendingRename(configPath: string): void {
  try {
    unlinkSync(configSidecarPath(configPath, PENDING_RENAME_SUFFIX));
  } catch (error) {
    if (errnoCode(error) !== "ENOENT") throw error;
  }
}

function isPendingRename(pending: PendingRename | undefined, rename: PendingRename): boolean {
  return pending !== undefined && pending.from === rename.from && pending.to === rename.to;
}

/**
 * What a mutation is allowed to touch, for the pending-rename evidence check.
 * Any mutation that would overwrite the marker or change the ownership of its
 * two names must wait until the rename's index half is finished; everything
 * else may proceed and leaves the marker alone.
 */
type ConfigIntent =
  | { readonly kind: "rename"; readonly from: string; readonly to: string }
  | { readonly kind: "names"; readonly names: readonly string[] }
  | { readonly kind: "any" };

/**
 * Thrown when a configuration mutation would destroy the only evidence of an
 * unfinished rename. Nothing is written: finish the rename first by repeating
 * it, which moves the old name's index state to the new name.
 */
export class PendingRenameError extends Error {
  constructor(readonly pending: PendingRename) {
    super(
      `The rename from '${pending.from}' to '${pending.to}' was published but its index move did not finish. ` +
        `Repeat that rename before changing either name; nothing was written.`,
    );
    this.name = "PendingRenameError";
  }
}

/**
 * Thrown when an existing pending-rename journal cannot be read or trusted.
 *
 * The journal is the only record of a rename whose index move may not have
 * finished, so it is never read as "no journal": a mutation fails closed
 * instead of clobbering the evidence or moving unrelated state. Nothing is
 * written; the file is left for a deliberate repair or removal.
 */
export class CorruptPendingRenameError extends Error {
  constructor(
    readonly markerPath: string,
    readonly reason: string,
  ) {
    super(
      `The pending-rename journal at ${markerPath} cannot be trusted (${reason}). ` +
        `It records a collection rename whose index move may not have finished. ` +
        `Repair or remove that file deliberately, then retry; nothing was written.`,
    );
    this.name = "CorruptPendingRenameError";
  }
}

/** Whether a configuration already carries the publication this marker names. */
function configHasRename(config: CollectionConfig, pending: PendingRename): boolean {
  return config.collections[pending.from] === undefined && config.collections[pending.to] !== undefined;
}

/**
 * Refuse a mutation that would overwrite or invalidate a live rename marker.
 * The matching rename is the only mutation allowed to touch those names.
 */
function assertRenameEvidenceIntact(
  pending: PendingRename | undefined,
  intent: ConfigIntent | undefined,
): void {
  if (pending === undefined || intent === undefined) return;
  if (intent.kind === "rename" && isPendingRename(pending, intent)) return;
  if (intent.kind === "names") {
    if (!intent.names.includes(pending.from) && !intent.names.includes(pending.to)) return;
  }
  throw new PendingRenameError(pending);
}

/** Thrown when another process keeps a config lock past the acquisition budget. */
export class ConfigLockTimeoutError extends Error {
  constructor(
    readonly lockPath: string,
    readonly timeoutMs: number,
  ) {
    super(
      `Timed out after ${timeoutMs}ms waiting for the configuration lock at ${lockPath}. ` +
        `Another qmd process is writing the config; retry once it finishes.`,
    );
    this.name = "ConfigLockTimeoutError";
  }
}

/** Thrown when one config mutation starts from inside another on the same file. */
export class NestedConfigMutationError extends Error {
  constructor(
    readonly configPath: string,
    readonly lockPath: string,
  ) {
    super(
      `Cannot start a config mutation for ${configPath} while another mutation of the same file ` +
        `is still running (lock ${lockPath}). Nothing was written: finish the enclosing ` +
        `update() before calling save(), update() or a collection/context mutation again.`,
    );
    this.name = "NestedConfigMutationError";
  }
}

/**
 * Locks this process already holds, so a mutation that publishes (or any other
 * nested acquisition) reuses the open transaction instead of self-deadlocking.
 * Keyed by the canonical sidecar path.
 */
const heldLocks = new Map<string, number>();

/**
 * Config files with a read-modify-write in flight in this process, keyed by the
 * canonical sidecar path. A second mutation on the same file is rejected
 * outright: it would publish its own snapshot, and the enclosing mutation would
 * then overwrite that publish with the stale snapshot it has been mutating.
 * Publishing (the outer mutation's own write) is not a mutation and is allowed.
 */
const mutationsInFlight = new Set<string>();

function errnoCode(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("code" in error)) return undefined;
  return typeof error.code === "string" ? error.code : undefined;
}

function readMsEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

/**
 * Resolve the file a save must replace. Writing to a config that is a symlink
 * must update the link's target, not replace the link with a regular file; the
 * atomic rename happens in the target's directory so it stays on one
 * filesystem. Parent-directory symlinks are left alone (equivalent path).
 */
function resolveWriteTarget(configPath: string): string {
  let current = configPath;
  for (let hop = 0; hop <= MAX_SYMLINK_HOPS; hop++) {
    let stats: Stats;
    try {
      stats = lstatSync(current);
    } catch (error) {
      if (errnoCode(error) === "ENOENT") return current;
      throw error;
    }
    if (!stats.isSymbolicLink()) return current;
    const link = readlinkSync(current);
    current = isAbsolute(link) ? link : resolve(dirname(current), link);
  }
  throw new Error(`Too many symlinks while resolving config path ${configPath}`);
}

/**
 * Path of a sidecar file that belongs to one config target.
 *
 * Derived from the *resolved* write target — a symlinked config and the file it
 * points at share it — and canonicalised through the real path of its
 * directory, so relative/absolute spellings and symlinked parents map to the
 * same inode.
 */
function configSidecarPath(configPath: string, suffix: string): string {
  const target = resolve(resolveWriteTarget(configPath));
  let dir = dirname(target);
  try {
    // The same directory reached through a different path (relative vs
    // absolute, symlinked parent) must map to one name, including for
    // in-process reentrancy.
    dir = realpathSync(dir);
  } catch {
    // Missing directory: `resolve` already produced the absolute form.
  }
  return join(dir, `${basename(target)}${suffix}`);
}

/**
 * Path of the sidecar database that serialises writers for a config target.
 * The file is created on demand and deliberately never unlinked.
 */
export function getConfigLockPath(configPath: string): string {
  return configSidecarPath(configPath, LOCK_SUFFIX);
}

/**
 * Run `fn` while holding the exclusive write transaction on the config's
 * sidecar database, handing it the canonical sidecar path so callers can key
 * their own per-target state on exactly the lock that is held. Reentrant within
 * this process (a mutation publishes while it already holds the lock); a writer
 * that cannot take it within its budget throws {@link ConfigLockTimeoutError}
 * instead of writing unlocked.
 *
 * SQLite takes the lock inside `BEGIN EXCLUSIVE` and the kernel drops it when
 * the process exits or dies, so there is no owner record to go stale, no age at
 * which a live writer looks abandoned, and no pathname anyone may unlink.
 */
function withConfigLock<T>(targetPath: string, fn: (lockPath: string) => T): T {
  // Create the directory first, then derive the sidecar path: canonicalisation
  // through the real path of a directory that exists is stable, so a nested
  // acquisition in this process (mutation → publish) finds the same key.
  const pendingLockPath = getConfigLockPath(targetPath);
  mkdirSync(dirname(pendingLockPath), { recursive: true });
  const lockPath = getConfigLockPath(targetPath);

  const depth = heldLocks.get(lockPath);
  if (depth !== undefined) {
    heldLocks.set(lockPath, depth + 1);
    try {
      return fn(lockPath);
    } finally {
      const next = (heldLocks.get(lockPath) ?? 1) - 1;
      if (next <= 0) heldLocks.delete(lockPath);
      else heldLocks.set(lockPath, next);
    }
  }

  const timeoutMs = readMsEnv("QMD_CONFIG_LOCK_TIMEOUT_MS", DEFAULT_LOCK_TIMEOUT_MS);
  const db = openDatabase(lockPath);
  try {
    // `openDatabase` already enables WAL; this budget is the fail-closed
    // contract for *this* lock, independent of the index-write timeout.
    db.exec(`PRAGMA busy_timeout = ${timeoutMs}`);
    try {
      db.exec("BEGIN EXCLUSIVE");
    } catch (error) {
      if (isBusyError(error)) throw new ConfigLockTimeoutError(lockPath, timeoutMs);
      throw error;
    }

    heldLocks.set(lockPath, 1);
    try {
      return fn(lockPath);
    } finally {
      heldLocks.delete(lockPath);
      if (db.inTransaction) {
        try {
          db.exec("ROLLBACK");
        } catch {
          // Connection already broken; closing it releases everything.
        }
      }
    }
  } finally {
    try {
      db.close();
    } catch {
      // best effort: the file stays for the next writer, by design.
    }
  }
}

/** Best-effort directory fsync so a completed rename survives a power loss. */
function fsyncDirectory(dir: string): void {
  let fd: number | undefined;
  try {
    fd = openSync(dir, "r");
    fsyncSync(fd);
  } catch {
    // Not supported on every platform; the atomic rename is the contract.
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // already closed
      }
    }
  }
}

/**
 * Publish `data` as the whole file atomically. Callers must hold the config
 * lock (the scope's publish path does); this function itself only guarantees
 * that readers see either the previous complete document or this one.
 */
function writeFileAtomically(filePath: string, data: string): void {
  const target = resolveWriteTarget(filePath);
  const dir = dirname(target);
  mkdirSync(dir, { recursive: true });

  // Preserve the mode of the file being replaced; a fresh file keeps the
  // process umask default that writeFileSync used to apply.
  let mode: number | undefined;
  try {
    mode = statSync(target).mode & 0o777;
  } catch (error) {
    if (errnoCode(error) !== "ENOENT") throw error;
  }

  const tempPath = join(
    dir,
    `.${basename(target)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`,
  );
  const bytes = Buffer.from(data, "utf-8");
  let fd: number | undefined;
  try {
    fd = openSync(tempPath, "wx", mode ?? 0o666);
    // `open`'s mode goes through the umask; the replaced file's exact mode wins.
    if (mode !== undefined) fchmodSync(fd, mode);
    let offset = 0;
    while (offset < bytes.length) offset += writeSync(fd, bytes, offset, bytes.length - offset);
    fsyncSync(fd);
    closeSync(fd);
    fd = undefined;
    renameSync(tempPath, target);
  } catch (error) {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // already closed
      }
    }
    try {
      unlinkSync(tempPath);
    } catch {
      // the temp file may not exist yet
    }
    throw error;
  }
  fsyncDirectory(dir);
}

/**
 * Effect that mirrors a published configuration change into derived state
 * (SQLite's `store_collections`) while the config lock is still held.
 *
 * A mutation publishes the configuration first and then runs the mirror, so:
 *   - a failing publish leaves the derived state untouched (no database-only
 *     edit survives an error);
 *   - the file (or inline object) stays authoritative, and every mirror is
 *     ordered by the same lock as the publishes, so a slower mirror can never
 *     overwrite derived state that a newer publish produced;
 *   - a throwing mirror leaves the published configuration in place: the error
 *     reaches the caller and the next sync repairs the derived state from the
 *     file. Cross-resource atomicity (one commit covering both the file and
 *     SQLite) is not claimed.
 *
 * The return value reports whether the derived state changed, so a mutation can
 * report the outcome of either half.
 */
export type ConfigMirror = (published: CollectionConfig) => boolean | void;

/**
 * Set model URIs in a YAML document's `models` map and render it. Only those
 * slots change, so comments, quoting and key order elsewhere survive.
 */
function setModelsInYaml(text: string, models: ModelsConfig): string {
  const doc = YAML.parseDocument(text);
  // setIn can't descend into a non-map `models`, e.g. a bare `models:` whose entries
  // are all commented out (a null scalar). Replace it, carrying its comments over.
  const current = doc.get("models", true);
  if (!YAML.isMap(current)) {
    const map = doc.createNode({});
    if (YAML.isNode(current)) {
      map.commentBefore = current.commentBefore;
      map.comment = current.comment;
    }
    doc.set("models", map);
  }
  for (const [slot, uri] of Object.entries(models)) {
    if (uri !== undefined) doc.setIn(["models", slot], uri);
  }
  return doc.toString({ indent: 2, lineWidth: 0 });
}

/**
 * One owner of a collection configuration.
 *
 * Every read and write in this module goes through a scope, so the target of
 * a mutation is always the scope that owns it. The CLI uses the single
 * module-level `defaultScope`; each SDK store creates its own scope and never
 * touches another store's target, even while both stores are open, and
 * closing a store only closes its own scope.
 */
export class CollectionConfigScope {
  private state: ConfigSourceState;
  private readonly resolveDefaultPath: () => string;
  /**
   * The index half of a rename this scope still owes for an inline config. An
   * inline config has no file to carry a marker, and the in-memory config lasts
   * exactly as long as this scope, so the scope itself is the durable record.
   */
  private pendingInlineRename?: PendingRename;

  constructor(
    options?: ConfigSourceOptions,
    resolveDefaultPath: () => string = getConfigFilePath,
  ) {
    this.resolveDefaultPath = resolveDefaultPath;
    this.state = sourceStateFrom(options);
  }

  /**
   * Bind the scope to a different target. Allowed from any state, including
   * closed, because the CLI reuses one default scope across invocations.
   */
  setSource(options?: ConfigSourceOptions): void {
    this.state = sourceStateFrom(options);
    // A different target must not inherit this scope's in-memory rename
    // evidence: it would authorize moving another config's index state.
    this.pendingInlineRename = undefined;
  }

  /** Release the scope. Idempotent; later reads and writes throw. */
  close(): void {
    this.state = { kind: "closed" };
  }

  get isClosed(): boolean {
    return this.state.kind === "closed";
  }

  /**
   * Load configuration from this scope's target.
   * Inline targets are returned as-is; missing files yield an empty config.
   *
   * Reads are lock-free: every publish is an atomic rename, so a reader always
   * gets one complete YAML document even while another process mutates.
   */
  load(): CollectionConfig {
    const state = this.openState();

    if (state.kind === "inline") {
      return state.config;
    }

    return this.readConfigFile(state.path ?? this.resolveDefaultPath());
  }

  /**
   * Save configuration to this scope's target.
   * Inline targets are replaced in memory; file targets are published as YAML
   * under the config lock, atomically.
   *
   * This replaces the whole file: a caller that loaded it earlier and then
   * saves can drop a concurrent process's change. Prefer {@link update} for
   * read-modify-write, which is what every in-tree mutation uses. Calling
   * `save` from inside `update` (or any mutation) on the same config is
   * rejected: the enclosing mutation is the transaction's only writer, and a
   * replacement there would silently discard the edits after it.
   */
  save(config: CollectionConfig): void {
    const state = this.openState();

    if (state.kind === "inline") {
      state.config = config;
      return;
    }

    const configPath = state.path ?? this.resolveDefaultPath();
    const lockPath = getConfigLockPath(resolveWriteTarget(configPath));
    if (mutationsInFlight.has(lockPath)) {
      throw new NestedConfigMutationError(configPath, lockPath);
    }

    // A whole-file replacement can produce or erase the very shapes a rename
    // marker speaks about, so it must not run while that evidence is live.
    withConfigLock(resolveWriteTarget(configPath), () => {
      const pending = readPendingRename(configPath);
      if (pending !== undefined && configHasRename(this.readConfigFile(configPath), pending)) {
        throw new PendingRenameError(pending);
      }
      if (pending !== undefined) clearPendingRename(configPath);
      this.publish(configPath, config);
    });
  }

  /**
   * Read, modify and publish this scope's config while holding its exclusive
   * lock. `mutator` sees the current on-disk version and its changes are
   * written even if another process mutated the same file after this scope's
   * caller last loaded it.
   *
   * A mutation started from inside `mutator` — an `update` or any collection or
   * context helper, through this scope or any other scope bound to the same
   * file — is rejected with {@link NestedConfigMutationError} before it reads
   * or writes. Reusing the lock alone would let the nested call publish its own
   * snapshot and the enclosing update overwrite it with its older one.
   *
   * A throwing `mutator` publishes nothing. Returns the configuration that was
   * published (for inline targets, the caller's live object, which is mutated
   * directly: no publish step exists, so nested inline mutations are allowed).
   */
  update(mutator: (config: CollectionConfig) => void): CollectionConfig {
    return this.mutateConfig((config) => {
      mutator(config);
      return { value: config, changed: true };
    }, undefined, { kind: "any" }).value;
  }

  /**
   * Set model URIs while holding this scope's lock, like {@link update}.
   * `resolve` sees the current on-disk config and returns the slots to set;
   * the returned models are handed back.
   *
   * An existing YAML file is edited as a parsed document and only the `models`
   * slots change, so its comments, quoting and key order survive (yaml still
   * normalizes whitespace). Inline targets and a missing file are written the
   * same way {@link update} writes them.
   */
  updateModels(resolve: (config: CollectionConfig) => ModelsConfig): ModelsConfig {
    const isFile = this.openState().kind === "file";
    const configPath = this.getConfigPath();
    return this.mutateConfig((config) => {
      const models = resolve(config);
      config.models = { ...config.models, ...models };
      // mutateConfig holds the lock, so this is the text `config` was parsed from.
      const yaml = isFile && existsSync(configPath)
        ? setModelsInYaml(readFileSync(configPath, "utf-8"), models)
        : undefined;
      return { value: models, changed: true, yaml };
    }, undefined, { kind: "any" }).value;
  }

  /**
   * Read this scope's authoritative configuration and run `fn` while the
   * config lock is held. Use this to keep a derived store (SQLite's
   * `store_collections`) in step with the file: a lock-free read followed by a
   * later sync would let a mutation another process publishes in between be
   * deleted from the derived store, and would stamp a hash for the stale
   * document.
   *
   * `fn` must not start a mutation of this config (that is rejected with
   * {@link NestedConfigMutationError}); it is for the mirror half of a sync.
   * Inline targets have no file to serialize on and pass their live object.
   */
  withLockedConfig<T>(fn: (config: CollectionConfig) => T): T {
    const state = this.openState();

    if (state.kind === "inline") {
      // No publish step and no lock: a nested inline mutation edits the same
      // live object the caller is deriving from, so no change can be lost.
      return fn(state.config);
    }

    const configPath = state.path ?? this.resolveDefaultPath();
    return withConfigLock(resolveWriteTarget(configPath), (lockPath) => {
      if (mutationsInFlight.has(lockPath)) {
        throw new NestedConfigMutationError(configPath, lockPath);
      }

      mutationsInFlight.add(lockPath);
      try {
        return fn(this.readConfigFile(configPath));
      } finally {
        mutationsInFlight.delete(lockPath);
      }
    });
  }

  /**
   * The rename this scope still owes its index half, read without mutating
   * anything: the exact old→new pair, or undefined when there is none.
   *
   * This is how a caller decides whether an operation has something to finish
   * *before* opening the index (or writing anything). Only a missing journal
   * means "none"; an unreadable or malformed one throws
   * {@link CorruptPendingRenameError} instead of being read as absence.
   */
  pendingRename(): PendingRename | undefined {
    const state = this.openState();
    if (state.kind === "inline") return this.pendingInlineRename;
    return readPendingRename(state.path ?? this.resolveDefaultPath());
  }

  /** Get this scope's config path, or `<inline>` for in-memory configs. */
  getConfigPath(): string {
    const state = this.openState();
    if (state.kind === "inline") return "<inline>";
    return state.path ?? this.resolveDefaultPath();
  }

  /** Check whether this scope's target exists. Inline configs always do. */
  configExists(): boolean {
    const state = this.openState();
    if (state.kind === "inline") return true;
    return existsSync(state.path ?? this.resolveDefaultPath());
  }

  /** Get a specific collection by name, or null if not found. */
  getCollection(name: string): NamedCollection | null {
    const collection = this.load().collections[name];
    if (!collection) return null;
    return { name, ...collection };
  }

  /** List all collections in this scope. */
  listCollections(): NamedCollection[] {
    return Object.entries(this.load().collections).map(([name, collection]) => ({
      name,
      ...collection,
    }));
  }

  /** List collections included by default in queries. */
  getDefaultCollections(): NamedCollection[] {
    return this.listCollections().filter(c => c.includeByDefault !== false);
  }

  /** List names of collections included by default in queries. */
  getDefaultCollectionNames(): string[] {
    return this.getDefaultCollections().map(c => c.name);
  }

  /** Update a collection's settings. Returns false when it does not exist. */
  updateCollectionSettings(
    name: string,
    settings: { update?: string | null; includeByDefault?: boolean }
  ): boolean {
    return this.mutateConfig((config) => {
      const collection = config.collections[name];
      if (!collection) return { value: false, changed: false };

      if (settings.update !== undefined) {
        if (settings.update === null) {
          delete collection.update;
        } else {
          collection.update = settings.update;
        }
      }

      if (settings.includeByDefault !== undefined) {
        if (settings.includeByDefault === true) {
          // true is default, remove the field
          delete collection.includeByDefault;
        } else {
          collection.includeByDefault = settings.includeByDefault;
        }
      }

      return { value: true, changed: true };
    }).value;
  }

  /**
   * Add or update a collection, preserving its existing context.
   *
   * `mirror` (see {@link ConfigMirror}) runs after the publish, under the same
   * lock, so a caller that keeps its own copy of the configuration in SQLite
   * steps in with the file rather than after an unlocked window.
   */
  addCollection(
    name: string,
    path: string,
    pattern: string = "**/*.md",
    mirror?: ConfigMirror,
  ): void {
    this.mutateConfig((config) => {
      config.collections[name] = {
        path,
        pattern,
        context: config.collections[name]?.context, // Preserve existing context
      };
      return { value: undefined, changed: true };
    }, mirror, { kind: "names", names: [name] });
  }

  /**
   * Remove a collection. Returns whether anything changed — in the
   * configuration or, through `mirror`, in the derived state — so a row that
   * only the derived store still held is still reported (and removed).
   */
  removeCollection(name: string, mirror?: ConfigMirror): boolean {
    const { value, mirrored } = this.mutateConfig((config) => {
      if (!config.collections[name]) {
        return { value: false, changed: false };
      }

      delete config.collections[name];
      return { value: true, changed: true };
    }, mirror, { kind: "names", names: [name] });
    return value || mirrored;
  }

  /**
   * Rename a collection. Throws when the new name is already taken, before
   * anything is published.
   *
   * A rename has an index half as well as a file half, so its `mirror` runs
   * only against durable evidence of that rename: the marker written before
   * this publication, or the publication itself. Without it, "the file no
   * longer lists the old name and lists another one" is not evidence — that
   * other name may be an unrelated collection, and moving the old name's
   * documents into it would corrupt its ownership — so `mirrored` is false and
   * the caller reports that nothing was renamed.
   */
  renameCollection(oldName: string, newName: string, mirror?: ConfigMirror): boolean {
    const { value, mirrored } = this.mutateConfig((config) => {
      if (!config.collections[oldName]) {
        return { value: false, changed: false };
      }

      if (config.collections[newName]) {
        throw new Error(`Collection '${newName}' already exists`);
      }

      config.collections[newName] = config.collections[oldName];
      delete config.collections[oldName];
      return { value: true, changed: true };
    }, mirror, { kind: "rename", from: oldName, to: newName });
    return value || mirrored;
  }

  /** Get the global context. */
  getGlobalContext(): string | undefined {
    return this.load().global_context;
  }

  /** Set the global context. `mirror` behaves as in {@link addCollection}. */
  setGlobalContext(context: string | undefined, mirror?: ConfigMirror): void {
    this.mutateConfig((config) => {
      config.global_context = context;
      return { value: undefined, changed: true };
    }, mirror);
  }

  /** Get all contexts for a collection. */
  getContexts(collectionName: string): ContextMap | undefined {
    return this.getCollection(collectionName)?.context;
  }

  /**
   * Add or update a context for a path in a collection. `mirror` behaves as in
   * {@link addCollection}.
   */
  addContext(
    collectionName: string,
    pathPrefix: string,
    contextText: string,
    mirror?: ConfigMirror,
  ): boolean {
    const { value, mirrored } = this.mutateConfig((config) => {
      const collection = config.collections[collectionName];

      if (!collection) {
        return { value: false, changed: false };
      }

      if (!collection.context) {
        collection.context = {};
      }

      collection.context[pathPrefix] = contextText;
      return { value: true, changed: true };
    }, mirror);
    return value || mirrored;
  }

  /**
   * Remove a context from a collection path. `mirror` behaves as in
   * {@link addCollection}.
   */
  removeContext(
    collectionName: string,
    pathPrefix: string,
    mirror?: ConfigMirror,
  ): boolean {
    const { value, mirrored } = this.mutateConfig((config) => {
      const collection = config.collections[collectionName];

      if (!collection?.context?.[pathPrefix]) {
        return { value: false, changed: false };
      }

      delete collection.context[pathPrefix];

      // Remove empty context object
      if (Object.keys(collection.context).length === 0) {
        delete collection.context;
      }

      return { value: true, changed: true };
    }, mirror);
    return value || mirrored;
  }

  /** List all contexts across all collections in this scope. */
  listAllContexts(): Array<{
    collection: string;
    path: string;
    context: string;
  }> {
    const config = this.load();
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
   * Find best matching context for a given collection and path.
   * Returns the most specific matching context (longest path prefix match).
   */
  findContextForPath(
    collectionName: string,
    filePath: string
  ): string | undefined {
    const config = this.load();
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

  /** Read and parse the YAML file, treating a missing file as empty. */
  private readConfigFile(configPath: string): CollectionConfig {
    if (!existsSync(configPath)) {
      return { collections: {} };
    }

    try {
      const content = readFileSync(configPath, "utf-8");
      const parsed = YAML.parse(content) as CollectionConfig | null | undefined;
      const config = parsed ?? { collections: {} };

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
   * Publish a whole configuration: serialise first, then replace the file under
   * the lock with an atomic rename. Reentrant with the mutation lock. `text`,
   * when given, is the already rendered document for `config`.
   */
  private publish(configPath: string, config: CollectionConfig, text?: string): void {
    try {
      const yaml = text ?? YAML.stringify(config, {
        indent: 2,
        lineWidth: 0,  // Don't wrap lines
      });
      withConfigLock(resolveWriteTarget(configPath), () => {
        writeFileAtomically(configPath, yaml);
      });
    } catch (error) {
      if (error instanceof ConfigLockTimeoutError) throw error;
      throw new Error(`Failed to write ${configPath}: ${error}`);
    }
  }

  /**
   * Shared read-modify-write path for file targets: the config is re-read from
   * disk under the exclusive lock, so a change another process published since
   * the caller last looked is preserved instead of being overwritten with a
   * stale copy. Inline targets are mutated in place, as before.
   *
   * A mutation started from inside another mutation on the same file is
   * rejected before it reads or writes: reusing the lock would let the nested
   * call publish its own snapshot, which the enclosing mutation would then
   * overwrite with the stale snapshot it has been mutating. The check runs
   * under the (reentrantly held) lock, so it cannot interleave with another
   * acquisition in this process.
   *
   * `mutator` returns the value to hand back and whether anything changed; an
   * unchanged result neither publishes nor touches the file. It may also return
   * `yaml`, the rendered file to publish in place of serialising the config. `mirror` runs
   * after the publish (after the in-place edit for inline targets) and still
   * under the same lock, whether or not the configuration changed: the derived
   * store must be able to catch up to a file that already holds the change. Its
   * throw propagates with the configuration already published; the next sync
   * repairs the derived store from it. A `rename` is the exception: see below.
   *
   * `rename` marks the mutation as one whose derived half is keyed by a name
   * pair. Its marker is written before the publication (so a crash between the
   * two leaves evidence) and removed only after the mirror succeeds; the mirror
   * runs when the mutation published, or when that marker proves the earlier
   * publication. See {@link CollectionConfigScope.renameCollection}.
   */
  private mutateConfig<T>(
    mutator: (config: CollectionConfig) => { value: T; changed: boolean; yaml?: string },
    mirror?: ConfigMirror,
    intent?: ConfigIntent,
  ): { value: T; changed: boolean; mirrored: boolean } {
    const state = this.openState();

    if (state.kind === "inline") {
      // No publish step: a nested inline mutation edits the caller's live
      // object, so no change can be lost and no rejection is needed. An inline
      // config has no file to carry the marker, so the scope object itself
      // remembers the rename it still owes while it stays open.
      let pending = this.pendingInlineRename;
      if (pending !== undefined && !configHasRename(state.config, pending)) {
        this.pendingInlineRename = undefined;
        pending = undefined;
      }
      assertRenameEvidenceIntact(pending, intent);

      const { value, changed } = mutator(state.config);
      if (intent?.kind === "rename" && mirror !== undefined && changed) {
        this.pendingInlineRename = { from: intent.from, to: intent.to };
      }
      const evidence =
        intent?.kind !== "rename"
        || changed
        || isPendingRename(pending, intent);
      if (!evidence) return { value, changed, mirrored: false };

      const mirrored = mirror?.(state.config) === true;
      if (intent?.kind === "rename") this.pendingInlineRename = undefined;
      return { value, changed, mirrored };
    }

    const configPath = state.path ?? this.resolveDefaultPath();
    return withConfigLock(resolveWriteTarget(configPath), (lockPath) => {
      if (mutationsInFlight.has(lockPath)) {
        throw new NestedConfigMutationError(configPath, lockPath);
      }

      const config = this.readConfigFile(configPath);
      mutationsInFlight.add(lockPath);
      try {
        // A marker is evidence only while the file reflects its publication. One
        // left by a publish that failed names an attempt, not a rename, so it is
        // dropped rather than allowed to block a change or authorize a move.
        let pending = readPendingRename(configPath);
        if (pending !== undefined && !configHasRename(config, pending)) {
          clearPendingRename(configPath);
          pending = undefined;
        }
        assertRenameEvidenceIntact(pending, intent);

        const { value, changed, yaml } = mutator(config);
        if (intent?.kind === "rename" && mirror !== undefined && changed) {
          // Durable intent first: a crash after the publication but before the
          // index half leaves this behind, and it is the only thing that may
          // authorize moving the old name's index state later.
          writePendingRename(configPath, { from: intent.from, to: intent.to });
        }
        if (changed) this.publish(configPath, config, yaml);

        const evidence =
          intent?.kind !== "rename"
          || changed
          || isPendingRename(pending, intent);
        if (!evidence) return { value, changed, mirrored: false };

        // The file is published before the derived state changes, so a failed
        // publish (lock budget, permissions, disk) leaves no database-only
        // edit behind. Holding the lock through the mirror keeps every mirror
        // ordered with the publishes it was derived from.
        const mirrored = mirror?.(config) === true;
        if (intent?.kind === "rename" && mirror !== undefined) clearPendingRename(configPath);
        return { value, changed, mirrored };
      } finally {
        mutationsInFlight.delete(lockPath);
      }
    });
  }

  private openState(): FileConfigSource | InlineConfigSource {
    if (this.state.kind === "closed") {
      throw new ConfigScopeClosedError();
    }
    return this.state;
  }
}

// ============================================================================
// Module-level API: the CLI's default scope
// ============================================================================

/**
 * The scope the module-level helpers below read and write. CLI code binds it
 * once per invocation via `setConfigSource` / `setConfigIndexName`; SDK
 * consumers should own a `CollectionConfigScope` per store instead.
 */
const defaultScope = new CollectionConfigScope();

/**
 * Set the config source for the default scope.
 * - File path: load/save from a specific YAML file
 * - Inline config: use an in-memory CollectionConfig (saveConfig updates in place, no file I/O)
 * - undefined: reset to default file-based config
 */
export function setConfigSource(source?: ConfigSourceOptions): void {
  defaultScope.setSource(source);
}

/** Load configuration from the default scope. */
export function loadConfig(): CollectionConfig {
  return defaultScope.load();
}

/**
 * Save configuration to the default scope.
 *
 * Replaces the whole file atomically under the config lock. A caller that
 * loaded the config earlier can drop another process's change; use
 * {@link updateConfig} for read-modify-write.
 */
export function saveConfig(config: CollectionConfig): void {
  defaultScope.save(config);
}

/**
 * Read, modify and publish the default scope's config under its exclusive
 * lock, so concurrent qmd processes cannot lose each other's changes.
 * `mutator` receives the version currently on disk. Starting another mutation
 * of the same file from inside `mutator` throws {@link NestedConfigMutationError}
 * before anything is written.
 */
export function updateConfig(mutator: (config: CollectionConfig) => void): CollectionConfig {
  return defaultScope.update(mutator);
}

/**
 * Fill in model URIs in the default scope's config under its exclusive lock
 * (see {@link CollectionConfigScope.updateModels}). Comments, quoting and key
 * order elsewhere in a YAML file survive.
 */
export function updateModelsConfig(resolve: (config: CollectionConfig) => ModelsConfig): ModelsConfig {
  return defaultScope.updateModels(resolve);
}

/**
 * Read the default scope's authoritative configuration and run `fn` while its
 * lock is held. This is how the CLI keeps SQLite in step with the file: a
 * lock-free {@link loadConfig} followed by a later sync can delete a
 * collection another qmd process published in between.
 *
 * `fn` must not start a mutation of the same file (rejected with
 * {@link NestedConfigMutationError}); it is for the mirror half of a sync.
 */
export function withConfigLocked<T>(fn: (config: CollectionConfig) => T): T {
  return defaultScope.withLockedConfig(fn);
}

/**
 * The default scope's unfinished rename, read without mutating anything
 * (see {@link CollectionConfigScope.pendingRename}).
 */
export function getPendingRename(): PendingRename | undefined {
  return defaultScope.pendingRename();
}

/**
 * Whether this exact rename still owes its index half. A caller that would
 * otherwise open the index (writing nothing yet) uses this to decide whether
 * the operation has anything to finish first — a name the config no longer
 * lists is only a rename worth acting on when this pair is recorded.
 */
export function hasPendingRename(from: string, to: string): boolean {
  return isPendingRename(defaultScope.pendingRename(), { from, to });
}

/** Get a specific collection by name from the default scope. */
export function getCollection(name: string): NamedCollection | null {
  return defaultScope.getCollection(name);
}

/** List all collections in the default scope. */
export function listCollections(): NamedCollection[] {
  return defaultScope.listCollections();
}

/** Get collections that are included by default in queries. */
export function getDefaultCollections(): NamedCollection[] {
  return defaultScope.getDefaultCollections();
}

/** Get collection names that are included by default. */
export function getDefaultCollectionNames(): string[] {
  return defaultScope.getDefaultCollectionNames();
}

/** Update a collection's settings in the default scope. */
export function updateCollectionSettings(
  name: string,
  settings: { update?: string | null; includeByDefault?: boolean }
): boolean {
  return defaultScope.updateCollectionSettings(name, settings);
}

/** Add or update a collection in the default scope. */
export function addCollection(
  name: string,
  path: string,
  pattern: string = "**/*.md",
  mirror?: ConfigMirror,
): void {
  defaultScope.addCollection(name, path, pattern, mirror);
}

/** Remove a collection from the default scope. */
export function removeCollection(name: string, mirror?: ConfigMirror): boolean {
  return defaultScope.removeCollection(name, mirror);
}

/** Rename a collection in the default scope. */
export function renameCollection(
  oldName: string,
  newName: string,
  mirror?: ConfigMirror,
): boolean {
  return defaultScope.renameCollection(oldName, newName, mirror);
}

/** Get global context from the default scope. */
export function getGlobalContext(): string | undefined {
  return defaultScope.getGlobalContext();
}

/** Set global context in the default scope. */
export function setGlobalContext(context: string | undefined, mirror?: ConfigMirror): void {
  defaultScope.setGlobalContext(context, mirror);
}

/** Get all contexts for a collection in the default scope. */
export function getContexts(collectionName: string): ContextMap | undefined {
  return defaultScope.getContexts(collectionName);
}

/** Add or update a context for a specific path in the default scope. */
export function addContext(
  collectionName: string,
  pathPrefix: string,
  contextText: string,
  mirror?: ConfigMirror,
): boolean {
  return defaultScope.addContext(collectionName, pathPrefix, contextText, mirror);
}

/** Remove a context from the default scope. */
export function removeContext(
  collectionName: string,
  pathPrefix: string,
  mirror?: ConfigMirror,
): boolean {
  return defaultScope.removeContext(collectionName, pathPrefix, mirror);
}

/** List all contexts across all collections in the default scope. */
export function listAllContexts(): Array<{
  collection: string;
  path: string;
  context: string;
}> {
  return defaultScope.listAllContexts();
}

/** Find best matching context for a given collection and path. */
export function findContextForPath(
  collectionName: string,
  filePath: string
): string | undefined {
  return defaultScope.findContextForPath(collectionName, filePath);
}

/** Get the default scope's config file path (useful for error messages). */
export function getConfigPath(): string {
  return defaultScope.getConfigPath();
}

/** Check if the default scope's config file exists. */
export function configExists(): boolean {
  return defaultScope.configExists();
}

/**
 * Validate a collection name
 * Collection names must be valid and not contain special characters
 */
export function isValidCollectionName(name: string): boolean {
  // Allow alphanumeric, hyphens, underscores
  return /^[a-zA-Z0-9_-]+$/.test(name);
}
