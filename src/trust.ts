/**
 * trust.ts — approval gate for project-local `.qmd` config.
 *
 * A project-local `.qmd/index.yml` arrives with a `git clone`, and
 * `findLocalConfigPath` adopts it automatically for any command run inside the
 * tree. Three fields in that file can reach outside the project:
 *
 * - `update:` — a shell command run by `qmd update` (#886)
 * - `collections.*.path` — any directory the process can read (#889)
 * - `models.embed` / `models.rerank` / `models.generate` — any `hf:` repo or
 *   local GGUF path (#889)
 *
 * Global `~/.config/qmd` is never gated. In-project collection paths and the
 * built-in default model URIs are also allowed without approval: those are
 * what a local config is for. Approvals are per config file and per gated
 * set, recorded in `<config dir>/trusted.json`. Editing a hook, pointing a
 * collection outside the project, or changing a custom model URI changes the
 * digest and re-arms the gate.
 *
 * Trust lifecycle. The persisted record is the whole state; nothing else is
 * stored per config path:
 *
 *   unapproved  no record for the resolved path               → gate prompts/skips
 *   approved    record whose digest equals the current digest → gate runs
 *   stale       record for a digest that is no longer current → gate re-arms
 *
 * Transitions: `recordTrust(digest)` moves unapproved/stale → approved (and
 * refreshes an approval of the same digest); `revokeTrust` moves
 * approved/stale → unapproved; editing a hook, an out-of-project path, or a
 * custom model URI moves approved → stale without any write, because the
 * digest in the record no longer matches the config. A missing, corrupt, or
 * malformed `trusted.json` is unapproved for every path — the store fails
 * closed and is never read as "everything is trusted".
 *
 * Several processes record approvals for the same file (`qmd trust` in two
 * terminals, or an interactive approval during `qmd update` while another
 * project is approved), so every write is a locked read-modify-write
 * published with an atomic rename. A concurrent writer can therefore neither
 * drop an approval nor revive a revoked one, and a writer killed mid-write
 * leaves the previous complete store behind instead of a truncated file.
 */

import { createHash, randomBytes } from "crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, renameSync, unlinkSync, writeFileSync } from "fs";
import { basename, dirname, isAbsolute, join, relative, resolve } from "path";
import { getConfigDir } from "./collections.js";
import { openDatabase } from "./db.js";
import { qmdHomedir } from "./paths.js";
import { copyCommandProcessorSpec, type CommandProcessorSpec } from "./processors.js";
import { copyRemoteEmbedOptions, isRemoteEmbedModel, parseRemoteEmbedUri, type RemoteEmbedOptions } from "./remote-embed.js";
import { copyJevRerankOptions, isJevModel, parseJevModel, type JevRerankOptions } from "./jev.js";

/** A collection's pre-update hook, as it will be executed. */
export type UpdateHook = {
  collection: string;
  command: string;
};

export type CollectionPath = {
  collection: string;
  path: string;
};

export type ModelSlot = "embed" | "rerank" | "generate";

export type ModelsSnapshot = {
  embed?: string;
  embedOptions?: RemoteEmbedOptions;
  rerank?: string;
  rerankOptions?: JevRerankOptions;
  generate?: string;
};

export type SensitiveSnapshot = {
  processors?: Array<{ collection: string; spec: CommandProcessorSpec }>;
  hooks: UpdateHook[];
  paths: CollectionPath[];
  models: ModelsSnapshot;
};

export type GatedItems = {
  processors?: Array<{ collection: string; spec: Readonly<CommandProcessorSpec> }>;
  hooks: UpdateHook[];
  paths: CollectionPath[];
  models: Array<{ slot: ModelSlot; uri: string }>;
  embedOptions?: Readonly<RemoteEmbedOptions>;
  rerankOptions?: Readonly<JevRerankOptions>;
};

export type BuiltinModels = Required<Pick<ModelsSnapshot, ModelSlot>>;

export type TrustRecord = {
  /** Digest of the gated set that was approved. */
  hooks: string;
  /** ISO timestamp of the approval. */
  trustedAt: string;
};

export type TrustStore = Record<string, TrustRecord>;

/** Path of the trust database. Lives beside the global config. */
export function getTrustFilePath(): string {
  return join(getConfigDir(), "trusted.json");
}

/**
 * Sidecar database whose open write transaction serialises trust-store writes,
 * like `.qmd-embed.lock.sqlite` does for `qmd embed`. Kept beside the store so
 * it shares its filesystem; leaving it on disk is deliberate — unlinking a
 * lock by pathname is what makes pidfile locks racy.
 */
export function getTrustLockPath(): string {
  return `${getTrustFilePath()}.lock.sqlite`;
}

/**
 * Whether a config path is project-local — i.e. discovered by walking up from
 * the working directory rather than written by the user in their config dir.
 *
 * `findLocalConfigPath` and `qmd init` both use `.qmd/index.y{a,}ml`, and the
 * global config lives in a directory named `qmd`, so the parent directory name
 * separates the two cases without extra bookkeeping.
 */
export function isLocalConfigPath(configPath: string): boolean {
  if (!configPath || configPath === "<inline>") return false;
  return basename(dirname(resolve(configPath))) === ".qmd";
}

/** Directory that contains the `.qmd` folder for a project-local config. */
export function projectRootFromConfig(configPath: string): string {
  return dirname(dirname(resolve(configPath)));
}

/** Expand a leading `~` to the user home directory. */
export function expandUserPath(raw: string): string {
  if (raw === "~") return qmdHomedir();
  if (raw.startsWith("~/") || raw.startsWith("~\\")) {
    return join(qmdHomedir(), raw.slice(2));
  }
  return raw;
}

/**
 * Resolve a collection `path` from a project-local config: `~` expansion,
 * then relative paths against the project root (the directory that contains
 * `.qmd`), not against the process cwd.
 */
export function resolveConfigCollectionPath(rawPath: string, configPath: string): string {
  const expanded = expandUserPath(String(rawPath ?? "").trim());
  if (!expanded) return projectRootFromConfig(configPath);
  if (isAbsolute(expanded)) return resolve(expanded);
  return resolve(projectRootFromConfig(configPath), expanded);
}

function realOrResolve(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return resolve(path);
  }
}

/**
 * True if `rawPath` from a project-local config stays inside that project
 * (the directory containing `.qmd`), after `~` expansion, relative
 * resolution, and symlink realpath.
 */
export function isCollectionPathInsideProject(configPath: string, rawPath: string): boolean {
  const root = realOrResolve(projectRootFromConfig(configPath));
  const target = realOrResolve(resolveConfigCollectionPath(rawPath, configPath));
  const rel = relative(root, target);
  if (rel === "") return true;
  if (isAbsolute(rel)) return false;
  return !rel.split(/[/\\]/).includes("..");
}

export function gatedModels(
  models: ModelsSnapshot,
  builtins: BuiltinModels,
): Array<{ slot: ModelSlot; uri: string }> {
  const out: Array<{ slot: ModelSlot; uri: string }> = [];
  for (const slot of ["embed", "rerank", "generate"] as const) {
    const uri = models[slot];
    if (!uri) continue;
    if (slot === "embed") {
      if (typeof uri !== "string") throw new Error("Embedding model must be a single URI string");
      if (isRemoteEmbedModel(uri)) parseRemoteEmbedUri(uri);
    }
    if (slot === "rerank") {
      if (typeof uri !== "string") throw new Error("Rerank model must be a single URI string");
      if (isJevModel(uri)) parseJevModel(uri);
    }
    if (uri === builtins[slot]) continue;
    out.push({ slot, uri });
  }
  return out;
}

export function gatedItems(
  configPath: string,
  snapshot: SensitiveSnapshot,
  builtins: BuiltinModels,
): GatedItems {
  return {
    ...(snapshot.processors?.length ? { processors: snapshot.processors.map(p => ({ collection: p.collection, spec: copyCommandProcessorSpec(p.spec) })) } : {}),
    hooks: snapshot.hooks,
    paths: snapshot.paths.filter(p => !isCollectionPathInsideProject(configPath, p.path)),
    models: gatedModels(snapshot.models, builtins),
    ...(snapshot.models.embedOptions !== undefined ? { embedOptions: copyRemoteEmbedOptions(snapshot.models.embedOptions) } : {}),
    ...(snapshot.models.rerankOptions !== undefined ? { rerankOptions: copyJevRerankOptions(snapshot.models.rerankOptions) } : {}),
  };
}

export function hasGatedItems(gated: GatedItems): boolean {
  return !!gated.processors?.length || gated.hooks.length > 0 || gated.paths.length > 0 || gated.models.length > 0 || gated.embedOptions !== undefined || gated.rerankOptions !== undefined;
}

function byFirst(a: string[], b: string[]): number {
  const left = a[0] ?? "";
  const right = b[0] ?? "";
  return left < right ? -1 : left > right ? 1 : 0;
}

/**
 * Stable digest over a hook set. Order-independent so that reordering
 * collections in the YAML does not invalidate an approval, while any change to
 * a command — or a new collection gaining one — does.
 */
export function hookDigest(hooks: UpdateHook[]): string {
  const canonical = JSON.stringify(
    hooks
      .map(h => [h.collection, h.command])
      .sort(byFirst),
  );
  return createHash("sha256").update(canonical).digest("hex");
}

/**
 * Digest of the gated surface of a project-local config: hooks, resolved
 * collection paths, and non-default model URIs. Missing model keys and the
 * built-in default URIs are equivalent so that `qmd init` filling defaults
 * into the YAML does not invalidate an approval.
 */
export function sensitiveDigest(
  snapshot: SensitiveSnapshot,
  configPath: string,
  builtins: BuiltinModels,
): string {
  const gated = gatedItems(configPath, snapshot, builtins);
  const canonical = JSON.stringify({
    ...(gated.processors?.length ? { processors: gated.processors.map(p => [p.collection, p.spec] as const).sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0) } : {}),
    hooks: gated.hooks.map(h => [h.collection, h.command]).sort(byFirst),
    paths: gated.paths
      .map(p => [p.collection, resolveConfigCollectionPath(p.path, configPath)] as [string, string])
      .sort(byFirst),
    models: gated.models
      .map(m => [m.slot, m.uri] as [string, string])
      .sort(byFirst),
    ...(gated.embedOptions !== undefined ? { embedOptions: gated.embedOptions } : {}),
    ...(gated.rerankOptions !== undefined ? { rerankOptions: gated.rerankOptions } : {}),
  });
  return createHash("sha256").update(canonical).digest("hex");
}

/** One stored record, or `undefined` for anything that is not a trust record. */
function asTrustRecord(value: unknown): TrustRecord | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  if (!("hooks" in value) || typeof value.hooks !== "string") return undefined;
  if (!("trustedAt" in value) || typeof value.trustedAt !== "string") return undefined;
  return { hooks: value.hooks, trustedAt: value.trustedAt };
}

/**
 * Read the store. Anything unreadable — a missing file, invalid JSON, a
 * top-level value that is not an object, or an entry that is not a trust
 * record — reads as "not approved", never as blanket trust. Entries are
 * copied out of the parsed document, so a caller never aliases parsed JSON.
 */
export function loadTrustStore(): TrustStore {
  const path = getTrustFilePath();
  if (!existsSync(path)) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const store: TrustStore = {};
    for (const [configPath, value] of Object.entries(parsed)) {
      const record = asTrustRecord(value);
      if (record) store[configPath] = record;
    }
    return store;
  } catch {
    // A corrupt trust file must not be treated as "everything is trusted".
    return {};
  }
}

/**
 * Publish a store atomically: write a sibling temp file, then rename it over
 * `trusted.json`. A reader sees either the previous complete store or the new
 * complete store — never a truncated file or a half-written record — and a
 * writer killed mid-write leaves the previous store behind.
 */
function saveTrustStore(store: TrustStore): void {
  const path = getTrustFilePath();
  mkdirSync(dirname(path), { recursive: true });
  const temp = `${path}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`;
  try {
    writeFileSync(temp, `${JSON.stringify(store, null, 2)}\n`, { encoding: "utf-8", flag: "wx" });
    renameSync(temp, path);
  } catch (err) {
    try { unlinkSync(temp); } catch { /* never created, or already renamed */ }
    throw err;
  }
}

/** Trust records are keyed on the resolved config path. */
function trustKey(configPath: string): string {
  return resolve(configPath);
}

export function isTrusted(configPath: string, digest: string): boolean {
  const record = loadTrustStore()[trustKey(configPath)];
  return record?.hooks === digest;
}

/**
 * Budget for taking the write lock. The critical section is a small file
 * publish, so a short queue covers a competing writer but a frozen holder is
 * reported instead of stalling an approval for the index-write default.
 */
const TRUST_LOCK_BUSY_TIMEOUT_MS = 5000;

/**
 * Open write transaction that serialises trust-store writers, mirroring
 * `cli/embed-lock.ts`: SQLite takes the lock inside `BEGIN EXCLUSIVE`, and the
 * kernel ends it when the process exits or dies. There is no pidfile to go
 * stale, no PID to recycle, and no window in which a live writer looks unowned.
 */
function withTrustStoreLock<T>(fn: () => T): T {
  const lockPath = getTrustLockPath();
  mkdirSync(dirname(lockPath), { recursive: true });
  const db = openDatabase(lockPath);
  try {
    db.exec(`PRAGMA busy_timeout = ${TRUST_LOCK_BUSY_TIMEOUT_MS}`);
    db.exec("BEGIN EXCLUSIVE");
    try {
      return fn();
    } finally {
      if (db.inTransaction) {
        try { db.exec("ROLLBACK"); } catch { /* connection already broken */ }
      }
    }
  } finally {
    try { db.close(); } catch { /* best effort */ }
  }
}

/**
 * Read-modify-write of the trust store under the write lock. The read happens
 * inside the lock, so two processes cannot both publish a store derived from
 * the same stale copy — which would silently drop one approval, or resurrect
 * one that a concurrent `revokeTrust` had just removed.
 *
 * A writer that cannot take the lock within the busy budget throws instead of
 * writing unlocked: an approval that was not recorded is reported, not lost.
 */
function mutateTrustStore(mutate: (store: TrustStore) => boolean): boolean {
  return withTrustStoreLock(() => {
    const store = loadTrustStore();
    if (!mutate(store)) return false;
    saveTrustStore(store);
    return true;
  });
}

/** Approve the gated set identified by `digest` for a config path. */
export function recordTrust(configPath: string, digest: string): void {
  mutateTrustStore(store => {
    store[trustKey(configPath)] = { hooks: digest, trustedAt: new Date().toISOString() };
    return true;
  });
}

/** Drop the record for a config path. Returns false if there was none. */
export function revokeTrust(configPath: string): boolean {
  return mutateTrustStore(store => {
    const key = trustKey(configPath);
    if (!(key in store)) return false;
    delete store[key];
    return true;
  });
}

export function listTrusted(): Array<{ path: string } & TrustRecord> {
  return Object.entries(loadTrustStore()).map(([path, record]) => ({ path, ...record }));
}

export type HookGateDecision =
  | { action: "run"; digest: string }
  | { action: "prompt"; digest: string }
  | { action: "skip"; digest: string };

export type LocalConfigGateDecision = HookGateDecision;

function isTruthyEnv(value: string | undefined): boolean {
  if (!value) return false;
  return !["0", "false", "off", "no", "none"].includes(value.trim().toLowerCase());
}

/** True when the process has opted in to trusting project-local config unattended. */
export function isLocalConfigTrustOptedIn(env: NodeJS.ProcessEnv = process.env): boolean {
  return isTruthyEnv(env.QMD_TRUST_LOCAL_CONFIG) || isTruthyEnv(env.QMD_TRUST_UPDATE_HOOKS);
}

/**
 * Decide what to do with a config's `update:` hooks.
 *
 * Non-interactive callers — agents, CI, the MCP server — get `skip` rather than
 * a hard failure: indexing is what they asked for, and failing the whole
 * command would only push people toward a blanket opt-out.
 */
export function decideHookGate(options: {
  configPath: string;
  hooks: UpdateHook[];
  isInteractive: boolean;
  env?: NodeJS.ProcessEnv;
  trustedCheck?: (configPath: string, digest: string) => boolean;
}): HookGateDecision {
  const env = options.env ?? process.env;
  const digest = hookDigest(options.hooks);

  if (options.hooks.length === 0) return { action: "run", digest };
  if (isTruthyEnv(env.QMD_TRUST_UPDATE_HOOKS) || isTruthyEnv(env.QMD_TRUST_LOCAL_CONFIG)) {
    return { action: "run", digest };
  }
  if (!isLocalConfigPath(options.configPath)) return { action: "run", digest };

  const trusted = options.trustedCheck ?? isTrusted;
  if (trusted(options.configPath, digest)) return { action: "run", digest };

  return { action: options.isInteractive ? "prompt" : "skip", digest };
}

/**
 * Decide whether a project-local config may use its gated fields (hooks,
 * out-of-project collection paths, non-default model URIs).
 *
 * In-project paths and built-in default models do not need a decision.
 * Non-interactive callers get `skip`: in-project indexing still proceeds,
 * hooks/outside paths/custom models do not.
 */
export function decideLocalConfigGate(options: {
  configPath: string;
  snapshot: SensitiveSnapshot;
  builtins: BuiltinModels;
  isInteractive: boolean;
  env?: NodeJS.ProcessEnv;
  trustedCheck?: (configPath: string, digest: string) => boolean;
}): LocalConfigGateDecision {
  const env = options.env ?? process.env;
  const digest = sensitiveDigest(options.snapshot, options.configPath, options.builtins);
  const gated = gatedItems(options.configPath, options.snapshot, options.builtins);

  if (!hasGatedItems(gated)) return { action: "run", digest };
  if (isLocalConfigTrustOptedIn(env)) return { action: "run", digest };
  if (!isLocalConfigPath(options.configPath)) return { action: "run", digest };

  const trusted = options.trustedCheck ?? isTrusted;
  if (trusted(options.configPath, digest)) return { action: "run", digest };

  return { action: options.isInteractive ? "prompt" : "skip", digest };
}
