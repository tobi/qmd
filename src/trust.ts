/**
 * trust.ts — approval gate for project-local `.qmd` config.
 *
 * A project-local `.qmd/index.yml` arrives with a `git clone`, and
 * `findLocalConfigPath` adopts it automatically for any command run anywhere
 * inside the tree. Three fields in that file are somebody else's say-so until
 * the user approves them:
 *
 * - `update:` hooks, which `qmd update` runs through `bash -c` (#886)
 * - `collections.*.path` that resolve outside the project, which `qmd update`
 *   would otherwise index (#889)
 * - `models.embed` / `models.rerank` / `models.generate` that are not the
 *   built-in defaults, which `qmd embed` / `qmd query` / `qmd pull` would
 *   otherwise download or load (#889)
 *
 * Paths that stay inside the project (the usual `path: ./docs`) and model
 * URIs that equal the built-in defaults are not gated — that is the intended
 * use of a checked-in config. Global `~/.config/qmd` is never gated.
 *
 * Approvals are per config file and per gated-surface set, recorded in
 * `<config dir>/trusted.json`. Editing a hook, an out-of-project path, or a
 * custom model URI — or a `git pull` that rewrites one — changes the digest
 * and re-arms the gate. A config that only has hooks keeps the #886 digest
 * so existing approvals still match.
 */

import { createHash } from "crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "path";
import { getConfigDir } from "./collections.js";
import { qmdHomedir } from "./paths.js";

/** A collection's pre-update hook, as it will be executed. */
export type UpdateHook = {
  collection: string;
  command: string;
};

/** A collection path that resolves outside the project and needs approval. */
export type GatedPath = {
  collection: string;
  path: string;
};

/** Model URIs from project-local config that are not the built-in defaults. */
export type GatedModels = {
  embed?: string;
  rerank?: string;
  generate?: string;
};

export type DefaultModels = {
  embed: string;
  generate: string;
  rerank: string;
};

/** Every surface a project-local config must not apply unattended. */
export type GatedConfig = {
  hooks: UpdateHook[];
  paths: GatedPath[];
  models: GatedModels;
};

export type TrustRecord = {
  /** Digest of the gated surfaces that were approved. */
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

/** Directory that contains the `.qmd/` folder for a project-local config. */
export function projectRootFromConfigPath(configPath: string): string {
  return dirname(dirname(resolve(configPath)));
}

function expandUserPath(p: string): string {
  if (p === "~") return qmdHomedir();
  if (p.startsWith("~/")) return join(qmdHomedir(), p.slice(2));
  return p;
}

function realOrResolve(p: string): string {
  try {
    return realpathSync(p);
  } catch {
    return resolve(p);
  }
}

/**
 * Resolve a collection `path` from a project-local config: `~` expands, relative
 * paths are against the project root (parent of `.qmd`), absolute paths stay
 * absolute.
 */
export function resolveConfigCollectionPath(configPath: string, collectionPath: string): string {
  const expanded = expandUserPath(collectionPath.trim());
  if (isAbsolute(expanded)) return resolve(expanded);
  return resolve(projectRootFromConfigPath(configPath), expanded);
}

/** True if `target` is `dir` or a descendant. Lexical after realpath/resolve. */
function isInsideDir(dir: string, target: string): boolean {
  const rel = relative(dir, target);
  if (rel === "" || rel === ".") return true;
  if (isAbsolute(rel)) return false;
  return rel !== ".." && !rel.startsWith(`..${sep}`);
}

/**
 * Whether a collection path from this config stays inside the project.
 * Used to decide if the path needs a trust approval (#889).
 */
export function isCollectionPathInsideProject(configPath: string, collectionPath: string): boolean {
  if (!collectionPath.trim()) return true;
  const projectRoot = realOrResolve(projectRootFromConfigPath(configPath));
  const target = realOrResolve(resolveConfigCollectionPath(configPath, collectionPath));
  return isInsideDir(projectRoot, target);
}

function comparePair(a: [string, string], b: [string, string]): number {
  if (a[0] < b[0]) return -1;
  if (a[0] > b[0]) return 1;
  if (a[1] < b[1]) return -1;
  if (a[1] > b[1]) return 1;
  return 0;
}

/**
 * Stable digest over a hook set. Order-independent so that reordering
 * collections in the YAML does not invalidate an approval, while any change to
 * a command — or a new collection gaining one — does.
 */
export function hookDigest(hooks: UpdateHook[]): string {
  const canonical = JSON.stringify(
    hooks
      .map(h => [h.collection, h.command] as [string, string])
      .sort(comparePair),
  );
  return createHash("sha256").update(canonical).digest("hex");
}

/**
 * Digest over every gated surface. When the config only has hooks (no
 * out-of-project paths, no custom models) this equals `hookDigest`, so
 * approvals recorded before #889 still match.
 */
export function configDigest(gated: GatedConfig): string {
  const hasPaths = gated.paths.length > 0;
  const hasModels = !!(gated.models.embed || gated.models.rerank || gated.models.generate);
  if (!hasPaths && !hasModels) return hookDigest(gated.hooks);

  const canonical = JSON.stringify({
    hooks: gated.hooks
      .map(h => [h.collection, h.command] as [string, string])
      .sort(comparePair),
    paths: gated.paths
      .map(p => [p.collection, p.path] as [string, string])
      .sort(comparePair),
    models: {
      embed: gated.models.embed ?? "",
      rerank: gated.models.rerank ?? "",
      generate: gated.models.generate ?? "",
    },
  });
  return createHash("sha256").update(canonical).digest("hex");
}

export function hasGatedSurfaces(gated: GatedConfig): boolean {
  return gated.hooks.length > 0
    || gated.paths.length > 0
    || !!(gated.models.embed || gated.models.rerank || gated.models.generate);
}

/**
 * Collect the surfaces a project-local config must not apply unattended.
 * Global configs and the SDK `<inline>` sentinel yield empty path/model sets
 * (hooks are still returned so callers can display them).
 */
export function collectGatedSurfaces(
  configPath: string,
  config: {
    collections?: Record<string, { path?: string; update?: string }>;
    models?: GatedModels;
  },
  defaultModels: DefaultModels,
): GatedConfig {
  const hooks: UpdateHook[] = [];
  const paths: GatedPath[] = [];
  for (const [name, col] of Object.entries(config.collections ?? {})) {
    if (col.update) hooks.push({ collection: name, command: col.update });
    if (
      col.path
      && isLocalConfigPath(configPath)
      && !isCollectionPathInsideProject(configPath, col.path)
    ) {
      paths.push({ collection: name, path: col.path });
    }
  }

  const models: GatedModels = {};
  if (isLocalConfigPath(configPath) && config.models) {
    if (config.models.embed && config.models.embed !== defaultModels.embed) {
      models.embed = config.models.embed;
    }
    if (config.models.rerank && config.models.rerank !== defaultModels.rerank) {
      models.rerank = config.models.rerank;
    }
    if (config.models.generate && config.models.generate !== defaultModels.generate) {
      models.generate = config.models.generate;
    }
  }

  return { hooks, paths, models };
}

export function loadTrustStore(): TrustStore {
  const path = getTrustFilePath();
  if (!existsSync(path)) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed as TrustStore;
  } catch {
    // A corrupt trust file must not be treated as "everything is trusted".
    return {};
  }
}

function saveTrustStore(store: TrustStore): void {
  const path = getTrustFilePath();
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(path, `${JSON.stringify(store, null, 2)}\n`, "utf-8");
}

/** Trust records are keyed on the resolved config path. */
function trustKey(configPath: string): string {
  return resolve(configPath);
}

export function isTrusted(configPath: string, digest: string): boolean {
  const record = loadTrustStore()[trustKey(configPath)];
  return record?.hooks === digest;
}

export function recordTrust(configPath: string, digest: string): void {
  const store = loadTrustStore();
  store[trustKey(configPath)] = { hooks: digest, trustedAt: new Date().toISOString() };
  saveTrustStore(store);
}

/** Drop the record for a config path. Returns false if there was none. */
export function revokeTrust(configPath: string): boolean {
  const store = loadTrustStore();
  const key = trustKey(configPath);
  if (!(key in store)) return false;
  delete store[key];
  saveTrustStore(store);
  return true;
}

export function listTrusted(): Array<{ path: string } & TrustRecord> {
  return Object.entries(loadTrustStore()).map(([path, record]) => ({ path, ...record }));
}

export type HookGateDecision =
  | { action: "run"; digest: string }
  | { action: "prompt"; digest: string }
  | { action: "skip"; digest: string };

/**
 * Decide what to do with a config's gated surfaces (hooks, out-of-project
 * paths, custom model URIs).
 *
 * Non-interactive callers — agents, CI, the MCP server — get `skip` rather than
 * a hard failure: indexing is what they asked for, and failing the whole
 * command would only push people toward a blanket opt-out.
 */
export function decideConfigGate(options: {
  configPath: string;
  gated: GatedConfig;
  isInteractive: boolean;
  env?: NodeJS.ProcessEnv;
  trustedCheck?: (configPath: string, digest: string) => boolean;
}): HookGateDecision {
  const env = options.env ?? process.env;
  const digest = configDigest(options.gated);

  if (!hasGatedSurfaces(options.gated)) return { action: "run", digest };
  if (isTruthyEnv(env.QMD_TRUST_UPDATE_HOOKS)) return { action: "run", digest };
  if (!isLocalConfigPath(options.configPath)) return { action: "run", digest };

  const trusted = options.trustedCheck ?? isTrusted;
  if (trusted(options.configPath, digest)) return { action: "run", digest };

  return { action: options.isInteractive ? "prompt" : "skip", digest };
}

/**
 * Decide what to do with a config's `update:` hooks.
 *
 * Thin wrapper around `decideConfigGate` for the hooks-only surface, so the
 * #886 tests and any caller that already has a hook list keep working.
 */
export function decideHookGate(options: {
  configPath: string;
  hooks: UpdateHook[];
  isInteractive: boolean;
  env?: NodeJS.ProcessEnv;
  trustedCheck?: (configPath: string, digest: string) => boolean;
}): HookGateDecision {
  return decideConfigGate({
    configPath: options.configPath,
    gated: { hooks: options.hooks, paths: [], models: {} },
    isInteractive: options.isInteractive,
    env: options.env,
    trustedCheck: options.trustedCheck,
  });
}

function isTruthyEnv(value: string | undefined): boolean {
  if (!value) return false;
  return !["0", "false", "off", "no", "none"].includes(value.trim().toLowerCase());
}
