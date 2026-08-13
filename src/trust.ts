/**
 * trust.ts — approval gate for `update:` hooks from project-local config.
 *
 * A collection's `update:` field is a shell command run by `qmd update`. That
 * is fine when the command came from the user's own global config, which only
 * `qmd collection update-cmd` writes. It is not fine for a project-local
 * `.qmd/index.yml`: that file arrives with a `git clone`, and `findLocalConfigPath`
 * adopts it automatically for any command run anywhere inside the tree. Without
 * a gate, cloning a repository and running `qmd update` executes shell commands
 * chosen by whoever wrote the repo (#886).
 *
 * The model is the one direnv, VS Code and `git safe.directory` converged on:
 * approvals are per config file and per command set, recorded in
 * `<config dir>/trusted.json`. Editing a hook — or a `git pull` that rewrites
 * one — changes the digest and re-arms the gate, so an approval cannot be
 * silently widened after the fact.
 */

import { createHash } from "crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { basename, dirname, join, resolve } from "path";
import { getConfigDir } from "./collections.js";

/** A collection's pre-update hook, as it will be executed. */
export type UpdateHook = {
  collection: string;
  command: string;
};

export type TrustRecord = {
  /** Digest of the hook set that was approved. */
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

/**
 * Stable digest over a hook set. Order-independent so that reordering
 * collections in the YAML does not invalidate an approval, while any change to
 * a command — or a new collection gaining one — does.
 */
export function hookDigest(hooks: UpdateHook[]): string {
  const canonical = JSON.stringify(
    hooks
      .map(h => [h.collection, h.command])
      .sort((a, b) => (a[0]! < b[0]! ? -1 : a[0]! > b[0]! ? 1 : 0)),
  );
  return createHash("sha256").update(canonical).digest("hex");
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
  if (isTruthyEnv(env.QMD_TRUST_UPDATE_HOOKS)) return { action: "run", digest };
  if (!isLocalConfigPath(options.configPath)) return { action: "run", digest };

  const trusted = options.trustedCheck ?? isTrusted;
  if (trusted(options.configPath, digest)) return { action: "run", digest };

  return { action: options.isInteractive ? "prompt" : "skip", digest };
}

function isTruthyEnv(value: string | undefined): boolean {
  if (!value) return false;
  return !["0", "false", "off", "no", "none"].includes(value.trim().toLowerCase());
}
