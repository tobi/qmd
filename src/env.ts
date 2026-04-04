/**
 * env.ts - Load QMD config from ~/.config/qmd/.env
 *
 * Two-tier precedence:
 *   QMD_* vars  → .env file always wins (overrides stale parent process vars)
 *   all others  → inherited environment wins (standard dotenv behaviour)
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

let _loaded = false;

/**
 * Returns the QMD config directory:
 *   $QMD_CONFIG_DIR  →  $XDG_CONFIG_HOME/qmd  →  ~/.config/qmd
 */
export function getQmdConfigDir(): string {
  return (
    process.env.QMD_CONFIG_DIR ||
    (process.env.XDG_CONFIG_HOME ? join(process.env.XDG_CONFIG_HOME, "qmd") : null) ||
    join(homedir(), ".config", "qmd")
  );
}

/**
 * Load ~/.config/qmd/.env (or $QMD_CONFIG_DIR/.env) into process.env.
 * Idempotent — safe to call multiple times; only reads the file once.
 */
export function loadQmdEnv(): void {
  if (_loaded) return;
  _loaded = true;

  const envPath = join(getQmdConfigDir(), ".env");
  if (!existsSync(envPath)) return;

  let content: string;
  try {
    content = readFileSync(envPath, "utf-8");
  } catch {
    return;
  }

  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
    if (!key) continue;
    if (key.startsWith("QMD_")) {
      // QMD's own config: .env is the source of truth, always override
      process.env[key] = val;
    } else if (!process.env[key]) {
      // Non-QMD vars: only set if not already present (standard dotenv)
      process.env[key] = val;
    }
  }
}
