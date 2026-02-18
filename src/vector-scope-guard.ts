import type { Database } from "./db.js";
import {
  DEFAULT_API_EMBED_BASE_URL,
  DEFAULT_API_EMBED_MODEL,
} from "./api-defaults.js";

export type ApiEmbeddingScope = {
  embedBaseUrl: string;
  embedModel: string;
};

function getConfiguredBackend(): string {
  return process.env.QMD_LLM_BACKEND?.trim().toLowerCase() || "local";
}

function resolveCurrentApiEmbeddingScopeFromEnv(): ApiEmbeddingScope {
  const embedBaseUrl = (
    process.env.QMD_EMBED_BASE_URL?.trim()
    || DEFAULT_API_EMBED_BASE_URL
  ).replace(/\/+$/, "");
  const embedModel = process.env.QMD_EMBED_MODEL?.trim() || DEFAULT_API_EMBED_MODEL;
  return { embedBaseUrl, embedModel };
}

function getApiMetaValue(db: Database, key: string): string | null {
  try {
    const row = db.prepare(`SELECT value FROM api_meta WHERE key = ?`).get(key) as { value: string } | null;
    return row?.value || null;
  } catch {
    // Older DBs or test fixtures may not include api_meta.
    return null;
  }
}

function setApiMetaValue(db: Database, key: string, value: string): void {
  db.prepare(`INSERT OR REPLACE INTO api_meta (key, value) VALUES (?, ?)`).run(key, value);
}

function hasAnyVectors(db: Database): boolean {
  const cvCount = db.prepare(`SELECT COUNT(*) as c FROM content_vectors`).get() as { c: number };
  if (cvCount.c > 0) return true;

  const tableExists = db.prepare(`
    SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'
  `).get();
  if (!tableExists) return false;

  try {
    const vvCount = db.prepare(`SELECT COUNT(*) as c FROM vectors_vec`).get() as { c: number };
    return vvCount.c > 0;
  } catch {
    // If vec table exists but count fails, treat as non-empty/unknown for safety.
    return true;
  }
}

function formatApiScope(scope: ApiEmbeddingScope): string {
  return `${scope.embedBaseUrl} | ${scope.embedModel}`;
}

export function getStoredApiEmbeddingScope(db: Database): ApiEmbeddingScope | null {
  const embedBaseUrl = getApiMetaValue(db, "embed_base_url");
  const embedModel = getApiMetaValue(db, "embed_model");
  if (!embedBaseUrl || !embedModel) return null;
  return { embedBaseUrl, embedModel };
}

export function setApiEmbeddingScopeFromCurrentEnv(db: Database): void {
  const scope = resolveCurrentApiEmbeddingScopeFromEnv();
  setApiMetaValue(db, "embed_base_url", scope.embedBaseUrl);
  setApiMetaValue(db, "embed_model", scope.embedModel);
}

export function clearApiEmbeddingScope(db: Database): void {
  db.exec(`DELETE FROM api_meta`);
}

export function getVectorScopeGuardMessage(db: Database): string | null {
  const backend = getConfiguredBackend();
  const storedScope = getStoredApiEmbeddingScope(db);

  if (backend === "local") {
    if (!storedScope) return null;
    return [
      "Index is marked for API embeddings, but current backend is local.",
      `Stored API embedding scope: ${formatApiScope(storedScope)}`,
      "Choose one:",
      "  1) Set QMD_LLM_BACKEND=api with matching embedding settings",
      "  2) Use a different index via --index",
      "  3) Run 'qmd embed -f' to clear vectors and remove API scope metadata",
    ].join("\n");
  }

  if (backend === "api") {
    const currentScope = resolveCurrentApiEmbeddingScopeFromEnv();

    if (!storedScope) {
      if (!hasAnyVectors(db)) return null;
      return [
        "This index has vectors but no API scope metadata (legacy/ambiguous state).",
        "Choose one:",
        "  1) Use a different index via --index",
        "  2) Run 'qmd embed -f' to reset vectors for the current API embedding scope",
      ].join("\n");
    }

    const isMatch = storedScope.embedBaseUrl === currentScope.embedBaseUrl
      && storedScope.embedModel === currentScope.embedModel;
    if (isMatch) return null;

    return [
      "API embedding scope mismatch for this index.",
      `Stored scope (in index db):  ${formatApiScope(storedScope)}`,
      `Current scope (from environment): ${formatApiScope(currentScope)}`,
      "Choose one:",
      "  1) Revert API embedding settings to match the stored scope",
      "  2) Use a different index via --index",
      "  3) Run 'qmd embed -f' to reset vectors for the current API embedding scope",
    ].join("\n");
  }

  // Unknown backend values are validated elsewhere; don't block here.
  return null;
}
