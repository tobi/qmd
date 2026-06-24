import { spawn } from "node:child_process";

type MaybePromise<T> = T | Promise<T>;

/**
 * Stable reference to a document already known to QMD.
 *
 * Custom lexical backends should return at least one of these identifiers so
 * QMD can hydrate the hit from its SQLite store before snippets, contexts, RRF,
 * and reranking run.
 */
export type LexicalDocumentRef = {
  /** `documents.id` in QMD's SQLite database. */
  documentId?: number;
  /** QMD content hash. */
  hash?: string;
  /** Short content hash prefix accepted by QMD, with or without leading `#`. */
  docid?: string;
  /** Full virtual path, e.g. `qmd://docs/readme.md`. */
  filepath?: string;
  /** Collection name plus collection-relative path. */
  collectionName?: string;
  path?: string;
};

export type LexicalSearchHit = LexicalDocumentRef & {
  /** Higher is better. QMD preserves ordering returned by the backend. */
  score: number;
  /** Optional unnormalized backend-native score for diagnostics. */
  rawScore?: number;
  /** Backend-specific metadata. QMD does not interpret this object. */
  metadata?: Record<string, unknown>;
};

export type LexicalSearchRequest = {
  query: string;
  limit: number;
  collectionName?: string;
  /** Absolute path to QMD's SQLite database. */
  dbPath: string;
};

export type LexicalSearchBackendContext = {
  /** Absolute path to QMD's SQLite database. */
  dbPath: string;
};

export interface LexicalSearchBackend {
  /**
   * Stable backend identifier. It is surfaced in SearchResult.lexicalBackend
   * for observability, while SearchResult.source remains "fts" for pipeline
   * compatibility.
   */
  name: string;
  search(
    request: LexicalSearchRequest,
    context: LexicalSearchBackendContext
  ): MaybePromise<LexicalSearchHit[]>;
}

export type ExternalCommandLexicalBackendOptions = {
  name?: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  timeoutMs?: number;
};

export type LexicalBackendConfig =
  | { type?: "sqlite-fts5" }
  | ({ type: "command" } & ExternalCommandLexicalBackendOptions);

export function isSqliteFts5LexicalBackendConfig(config: LexicalBackendConfig | undefined): boolean {
  return !config || !config.type || config.type === "sqlite-fts5";
}

export function createExternalCommandLexicalBackend(
  options: ExternalCommandLexicalBackendOptions
): LexicalSearchBackend {
  return {
    name: options.name ?? "external-command",
    search: async (request) => {
      const stdout = await runBackendCommand(options, request);
      const parsed = JSON.parse(stdout) as unknown;
      const hits = Array.isArray(parsed)
        ? parsed
        : isRecord(parsed) && Array.isArray(parsed.hits)
          ? parsed.hits
          : null;
      if (!hits) {
        throw new Error("Lexical backend command must return a JSON array or an object with a hits array");
      }
      return hits.map(parseHit);
    },
  };
}

export function createLexicalSearchBackendFromConfig(
  config: LexicalBackendConfig | undefined
): LexicalSearchBackend | undefined {
  if (!config || !config.type || config.type === "sqlite-fts5") return undefined;
  switch (config.type) {
    case "command":
      return createExternalCommandLexicalBackend(config);
    default:
      throw new Error(`Unsupported lexical backend config: ${JSON.stringify(config)}`);
  }
}

function runBackendCommand(
  options: ExternalCommandLexicalBackendOptions,
  request: LexicalSearchRequest
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(options.command, options.args ?? [], {
      env: { ...process.env, ...options.env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = options.timeoutMs
      ? setTimeout(() => {
        settled = true;
        child.kill("SIGTERM");
        reject(new Error(`Lexical backend command timed out after ${options.timeoutMs}ms`));
      }, options.timeoutMs)
      : undefined;

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", error => {
      if (timer) clearTimeout(timer);
      if (!settled) reject(error);
    });
    child.on("close", code => {
      if (timer) clearTimeout(timer);
      if (settled) return;
      if (code !== 0) {
        reject(new Error(`Lexical backend command exited with ${code}: ${stderr.trim()}`));
        return;
      }
      resolve(stdout);
    });
    child.stdin.end(JSON.stringify(request));
  });
}

function parseHit(hit: unknown): LexicalSearchHit {
  if (!isRecord(hit)) {
    throw new Error("Lexical backend hit must be an object");
  }
  const score = Number(hit.score);
  if (!Number.isFinite(score)) {
    throw new Error("Lexical backend hit must include a finite numeric score");
  }
  return {
    documentId: typeof hit.documentId === "number" ? hit.documentId : undefined,
    hash: typeof hit.hash === "string" ? hit.hash : undefined,
    docid: typeof hit.docid === "string" ? hit.docid : undefined,
    filepath: typeof hit.filepath === "string" ? hit.filepath : undefined,
    collectionName: typeof hit.collectionName === "string" ? hit.collectionName : undefined,
    path: typeof hit.path === "string" ? hit.path : undefined,
    score,
    rawScore: typeof hit.rawScore === "number" ? hit.rawScore : undefined,
    metadata: isRecord(hit.metadata) ? hit.metadata : undefined,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
