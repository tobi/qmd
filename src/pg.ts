/**
 * pg.ts — PostgreSQL adapter implementing the Database/Statement interfaces.
 *
 * Uses a Worker thread to run async postgres queries and Atomics.wait() to
 * block the caller, exposing a synchronous API compatible with SQLite adapters.
 *
 * SQL differences handled here:
 *   - ? placeholders -> $1, $2, ...
 *   - Float32Array params -> pgvector literal '[f1,f2,...]'
 *   - loadExtension() is a no-op
 */

import {
  MessageChannel,
  receiveMessageOnPort,
  type MessagePort,
  Worker,
} from "node:worker_threads";
import { fileURLToPath } from "node:url";
import type { Database, Statement } from "./db.js";

type QueryType = "exec" | "run" | "get" | "all" | "close";

type WorkerResponse = {
  result: unknown;
  error: string | null;
};

/**
 * Translate SQLite-style `?` placeholders to PostgreSQL `$N` placeholders.
 * Skips placeholders inside SQL string literals.
 */
function translatePlaceholders(sql: string): string {
  let i = 0;
  let index = 0;
  let out = "";

  while (i < sql.length) {
    const ch = sql[i]!;

    if (ch === "'") {
      out += ch;
      i++;
      while (i < sql.length) {
        const sch = sql[i]!;
        out += sch;
        if (sch === "'") {
          if (sql[i + 1] === "'") {
            out += "'";
            i += 2;
          } else {
            i++;
            break;
          }
        } else {
          i++;
        }
      }
      continue;
    }

    if (ch === "?") {
      index += 1;
      out += `$${index}`;
      i++;
      continue;
    }

    out += ch;
    i++;
  }

  return out;
}

/**
 * Convert Float32Array params to pgvector text literal.
 */
function convertParams(params: unknown[]): unknown[] {
  return params.map((param) => {
    if (param instanceof Float32Array) {
      return `[${Array.from(param).join(",")}]`;
    }
    return param;
  });
}

function resolveWorkerPath(): string {
  const thisFile = fileURLToPath(import.meta.url);
  const workerFile = thisFile.endsWith(".ts") ? "pg-worker.ts" : "pg-worker.js";
  return fileURLToPath(new URL(`./${workerFile}`, import.meta.url));
}

class PgStatement implements Statement {
  constructor(
    private readonly db: PgDatabase,
    private readonly sql: string,
  ) {}

  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint } {
    return this.db.syncQuery("run", this.sql, params) as {
      changes: number;
      lastInsertRowid: number | bigint;
    };
  }

  get(...params: unknown[]): unknown {
    return this.db.syncQuery("get", this.sql, params);
  }

  all(...params: unknown[]): unknown[] {
    return this.db.syncQuery("all", this.sql, params) as unknown[];
  }
}

export class PgDatabase implements Database {
  private readonly worker: Worker;
  private readonly port: MessagePort;
  private readonly waitState: Int32Array;

  constructor(url: string) {
    const sharedBuffer = new SharedArrayBuffer(4);
    this.waitState = new Int32Array(sharedBuffer);

    const { port1, port2 } = new MessageChannel();
    this.port = port1;

    const workerPath = resolveWorkerPath();
    const isBunRuntime = typeof (globalThis as Record<string, unknown>).Bun !== "undefined";
    const workerNeedsTsx = !isBunRuntime && workerPath.endsWith(".ts");

    this.worker = new Worker(workerPath, {
      workerData: {
        pgUrl: url,
        sharedBuffer,
        port: port2,
      },
      transferList: [port2],
      execArgv: workerNeedsTsx ? ["--import", "tsx/esm"] : [],
    });
  }

  syncQuery(type: QueryType, query: string, params: unknown[]): unknown {
    Atomics.store(this.waitState, 0, 0);

    this.port.postMessage({
      type,
      query,
      params: convertParams(params),
    });

    Atomics.wait(this.waitState, 0, 0);

    const response = receiveMessageOnPort(this.port);
    if (!response?.message) {
      throw new Error("[PgDatabase] no response from postgres worker");
    }

    const payload = response.message as WorkerResponse;
    if (payload.error) {
      throw new Error(payload.error);
    }

    return payload.result;
  }

  exec(sql: string): void {
    this.syncQuery("exec", sql, []);
  }

  prepare(sql: string): Statement {
    return new PgStatement(this, translatePlaceholders(sql));
  }

  // PostgreSQL extensions are managed by CREATE EXTENSION.
  loadExtension(_path: string): void {}

  close(): void {
    try {
      this.syncQuery("close", "", []);
    } catch {
      // Ignore close errors (worker may already be terminating).
    }
    void this.worker.terminate();
  }
}

export function openPgDatabase(url: string): Database {
  return new PgDatabase(url);
}
