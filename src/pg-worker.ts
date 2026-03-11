/**
 * pg-worker.ts — Worker thread that manages a PostgreSQL connection pool.
 *
 * Runs inside a worker_threads Worker. The main thread sends query messages
 * and blocks on a SharedArrayBuffer using Atomics. This worker executes the
 * async postgres query, writes the result to the message port, then
 * signals the main thread via Atomics.notify().
 *
 * Protocol:
 *   Main → Worker: { type, query, params }
 *   Worker → Main: { result, error }
 *   Worker: Atomics.store(sharedInt32, 0, 1); Atomics.notify(sharedInt32, 0)
 */

import { workerData } from 'node:worker_threads';
import postgres from 'postgres';

const pgUrl: string = workerData.pgUrl;
const sharedBuffer: SharedArrayBuffer = workerData.sharedBuffer;
const port = workerData.port;

const sharedInt32 = new Int32Array(sharedBuffer);

// Single connection — one query at a time (matching synchronous caller semantics)
const sql = postgres(pgUrl, {
  max: 1,
  idle_timeout: 60,
  connect_timeout: 10,
  // Parse int8 (bigint) as regular JS numbers to match SQLite behavior
  types: {
    bigint: {
      to: 20,
      from: [20],
      serialize: (x: bigint | number | string) => String(x),
      parse: (x: string) => Number(x),
    },
  },
});

/**
 * Convert BigInt values in a row to Number to ensure postMessage
 * compatibility and match SQLite's numeric behavior.
 */
function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(row)) {
    out[k] = typeof v === 'bigint' ? Number(v) : v;
  }
  return out;
}

function normalizeRows(rows: readonly Record<string, unknown>[]): Record<string, unknown>[] {
  return rows.map(normalizeRow);
}

type QueryMessage = {
  type: 'exec' | 'run' | 'get' | 'all' | 'close';
  query: string;
  params: unknown[];
};

port.on('message', async (msg: QueryMessage) => {
  const { type, query, params } = msg;
  let result: unknown = null;
  let error: string | null = null;

  try {
    if (type === 'close') {
      await sql.end({ timeout: 5 });
      result = null;
    } else if (type === 'exec') {
      await sql.unsafe(query, []);
      result = { changes: 0, lastInsertRowid: 0 };
    } else if (type === 'run') {
      const rows = await sql.unsafe(query, params as postgres.ParameterOrJSON<never>[]);
      result = {
        changes: (rows as unknown as { count: number }).count ?? 0,
        lastInsertRowid: 0,
      };
    } else if (type === 'get') {
      const rows = await sql.unsafe(query, params as postgres.ParameterOrJSON<never>[]);
      result = rows.length > 0 ? normalizeRow(rows[0] as Record<string, unknown>) : null;
    } else if (type === 'all') {
      const rows = await sql.unsafe(query, params as postgres.ParameterOrJSON<never>[]);
      result = normalizeRows(rows as readonly Record<string, unknown>[]);
    }
  } catch (err: unknown) {
    error = err instanceof Error ? err.message : String(err);
    if (type !== 'close') {
      console.error('[pg-worker] query error:', error, '\nSQL:', query, '\nParams:', params);
    }
  }

  // Post result before signalling so main thread can receiveMessageOnPort
  port.postMessage({ result, error });

  // Signal main thread that result is ready
  Atomics.store(sharedInt32, 0, 1);
  Atomics.notify(sharedInt32, 0, 1);
});
