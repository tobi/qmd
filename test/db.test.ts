/**
 * db.test.ts - openDatabase configuration
 */

import { describe, test, expect, afterEach } from "vitest";
import { openDatabase } from "../src/db.js";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULT_BUSY_TIMEOUT_MS = 120_000;

function readBusyTimeout(db: ReturnType<typeof openDatabase>): number {
  const row = db.prepare("PRAGMA busy_timeout").get() as Record<string, number>;
  const value = Object.values(row)[0];
  return typeof value === "number" ? value : Number(value);
}

describe("openDatabase", () => {
  const originalEnv = process.env.QMD_SQLITE_BUSY_TIMEOUT;
  afterEach(() => {
    if (originalEnv === undefined) delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
    else process.env.QMD_SQLITE_BUSY_TIMEOUT = originalEnv;
  });

  test("sets the default busy_timeout so concurrent writers wait for the lock", () => {
    delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
    const db = openDatabase(":memory:");
    try {
      expect(readBusyTimeout(db)).toBe(DEFAULT_BUSY_TIMEOUT_MS);
    } finally {
      db.close();
    }
  });

  test("applies the busy_timeout to each independently opened connection", async () => {
    delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
    const dir = await mkdtemp(join(tmpdir(), "qmd-busy-"));
    const dbPath = join(dir, "shared.sqlite");
    try {
      const a = openDatabase(dbPath);
      const b = openDatabase(dbPath);
      try {
        expect(readBusyTimeout(a)).toBe(DEFAULT_BUSY_TIMEOUT_MS);
        expect(readBusyTimeout(b)).toBe(DEFAULT_BUSY_TIMEOUT_MS);
      } finally {
        a.close();
        b.close();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("QMD_SQLITE_BUSY_TIMEOUT overrides the default", () => {
    process.env.QMD_SQLITE_BUSY_TIMEOUT = "750";
    const db = openDatabase(":memory:");
    try {
      expect(readBusyTimeout(db)).toBe(750);
    } finally {
      db.close();
    }
  });

  test("QMD_SQLITE_BUSY_TIMEOUT=0 restores fail-fast", () => {
    process.env.QMD_SQLITE_BUSY_TIMEOUT = "0";
    const db = openDatabase(":memory:");
    try {
      expect(readBusyTimeout(db)).toBe(0);
    } finally {
      db.close();
    }
  });

  test("ignores unparseable QMD_SQLITE_BUSY_TIMEOUT and falls back to the default", () => {
    process.env.QMD_SQLITE_BUSY_TIMEOUT = "not-a-number";
    const db = openDatabase(":memory:");
    try {
      expect(readBusyTimeout(db)).toBe(DEFAULT_BUSY_TIMEOUT_MS);
    } finally {
      db.close();
    }
  });

  test("SQLite honors the configured busy_timeout when another connection holds the write lock", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-busy-"));
    const dbPath = join(dir, "contention.sqlite");
    try {
      const setup = openDatabase(dbPath);
      setup.exec("PRAGMA journal_mode = WAL");
      setup.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)");
      setup.close();

      const holder = openDatabase(dbPath);
      const waiter = openDatabase(dbPath);
      try {
        // The synchronous SQLite API blocks the thread while it waits for the
        // lock, so the test can't release the holder mid-wait. Shorten the
        // waiter's timeout so the test finishes quickly; openDatabase already
        // proved (above) that the default is the full 120_000ms.
        waiter.exec("PRAGMA busy_timeout = 250");

        holder.exec("BEGIN IMMEDIATE");
        holder.prepare("INSERT INTO t (v) VALUES ('holder')").run();

        const start = Date.now();
        let threw: unknown = null;
        try {
          waiter.exec("BEGIN IMMEDIATE");
        } catch (err) {
          threw = err;
        }
        const elapsed = Date.now() - start;

        expect(threw).toBeTruthy();
        expect(elapsed).toBeGreaterThanOrEqual(200);
        expect(elapsed).toBeLessThan(2000);

        holder.exec("ROLLBACK");
      } finally {
        holder.close();
        waiter.close();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
