/**
 * store-cjk-normalization.test.ts - CJK text indexed by the sync pipeline
 * must be per-character normalized in documents_fts so the query-side CJK
 * phrase builder (buildFTS5Query → sanitizeFTS5Phrase) can match it.
 *
 * Regression context: the sync triggers (documents_ai / documents_au) insert
 * the raw content doc, and insertDocument relies on rebuildDocumentFTS to
 * replace that row with the normalized text. If that replacement is skipped
 * or runs before the content row is visible, CJK queries return zero hits
 * while the raw run tokens are searchable (tobi/qmd#617 follow-up).
 */

import { describe, test, expect } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  searchFTS,
  type Store,
} from "../src/store.js";
import { openDatabase } from "../src/db.js";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import YAML from "yaml";

const CJK_BODY = [
  "# 报文处理",
  "",
  "手工发报支持 pacs.008 报文，支付取消走 camt.056。",
].join("\n");

function ftsCount(store: Store, match: string): number {
  const row = store.db
    .prepare(`SELECT count(*) AS c FROM documents_fts WHERE documents_fts MATCH ?`)
    .get(match) as { c: number };
  return row.c;
}

describe("CJK documents are searchable as character phrases", () => {
  test("insertDocument replaces the raw trigger row with a normalized FTS row", async () => {
    const store = createStore(
      `:memory:`,
    );
    const now = new Date().toISOString();
    const hash = await hashContent(CJK_BODY);
    insertContent(store.db, hash, CJK_BODY, now);
    insertDocument(store.db, "probes", "cjk/baowen.md", "报文处理", hash, now, now);

    // Normalized (per-character) text must be searchable through the exact
    // phrase form buildFTS5Query emits for CJK runs.
    expect(ftsCount(store, `"报 文"`)).toBeGreaterThan(0);
    expect(ftsCount(store, `"支 付 取 消"`)).toBeGreaterThan(0);

    // Raw run tokens must NOT remain in the index.
    expect(ftsCount(store, `"报文"`)).toBe(0);

    // End-to-end: the shipped lexical query path finds the document.
    const hits = searchFTS(store.db, "报文处理", 10);
    expect(hits.length).toBeGreaterThan(0);
  });

  test("collection add keeps CJK normalized through the real CLI pipeline", async () => {
    const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
    const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
    const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
    const base = await mkdtemp(join(tmpdir(), "qmd-cjk-cli-"));
    try {
      const collectionDir = join(base, "corpus");
      await mkdir(collectionDir, { recursive: true });
      await writeFile(join(collectionDir, "probe.md"), CJK_BODY, "utf-8");
      const configDir = join(base, "config");
      await mkdir(configDir, { recursive: true });
      const dbPath = join(base, "index.sqlite");
      await writeFile(join(configDir, "index.yml"), "collections: {}\n");

      const runQmd = (args: string[]): Promise<number> => {
        const proc2 = spawn(process.execPath, [tsxCli, qmdScript, ...args], {
          cwd: collectionDir,
          env: { ...process.env, INDEX_PATH: dbPath, QMD_CONFIG_DIR: configDir, CI: "true", QMD_DOCTOR_DEVICE_PROBE: "0" },
          stdio: ["ignore", "pipe", "pipe"],
        });
        let err = "";
        proc2.stderr?.on("data", (c: Buffer) => { err += c.toString(); });
        return new Promise<number>((res, rej) => {
          proc2.once("error", rej);
          proc2.on("close", (code) => {
            if (code !== 0) console.log(`QMD-ERR ${args.join(" ")}:\n${err}`);
            res(code ?? 1);
          });
        });
      };

      expect(await runQmd(["collection", "add", collectionDir, "--name", "cjk"])).toBe(0);

      // Probe AFTER plain add, BEFORE rename: proves the plain-add path
      // normalizes on its own, so the later rename sweep cannot mask a
      // plain-add defect.
      const probe = (): void => {
        const db = openDatabase(dbPath);
        try {
          const count = (match: string): number => {
            const row = db.prepare(`SELECT count(*) AS c FROM documents_fts WHERE documents_fts MATCH ?`).get(match) as { c: number };
            return row.c;
          };
          expect(count(`"报 文"`)).toBeGreaterThan(0);
          expect(count(`"报文"`)).toBe(0);
        } finally {
          db.close();
        }
      };
      probe();

      // The IBS refresh flow does exactly this: add, then rename the
      // collection to its final name. renameCollection UPDATEs documents
      // rows directly, which fires documents_au and rewrites FTS rows from
      // the raw content doc.
      expect(await runQmd(["collection", "rename", "cjk", "cjk2"])).toBe(0);

      // Probe AFTER rename: the sweep must have restored normalization.
      probe();
    } finally {
      await rm(base, { recursive: true, force: true });
    }
  });
});
