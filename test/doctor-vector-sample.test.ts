import { describe, expect, test } from "vitest";

import { selectDoctorEmbeddingVectorSamples } from "../src/cli/qmd.ts";
import { openDatabase, type Database } from "../src/db.ts";

const MODEL = "test-embed-model";
const FINGERPRINT = "test-fingerprint";
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

function createSamplingSchema(db: Database, contentTableSql: string = `
  CREATE TABLE content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL
  );
`): void {
  db.exec(`
    ${contentTableSql}
    CREATE TABLE documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      hash TEXT NOT NULL,
      path TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX idx_documents_hash ON documents(hash);
    CREATE TABLE content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL,
      model TEXT NOT NULL,
      embed_fingerprint TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
    );
    CREATE INDEX idx_content_vectors_model_fingerprint
      ON content_vectors(model, embed_fingerprint, hash);
  `);
}

function insertSample(
  db: Database,
  {
    hash,
    seq = 0,
    body,
    path,
    active = 1,
    model = MODEL,
    fingerprint = FINGERPRINT,
  }: {
    hash: string;
    seq?: number;
    body: string;
    path: string;
    active?: number;
    model?: string;
    fingerprint?: string;
  },
  contentTable: string = "content",
): void {
  db.prepare(`INSERT INTO ${contentTable} (hash, doc) VALUES (?, ?)`).run(hash, body);
  db.prepare(`INSERT INTO documents (hash, path, active) VALUES (?, ?, ?)`).run(hash, path, active);
  db.prepare(`
    INSERT INTO content_vectors (hash, seq, model, embed_fingerprint)
    VALUES (?, ?, ?, ?)
  `).run(hash, seq, model, fingerprint);
}

describe("doctor embedding vector sampling", () => {
  test("preserves active-document, model, fingerprint, and minimum-path selection", () => {
    const db = openDatabase(":memory:");
    try {
      createSamplingSchema(db);
      insertSample(db, { hash: "active-a", body: "body a", path: "z.md" });
      db.prepare(`INSERT INTO documents (hash, path, active) VALUES (?, ?, 1)`).run("active-a", "a.md");
      insertSample(db, { hash: "active-b", seq: 1, body: "body b", path: "b.md" });
      insertSample(db, { hash: "inactive", body: "inactive body", path: "inactive.md", active: 0 });
      insertSample(db, { hash: "wrong-model", body: "wrong model", path: "wrong-model.md", model: "other-model" });
      insertSample(db, { hash: "wrong-fingerprint", body: "wrong fingerprint", path: "wrong-fingerprint.md", fingerprint: "other-fingerprint" });

      const samples = selectDoctorEmbeddingVectorSamples(db, MODEL, FINGERPRINT, 10)
        .sort((a, b) => a.hash.localeCompare(b.hash));

      expect(samples).toEqual([
        { hash: "active-a", seq: 0, body: "body a", path: "a.md" },
        { hash: "active-b", seq: 1, body: "body b", path: "b.md" },
      ]);
    } finally {
      db.close();
    }
  });

  test.skipIf(isBunRuntime)("reads document bodies only after limiting sample identifiers", () => {
    const db = openDatabase(":memory:") as Database & {
      function(name: string, fn: (value: string) => string): void;
    };
    let bodyReads = 0;
    db.function("observe_body", (value: string) => {
      bodyReads += 1;
      return value;
    });

    try {
      createSamplingSchema(db, `
        CREATE TABLE raw_content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL
        );
        CREATE VIEW content AS
          SELECT hash, observe_body(doc) AS doc
          FROM raw_content;
      `);
      for (let index = 0; index < 40; index += 1) {
        insertSample(db, {
          hash: `hash-${index.toString().padStart(2, "0")}`,
          body: `body-${index}`,
          path: `doc-${index}.md`,
        }, "raw_content");
      }

      const samples = selectDoctorEmbeddingVectorSamples(db, MODEL, FINGERPRINT, 3);

      expect(samples).toHaveLength(3);
      expect(bodyReads).toBe(samples.length);
    } finally {
      db.close();
    }
  });
});
