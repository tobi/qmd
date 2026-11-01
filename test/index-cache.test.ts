/** Index maintenance preserves model responses keyed by their actual inputs. */
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, Maintenance, type ExpandedQuery } from "../src/index.js";
import { qmdRunnerArgs } from "./qmd-runner.js";

let directory: string;

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), "qmd-index-cache-"));
});

afterEach(async () => {
  await rm(directory, { recursive: true, force: true });
});

describe("index model caches", () => {
  test.each([
    ["sdk", false], ["sdk", true], ["cli", false], ["cli", true], ["add", false],
  ] as const)("%s preserves cached expansions (changed=%s)", async (entrypoint, changed) => {
    const documents = join(directory, "documents");
    await mkdir(documents);
    await writeFile(join(documents, "note.md"), "# Authentication\n\nA scoped test document.\n");
    const configPath = join(directory, "index.yml");
    await writeFile(configPath, YAML.stringify({ collections: { documents: { path: documents, pattern: "**/*.md" } } }));
    const store = await createStore({ dbPath: join(directory, "index.sqlite"), configPath });
    try {
      await store.update();
      const llm = store.internal.llm;
      assert(llm);
      const query = "How does authentication work?";
      const expansion: ExpandedQuery[] = [{ type: "vec", query: "authentication overview" }];
      const cached = JSON.stringify(expansion);
      const key = store.internal.getCacheKey("expandQuery", { query, model: llm.generateModelName });
      store.internal.setCachedResult(key, cached);
      llm.expandQuery = async () => { throw new Error("A valid cached expansion must not invoke generation"); };
      const rerankCalls: string[][] = [];
      llm.rerank = async (_query, docs) => {
        rerankCalls.push(docs.map(doc => doc.text));
        return {
          model: llm.rerankModelName,
          results: docs.map((doc, index) => ({ file: doc.file, index, score: 0.8 })),
        };
      };
      const original = [{ file: "note.md", text: "A scoped test document." }];
      expect(await store.internal.rerank(query, original)).toEqual([{ file: "note.md", score: 0.8 }]);

      if (changed) {
        await writeFile(join(documents, "note.md"), "# Authentication\n\nThe source content changed.\n");
      }
      if (entrypoint === "sdk") {
        const result = await store.update();
        expect(result.updated).toBe(changed ? 1 : 0);
        expect(result.unchanged).toBe(changed ? 0 : 1);
      } else {
        let args = ["update"];
        if (entrypoint === "add") {
          const additional = join(directory, "additional");
          await mkdir(additional);
          await writeFile(join(additional, "more.md"), "# More evidence\n\nAnother collection.\n");
          args = ["collection", "add", additional, "--name", "additional"];
        }
        const runner = qmdRunnerArgs(args);
        execFileSync(runner.command, runner.args, {
          cwd: directory, timeout: 10_000, encoding: "utf8",
          env: { ...process.env, CI: "true", INDEX_PATH: store.dbPath, QMD_CONFIG_DIR: directory, XDG_CACHE_HOME: join(directory, "cache"), PWD: directory },
        });
      }

      expect(store.internal.getCachedResult(key)).toBe(cached);
      expect(await store.expandQuery(query)).toEqual(expansion);
      expect(await store.internal.rerank(query, original)).toEqual([{ file: "note.md", score: 0.8 }]);
      expect(rerankCalls).toEqual([original.map(doc => doc.text)]);
      const edited = [{ file: "note.md", text: "The source content changed." }];
      expect(await store.internal.rerank(query, edited)).toEqual([{ file: "note.md", score: 0.8 }]);
      expect(rerankCalls).toEqual([original.map(doc => doc.text), edited.map(doc => doc.text)]);
      expect(new Maintenance(store.internal).clearLLMCache()).toBeGreaterThan(0);
      expect(store.internal.getCachedResult(key)).toBeNull();
    } finally {
      await store.close();
    }
  });

  test("fallback reranking is retried instead of cached as a real model score", async () => {
    const store = await createStore({ dbPath: join(directory, "index.sqlite"), config: { collections: {} } });
    try {
      const llm = store.internal.llm;
      assert(llm);
      let calls = 0;
      llm.rerank = async (_query, docs) => {
        calls++;
        return {
          model: calls === 1 ? "fallback" : llm.rerankModelName,
          results: docs.map((doc, index) => ({ file: doc.file, index, score: calls === 1 ? 0.5 : 0.9 })),
        };
      };
      const docs = [{ file: "note.md", text: "A relevant passage." }];
      expect(await store.internal.rerank("relevance", docs)).toEqual([{ file: "note.md", score: 0.5 }]);
      expect(await store.internal.rerank("relevance", docs)).toEqual([{ file: "note.md", score: 0.9 }]);
      expect(await store.internal.rerank("relevance", docs)).toEqual([{ file: "note.md", score: 0.9 }]);
      expect(calls).toBe(2);
    } finally {
      await store.close();
    }
  });
});
