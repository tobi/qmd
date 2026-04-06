/**
 * Integration tests for AST-aware chunking.
 *
 * Migrated from the standalone test-ast-chunking.mjs script into the
 * vitest suite. Covers the integration between AST break point extraction
 * and the chunking pipeline — areas not tested by the unit-level ast.test.ts.
 */

import { describe, test, expect } from "vitest";
import { getASTBreakPoints } from "../src/ast.js";
import {
  chunkDocument,
  chunkDocumentAsync,
  chunkDocumentWithBreakPoints,
  mergeBreakPoints,
  scanBreakPoints,
  findCodeFences,
} from "../src/store.js";

// ==========================================================================
// mergeBreakPoints
// ==========================================================================

describe("mergeBreakPoints", () => {
  test("merges regex and AST break points, higher score wins at same position", () => {
    const regexPoints = [
      { pos: 10, score: 20, type: "blank" },
      { pos: 50, score: 1, type: "newline" },
      { pos: 100, score: 20, type: "blank" },
    ];
    const astPoints = [
      { pos: 10, score: 90, type: "ast:func" },
      { pos: 75, score: 100, type: "ast:class" },
      { pos: 100, score: 60, type: "ast:import" },
    ];

    const merged = mergeBreakPoints(regexPoints, astPoints);

    expect(merged).toHaveLength(4);
    expect(merged.find(p => p.pos === 10)?.score).toBe(90);   // AST wins (90 > 20)
    expect(merged.find(p => p.pos === 50)?.score).toBe(1);    // regex only
    expect(merged.find(p => p.pos === 75)?.score).toBe(100);  // AST only
    expect(merged.find(p => p.pos === 100)?.score).toBe(60);  // AST wins (60 > 20)
  });

  test("result is sorted by position", () => {
    const merged = mergeBreakPoints(
      [{ pos: 100, score: 10, type: "a" }],
      [{ pos: 5, score: 50, type: "b" }],
    );
    expect(merged[0]!.pos).toBeLessThan(merged[1]!.pos);
  });
});

// ==========================================================================
// AST vs Regex chunking comparison
// ==========================================================================

describe("AST vs Regex chunking", () => {
  // Generate a large TS file with 30 functions
  const parts: string[] = [];
  for (let i = 0; i < 30; i++) {
    parts.push(`
export function handler${i}(req: Request, res: Response): void {
  const startTime = Date.now();
  const userId = req.params.userId;
  const sessionToken = req.headers.authorization;

  if (!userId || !sessionToken) {
    res.status(400).json({ error: "Missing required parameters" });
    return;
  }

  console.log(\`Processing request ${i} for user \${userId}\`);
  const result = processBusinessLogic${i}(userId, sessionToken);

  const elapsed = Date.now() - startTime;
  res.json({ data: result, processingTimeMs: elapsed });
}
`);
  }
  const largeTS = parts.join("\n");

  function countSplitFunctions(chunks: { text: string; pos: number }[]): number {
    let splits = 0;
    for (let i = 0; i < 30; i++) {
      const funcStart = largeTS.indexOf(`function handler${i}(`);
      const nextFunc = largeTS.indexOf(`function handler${i + 1}(`, funcStart + 1);
      const funcEnd = nextFunc > 0 ? nextFunc : largeTS.length;
      const chunkIndices = new Set<number>();
      for (let ci = 0; ci < chunks.length; ci++) {
        const chunkStart = chunks[ci]!.pos;
        const chunkEnd = chunkStart + chunks[ci]!.text.length;
        if (chunkStart < funcEnd && chunkEnd > funcStart) {
          chunkIndices.add(ci);
        }
      }
      if (chunkIndices.size > 1) splits++;
    }
    return splits;
  }

  test("AST splits fewer functions across chunk boundaries than regex", async () => {
    const regexChunks = chunkDocument(largeTS);
    const astChunks = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, "handlers.ts", "auto");

    const regexSplits = countSplitFunctions(regexChunks);
    const astSplits = countSplitFunctions(astChunks);

    expect(astSplits).toBeLessThanOrEqual(regexSplits);
  });

  test("markdown files produce identical chunks in auto vs regex mode", async () => {
    const sections: string[] = [];
    for (let i = 0; i < 15; i++) {
      sections.push(`# Section ${i}\n\n${"Lorem ipsum dolor sit amet. ".repeat(40)}\n`);
    }
    const largeMD = sections.join("\n");

    const mdRegex = chunkDocument(largeMD);
    const mdAst = await chunkDocumentAsync(largeMD, undefined, undefined, undefined, "readme.md", "auto");

    expect(mdAst).toHaveLength(mdRegex.length);
    for (let i = 0; i < mdRegex.length; i++) {
      expect(mdAst[i]?.text).toBe(mdRegex[i]?.text);
      expect(mdAst[i]?.pos).toBe(mdRegex[i]?.pos);
    }
  });

  test("regex strategy bypasses AST entirely", async () => {
    const regexOnly = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, "handlers.ts", "regex");
    const syncRegex = chunkDocument(largeTS);

    expect(regexOnly).toHaveLength(syncRegex.length);
    for (let i = 0; i < syncRegex.length; i++) {
      expect(regexOnly[i]?.text).toBe(syncRegex[i]?.text);
    }
  });

  test("no filepath falls back to regex", async () => {
    const noPathChunks = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, undefined, "auto");
    const syncRegex = chunkDocument(largeTS);
    expect(noPathChunks).toHaveLength(syncRegex.length);
  });

  test("small file produces single chunk", async () => {
    const smallChunks = await chunkDocumentAsync("export const x = 1;", undefined, undefined, undefined, "s.ts", "auto");
    expect(smallChunks).toHaveLength(1);
  });
});

// ==========================================================================
// chunkDocumentWithBreakPoints equivalence
// ==========================================================================

describe("chunkDocumentWithBreakPoints equivalence", () => {
  test("produces identical output to chunkDocument for the same content", () => {
    const content = "a".repeat(5000) + "\n\n" + "b".repeat(5000);
    const old = chunkDocument(content);
    const withBP = chunkDocumentWithBreakPoints(content, scanBreakPoints(content), findCodeFences(content));

    expect(withBP).toHaveLength(old.length);
    for (let i = 0; i < old.length; i++) {
      expect(withBP[i]?.text).toBe(old[i]?.text);
      expect(withBP[i]?.pos).toBe(old[i]?.pos);
    }
  });
});

// ==========================================================================
// Score assertions not covered by ast.test.ts unit tests
// ==========================================================================

describe("AST break point scores", () => {
  test("TypeScript export (class) scores 90", async () => {
    const code = `export class Foo {}\nexport function bar() {}`;
    const points = await getASTBreakPoints(code, "a.ts");
    const exportPoint = points.find(p => p.type === "ast:export");
    expect(exportPoint?.score).toBe(90);
  });

  test("Python class scores 100", async () => {
    const code = `class Foo:\n    pass\n\ndef bar():\n    pass`;
    const points = await getASTBreakPoints(code, "a.py");
    expect(points.find(p => p.type === "ast:class")?.score).toBe(100);
  });

  test("Go type scores 80", async () => {
    const code = `package main\n\ntype Server struct {\n    port int\n}\n\nfunc main() {}`;
    const points = await getASTBreakPoints(code, "a.go");
    expect(points.find(p => p.type === "ast:type")?.score).toBe(80);
  });

  test("Rust enum scores 80", async () => {
    const code = `enum State {\n    On,\n    Off,\n}\n\nfn main() {}`;
    const points = await getASTBreakPoints(code, "a.rs");
    expect(points.find(p => p.type === "ast:enum")?.score).toBe(80);
  });
});
