/**
 * Integration tests for AST-aware chunking and symbol enrichment.
 *
 * These tests cover the integration between AST parsing and the chunking
 * pipeline — areas not covered by the unit-level ast.test.ts or store.test.ts.
 */

import { describe, test, expect } from "vitest";
import { detectLanguage, getASTBreakPoints, extractAllSymbols, extractSymbols, parseCodeFile } from "../src/ast.js";
import {
  chunkDocument,
  chunkDocumentAsync,
  chunkDocumentWithBreakPoints,
  mergeBreakPoints,
  scanBreakPoints,
  findCodeFences,
} from "../src/store.js";
import { formatDocForEmbedding } from "../src/llm.js";

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
// Overlapping chunks get correct symbols
// ==========================================================================

describe("overlapping chunks symbol mapping", () => {
  test("a symbol in the overlap region appears in both chunks", async () => {
    const code = `function alpha() { return "a"; }

function beta() { return "b"; }

function gamma() { return "c"; }
`;
    const allSym = await extractAllSymbols(code, "overlap.ts");
    expect(allSym).toHaveLength(3);

    // Simulate two overlapping chunks where beta falls in the overlap
    const gammaPos = allSym.find(s => s.name === "gamma")!.pos;
    const betaPos = allSym.find(s => s.name === "beta")!.pos;

    const chunk1Sym = allSym.filter(s => s.pos >= 0 && s.pos < gammaPos);
    const chunk2Sym = allSym.filter(s => s.pos >= betaPos && s.pos < code.length);

    expect(chunk1Sym.some(s => s.name === "beta")).toBe(true);
    expect(chunk2Sym.some(s => s.name === "beta")).toBe(true);
  });
});

// ==========================================================================
// formatDocForEmbedding with symbol enrichment
// ==========================================================================

describe("formatDocForEmbedding symbol enrichment", () => {
  test("without symbols, no 'symbols:' prefix in text", () => {
    const text = formatDocForEmbedding("some code", "auth.ts");
    expect(text).not.toContain("symbols:");
  });

  test("with symbols, includes symbol names, title, and original text", () => {
    const text = formatDocForEmbedding("some code", "auth.ts", undefined, ["authenticate", "validateToken"]);
    expect(text).toContain("symbols:");
    expect(text).toContain("authenticate");
    expect(text).toContain("validateToken");
    expect(text).toContain("auth.ts");
    expect(text).toContain("some code");
  });
});
