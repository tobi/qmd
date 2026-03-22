#!/usr/bin/env npx tsx
/**
 * Thorough integration test + real-collection performance report for
 * AST-aware chunking.
 *
 * Usage:
 *   npx tsx test-ast-chunking.mjs                  # synthetic tests only
 *   npx tsx test-ast-chunking.mjs /path/to/code    # + scan a real directory
 *   npx tsx test-ast-chunking.mjs ~/dev/myproject   # works with ~
 *   npx tsx test-ast-chunking.mjs --help
 *
 * The real-collection scan walks the directory tree, finds supported code
 * files (.ts/.js/.py/.go/.rs) and markdown (.md), chunks each file with
 * both strategies, and prints a comparative performance report.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, extname, resolve } from "node:path";
import { homedir } from "node:os";
import { detectLanguage, getASTBreakPoints } from "./src/ast.js";
import {
  chunkDocument,
  chunkDocumentAsync,
  chunkDocumentWithBreakPoints,
  mergeBreakPoints,
  scanBreakPoints,
  findCodeFences,
  CHUNK_SIZE_CHARS,
} from "./src/store.js";

// ============================================================================
// Helpers
// ============================================================================

let passed = 0;
let failed = 0;

function section(title) {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`  ${title}`);
  console.log("=".repeat(70));
}

function check(label, condition, detail) {
  if (condition) {
    console.log(`  PASS  ${label}`);
    passed++;
  } else {
    console.log(`  FAIL  ${label}`);
    if (detail) console.log(`        ${detail}`);
    failed++;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function pct(n, d) {
  if (d === 0) return "N/A";
  return `${((n / d) * 100).toFixed(1)}%`;
}

const SKIP_DIRS = new Set([
  "node_modules", ".git", ".cache", "vendor", "dist", "build",
  "__pycache__", ".tox", ".venv", "venv", ".mypy_cache", "target",
  ".next", ".nuxt", "coverage", ".turbo",
]);

const CODE_EXTS = new Set([
  ".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs",
  ".py", ".go", ".rs",
]);

const ALL_EXTS = new Set([...CODE_EXTS, ".md"]);

function walkDir(dir, maxFiles = 5000) {
  const results = [];
  const queue = [dir];
  while (queue.length > 0 && results.length < maxFiles) {
    const current = queue.shift();
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (results.length >= maxFiles) break;
      if (entry.name.startsWith(".")) continue;
      const full = join(current, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) queue.push(full);
      } else if (entry.isFile()) {
        const ext = extname(entry.name).toLowerCase();
        if (ALL_EXTS.has(ext)) results.push(full);
      }
    }
  }
  return results;
}

// ============================================================================
// Parse CLI args
// ============================================================================

const args = process.argv.slice(2);
let scanDir = null;
let skipSynthetic = false;

for (const arg of args) {
  if (arg === "--help" || arg === "-h") {
    console.log(`Usage: npx tsx test-ast-chunking.mjs [options] [directory]

Options:
  --help, -h          Show this help
  --scan-only         Skip synthetic tests, only scan directory

Arguments:
  directory           Path to scan for a real-collection performance report.
                      Walks the tree for .ts/.tsx/.js/.jsx/.py/.go/.rs/.md files.

Examples:
  npx tsx test-ast-chunking.mjs                    # synthetic tests only
  npx tsx test-ast-chunking.mjs ~/dev/myproject     # synthetic + real scan
  npx tsx test-ast-chunking.mjs --scan-only ~/dev   # real scan only
`);
    process.exit(0);
  }
  if (arg === "--scan-only") {
    skipSynthetic = true;
  } else if (!arg.startsWith("-")) {
    scanDir = arg.startsWith("~") ? arg.replace("~", homedir()) : resolve(arg);
  }
}

// ============================================================================
// PART 1: Synthetic Tests
// ============================================================================

if (!skipSynthetic) {

// --------------------------------------------------------------------------
// 1. Language Detection
// --------------------------------------------------------------------------
section("1. Language Detection");

const langTests = [
  ["src/auth.ts", "typescript"],
  ["src/App.tsx", "tsx"],
  ["src/util.js", "javascript"],
  ["src/App.jsx", "tsx"],
  ["src/auth.mts", "typescript"],
  ["src/auth.cjs", "javascript"],
  ["src/auth.py", "python"],
  ["src/auth.go", "go"],
  ["src/auth.rs", "rust"],
  ["docs/README.md", null],
  ["data/file.csv", null],
  ["Makefile", null],
  ["qmd://myproject/src/auth.ts", "typescript"],
  ["qmd://docs/notes.md", null],
];

for (const [path, expected] of langTests) {
  const result = detectLanguage(path);
  check(`detectLanguage("${path}") = ${result}`, result === expected,
    `expected ${expected}, got ${result}`);
}

// --------------------------------------------------------------------------
// 2. AST Break Points - TypeScript
// --------------------------------------------------------------------------
section("2. AST Break Points - TypeScript");

const TS_SAMPLE = `import { Database } from './db';
import type { User } from './types';

interface AuthConfig {
  secret: string;
  ttl: number;
}

type UserId = string;

export class AuthService {
  constructor(private db: Database) {}

  async authenticate(user: User, token: string): Promise<boolean> {
    const session = await this.db.findSession(token);
    return session?.userId === user.id;
  }

  validateToken(token: string): boolean {
    return token.length === 64;
  }
}

export function hashPassword(password: string): string {
  return crypto.createHash('sha256').update(password).digest('hex');
}

const helper = (x: number) => x * 2;
`;

const tsPoints = await getASTBreakPoints(TS_SAMPLE, "auth.ts");
console.log(`\n  TypeScript break points (${tsPoints.length} total):`);
for (const p of tsPoints) {
  const snippet = TS_SAMPLE.slice(p.pos, p.pos + 40).replace(/\n/g, "\\n");
  console.log(`    pos=${String(p.pos).padStart(4)} score=${String(p.score).padStart(3)} type=${p.type.padEnd(15)} text="${snippet}..."`);
}

check("Has import break points", tsPoints.some(p => p.type === "ast:import"));
check("Has interface break point", tsPoints.some(p => p.type === "ast:iface"));
check("Has type break point", tsPoints.some(p => p.type === "ast:type"));
check("Has export break point (class)", tsPoints.some(p => p.type === "ast:export"));
check("Has method break points", tsPoints.filter(p => p.type === "ast:method").length >= 2);
check("Import scores 60", tsPoints.find(p => p.type === "ast:import")?.score === 60);
check("Interface scores 100", tsPoints.find(p => p.type === "ast:iface")?.score === 100);
check("Method scores 90", tsPoints.find(p => p.type === "ast:method")?.score === 90);
check("Export scores 90", tsPoints.find(p => p.type === "ast:export")?.score === 90);
check("Break points sorted by position", tsPoints.every((p, i) => i === 0 || p.pos >= tsPoints[i-1].pos));

const firstImport = tsPoints.find(p => p.type === "ast:import");
check("First import position is correct",
  TS_SAMPLE.slice(firstImport.pos, firstImport.pos + 6) === "import",
  `at pos ${firstImport.pos}: "${TS_SAMPLE.slice(firstImport.pos, firstImport.pos + 10)}"`);

// --------------------------------------------------------------------------
// 3. AST Break Points - Python
// --------------------------------------------------------------------------
section("3. AST Break Points - Python");

const PY_SAMPLE = `import os
from typing import Optional, List

class UserService:
    def __init__(self, db):
        self.db = db

    async def find_user(self, user_id: str) -> Optional[dict]:
        return await self.db.find(user_id)

    def validate(self, user: dict) -> bool:
        return "id" in user and "name" in user

def create_user(name: str, email: str) -> dict:
    return {"name": name, "email": email}

@login_required
def protected_endpoint():
    return "secret"
`;

const pyPoints = await getASTBreakPoints(PY_SAMPLE, "service.py");
console.log(`\n  Python break points (${pyPoints.length} total):`);
for (const p of pyPoints) {
  const snippet = PY_SAMPLE.slice(p.pos, p.pos + 40).replace(/\n/g, "\\n");
  console.log(`    pos=${String(p.pos).padStart(4)} score=${String(p.score).padStart(3)} type=${p.type.padEnd(15)} text="${snippet}..."`);
}

check("Has import break points", pyPoints.filter(p => p.type === "ast:import").length >= 2);
check("Has class break point", pyPoints.some(p => p.type === "ast:class"));
check("Has function break points (methods)", pyPoints.filter(p => p.type === "ast:func").length >= 3);
check("Has decorated definition", pyPoints.some(p => p.type === "ast:decorated"));
check("Class scores 100", pyPoints.find(p => p.type === "ast:class")?.score === 100);

// --------------------------------------------------------------------------
// 4. AST Break Points - Go
// --------------------------------------------------------------------------
section("4. AST Break Points - Go");

const GO_SAMPLE = `package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
    db   *Database
}

type Config interface {
    GetPort() int
}

func NewServer(port int) *Server {
    return &Server{port: port}
}

func (s *Server) Start() error {
    return http.ListenAndServe(fmt.Sprintf(":%d", s.port), nil)
}

func (s *Server) Stop() {
    fmt.Println("stopping")
}
`;

const goPoints = await getASTBreakPoints(GO_SAMPLE, "server.go");
console.log(`\n  Go break points (${goPoints.length} total):`);
for (const p of goPoints) {
  const snippet = GO_SAMPLE.slice(p.pos, p.pos + 40).replace(/\n/g, "\\n");
  console.log(`    pos=${String(p.pos).padStart(4)} score=${String(p.score).padStart(3)} type=${p.type.padEnd(15)} text="${snippet}..."`);
}

check("Has import break point", goPoints.some(p => p.type === "ast:import"));
check("Has type break points", goPoints.filter(p => p.type === "ast:type").length >= 2);
check("Has function break point", goPoints.some(p => p.type === "ast:func"));
check("Has method break points", goPoints.filter(p => p.type === "ast:method").length >= 2);
check("Type scores 80", goPoints.find(p => p.type === "ast:type")?.score === 80);

// --------------------------------------------------------------------------
// 5. AST Break Points - Rust
// --------------------------------------------------------------------------
section("5. AST Break Points - Rust");

const RS_SAMPLE = `use std::collections::HashMap;
use std::io;

pub struct Config {
    port: u16,
    host: String,
}

impl Config {
    pub fn new(port: u16, host: String) -> Self {
        Config { port, host }
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

pub trait Configurable {
    fn configure(&mut self, config: &Config);
}

pub enum ServerState {
    Running,
    Stopped,
    Error(String),
}

pub fn start_server(config: Config) -> io::Result<()> {
    Ok(())
}
`;

const rsPoints = await getASTBreakPoints(RS_SAMPLE, "config.rs");
console.log(`\n  Rust break points (${rsPoints.length} total):`);
for (const p of rsPoints) {
  const snippet = RS_SAMPLE.slice(p.pos, p.pos + 40).replace(/\n/g, "\\n");
  console.log(`    pos=${String(p.pos).padStart(4)} score=${String(p.score).padStart(3)} type=${p.type.padEnd(15)} text="${snippet}..."`);
}

check("Has use/import break points", rsPoints.filter(p => p.type === "ast:import").length >= 2);
check("Has struct break point", rsPoints.some(p => p.type === "ast:struct"));
check("Has impl break point", rsPoints.some(p => p.type === "ast:impl"));
check("Has trait break point", rsPoints.some(p => p.type === "ast:trait"));
check("Has enum break point", rsPoints.some(p => p.type === "ast:enum"));
check("Has function break point", rsPoints.some(p => p.type === "ast:func"));
check("Struct scores 100", rsPoints.find(p => p.type === "ast:struct")?.score === 100);
check("Impl scores 100", rsPoints.find(p => p.type === "ast:impl")?.score === 100);
check("Trait scores 100", rsPoints.find(p => p.type === "ast:trait")?.score === 100);
check("Enum scores 80", rsPoints.find(p => p.type === "ast:enum")?.score === 80);

// --------------------------------------------------------------------------
// 6. Merge Break Points
// --------------------------------------------------------------------------
section("6. mergeBreakPoints");

const regexPoints = [
  { pos: 10, score: 20, type: "blank" },
  { pos: 50, score: 1, type: "newline" },
  { pos: 100, score: 20, type: "blank" },
];
const astPointsMerge = [
  { pos: 10, score: 90, type: "ast:func" },
  { pos: 75, score: 100, type: "ast:class" },
  { pos: 100, score: 60, type: "ast:import" },
];

const merged = mergeBreakPoints(regexPoints, astPointsMerge);
console.log(`\n  Merged break points (${merged.length} total):`);
for (const p of merged) {
  console.log(`    pos=${String(p.pos).padStart(4)} score=${String(p.score).padStart(3)} type=${p.type}`);
}

check("Merge has 4 unique positions", merged.length === 4);
check("pos 10: AST wins (90 > 20)", merged.find(p => p.pos === 10)?.score === 90);
check("pos 50: regex only (1)", merged.find(p => p.pos === 50)?.score === 1);
check("pos 75: AST only (100)", merged.find(p => p.pos === 75)?.score === 100);
check("pos 100: AST wins (60 > 20)", merged.find(p => p.pos === 100)?.score === 60);
check("Sorted by position", merged.every((p, i) => i === 0 || p.pos >= merged[i-1].pos));

// --------------------------------------------------------------------------
// 7. AST vs Regex Chunking Comparison (Large Synthetic File)
// --------------------------------------------------------------------------
section("7. AST vs Regex Chunking Comparison");

const largeTSParts = [];
for (let i = 0; i < 30; i++) {
  largeTSParts.push(`
export function handler${i}(req: Request, res: Response): void {
  const startTime = Date.now();
  const userId = req.params.userId;
  const sessionToken = req.headers.authorization;

  // Validate the incoming request parameters
  if (!userId || !sessionToken) {
    res.status(400).json({ error: "Missing required parameters" });
    return;
  }

  // Process the request with detailed logging
  console.log(\`Processing request \${i} for user \${userId}\`);
  const result = processBusinessLogic${i}(userId, sessionToken);

  // Return the response with timing info
  const elapsed = Date.now() - startTime;
  res.json({ data: result, processingTimeMs: elapsed });
}
`);
}
const largeTS = largeTSParts.join("\n");

console.log(`\n  Large TS file: ${largeTS.length} chars, ${largeTSParts.length} functions`);

const regexChunks = chunkDocument(largeTS);
const astChunks = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, "handlers.ts", "auto");

console.log(`  Regex chunks: ${regexChunks.length}`);
console.log(`  AST chunks:   ${astChunks.length}`);

function countSplitFunctions(chunks, source) {
  let splits = 0;
  for (let i = 0; i < 30; i++) {
    const funcStart = source.indexOf(`function handler${i}(`);
    const nextFunc = source.indexOf(`function handler${i + 1}(`, funcStart + 1);
    const funcEnd = nextFunc > 0 ? nextFunc : source.length;
    const chunkIndices = new Set();
    for (let ci = 0; ci < chunks.length; ci++) {
      const chunkStart = chunks[ci].pos;
      const chunkEnd = chunkStart + chunks[ci].text.length;
      if (chunkStart < funcEnd && chunkEnd > funcStart) {
        chunkIndices.add(ci);
      }
    }
    if (chunkIndices.size > 1) splits++;
  }
  return splits;
}

const regexSplits = countSplitFunctions(regexChunks, largeTS);
const astSplitsSynth = countSplitFunctions(astChunks, largeTS);

console.log(`\n  Functions split across chunks:`);
console.log(`    Regex: ${regexSplits} / 30`);
console.log(`    AST:   ${astSplitsSynth} / 30`);

check("AST splits fewer functions than regex", astSplitsSynth <= regexSplits,
  `AST split ${astSplitsSynth}, regex split ${regexSplits}`);

// --------------------------------------------------------------------------
// 8. Markdown Files Unchanged
// --------------------------------------------------------------------------
section("8. Markdown Files Unchanged in Auto Mode");

const mdContent = [];
for (let i = 0; i < 15; i++) {
  mdContent.push(`# Section ${i}\n\n${"Lorem ipsum dolor sit amet. ".repeat(40)}\n`);
}
const largeMD = mdContent.join("\n");

const mdRegex = chunkDocument(largeMD);
const mdAst = await chunkDocumentAsync(largeMD, undefined, undefined, undefined, "readme.md", "auto");

check("Same number of chunks", mdRegex.length === mdAst.length,
  `regex=${mdRegex.length}, ast=${mdAst.length}`);

let mdIdentical = true;
for (let i = 0; i < mdRegex.length; i++) {
  if (mdRegex[i]?.text !== mdAst[i]?.text || mdRegex[i]?.pos !== mdAst[i]?.pos) {
    mdIdentical = false;
    break;
  }
}
check("Chunk content is identical", mdIdentical);

// --------------------------------------------------------------------------
// 9-11. Strategy bypass, no-filepath fallback, error handling
// --------------------------------------------------------------------------
section("9. Regex Strategy Bypass");
const regexOnly = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, "handlers.ts", "regex");
const syncRegex = chunkDocument(largeTS);
check("Same chunks as sync regex", regexOnly.length === syncRegex.length &&
  regexOnly.every((c, i) => c.text === syncRegex[i]?.text));

section("10. No Filepath Falls Back to Regex");
const noPathChunks = await chunkDocumentAsync(largeTS, undefined, undefined, undefined, undefined, "auto");
check("Same chunks as regex", noPathChunks.length === syncRegex.length);

section("11. Error Handling & Edge Cases");
check("Empty file -> []", (await getASTBreakPoints("", "e.ts")).length === 0);
check("Broken syntax doesn't crash", Array.isArray(await getASTBreakPoints("function { %%", "x.ts")));
check("Unknown ext -> []", (await getASTBreakPoints("data", "f.csv")).length === 0);
check("Markdown -> []", (await getASTBreakPoints("# H", "r.md")).length === 0);
const smallChunks = await chunkDocumentAsync("export const x = 1;", undefined, undefined, undefined, "s.ts", "auto");
check("Small file -> 1 chunk", smallChunks.length === 1);

// --------------------------------------------------------------------------
// 12. chunkDocumentWithBreakPoints Equivalence
// --------------------------------------------------------------------------
section("12. chunkDocumentWithBreakPoints Equivalence");
const eqContent = "a".repeat(5000) + "\n\n" + "b".repeat(5000);
const eqOld = chunkDocument(eqContent);
const eqNew = chunkDocumentWithBreakPoints(eqContent, scanBreakPoints(eqContent), findCodeFences(eqContent));
check("Identical output", eqOld.length === eqNew.length &&
  eqOld.every((c, i) => c.text === eqNew[i]?.text && c.pos === eqNew[i]?.pos));

// --------------------------------------------------------------------------
// 13. Synthetic performance
// --------------------------------------------------------------------------
section("13. Synthetic Performance");

const t0 = performance.now();
for (let i = 0; i < 10; i++) await getASTBreakPoints(largeTS, "p.ts");
const astExtractMs = (performance.now() - t0) / 10;

const t1 = performance.now();
for (let i = 0; i < 10; i++) scanBreakPoints(largeTS);
const regexExtractMs = (performance.now() - t1) / 10;

const t2 = performance.now();
for (let i = 0; i < 10; i++) await chunkDocumentAsync(largeTS, undefined, undefined, undefined, "p.ts", "auto");
const astFullMs = (performance.now() - t2) / 10;

const t3 = performance.now();
for (let i = 0; i < 10; i++) chunkDocument(largeTS);
const regexFullMs = (performance.now() - t3) / 10;

console.log(`\n  File size:                    ${formatBytes(largeTS.length)}`);
console.log(`  AST break point extraction:   ${astExtractMs.toFixed(1)}ms`);
console.log(`  Regex break point extraction: ${regexExtractMs.toFixed(1)}ms`);
console.log(`  Full AST chunking:            ${astFullMs.toFixed(1)}ms`);
console.log(`  Full regex chunking:          ${regexFullMs.toFixed(1)}ms`);
console.log(`  Overhead per file:            ${(astFullMs - regexFullMs).toFixed(1)}ms`);

check("AST chunking < 50ms per file", astFullMs < 50, `was ${astFullMs.toFixed(1)}ms`);

// End of synthetic tests
section("Synthetic Test Results");
console.log(`\n  ${passed} passed, ${failed} failed`);

} // end if (!skipSynthetic)


// ============================================================================
// PART 2: Real Collection Scan
// ============================================================================

if (scanDir) {

section(`Real Collection Scan: ${scanDir}`);

console.log(`\n  Discovering files...`);
const realFiles = walkDir(scanDir);
console.log(`  Found ${realFiles.length} files\n`);

if (realFiles.length === 0) {
  console.log("  No supported files found. Supported: .ts .tsx .js .jsx .py .go .rs .md");
} else {

  // Classify files
  const byLang = {};
  let totalBytes = 0;
  const fileEntries = [];

  for (const filepath of realFiles) {
    let content;
    try {
      const stat = statSync(filepath);
      if (stat.size > 500_000) continue; // skip files > 500KB
      content = readFileSync(filepath, "utf-8");
    } catch {
      continue;
    }
    if (!content.trim()) continue;

    const rel = relative(scanDir, filepath);
    const lang = detectLanguage(filepath);
    const langLabel = lang ?? "markdown";

    byLang[langLabel] = (byLang[langLabel] || 0) + 1;
    totalBytes += content.length;
    fileEntries.push({ filepath, rel, lang, langLabel, content });
  }

  // Print file distribution
  console.log("  File distribution:");
  for (const [lang, count] of Object.entries(byLang).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${lang.padEnd(14)} ${count} files`);
  }
  console.log(`    ${"total".padEnd(14)} ${fileEntries.length} files (${formatBytes(totalBytes)})`);

  // ---- Per-file analysis ----

  // Accumulators
  const perLang = {};
  let totalRegexChunks = 0;
  let totalAstChunks = 0;
  let totalRegexMs = 0;
  let totalAstMs = 0;
  let filesWithDifference = 0;
  let multiChunkFiles = 0;
  const bigDiffs = [];  // files where AST made the biggest difference

  console.log(`\n  Analyzing ${fileEntries.length} files...\n`);

  for (const entry of fileEntries) {
    const { rel, lang, langLabel, content } = entry;
    const isCode = lang !== null;

    // Regex chunking
    const rt0 = performance.now();
    const rChunks = chunkDocument(content);
    const rMs = performance.now() - rt0;

    // AST chunking
    const at0 = performance.now();
    const aChunks = await chunkDocumentAsync(content, undefined, undefined, undefined, rel, "auto");
    const aMs = performance.now() - at0;

    totalRegexChunks += rChunks.length;
    totalAstChunks += aChunks.length;
    totalRegexMs += rMs;
    totalAstMs += aMs;

    if (rChunks.length > 1 || aChunks.length > 1) multiChunkFiles++;

    const chunkDiff = aChunks.length - rChunks.length;
    const contentDiffers = rChunks.length !== aChunks.length ||
      rChunks.some((c, i) => c.text !== aChunks[i]?.text);

    if (contentDiffers) filesWithDifference++;

    // Per-language stats
    if (!perLang[langLabel]) {
      perLang[langLabel] = {
        files: 0, bytes: 0, regexChunks: 0, astChunks: 0,
        regexMs: 0, astMs: 0, astBreakpoints: 0, diffs: 0,
      };
    }
    const s = perLang[langLabel];
    s.files++;
    s.bytes += content.length;
    s.regexChunks += rChunks.length;
    s.astChunks += aChunks.length;
    s.regexMs += rMs;
    s.astMs += aMs;
    if (contentDiffers) s.diffs++;

    // Count AST breakpoints for code files
    if (isCode) {
      const bp = await getASTBreakPoints(content, rel);
      s.astBreakpoints += bp.length;
    }

    // Track big differences for the detailed report
    if (contentDiffers && isCode && (rChunks.length > 1 || aChunks.length > 1)) {
      bigDiffs.push({
        rel, lang: langLabel, bytes: content.length,
        regexN: rChunks.length, astN: aChunks.length,
        diff: chunkDiff, overheadMs: aMs - rMs,
      });
    }
  }

  // ---- Aggregate report ----

  section("Per-Language Summary");

  const langOrder = Object.entries(perLang).sort((a, b) => b[1].files - a[1].files);
  const colW = { lang: 14, files: 7, bytes: 10, rChunks: 9, aChunks: 9, bps: 6, diffs: 6, rMs: 9, aMs: 9 };

  console.log(
    `\n  ${"Language".padEnd(colW.lang)}${"Files".padStart(colW.files)}${"Size".padStart(colW.bytes)}` +
    `${"Rx Chnk".padStart(colW.rChunks)}${"AST Chnk".padStart(colW.aChunks)}` +
    `${"BPs".padStart(colW.bps)}${"Diffs".padStart(colW.diffs)}` +
    `${"Rx ms".padStart(colW.rMs)}${"AST ms".padStart(colW.aMs)}`
  );
  console.log("  " + "-".repeat(Object.values(colW).reduce((a, b) => a + b, 0)));

  for (const [lang, s] of langOrder) {
    console.log(
      `  ${lang.padEnd(colW.lang)}` +
      `${String(s.files).padStart(colW.files)}` +
      `${formatBytes(s.bytes).padStart(colW.bytes)}` +
      `${String(s.regexChunks).padStart(colW.rChunks)}` +
      `${String(s.astChunks).padStart(colW.aChunks)}` +
      `${String(s.astBreakpoints).padStart(colW.bps)}` +
      `${String(s.diffs).padStart(colW.diffs)}` +
      `${s.regexMs.toFixed(1).padStart(colW.rMs)}` +
      `${s.astMs.toFixed(1).padStart(colW.aMs)}`
    );
  }

  console.log("  " + "-".repeat(Object.values(colW).reduce((a, b) => a + b, 0)));
  console.log(
    `  ${"TOTAL".padEnd(colW.lang)}` +
    `${String(fileEntries.length).padStart(colW.files)}` +
    `${formatBytes(totalBytes).padStart(colW.bytes)}` +
    `${String(totalRegexChunks).padStart(colW.rChunks)}` +
    `${String(totalAstChunks).padStart(colW.aChunks)}` +
    `${"".padStart(colW.bps)}` +
    `${String(filesWithDifference).padStart(colW.diffs)}` +
    `${totalRegexMs.toFixed(1).padStart(colW.rMs)}` +
    `${totalAstMs.toFixed(1).padStart(colW.aMs)}`
  );

  // ---- Headline stats ----

  section("Headline Stats");

  const codeFiles = fileEntries.filter(e => e.lang !== null).length;
  const mdFiles = fileEntries.filter(e => e.lang === null).length;
  const avgOverheadMs = codeFiles > 0
    ? (langOrder.filter(([l]) => l !== "markdown").reduce((s, [, v]) => s + v.astMs - v.regexMs, 0)) / codeFiles
    : 0;

  console.log(`
  Files scanned:         ${fileEntries.length} (${codeFiles} code, ${mdFiles} markdown)
  Multi-chunk files:     ${multiChunkFiles} (files large enough to produce >1 chunk)
  Files where AST differed: ${filesWithDifference} / ${fileEntries.length} (${pct(filesWithDifference, fileEntries.length)})
  Total chunks (regex):  ${totalRegexChunks}
  Total chunks (AST):    ${totalAstChunks}  (${totalAstChunks > totalRegexChunks ? "+" : ""}${totalAstChunks - totalRegexChunks})
  Total time (regex):    ${totalRegexMs.toFixed(1)}ms
  Total time (AST):      ${totalAstMs.toFixed(1)}ms  (+${(totalAstMs - totalRegexMs).toFixed(1)}ms overhead)
  Avg overhead per code file: ${avgOverheadMs.toFixed(2)}ms
  `);

  // ---- Top differences ----

  if (bigDiffs.length > 0) {
    section("Top Files Where AST Changed Chunking");

    bigDiffs.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
    const topN = bigDiffs.slice(0, 20);

    console.log(
      `\n  ${"File".padEnd(50)} ${"Lang".padEnd(12)} ${"Size".padStart(8)} ` +
      `${"Rx".padStart(4)} ${"AST".padStart(4)} ${"Diff".padStart(5)} ` +
      `${"OH ms".padStart(7)}`
    );
    console.log("  " + "-".repeat(94));

    for (const d of topN) {
      const sign = d.diff > 0 ? "+" : "";
      console.log(
        `  ${d.rel.slice(0, 49).padEnd(50)} ${d.lang.padEnd(12)} ${formatBytes(d.bytes).padStart(8)} ` +
        `${String(d.regexN).padStart(4)} ${String(d.astN).padStart(4)} ${(sign + d.diff).padStart(5)} ` +
        `${d.overheadMs.toFixed(1).padStart(7)}`
      );
    }

    if (bigDiffs.length > 20) {
      console.log(`\n  ... and ${bigDiffs.length - 20} more files with differences`);
    }
  }

  // ---- Markdown regression check ----

  const mdEntries = fileEntries.filter(e => e.lang === null);
  if (mdEntries.length > 0) {
    section("Markdown Regression Check");

    let mdRegressions = 0;
    for (const entry of mdEntries) {
      const rChunks = chunkDocument(entry.content);
      const aChunks = await chunkDocumentAsync(entry.content, undefined, undefined, undefined, entry.rel, "auto");
      const same = rChunks.length === aChunks.length &&
        rChunks.every((c, i) => c.text === aChunks[i]?.text);
      if (!same) {
        mdRegressions++;
        console.log(`  REGRESSION: ${entry.rel} (regex=${rChunks.length}, ast=${aChunks.length})`);
      }
    }

    if (mdRegressions === 0) {
      console.log(`\n  All ${mdEntries.length} markdown files produce identical chunks. No regressions.`);
    } else {
      console.log(`\n  ${mdRegressions} / ${mdEntries.length} markdown files differ (unexpected!)`);
    }
  }

} // end if realFiles.length > 0

} // end if scanDir

// ============================================================================
// Final Summary
// ============================================================================

console.log(`\n${"=".repeat(70)}`);
if (!skipSynthetic) {
  console.log(`  SYNTHETIC TESTS: ${passed} passed, ${failed} failed`);
}
if (scanDir) {
  console.log(`  COLLECTION SCAN: complete (see report above)`);
}
if (!scanDir && !skipSynthetic) {
  console.log(`\n  Tip: Run with a directory argument to scan real files:`);
  console.log(`    npx tsx test-ast-chunking.mjs ~/dev/my-project`);
}
console.log("=".repeat(70));

if (failed > 0) process.exit(1);
