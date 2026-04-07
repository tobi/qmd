/**
 * AST-aware chunking support via web-tree-sitter.
 *
 * Provides language detection, AST break point extraction for supported
 * code file types, and a stub for future symbol extraction.
 *
 * All functions degrade gracefully: parse failures or unsupported languages
 * return empty arrays, falling back to regex-only chunking.
 *
 * ## Dependency Note
 *
 * Grammar packages (tree-sitter-typescript, etc.) are listed as
 * optionalDependencies with pinned versions. They ship native prebuilds
 * and source files (~72 MB total) but QMD only uses the .wasm files
 * (~5 MB). If install size becomes a concern, the .wasm files can be
 * bundled directly in the repo (e.g. assets/grammars/) and resolved
 * via import.meta.url instead of require.resolve(), eliminating the
 * grammar packages entirely.
 */

import { createRequire } from "node:module";
import { extname } from "node:path";
import { callCSharpSidecar } from "./csharp-sidecar.js";
import { mergeBreakPoints, type BreakPoint } from "./store.js";

// web-tree-sitter types — imported dynamically to avoid top-level WASM init
type ParserType = import("web-tree-sitter").Parser;
type LanguageType = import("web-tree-sitter").Language;
type QueryType = import("web-tree-sitter").Query;

// =============================================================================
// Language Detection
// =============================================================================

export type SupportedLanguage = "typescript" | "tsx" | "javascript" | "python" | "go" | "rust" | "csharp";

const EXTENSION_MAP: Record<string, SupportedLanguage> = {
  ".ts": "typescript",
  ".tsx": "tsx",
  ".js": "javascript",
  ".jsx": "tsx",
  ".mts": "typescript",
  ".cts": "typescript",
  ".mjs": "javascript",
  ".cjs": "javascript",
  ".py": "python",
  ".go": "go",
  ".rs": "rust",
  ".cs": "csharp",
};

/**
 * Detect language from file path extension.
 * Returns null for unsupported or unknown extensions (including .md).
 */
export function detectLanguage(filepath: string): SupportedLanguage | null {
  const ext = extname(filepath).toLowerCase();
  return EXTENSION_MAP[ext] ?? null;
}

// =============================================================================
// Grammar Resolution
// =============================================================================

/**
 * Maps language to the npm package and wasm filename for the grammar.
 */
const GRAMMAR_MAP: Record<SupportedLanguage, { pkg: string; wasm: string }> = {
  typescript: { pkg: "tree-sitter-typescript", wasm: "tree-sitter-typescript.wasm" },
  tsx:        { pkg: "tree-sitter-typescript", wasm: "tree-sitter-tsx.wasm" },
  javascript: { pkg: "tree-sitter-typescript", wasm: "tree-sitter-typescript.wasm" },
  python:     { pkg: "tree-sitter-python",     wasm: "tree-sitter-python.wasm" },
  go:         { pkg: "tree-sitter-go",         wasm: "tree-sitter-go.wasm" },
  rust:       { pkg: "tree-sitter-rust",        wasm: "tree-sitter-rust.wasm" },
  csharp:     { pkg: "tree-sitter-c-sharp",    wasm: "tree-sitter-c_sharp.wasm" },
};

// =============================================================================
// Per-Language Query Definitions
// =============================================================================

/**
 * Tree-sitter S-expression queries for each language.
 * Each capture name maps to a break point score via SCORE_MAP.
 *
 * For TypeScript/JavaScript, we match export_statement wrappers to get the
 * correct start position (before `export`), plus bare declarations for
 * non-exported code.
 */
const LANGUAGE_QUERIES: Record<SupportedLanguage, string> = {
  typescript: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
  tsx: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
  javascript: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
  python: `
    (class_definition) @class
    (function_definition) @func
    (decorated_definition) @decorated
    (import_statement) @import
    (import_from_statement) @import
  `,
  go: `
    (type_declaration) @type
    (function_declaration) @func
    (method_declaration) @method
    (import_declaration) @import
  `,
  rust: `
    (struct_item) @struct
    (impl_item) @impl
    (function_item) @func
    (trait_item) @trait
    (enum_item) @enum
    (use_declaration) @import
    (type_item) @type
    (mod_item) @mod
  `,
  csharp: `
    (using_directive) @import
    (namespace_declaration) @namespace
    (file_scoped_namespace_declaration) @namespace
    (class_declaration) @type
    (struct_declaration) @type
    (interface_declaration) @type
    (record_declaration) @type
    (enum_declaration) @enum
    (constructor_declaration) @ctor
    (method_declaration) @method
  `,
};

/**
 * Score mapping from capture names to break point scores.
 * Aligned with the markdown BREAK_PATTERNS scale (h1=100, h2=90, etc.)
 * so findBestCutoff() decay works unchanged.
 */
const SCORE_MAP: Record<string, number> = {
  class:     100,
  iface:     100,
  struct:    100,
  trait:     100,
  impl:      100,
  mod:       100,
  namespace: 100,
  export:     90,
  func:       90,
  method:     90,
  ctor:       90,
  decorated:  90,
  type:       80,
  enum:       80,
  import:     60,
};

/**
 * Language-specific score overrides for capture names.
 * Keep baseline scores stable across existing languages and only
 * elevate scores where the language semantics require it.
 */
const LANGUAGE_SCORE_OVERRIDES: Partial<Record<SupportedLanguage, Partial<Record<string, number>>>> = {
  csharp: {
    type: 100,
  },
};

// =============================================================================
// Parser Caching & Initialization
// =============================================================================

let ParserClass: typeof import("web-tree-sitter").Parser | null = null;
let LanguageClass: typeof import("web-tree-sitter").Language | null = null;
let QueryClass: typeof import("web-tree-sitter").Query | null = null;
let initPromise: Promise<void> | null = null;

/** Languages that have already failed to load — warn only once per process. */
const failedLanguages = new Set<string>();

/** Cached grammar load promises. */
const grammarCache = new Map<string, Promise<LanguageType>>();

/** Cached compiled queries per language. */
const queryCache = new Map<string, QueryType>();

/**
 * Initialize web-tree-sitter. Called once and cached.
 */
async function ensureInit(): Promise<void> {
  if (!initPromise) {
    initPromise = (async () => {
      const mod = await import("web-tree-sitter");
      ParserClass = mod.Parser;
      LanguageClass = mod.Language;
      QueryClass = mod.Query;
      await ParserClass.init();
    })();
  }
  return initPromise;
}

/**
 * Resolve the filesystem path to a grammar .wasm file.
 * Uses createRequire to resolve from installed dependency packages.
 */
function resolveGrammarPath(language: SupportedLanguage): string {
  const { pkg, wasm } = GRAMMAR_MAP[language];
  const require = createRequire(import.meta.url);
  return require.resolve(`${pkg}/${wasm}`);
}

/**
 * Load and cache a grammar for the given language.
 * Returns null on failure (logs once per language).
 */
async function loadGrammar(language: SupportedLanguage): Promise<LanguageType | null> {
  if (failedLanguages.has(language)) return null;

  const wasmKey = GRAMMAR_MAP[language].wasm;
  if (!grammarCache.has(wasmKey)) {
    grammarCache.set(wasmKey, (async () => {
      const path = resolveGrammarPath(language);
      return LanguageClass!.load(path);
    })());
  }

  try {
    return await grammarCache.get(wasmKey)!;
  } catch (err) {
    failedLanguages.add(language);
    grammarCache.delete(wasmKey);
    console.warn(`[qmd] Failed to load tree-sitter grammar for ${language}: ${err}`);
    return null;
  }
}

/**
 * Get or create a compiled query for the given language.
 */
function getQuery(language: SupportedLanguage, grammar: LanguageType): QueryType {
  if (!queryCache.has(language)) {
    const source = LANGUAGE_QUERIES[language];
    const query = new QueryClass!(grammar, source);
    queryCache.set(language, query);
  }
  return queryCache.get(language)!;
}

// =============================================================================
// AST Break Point Extraction
// =============================================================================

/**
 * Parse a source file and return break points at AST node boundaries.
 *
 * Returns an empty array for unsupported languages, parse failures,
 * or grammar loading failures. Never throws.
 *
 * @param content - The file content to parse.
 * @param filepath - The file path (used for language detection).
 * @returns Array of BreakPoint objects suitable for merging with regex break points.
 */
export async function getASTBreakPoints(
  content: string,
  filepath: string,
): Promise<BreakPoint[]> {
  const language = detectLanguage(filepath);
  if (!language) return [];

  try {
    await ensureInit();

    const grammar = await loadGrammar(language);
    if (!grammar) return [];

    const parser = new ParserClass!();
    parser.setLanguage(grammar);

    const tree = parser.parse(content);
    if (!tree) {
      parser.delete();
      return [];
    }

    const query = getQuery(language, grammar);
    const captures = query.captures(tree.rootNode);

    // Deduplicate: at each byte position, keep the highest-scoring capture.
    // This handles cases like export_statement wrapping a class_declaration
    // at different offsets — we want the outermost (earliest) position.
    const seen = new Map<number, BreakPoint>();

    for (const cap of captures) {
      const pos = cap.node.startIndex;
      const score = LANGUAGE_SCORE_OVERRIDES[language]?.[cap.name] ?? SCORE_MAP[cap.name] ?? 20;
      const type = `ast:${cap.name}`;

      const existing = seen.get(pos);
      if (!existing || score > existing.score) {
        seen.set(pos, { pos, score, type });
      }
    }

    tree.delete();
    parser.delete();

    const points = Array.from(seen.values()).sort((a, b) => a.pos - b.pos);
    if (language !== "csharp") {
      return points;
    }

    try {
      const enhanced = await callCSharpSidecar(filepath, content);
      if (enhanced?.breakpoints.length) {
        return mergeBreakPoints(points, enhanced.breakpoints);
      }
    } catch {
      // Fall back to baseline tree-sitter breakpoints if the sidecar is unavailable.
    }

    return points;
  } catch (err) {
    console.warn(`[qmd] AST parse failed for ${filepath}, falling back to regex: ${err instanceof Error ? err.message : err}`);
    return [];
  }
}

// =============================================================================
// Health / Status
// =============================================================================

/**
 * Check which tree-sitter grammars are available.
 * Returns a status object for each supported language.
 */
export async function getASTStatus(): Promise<{
  available: boolean;
  languages: { language: SupportedLanguage; available: boolean; error?: string }[];
}> {
  const languages: { language: SupportedLanguage; available: boolean; error?: string }[] = [];

  try {
    await ensureInit();
  } catch (err) {
    return {
      available: false,
      languages: (Object.keys(GRAMMAR_MAP) as SupportedLanguage[]).map(lang => ({
        language: lang,
        available: false,
        error: `web-tree-sitter init failed: ${err instanceof Error ? err.message : err}`,
      })),
    };
  }

  for (const lang of Object.keys(GRAMMAR_MAP) as SupportedLanguage[]) {
    try {
      const grammar = await loadGrammar(lang);
      if (grammar) {
        // Also verify the query compiles
        getQuery(lang, grammar);
        languages.push({ language: lang, available: true });
      } else {
        languages.push({ language: lang, available: false, error: "grammar failed to load" });
      }
    } catch (err) {
      languages.push({
        language: lang,
        available: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return {
    available: languages.some(l => l.available),
    languages,
  };
}

// =============================================================================
// Symbol Extraction (Phase 2 Stub)
// =============================================================================

/**
 * Metadata about a code symbol within a chunk.
 * Stubbed for Phase 2 — always returns empty array in Phase 1.
 */
export interface SymbolInfo {
  name: string;
  kind: string;
  signature?: string;
  line: number;
}

/**
 * Extract symbol metadata for code within a byte range.
 * Stubbed for Phase 2 — returns empty array.
 */
export function extractSymbols(
  _content: string,
  _language: string,
  _startPos: number,
  _endPos: number,
): SymbolInfo[] {
  return [];
}
