/**
 * AST-aware chunking and symbol extraction via web-tree-sitter.
 *
 * Provides language detection, AST break point extraction for supported
 * code file types, and symbol extraction (names, kinds, signatures).
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
import type { BreakPoint } from "./store.js";

// web-tree-sitter types — imported dynamically to avoid top-level WASM init
type ParserType = import("web-tree-sitter").Parser;
type LanguageType = import("web-tree-sitter").Language;
type QueryType = import("web-tree-sitter").Query;
type NodeType = import("web-tree-sitter").Node;

// =============================================================================
// Language Detection
// =============================================================================

export type SupportedLanguage = "typescript" | "tsx" | "javascript" | "python" | "go" | "rust";

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

const GRAMMAR_MAP: Record<SupportedLanguage, { pkg: string; wasm: string }> = {
  typescript: { pkg: "tree-sitter-typescript", wasm: "tree-sitter-typescript.wasm" },
  tsx:        { pkg: "tree-sitter-typescript", wasm: "tree-sitter-tsx.wasm" },
  // JavaScript uses the TypeScript grammar — TS is a superset so the parser handles
  // plain JS correctly, and we avoid an extra grammar dependency. Symbol queries use
  // TS node types (e.g. type_identifier) which also work for JS ASTs.
  javascript: { pkg: "tree-sitter-typescript", wasm: "tree-sitter-typescript.wasm" },
  python:     { pkg: "tree-sitter-python",     wasm: "tree-sitter-python.wasm" },
  go:         { pkg: "tree-sitter-go",         wasm: "tree-sitter-go.wasm" },
  rust:       { pkg: "tree-sitter-rust",        wasm: "tree-sitter-rust.wasm" },
};

// =============================================================================
// Per-Language Query Definitions
// =============================================================================

/**
 * Breakpoint queries — capture outer nodes for chunk boundary scoring.
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
};

/**
 * Symbol queries — capture both outer node and name child for extraction.
 */
const SYMBOL_QUERIES: Record<SupportedLanguage, string> = {
  typescript: `
    (class_declaration name: (type_identifier) @class_name) @class
    (function_declaration name: (identifier) @func_name) @func
    (method_definition name: (property_identifier) @method_name) @method
    (interface_declaration name: (type_identifier) @iface_name) @iface
    (type_alias_declaration name: (type_identifier) @type_name) @type
    (enum_declaration name: (identifier) @enum_name) @enum
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (arrow_function))) @func
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (function_expression))) @func
  `,
  tsx: `
    (class_declaration name: (type_identifier) @class_name) @class
    (function_declaration name: (identifier) @func_name) @func
    (method_definition name: (property_identifier) @method_name) @method
    (interface_declaration name: (type_identifier) @iface_name) @iface
    (type_alias_declaration name: (type_identifier) @type_name) @type
    (enum_declaration name: (identifier) @enum_name) @enum
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (arrow_function))) @func
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (function_expression))) @func
  `,
  javascript: `
    (class_declaration name: (type_identifier) @class_name) @class
    (function_declaration name: (identifier) @func_name) @func
    (method_definition name: (property_identifier) @method_name) @method
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (arrow_function))) @func
    (lexical_declaration (variable_declarator name: (identifier) @func_name value: (function_expression))) @func
  `,
  python: `
    (class_definition name: (identifier) @class_name) @class
    (function_definition name: (identifier) @func_name) @func
  `,
  go: `
    (function_declaration name: (identifier) @func_name) @func
    (method_declaration name: (field_identifier) @method_name) @method
    (type_declaration (type_spec name: (type_identifier) @type_name)) @type
  `,
  rust: `
    (function_item name: (identifier) @func_name) @func
    (struct_item name: (type_identifier) @struct_name) @struct
    (impl_item type: (type_identifier) @impl_name) @impl
    (trait_item name: (type_identifier) @trait_name) @trait
    (enum_item name: (type_identifier) @enum_name) @enum
  `,
};

/**
 * Score mapping from capture names to break point scores.
 */
const SCORE_MAP: Record<string, number> = {
  class:     100,
  iface:     100,
  struct:    100,
  trait:     100,
  impl:      100,
  mod:       100,
  export:     90,
  func:       90,
  method:     90,
  decorated:  90,
  type:       80,
  enum:       80,
  import:     60,
};

/**
 * Maps symbol capture names to user-facing kind strings.
 */
const KIND_MAP: Record<string, string> = {
  class:  "class",
  iface:  "interface",
  struct: "struct",
  trait:  "trait",
  impl:   "impl",
  func:   "function",
  method: "method",
  type:   "type",
  enum:   "enum",
};

// =============================================================================
// Types
// =============================================================================

/**
 * Internal symbol with byte offset — used for chunk-to-symbol mapping.
 * Never exposed through SDK/MCP/CLI.
 */
export interface InternalSymbol {
  name: string;
  kind: string;
  signature?: string;
  line: number;
  pos: number;
}

/**
 * Public symbol type — user-facing, no internal byte offset.
 */
export interface SymbolInfo {
  name: string;
  kind: string;
  signature?: string;
  line: number;
}

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

/** Cached compiled breakpoint queries per language. */
const queryCache = new Map<string, QueryType>();

/** Cached compiled symbol queries per language. */
const symbolQueryCache = new Map<string, QueryType>();

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
 * Get or create a compiled breakpoint query for the given language.
 */
function getBreakpointQuery(language: SupportedLanguage, grammar: LanguageType): QueryType {
  if (!queryCache.has(language)) {
    const source = LANGUAGE_QUERIES[language];
    const query = new QueryClass!(grammar, source);
    queryCache.set(language, query);
  }
  return queryCache.get(language)!;
}

/**
 * Get or create a compiled symbol query for the given language.
 */
function getSymbolQuery(language: SupportedLanguage, grammar: LanguageType): QueryType {
  if (!symbolQueryCache.has(language)) {
    const source = SYMBOL_QUERIES[language];
    const query = new QueryClass!(grammar, source);
    symbolQueryCache.set(language, query);
  }
  return symbolQueryCache.get(language)!;
}

// =============================================================================
// Signature Extraction
// =============================================================================

/**
 * Child node types to include when building a signature.
 * These contain parameters and return types.
 */
const SIGNATURE_INCLUDE_TYPES = new Set([
  // TypeScript / JavaScript
  "formal_parameters", "type_annotation",
  // Python
  "parameters",
  // Go
  "parameter_list", "pointer_type", "qualified_type",
  "slice_type", "map_type", "array_type", "generic_type",
  // Rust
  "parameters",
]);

/**
 * Child node types to skip when building a signature.
 * These contain implementation bodies, keywords, or name nodes we don't need.
 */
const SIGNATURE_SKIP_TYPES = new Set([
  "statement_block", "block", "body", "declaration_list",
  "field_declaration_list", "enum_variant_list",
  "visibility_modifier", "pub", "async", "def", "fn", "func", "function",
  "class", "interface", "type", "struct", "trait", "impl", "enum",
  "identifier", "type_identifier", "property_identifier", "field_identifier",
  ":", "comment",
  // Go/Rust type-related keywords
  "type_spec",
]);

/**
 * Symbol kinds that are "type-like" — they don't have meaningful signatures.
 * The name field already carries all the identity info; a signature would
 * just repeat the name or capture irrelevant child text.
 */
const NO_SIGNATURE_KINDS = new Set([
  "class", "interface", "struct", "enum", "trait", "impl", "type",
]);

/**
 * Extract a compact signature from an AST node by concatenating
 * parameter and type children, skipping body/keyword children.
 */
function extractSignature(node: NodeType, kind: string, language: SupportedLanguage): string | undefined {
  // Type-like declarations don't have meaningful signatures
  if (NO_SIGNATURE_KINDS.has(kind)) return undefined;

  // For lexical_declaration wrapping arrow/function expressions,
  // drill into the actual function node for signature children
  let sigNode = node;
  if (node.type === "lexical_declaration") {
    for (const child of node.children) {
      if (child.type === "variable_declarator") {
        for (const grandchild of child.children) {
          if (grandchild.type === "arrow_function" || grandchild.type === "function_expression") {
            sigNode = grandchild;
            break;
          }
        }
        break;
      }
    }
  }

  const parts: string[] = [];

  for (const child of sigNode.children) {
    const t = child.type;

    if (SIGNATURE_INCLUDE_TYPES.has(t)) {
      parts.push(child.text);
    } else if (t === "->" && (language === "python" || language === "rust")) {
      parts.push("->");
    } else if (t === "type" && language === "python") {
      // Python return type after ->
      parts.push(child.text);
    } else if (language === "go" && t === "type_identifier") {
      // Go return type (e.g., "error", "string") — must check before SKIP set
      parts.push(child.text);
    } else if (language === "rust" && (t === "type_identifier" || t === "generic_type" || t === "scoped_type_identifier")) {
      // Rust return type after -> — must check before SKIP set
      parts.push(child.text);
    } else if (SIGNATURE_SKIP_TYPES.has(t)) {
      continue;
    }
  }

  if (parts.length === 0) return undefined;
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

// =============================================================================
// Unified Parse: Breakpoints + Symbols from One Tree
// =============================================================================

/**
 * Parse a code file once and extract both break points and symbols.
 * This is the single-parse optimization — avoids parsing the same file twice
 * during embedding (once for chunking, once for symbols).
 *
 * Returns empty results for unsupported languages or parse failures.
 */
export async function parseCodeFile(
  content: string,
  filepath: string,
): Promise<{ breakPoints: BreakPoint[]; symbols: InternalSymbol[] }> {
  const empty = { breakPoints: [], symbols: [] };
  const language = detectLanguage(filepath);
  if (!language) return empty;

  try {
    await ensureInit();

    const grammar = await loadGrammar(language);
    if (!grammar) return empty;

    const parser = new ParserClass!();
    let tree: ReturnType<typeof parser.parse> | null = null;
    try {
      parser.setLanguage(grammar);

      tree = parser.parse(content);
      if (!tree) {
        return empty;
      }

      // --- Break points (same logic as getASTBreakPoints) ---
      const bpQuery = getBreakpointQuery(language, grammar);
      const bpCaptures = bpQuery.captures(tree.rootNode);

      const bpSeen = new Map<number, BreakPoint>();
      for (const cap of bpCaptures) {
        const pos = cap.node.startIndex;
        const score = SCORE_MAP[cap.name] ?? 20;
        const type = `ast:${cap.name}`;
        const existing = bpSeen.get(pos);
        if (!existing || score > existing.score) {
          bpSeen.set(pos, { pos, score, type });
        }
      }
      const breakPoints = Array.from(bpSeen.values()).sort((a, b) => a.pos - b.pos);

      // --- Symbols ---
      const symQuery = getSymbolQuery(language, grammar);
      const symMatches = symQuery.matches(tree.rootNode);
      const symbols = extractSymbolsFromMatches(symMatches, language);

      return { breakPoints, symbols };
    } finally {
      tree?.delete();
      parser.delete();
    }
  } catch (err) {
    console.warn(`[qmd] AST parse failed for ${filepath}, falling back to regex: ${err instanceof Error ? err.message : err}`);
    return empty;
  }
}

/**
 * Process tree-sitter query matches into InternalSymbol[].
 */
function extractSymbolsFromMatches(
  matches: import("web-tree-sitter").QueryMatch[],
  language: SupportedLanguage,
): InternalSymbol[] {
  const symbols: InternalSymbol[] = [];

  for (const match of matches) {
    let outerNode: NodeType | null = null;
    let nameText: string | null = null;
    let captureName: string | null = null;

    for (const cap of match.captures) {
      if (cap.name.endsWith("_name")) {
        nameText = cap.node.text;
      } else {
        outerNode = cap.node;
        captureName = cap.name;
      }
    }

    if (!outerNode || !nameText || !captureName) continue;

    const kind = KIND_MAP[captureName];
    if (!kind) continue;

    const signature = extractSignature(outerNode, kind, language);

    symbols.push({
      name: nameText,
      kind,
      signature,
      line: outerNode.startPosition.row + 1,
      pos: outerNode.startIndex,
    });
  }

  return symbols.sort((a, b) => a.pos - b.pos);
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
        // Verify both breakpoint and symbol queries compile
        getBreakpointQuery(lang, grammar);
        getSymbolQuery(lang, grammar);
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
// Public API: Break Point Extraction
// =============================================================================

/**
 * Parse a source file and return break points at AST node boundaries.
 * Delegates to parseCodeFile() for the actual parsing.
 */
export async function getASTBreakPoints(
  content: string,
  filepath: string,
): Promise<BreakPoint[]> {
  const { breakPoints } = await parseCodeFile(content, filepath);
  return breakPoints;
}

// =============================================================================
// Public API: Symbol Extraction
// =============================================================================

/**
 * Extract all symbols from a source file. Returns InternalSymbol[] with
 * byte offsets for chunk-to-symbol mapping during embedding.
 */
export async function extractAllSymbols(
  content: string,
  filepath: string,
): Promise<InternalSymbol[]> {
  const { symbols } = await parseCodeFile(content, filepath);
  return symbols;
}

/**
 * Extract symbols within a byte range. Returns public SymbolInfo[] (no pos).
 * Used at query time to enrich search results.
 */
export async function extractSymbols(
  content: string,
  filepath: string,
  startPos: number,
  endPos: number,
): Promise<SymbolInfo[]> {
  const all = await extractAllSymbols(content, filepath);
  return all
    .filter(s => s.pos >= startPos && s.pos < endPos)
    .map(({ name, kind, signature, line }) => ({ name, kind, signature, line }));
}
