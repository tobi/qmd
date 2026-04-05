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
import type { BreakPoint } from "./store.js";
export type SupportedLanguage = "typescript" | "tsx" | "javascript" | "python" | "go" | "rust";
/**
 * Detect language from file path extension.
 * Returns null for unsupported or unknown extensions (including .md).
 */
export declare function detectLanguage(filepath: string): SupportedLanguage | null;
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
export declare function getASTBreakPoints(content: string, filepath: string): Promise<BreakPoint[]>;
/**
 * Check which tree-sitter grammars are available.
 * Returns a status object for each supported language.
 */
export declare function getASTStatus(): Promise<{
    available: boolean;
    languages: {
        language: SupportedLanguage;
        available: boolean;
        error?: string;
    }[];
}>;
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
export declare function extractSymbols(_content: string, _language: string, _startPos: number, _endPos: number): SymbolInfo[];
