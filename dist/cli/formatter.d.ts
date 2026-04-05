/**
 * formatter.ts - Output formatting utilities for QMD
 *
 * Provides methods to format search results and documents into various output formats:
 * JSON, CSV, XML, Markdown, files list, and CLI (colored terminal output).
 */
import type { SearchResult, MultiGetResult, DocumentResult } from "../store.js";
export type { SearchResult, MultiGetResult, DocumentResult };
export type MultiGetFile = {
    filepath: string;
    displayPath: string;
    title: string;
    body: string;
    context?: string | null;
    skipped: false;
} | {
    filepath: string;
    displayPath: string;
    title: string;
    body: string;
    context?: string | null;
    skipped: true;
    skipReason: string;
};
export type OutputFormat = "cli" | "csv" | "md" | "xml" | "files" | "json";
export type FormatOptions = {
    full?: boolean;
    query?: string;
    useColor?: boolean;
    lineNumbers?: boolean;
    intent?: string;
};
/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 * @param text The text to add line numbers to
 * @param startLine Optional starting line number (default: 1)
 */
export declare function addLineNumbers(text: string, startLine?: number): string;
/**
 * Extract short docid from a full hash (first 6 characters).
 */
export declare function getDocid(hash: string): string;
export declare function escapeCSV(value: string | null | number): string;
export declare function escapeXml(str: string): string;
/**
 * Format search results as JSON
 */
export declare function searchResultsToJson(results: SearchResult[], opts?: FormatOptions): string;
/**
 * Format search results as CSV
 */
export declare function searchResultsToCsv(results: SearchResult[], opts?: FormatOptions): string;
/**
 * Format search results as simple files list (docid,score,filepath,context)
 */
export declare function searchResultsToFiles(results: SearchResult[]): string;
/**
 * Format search results as Markdown
 */
export declare function searchResultsToMarkdown(results: SearchResult[], opts?: FormatOptions): string;
/**
 * Format search results as XML
 */
export declare function searchResultsToXml(results: SearchResult[], opts?: FormatOptions): string;
/**
 * Format search results for MCP (simpler CSV format with pre-extracted snippets)
 */
export declare function searchResultsToMcpCsv(results: {
    docid: string;
    file: string;
    title: string;
    score: number;
    context: string | null;
    snippet: string;
}[]): string;
/**
 * Format documents as JSON
 */
export declare function documentsToJson(results: MultiGetFile[]): string;
/**
 * Format documents as CSV
 */
export declare function documentsToCsv(results: MultiGetFile[]): string;
/**
 * Format documents as files list
 */
export declare function documentsToFiles(results: MultiGetFile[]): string;
/**
 * Format documents as Markdown
 */
export declare function documentsToMarkdown(results: MultiGetFile[]): string;
/**
 * Format documents as XML
 */
export declare function documentsToXml(results: MultiGetFile[]): string;
/**
 * Format a single DocumentResult as JSON
 */
export declare function documentToJson(doc: DocumentResult): string;
/**
 * Format a single DocumentResult as Markdown
 */
export declare function documentToMarkdown(doc: DocumentResult): string;
/**
 * Format a single DocumentResult as XML
 */
export declare function documentToXml(doc: DocumentResult): string;
/**
 * Format a single document to the specified format
 */
export declare function formatDocument(doc: DocumentResult, format: OutputFormat): string;
/**
 * Format search results to the specified output format
 */
export declare function formatSearchResults(results: SearchResult[], format: OutputFormat, opts?: FormatOptions): string;
/**
 * Format documents to the specified output format
 */
export declare function formatDocuments(results: MultiGetFile[], format: OutputFormat): string;
