/**
 * formatter.test.ts - Unit tests verifying context is shown in all output formats
 *
 * Run with: bun test formatter.test.ts
 */

import { describe, test, expect } from "bun:test";
import {
  // Search result formatters
  searchResultsToJson,
  searchResultsToCsv,
  searchResultsToFiles,
  searchResultsToMarkdown,
  searchResultsToXml,
  searchResultsToMcpCsv,
  formatSearchResults,
  // Document (multi-get) formatters
  documentsToJson,
  documentsToCsv,
  documentsToFiles,
  documentsToMarkdown,
  documentsToXml,
  formatDocuments,
  // Single document formatters
  documentToJson,
  documentToMarkdown,
  documentToXml,
  formatDocument,
  type MultiGetFile,
} from "./formatter.js";
import type { SearchResult, DocumentResult } from "./store.js";

// =============================================================================
// Test Fixtures
// =============================================================================

const TEST_CONTEXT = "Internal engineering keynotes from company summit events";

function makeSearchResult(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    filepath: "qmd://archive/summit/keynote.md",
    displayPath: "qmd://archive/summit/keynote.md",
    title: "Summit Keynote",
    context: TEST_CONTEXT,
    hash: "dc5590abcdef",
    docid: "dc5590",
    collectionName: "archive",
    modifiedAt: "2024-01-01T00:00:00Z",
    bodyLength: 100,
    body: "---\ntitle: Summit Keynote\n---\n\nThis is the keynote content.",
    score: 0.84,
    source: "fts",
    ...overrides,
  };
}

function makeDocumentResult(overrides: Partial<DocumentResult> = {}): DocumentResult {
  return {
    filepath: "qmd://archive/summit/keynote.md",
    displayPath: "qmd://archive/summit/keynote.md",
    title: "Summit Keynote",
    context: TEST_CONTEXT,
    hash: "dc5590abcdef",
    docid: "dc5590",
    collectionName: "archive",
    modifiedAt: "2024-01-01T00:00:00Z",
    bodyLength: 100,
    body: "---\ntitle: Summit Keynote\n---\n\nThis is the keynote content.",
    ...overrides,
  };
}

function makeMultiGetFile(overrides: Partial<MultiGetFile & { skipped: false }> = {}): MultiGetFile {
  return {
    filepath: "qmd://archive/summit/keynote.md",
    displayPath: "qmd://archive/summit/keynote.md",
    title: "Summit Keynote",
    context: TEST_CONTEXT,
    body: "---\ntitle: Summit Keynote\n---\n\nThis is the keynote content.",
    skipped: false,
    ...overrides,
  };
}

// =============================================================================
// Search Results: Context in Every Format
// =============================================================================

describe("search results include context in all formats", () => {
  const results = [makeSearchResult()];

  test("JSON format includes context", () => {
    const output = searchResultsToJson(results, { query: "keynote" });
    const parsed = JSON.parse(output);
    expect(parsed[0].context).toBe(TEST_CONTEXT);
  });

  test("CSV format includes context", () => {
    const output = searchResultsToCsv(results, { query: "keynote" });
    // Header should have context column
    const lines = output.split("\n");
    expect(lines[0]).toContain("context");
    // Data row should contain the context text
    expect(output).toContain(TEST_CONTEXT);
  });

  test("files format includes context", () => {
    const output = searchResultsToFiles(results);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("Markdown format includes context", () => {
    const output = searchResultsToMarkdown(results, { query: "keynote" });
    expect(output).toContain(TEST_CONTEXT);
  });

  test("XML format includes context", () => {
    const output = searchResultsToXml(results, { query: "keynote" });
    expect(output).toContain(TEST_CONTEXT);
  });

  test("MCP CSV format includes context", () => {
    const mcpResults = [{
      docid: "dc5590",
      file: "qmd://archive/summit/keynote.md",
      title: "Summit Keynote",
      score: 0.84,
      context: TEST_CONTEXT,
      snippet: "This is the keynote content.",
    }];
    const output = searchResultsToMcpCsv(mcpResults);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatSearchResults (JSON) includes context", () => {
    const output = formatSearchResults(results, "json", { query: "keynote" });
    const parsed = JSON.parse(output);
    expect(parsed[0].context).toBe(TEST_CONTEXT);
  });

  test("formatSearchResults (CSV) includes context", () => {
    const output = formatSearchResults(results, "csv", { query: "keynote" });
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatSearchResults (files) includes context", () => {
    const output = formatSearchResults(results, "files");
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatSearchResults (md) includes context", () => {
    const output = formatSearchResults(results, "md", { query: "keynote" });
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatSearchResults (xml) includes context", () => {
    const output = formatSearchResults(results, "xml", { query: "keynote" });
    expect(output).toContain(TEST_CONTEXT);
  });
});

// =============================================================================
// Search Results: No Context When Absent
// =============================================================================

describe("search results omit context when null", () => {
  const results = [makeSearchResult({ context: null })];

  test("JSON format omits context field when null", () => {
    const output = searchResultsToJson(results, { query: "keynote" });
    const parsed = JSON.parse(output);
    expect(parsed[0].context).toBeUndefined();
  });

  test("files format does not include trailing context when null", () => {
    const output = searchResultsToFiles(results);
    // Should just be docid,score,path - no trailing comma/context
    expect(output).not.toContain(",\"");
  });
});

// =============================================================================
// Multi-Get Documents: Context in Every Format
// =============================================================================

describe("multi-get documents include context in all formats", () => {
  const docs = [makeMultiGetFile()];

  test("JSON format includes context", () => {
    const output = documentsToJson(docs);
    const parsed = JSON.parse(output);
    expect(parsed[0].context).toBe(TEST_CONTEXT);
  });

  test("CSV format includes context", () => {
    const output = documentsToCsv(docs);
    const lines = output.split("\n");
    expect(lines[0]).toContain("context");
    expect(output).toContain(TEST_CONTEXT);
  });

  test("files format includes context", () => {
    const output = documentsToFiles(docs);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("Markdown format includes context", () => {
    const output = documentsToMarkdown(docs);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("XML format includes context", () => {
    const output = documentsToXml(docs);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatDocuments (JSON) includes context", () => {
    const output = formatDocuments(docs, "json");
    const parsed = JSON.parse(output);
    expect(parsed[0].context).toBe(TEST_CONTEXT);
  });

  test("formatDocuments (md) includes context", () => {
    const output = formatDocuments(docs, "md");
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatDocuments (xml) includes context", () => {
    const output = formatDocuments(docs, "xml");
    expect(output).toContain(TEST_CONTEXT);
  });
});

// =============================================================================
// Single Document: Context in Every Format
// =============================================================================

describe("single document includes context in all formats", () => {
  const doc = makeDocumentResult();

  test("JSON format includes context", () => {
    const output = documentToJson(doc);
    const parsed = JSON.parse(output);
    expect(parsed.context).toBe(TEST_CONTEXT);
  });

  test("Markdown format includes context", () => {
    const output = documentToMarkdown(doc);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("XML format includes context", () => {
    const output = documentToXml(doc);
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatDocument (JSON) includes context", () => {
    const output = formatDocument(doc, "json");
    const parsed = JSON.parse(output);
    expect(parsed.context).toBe(TEST_CONTEXT);
  });

  test("formatDocument (md) includes context", () => {
    const output = formatDocument(doc, "md");
    expect(output).toContain(TEST_CONTEXT);
  });

  test("formatDocument (xml) includes context", () => {
    const output = formatDocument(doc, "xml");
    expect(output).toContain(TEST_CONTEXT);
  });
});

// =============================================================================
// Single Document: No Context When Absent
// =============================================================================

describe("single document omits context when null", () => {
  const doc = makeDocumentResult({ context: null });

  test("JSON format omits context field when null", () => {
    const output = documentToJson(doc);
    const parsed = JSON.parse(output);
    expect(parsed.context).toBeUndefined();
  });

  test("Markdown format does not show Context line when null", () => {
    const output = documentToMarkdown(doc);
    expect(output).not.toContain("Context:");
  });

  test("XML format does not show context element when null", () => {
    const output = documentToXml(doc);
    expect(output).not.toContain("<context>");
  });
});
