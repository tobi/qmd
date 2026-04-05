/**
 * formatter.ts - Output formatting utilities for QMD
 *
 * Provides methods to format search results and documents into various output formats:
 * JSON, CSV, XML, Markdown, files list, and CLI (colored terminal output).
 */
import { extractSnippet } from "../store.js";
// =============================================================================
// Helper Functions
// =============================================================================
/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 * @param text The text to add line numbers to
 * @param startLine Optional starting line number (default: 1)
 */
export function addLineNumbers(text, startLine = 1) {
    const lines = text.split('\n');
    return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}
/**
 * Extract short docid from a full hash (first 6 characters).
 */
export function getDocid(hash) {
    return hash.slice(0, 6);
}
// =============================================================================
// Escape Helpers
// =============================================================================
export function escapeCSV(value) {
    if (value === null || value === undefined)
        return "";
    const str = String(value);
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
        return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
}
export function escapeXml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
}
// =============================================================================
// Search Results Formatters
// =============================================================================
/**
 * Format search results as JSON
 */
export function searchResultsToJson(results, opts = {}) {
    const query = opts.query || "";
    const output = results.map(row => {
        const bodyStr = row.body || "";
        const snippetInfo = bodyStr
            ? extractSnippet(bodyStr, query, 300, row.chunkPos, undefined, opts.intent)
            : undefined;
        let body = opts.full ? bodyStr : undefined;
        let snippet = !opts.full ? snippetInfo?.snippet : undefined;
        if (opts.lineNumbers) {
            if (body)
                body = addLineNumbers(body);
            if (snippet)
                snippet = addLineNumbers(snippet);
        }
        return {
            docid: `#${row.docid}`,
            score: Math.round(row.score * 100) / 100,
            file: row.displayPath,
            ...(snippetInfo && { line: snippetInfo.line }),
            title: row.title,
            ...(row.context && { context: row.context }),
            ...(body && { body }),
            ...(snippet && { snippet }),
        };
    });
    return JSON.stringify(output, null, 2);
}
/**
 * Format search results as CSV
 */
export function searchResultsToCsv(results, opts = {}) {
    const query = opts.query || "";
    const header = "docid,score,file,title,context,line,snippet";
    const rows = results.map(row => {
        const bodyStr = row.body || "";
        const { line, snippet } = extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent);
        let content = opts.full ? bodyStr : snippet;
        if (opts.lineNumbers && content) {
            content = addLineNumbers(content);
        }
        return [
            `#${row.docid}`,
            row.score.toFixed(4),
            escapeCSV(row.displayPath),
            escapeCSV(row.title),
            escapeCSV(row.context || ""),
            line,
            escapeCSV(content),
        ].join(",");
    });
    return [header, ...rows].join("\n");
}
/**
 * Format search results as simple files list (docid,score,filepath,context)
 */
export function searchResultsToFiles(results) {
    return results.map(row => {
        const ctx = row.context ? `,"${row.context.replace(/"/g, '""')}"` : "";
        return `#${row.docid},${row.score.toFixed(2)},${row.displayPath}${ctx}`;
    }).join("\n");
}
/**
 * Format search results as Markdown
 */
export function searchResultsToMarkdown(results, opts = {}) {
    const query = opts.query || "";
    return results.map(row => {
        const heading = row.title || row.displayPath;
        const bodyStr = row.body || "";
        let content;
        if (opts.full) {
            content = bodyStr;
        }
        else {
            content = extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent).snippet;
        }
        if (opts.lineNumbers) {
            content = addLineNumbers(content);
        }
        const contextLine = row.context ? `**context:** ${row.context}\n` : "";
        return `---\n# ${heading}\n\n**docid:** \`#${row.docid}\`\n${contextLine}\n${content}\n`;
    }).join("\n");
}
/**
 * Format search results as XML
 */
export function searchResultsToXml(results, opts = {}) {
    const query = opts.query || "";
    const items = results.map(row => {
        const titleAttr = row.title ? ` title="${escapeXml(row.title)}"` : "";
        const bodyStr = row.body || "";
        let content = opts.full ? bodyStr : extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent).snippet;
        if (opts.lineNumbers) {
            content = addLineNumbers(content);
        }
        const contextAttr = row.context ? ` context="${escapeXml(row.context)}"` : "";
        return `<file docid="#${row.docid}" name="${escapeXml(row.displayPath)}"${titleAttr}${contextAttr}>\n${escapeXml(content)}\n</file>`;
    });
    return items.join("\n\n");
}
/**
 * Format search results for MCP (simpler CSV format with pre-extracted snippets)
 */
export function searchResultsToMcpCsv(results) {
    const header = "docid,file,title,score,context,snippet";
    const rows = results.map(r => [`#${r.docid}`, r.file, r.title, r.score, r.context || "", r.snippet].map(escapeCSV).join(","));
    return [header, ...rows].join("\n");
}
// =============================================================================
// Document Formatters (for multi-get using MultiGetFile from store)
// =============================================================================
/**
 * Format documents as JSON
 */
export function documentsToJson(results) {
    const output = results.map(r => ({
        file: r.displayPath,
        title: r.title,
        ...(r.context && { context: r.context }),
        ...(r.skipped ? { skipped: true, reason: r.skipReason } : { body: r.body }),
    }));
    return JSON.stringify(output, null, 2);
}
/**
 * Format documents as CSV
 */
export function documentsToCsv(results) {
    const header = "file,title,context,skipped,body";
    const rows = results.map(r => [
        r.displayPath,
        r.title,
        r.context || "",
        r.skipped ? "true" : "false",
        r.skipped ? (r.skipReason || "") : r.body
    ].map(escapeCSV).join(","));
    return [header, ...rows].join("\n");
}
/**
 * Format documents as files list
 */
export function documentsToFiles(results) {
    return results.map(r => {
        const ctx = r.context ? `,"${r.context.replace(/"/g, '""')}"` : "";
        const status = r.skipped ? ",[SKIPPED]" : "";
        return `${r.displayPath}${ctx}${status}`;
    }).join("\n");
}
/**
 * Format documents as Markdown
 */
export function documentsToMarkdown(results) {
    return results.map(r => {
        let md = `## ${r.displayPath}\n\n`;
        if (r.title && r.title !== r.displayPath)
            md += `**Title:** ${r.title}\n\n`;
        if (r.context)
            md += `**Context:** ${r.context}\n\n`;
        if (r.skipped) {
            md += `> ${r.skipReason}\n`;
        }
        else {
            md += "```\n" + r.body + "\n```\n";
        }
        return md;
    }).join("\n");
}
/**
 * Format documents as XML
 */
export function documentsToXml(results) {
    const items = results.map(r => {
        let xml = "  <document>\n";
        xml += `    <file>${escapeXml(r.displayPath)}</file>\n`;
        xml += `    <title>${escapeXml(r.title)}</title>\n`;
        if (r.context)
            xml += `    <context>${escapeXml(r.context)}</context>\n`;
        if (r.skipped) {
            xml += `    <skipped>true</skipped>\n`;
            xml += `    <reason>${escapeXml(r.skipReason || "")}</reason>\n`;
        }
        else {
            xml += `    <body>${escapeXml(r.body)}</body>\n`;
        }
        xml += "  </document>";
        return xml;
    });
    return `<?xml version="1.0" encoding="UTF-8"?>\n<documents>\n${items.join("\n")}\n</documents>`;
}
// =============================================================================
// Single Document Formatters
// =============================================================================
/**
 * Format a single DocumentResult as JSON
 */
export function documentToJson(doc) {
    return JSON.stringify({
        file: doc.displayPath,
        title: doc.title,
        ...(doc.context && { context: doc.context }),
        hash: doc.hash,
        modifiedAt: doc.modifiedAt,
        bodyLength: doc.bodyLength,
        ...(doc.body !== undefined && { body: doc.body }),
    }, null, 2);
}
/**
 * Format a single DocumentResult as Markdown
 */
export function documentToMarkdown(doc) {
    let md = `# ${doc.title || doc.displayPath}\n\n`;
    if (doc.context)
        md += `**Context:** ${doc.context}\n\n`;
    md += `**File:** ${doc.displayPath}\n`;
    md += `**Modified:** ${doc.modifiedAt}\n\n`;
    if (doc.body !== undefined) {
        md += "---\n\n" + doc.body + "\n";
    }
    return md;
}
/**
 * Format a single DocumentResult as XML
 */
export function documentToXml(doc) {
    let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<document>\n`;
    xml += `  <file>${escapeXml(doc.displayPath)}</file>\n`;
    xml += `  <title>${escapeXml(doc.title)}</title>\n`;
    if (doc.context)
        xml += `  <context>${escapeXml(doc.context)}</context>\n`;
    xml += `  <hash>${escapeXml(doc.hash)}</hash>\n`;
    xml += `  <modifiedAt>${escapeXml(doc.modifiedAt)}</modifiedAt>\n`;
    xml += `  <bodyLength>${doc.bodyLength}</bodyLength>\n`;
    if (doc.body !== undefined) {
        xml += `  <body>${escapeXml(doc.body)}</body>\n`;
    }
    xml += `</document>`;
    return xml;
}
/**
 * Format a single document to the specified format
 */
export function formatDocument(doc, format) {
    switch (format) {
        case "json":
            return documentToJson(doc);
        case "md":
            return documentToMarkdown(doc);
        case "xml":
            return documentToXml(doc);
        default:
            // Default to markdown for CLI and other formats
            return documentToMarkdown(doc);
    }
}
// =============================================================================
// Universal Format Function
// =============================================================================
/**
 * Format search results to the specified output format
 */
export function formatSearchResults(results, format, opts = {}) {
    switch (format) {
        case "json":
            return searchResultsToJson(results, opts);
        case "csv":
            return searchResultsToCsv(results, opts);
        case "files":
            return searchResultsToFiles(results);
        case "md":
            return searchResultsToMarkdown(results, opts);
        case "xml":
            return searchResultsToXml(results, opts);
        case "cli":
            // CLI format should be handled separately with colors
            // Return a simple text version as fallback
            return searchResultsToMarkdown(results, opts);
        default:
            return searchResultsToJson(results, opts);
    }
}
/**
 * Format documents to the specified output format
 */
export function formatDocuments(results, format) {
    switch (format) {
        case "json":
            return documentsToJson(results);
        case "csv":
            return documentsToCsv(results);
        case "files":
            return documentsToFiles(results);
        case "md":
            return documentsToMarkdown(results);
        case "xml":
            return documentsToXml(results);
        case "cli":
            // CLI format should be handled separately with colors
            return documentsToMarkdown(results);
        default:
            return documentsToJson(results);
    }
}
