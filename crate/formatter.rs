//! formatter.rs - Output formatting utilities
//!
//! Formats search results and documents into JSON, CSV, XML, Markdown,
//! files list, and CLI (colored terminal) output.

use crate::search::extract_snippet;
use crate::store::{DocumentResult, MultiGetResult, SearchResult};

// =============================================================================
// Types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Cli,
    Json,
    Csv,
    Md,
    Xml,
    Files,
}

impl OutputFormat {
    pub fn from_flags(json: bool, csv: bool, md: bool, xml: bool, files: bool) -> Self {
        if csv {
            Self::Csv
        } else if md {
            Self::Md
        } else if xml {
            Self::Xml
        } else if files {
            Self::Files
        } else if json {
            Self::Json
        } else {
            Self::Cli
        }
    }

    /// Default result limit for this format.
    pub fn default_limit(&self) -> usize {
        match self {
            Self::Files | Self::Json => 20,
            _ => 5,
        }
    }
}

pub struct FormatOptions {
    pub full: bool,
    pub query: String,
    pub line_numbers: bool,
    pub intent: Option<String>,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            full: false,
            query: String::new(),
            line_numbers: false,
            intent: None,
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Add line numbers to text content.
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {line}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// =============================================================================
// Search Results Formatters
// =============================================================================

pub fn format_search_results(
    results: &[SearchResult],
    format: OutputFormat,
    opts: &FormatOptions,
) -> String {
    match format {
        OutputFormat::Json => search_results_to_json(results, opts),
        OutputFormat::Csv => search_results_to_csv(results, opts),
        OutputFormat::Files => search_results_to_files(results),
        OutputFormat::Md => search_results_to_markdown(results, opts),
        OutputFormat::Xml => search_results_to_xml(results, opts),
        OutputFormat::Cli => search_results_to_cli(results, opts),
    }
}

fn search_results_to_json(results: &[SearchResult], opts: &FormatOptions) -> String {
    let items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let body_str = r.body.as_deref().unwrap_or("");
            let (snippet, _line) =
                extract_snippet(body_str, &opts.query, 300, r.chunk_pos, None, opts.intent.as_deref());

            let mut obj = serde_json::json!({
                "docid": format!("#{}", r.docid),
                "score": (r.score * 100.0).round() / 100.0,
                "file": r.display_path,
                "title": r.title,
            });

            if let Some(ctx) = &r.context {
                obj["context"] = serde_json::Value::String(ctx.clone());
            }

            if opts.full {
                let mut body = body_str.to_string();
                if opts.line_numbers {
                    body = add_line_numbers(&body, 1);
                }
                obj["body"] = serde_json::Value::String(body);
            } else {
                let mut s = snippet;
                if opts.line_numbers {
                    s = add_line_numbers(&s, 1);
                }
                obj["snippet"] = serde_json::Value::String(s);
            }

            obj
        })
        .collect();

    serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
}

fn search_results_to_csv(results: &[SearchResult], opts: &FormatOptions) -> String {
    let header = "docid,score,file,title,context,line,snippet";
    let rows: Vec<String> = results
        .iter()
        .map(|r| {
            let body_str = r.body.as_deref().unwrap_or("");
            let (snippet, line) =
                extract_snippet(body_str, &opts.query, 500, r.chunk_pos, None, opts.intent.as_deref());
            let content = if opts.full {
                body_str.to_string()
            } else {
                snippet
            };
            format!(
                "#{},{:.4},{},{},{},{},{}",
                r.docid,
                r.score,
                escape_csv(&r.display_path),
                escape_csv(&r.title),
                escape_csv(r.context.as_deref().unwrap_or("")),
                line,
                escape_csv(&content),
            )
        })
        .collect();

    std::iter::once(header.to_string())
        .chain(rows)
        .collect::<Vec<_>>()
        .join("\n")
}

fn search_results_to_files(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|r| {
            let ctx = r
                .context
                .as_ref()
                .map(|c| format!(",\"{}\"", c.replace('"', "\"\"")))
                .unwrap_or_default();
            format!("#{},{:.2},{}{ctx}", r.docid, r.score, r.display_path)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn search_results_to_markdown(results: &[SearchResult], opts: &FormatOptions) -> String {
    results
        .iter()
        .map(|r| {
            let heading = if r.title.is_empty() {
                &r.display_path
            } else {
                &r.title
            };
            let body_str = r.body.as_deref().unwrap_or("");
            let mut content = if opts.full {
                body_str.to_string()
            } else {
                extract_snippet(body_str, &opts.query, 500, r.chunk_pos, None, opts.intent.as_deref()).0
            };
            if opts.line_numbers {
                content = add_line_numbers(&content, 1);
            }
            let ctx_line = r
                .context
                .as_ref()
                .map(|c| format!("**context:** {c}\n"))
                .unwrap_or_default();
            format!(
                "---\n# {heading}\n\n**docid:** `#{}`\n{ctx_line}\n{content}\n",
                r.docid
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn search_results_to_xml(results: &[SearchResult], opts: &FormatOptions) -> String {
    results
        .iter()
        .map(|r| {
            let title_attr = if r.title.is_empty() {
                String::new()
            } else {
                format!(" title=\"{}\"", escape_xml(&r.title))
            };
            let body_str = r.body.as_deref().unwrap_or("");
            let mut content = if opts.full {
                body_str.to_string()
            } else {
                extract_snippet(body_str, &opts.query, 500, r.chunk_pos, None, opts.intent.as_deref()).0
            };
            if opts.line_numbers {
                content = add_line_numbers(&content, 1);
            }
            let ctx_attr = r
                .context
                .as_ref()
                .map(|c| format!(" context=\"{}\"", escape_xml(c)))
                .unwrap_or_default();
            format!(
                "<file docid=\"#{}\" name=\"{}\"{title_attr}{ctx_attr}>\n{}\n</file>",
                r.docid,
                escape_xml(&r.display_path),
                escape_xml(&content),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn search_results_to_cli(results: &[SearchResult], opts: &FormatOptions) -> String {
    use colored::Colorize;

    if results.is_empty() {
        return "No results found.".dimmed().to_string();
    }

    results
        .iter()
        .map(|r| {
            let score_pct = (r.score * 100.0).round() as u32;
            let score_str = format!("{score_pct}%").yellow().to_string();
            let docid_str = format!("#{}", r.docid).dimmed().to_string();
            let path_str = r.display_path.cyan().to_string();
            let title_str = if r.title.is_empty() {
                String::new()
            } else {
                format!(" - {}", r.title)
            };

            let body_str = r.body.as_deref().unwrap_or("");
            let content = if opts.full {
                body_str.to_string()
            } else {
                extract_snippet(body_str, &opts.query, 300, r.chunk_pos, None, opts.intent.as_deref()).0
            };

            format!("{score_str} {docid_str} {path_str}{title_str}\n  {content}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

// =============================================================================
// Document Formatters (multi-get)
// =============================================================================

pub fn format_documents(results: &[MultiGetResult], format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => documents_to_json(results),
        OutputFormat::Csv => documents_to_csv(results),
        OutputFormat::Files => documents_to_files(results),
        OutputFormat::Md => documents_to_markdown(results),
        OutputFormat::Xml => documents_to_xml(results),
        OutputFormat::Cli => documents_to_markdown(results),
    }
}

fn documents_to_json(results: &[MultiGetResult]) -> String {
    let items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let mut obj = serde_json::json!({
                "file": r.display_path,
                "title": r.title,
            });
            if let Some(ctx) = &r.context {
                obj["context"] = serde_json::Value::String(ctx.clone());
            }
            if r.skipped {
                obj["skipped"] = serde_json::Value::Bool(true);
                if let Some(reason) = &r.skip_reason {
                    obj["reason"] = serde_json::Value::String(reason.clone());
                }
            } else {
                obj["body"] = serde_json::Value::String(r.body.clone());
            }
            obj
        })
        .collect();

    serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
}

fn documents_to_csv(results: &[MultiGetResult]) -> String {
    let header = "file,title,context,skipped,body";
    let rows: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{},{},{},{},{}",
                escape_csv(&r.display_path),
                escape_csv(&r.title),
                escape_csv(r.context.as_deref().unwrap_or("")),
                if r.skipped { "true" } else { "false" },
                escape_csv(if r.skipped {
                    r.skip_reason.as_deref().unwrap_or("")
                } else {
                    &r.body
                }),
            )
        })
        .collect();

    std::iter::once(header.to_string())
        .chain(rows)
        .collect::<Vec<_>>()
        .join("\n")
}

fn documents_to_files(results: &[MultiGetResult]) -> String {
    results
        .iter()
        .map(|r| {
            let ctx = r
                .context
                .as_ref()
                .map(|c| format!(",\"{}\"", c.replace('"', "\"\"")))
                .unwrap_or_default();
            let status = if r.skipped { ",[SKIPPED]" } else { "" };
            format!("{}{ctx}{status}", r.display_path)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn documents_to_markdown(results: &[MultiGetResult]) -> String {
    results
        .iter()
        .map(|r| {
            let mut md = format!("## {}\n\n", r.display_path);
            if !r.title.is_empty() && r.title != r.display_path {
                md += &format!("**Title:** {}\n\n", r.title);
            }
            if let Some(ctx) = &r.context {
                md += &format!("**Context:** {ctx}\n\n");
            }
            if r.skipped {
                if let Some(reason) = &r.skip_reason {
                    md += &format!("> {reason}\n");
                }
            } else {
                md += &format!("```\n{}\n```\n", r.body);
            }
            md
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn documents_to_xml(results: &[MultiGetResult]) -> String {
    let items: Vec<String> = results
        .iter()
        .map(|r| {
            let mut xml = "  <document>\n".to_string();
            xml += &format!("    <file>{}</file>\n", escape_xml(&r.display_path));
            xml += &format!("    <title>{}</title>\n", escape_xml(&r.title));
            if let Some(ctx) = &r.context {
                xml += &format!("    <context>{}</context>\n", escape_xml(ctx));
            }
            if r.skipped {
                xml += "    <skipped>true</skipped>\n";
                if let Some(reason) = &r.skip_reason {
                    xml += &format!("    <reason>{}</reason>\n", escape_xml(reason));
                }
            } else {
                xml += &format!("    <body>{}</body>\n", escape_xml(&r.body));
            }
            xml += "  </document>";
            xml
        })
        .collect();

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<documents>\n{}\n</documents>",
        items.join("\n")
    )
}

// =============================================================================
// Single Document Formatters
// =============================================================================

pub fn format_document(doc: &DocumentResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => document_to_json(doc),
        OutputFormat::Md | OutputFormat::Cli => document_to_markdown(doc),
        OutputFormat::Xml => document_to_xml(doc),
        _ => document_to_markdown(doc),
    }
}

fn document_to_json(doc: &DocumentResult) -> String {
    let mut obj = serde_json::json!({
        "file": doc.display_path,
        "title": doc.title,
        "hash": doc.hash,
        "modifiedAt": doc.modified_at,
        "bodyLength": doc.body_length,
    });
    if let Some(ctx) = &doc.context {
        obj["context"] = serde_json::Value::String(ctx.clone());
    }
    if let Some(body) = &doc.body {
        obj["body"] = serde_json::Value::String(body.clone());
    }
    serde_json::to_string_pretty(&obj).unwrap_or_else(|_| "{}".to_string())
}

fn document_to_markdown(doc: &DocumentResult) -> String {
    let heading = if doc.title.is_empty() {
        &doc.display_path
    } else {
        &doc.title
    };
    let mut md = format!("# {heading}\n\n");
    if let Some(ctx) = &doc.context {
        md += &format!("**Context:** {ctx}\n\n");
    }
    md += &format!("**File:** {}\n", doc.display_path);
    md += &format!("**Modified:** {}\n\n", doc.modified_at);
    if let Some(body) = &doc.body {
        md += &format!("---\n\n{body}\n");
    }
    md
}

fn document_to_xml(doc: &DocumentResult) -> String {
    let mut xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<document>\n".to_string();
    xml += &format!("  <file>{}</file>\n", escape_xml(&doc.display_path));
    xml += &format!("  <title>{}</title>\n", escape_xml(&doc.title));
    if let Some(ctx) = &doc.context {
        xml += &format!("  <context>{}</context>\n", escape_xml(ctx));
    }
    xml += &format!("  <hash>{}</hash>\n", escape_xml(&doc.hash));
    xml += &format!("  <modifiedAt>{}</modifiedAt>\n", escape_xml(&doc.modified_at));
    xml += &format!("  <bodyLength>{}</bodyLength>\n", doc.body_length);
    if let Some(body) = &doc.body {
        xml += &format!("  <body>{}</body>\n", escape_xml(body));
    }
    xml += "</document>";
    xml
}
