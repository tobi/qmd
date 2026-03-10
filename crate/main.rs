//! QMD - Query Markup Documents
//!
//! On-device hybrid search for markdown files with BM25, vector search,
//! and LLM reranking.

use anyhow::Result;
use clap::Parser;
use colored::Colorize;

use qmd::cli::*;
use qmd::collections;
use qmd::context;
use qmd::formatter::{self, OutputFormat, FormatOptions};
use qmd::search;
use qmd::store::{self, Store};

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set index name if provided
    if let Some(index) = &cli.index {
        collections::set_config_index_name(index);
    }

    match cli.command {
        Command::Collection { action } => handle_collection(action),
        Command::Ls { path } => handle_ls(path),
        Command::Context { action } => handle_context(action),
        Command::Get { file, opts } => handle_get(&file, opts),
        Command::MultiGet { pattern, opts } => handle_multi_get(&pattern, opts),
        Command::Status => handle_status(),
        Command::Update { pull } => handle_update(pull),
        Command::Embed { force } => handle_embed(force),
        Command::Search { query, opts } => handle_search(&query, opts),
        Command::Vsearch { query, opts } => handle_vsearch(&query, opts),
        Command::Query {
            query,
            opts,
            candidate_limit,
            intent,
        } => handle_query(&query, opts, candidate_limit, intent),
        Command::Mcp {
            subcommand,
            http,
            daemon,
            port,
        } => handle_mcp(subcommand, http, daemon, port),
        Command::Cleanup => handle_cleanup(),
    }
}

// =============================================================================
// Command handlers
// =============================================================================

fn handle_collection(action: CollectionAction) -> Result<()> {
    match action {
        CollectionAction::Add { path, name, mask } => {
            let name = name.unwrap_or_else(|| {
                std::path::Path::new(&path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unnamed")
                    .to_string()
            });
            collections::add_collection(&name, &path, &mask)?;
            println!("Collection '{}' added.", name.green());

            // Index the collection
            let store = Store::new(None)?;
            let named = collections::get_collection(&name)?
                .ok_or_else(|| anyhow::anyhow!("Collection not found after adding"))?;
            let result = store::index_collection(&store, &named)?;
            println!(
                "Indexed: {} added, {} updated, {} unchanged, {} removed",
                result.added, result.updated, result.unchanged, result.removed
            );
            Ok(())
        }
        CollectionAction::List => {
            let collections = collections::list_collections()?;
            if collections.is_empty() {
                println!("No collections configured.");
                return Ok(());
            }
            for nc in &collections {
                println!(
                    "  {} ({})",
                    nc.name.cyan(),
                    nc.collection.path.dimmed()
                );
                println!("    Pattern: {}", nc.collection.pattern);
                if let Some(ctx) = &nc.collection.context {
                    for (path, text) in ctx {
                        println!("    Context {}: {}", path.dimmed(), text);
                    }
                }
            }
            Ok(())
        }
        CollectionAction::Remove { name } => {
            // Also clean up database
            let store = Store::new(None)?;
            store.conn.execute(
                "UPDATE documents SET active = 0 WHERE collection = ?1",
                rusqlite::params![name],
            )?;
            if collections::remove_collection(&name)? {
                println!("Collection '{}' removed.", name.green());
            } else {
                println!("Collection '{}' not found.", name.red());
            }
            Ok(())
        }
        CollectionAction::Rename {
            old_name,
            new_name,
        } => {
            if collections::rename_collection(&old_name, &new_name)? {
                // Update database
                let store = Store::new(None)?;
                store.conn.execute(
                    "UPDATE documents SET collection = ?1 WHERE collection = ?2",
                    rusqlite::params![new_name, old_name],
                )?;
                println!("Renamed '{}' → '{}'", old_name, new_name.green());
            } else {
                println!("Collection '{}' not found.", old_name.red());
            }
            Ok(())
        }
    }
}

fn handle_ls(path: Option<String>) -> Result<()> {
    let store = Store::new(None)?;

    match path {
        None => {
            // List all collections with file counts
            let collections = collections::list_collections()?;
            if collections.is_empty() {
                println!("No collections. Run: qmd collection add <path> --name <name>");
                return Ok(());
            }
            println!("{}", "Collections:".bold());
            for nc in &collections {
                let count: usize = store.conn.query_row(
                    "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND active = 1",
                    rusqlite::params![nc.name],
                    |r| r.get(0),
                )?;
                println!(
                    "  {}  ({} files)",
                    format!("qmd://{}/", nc.name).cyan(),
                    count
                );
            }
        }
        Some(path_arg) => {
            // Parse collection/path
            let (collection, prefix) = if let Some(rest) = path_arg.strip_prefix("qmd://") {
                if let Some(pos) = rest.find('/') {
                    (rest[..pos].to_string(), Some(rest[pos + 1..].to_string()))
                } else {
                    (rest.to_string(), None)
                }
            } else if let Some(pos) = path_arg.find('/') {
                (
                    path_arg[..pos].to_string(),
                    Some(path_arg[pos + 1..].to_string()),
                )
            } else {
                (path_arg, None)
            };

            let mut stmt = if let Some(prefix) = &prefix {
                let pattern = format!("{prefix}%");
                let mut s = store.conn.prepare(
                    "SELECT d.path, d.title, LENGTH(c.doc), d.modified_at
                     FROM documents d
                     LEFT JOIN content c ON c.hash = d.hash
                     WHERE d.collection = ?1 AND d.active = 1 AND d.path LIKE ?2
                     ORDER BY d.path",
                )?;
                // Need to use a different approach since we can't return stmt with borrowed params
                let rows: Vec<(String, String, usize, String)> = s
                    .query_map(rusqlite::params![collection, pattern], |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get::<_, usize>(2).unwrap_or(0),
                            row.get(3)?,
                        ))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                for (path, _title, size, modified) in &rows {
                    let size_str = format_size(*size);
                    let date_str = &modified[..10];
                    println!(
                        "  {:>6}  {}  {}",
                        size_str.dimmed(),
                        date_str.dimmed(),
                        format!("qmd://{collection}/{path}").cyan(),
                    );
                }
                return Ok(());
            } else {
                store.conn.prepare(
                    "SELECT d.path, d.title, LENGTH(c.doc), d.modified_at
                     FROM documents d
                     LEFT JOIN content c ON c.hash = d.hash
                     WHERE d.collection = ?1 AND d.active = 1
                     ORDER BY d.path",
                )?
            };

            let rows: Vec<(String, String, usize, String)> = stmt
                .query_map(rusqlite::params![collection], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get::<_, usize>(2).unwrap_or(0),
                        row.get(3)?,
                    ))
                })?
                .filter_map(|r| r.ok())
                .collect();

            for (path, _title, size, modified) in &rows {
                let size_str = format_size(*size);
                let date_str = &modified[..10.min(modified.len())];
                println!(
                    "  {:>6}  {}  {}",
                    size_str.dimmed(),
                    date_str.dimmed(),
                    format!("qmd://{collection}/{path}").cyan(),
                );
            }
        }
    }
    Ok(())
}

fn handle_context(action: ContextAction) -> Result<()> {
    match action {
        ContextAction::Add { path, text } => {
            let path = path.unwrap_or_else(|| ".".to_string());

            // Resolve path to collection + prefix
            if path == "/" {
                collections::set_global_context(&text)?;
                println!("Global context set.");
            } else if let Some((collection, prefix)) = store::parse_virtual_path(&path) {
                context::add_context(&collection, &prefix, &text)?;
                println!("Context added for qmd://{collection}/{prefix}");
            } else {
                // Try to detect collection from current directory
                let cwd = std::env::current_dir()?;
                let abs_path = if path == "." {
                    cwd.clone()
                } else {
                    cwd.join(&path)
                };

                // Find which collection this path belongs to
                let configs = collections::list_collections()?;
                let mut found = false;
                for nc in &configs {
                    let coll_path = std::path::Path::new(&nc.collection.path);
                    if let Ok(rel) = abs_path.strip_prefix(coll_path) {
                        let prefix = if rel.as_os_str().is_empty() {
                            "/".to_string()
                        } else {
                            format!("/{}", rel.display())
                        };
                        context::add_context(&nc.name, &prefix, &text)?;
                        println!("Context added for qmd://{}{prefix}", nc.name);
                        found = true;
                        break;
                    }
                }
                if !found {
                    anyhow::bail!("Could not detect collection for path: {path}");
                }
            }
            Ok(())
        }
        ContextAction::List => {
            let entries = context::list_all_contexts()?;
            if entries.is_empty() {
                println!("No contexts set.");
                return Ok(());
            }
            for entry in &entries {
                if entry.is_global {
                    println!("  {} {}", "/".dimmed(), entry.context);
                } else {
                    println!(
                        "  {} {}",
                        format!("qmd://{}{}", entry.collection, entry.path)
                            .dimmed(),
                        entry.context,
                    );
                }
            }
            Ok(())
        }
        ContextAction::Check => {
            let missing = context::check_missing_contexts()?;
            if missing.is_empty() {
                println!("All collections and paths have context. {}", "✓".green());
            } else {
                println!("Missing context:");
                for m in &missing {
                    println!("  {}", m.yellow());
                }
            }
            Ok(())
        }
        ContextAction::Rm { path } => {
            if path == "/" {
                let mut config = collections::load_config()?;
                config.global_context = None;
                collections::save_config(&config)?;
                println!("Global context removed.");
            } else if let Some((collection, prefix)) = store::parse_virtual_path(&path) {
                if context::remove_context(&collection, &prefix)? {
                    println!("Context removed.");
                } else {
                    println!("{}", "Context not found.".red());
                }
            } else {
                println!("{}", "Invalid path. Use qmd://collection/path or /".red());
            }
            Ok(())
        }
    }
}

fn handle_get(file: &str, opts: GetOptions) -> Result<()> {
    let store = Store::new(None)?;

    // Parse optional :line suffix
    let (filepath, from_line) = if let Some(colon_pos) = file.rfind(':') {
        if let Ok(line) = file[colon_pos + 1..].parse::<usize>() {
            (&file[..colon_pos], Some(line))
        } else {
            (file, None)
        }
    } else {
        (file, None)
    };

    let from_line = opts.from.or(from_line);

    match store::find_document(&store.conn, filepath, true)? {
        Some(mut doc) => {
            // Apply line limits
            if let Some(body) = &doc.body {
                let mut lines: Vec<&str> = body.lines().collect();
                if let Some(from) = from_line {
                    let skip = from.saturating_sub(1);
                    lines = lines.into_iter().skip(skip).collect();
                }
                if let Some(max) = opts.l {
                    lines.truncate(max);
                }
                let mut text = lines.join("\n");
                if opts.line_numbers {
                    let start = from_line.unwrap_or(1);
                    text = formatter::add_line_numbers(&text, start);
                }
                doc.body = Some(text);
            }

            let format = if opts.json {
                OutputFormat::Json
            } else if opts.xml {
                OutputFormat::Xml
            } else if opts.md {
                OutputFormat::Md
            } else {
                OutputFormat::Cli
            };

            println!("{}", formatter::format_document(&doc, format));
        }
        None => {
            eprintln!("{}", format!("Document not found: {file}").red());
            std::process::exit(1);
        }
    }
    Ok(())
}

fn handle_multi_get(pattern: &str, opts: MultiGetOptions) -> Result<()> {
    let store = Store::new(None)?;
    let (results, errors) = store::find_documents(
        &store.conn,
        pattern,
        Some(opts.max_bytes),
        opts.l,
    )?;

    let format = OutputFormat::from_flags(opts.json, opts.csv, opts.md, opts.xml, opts.files);

    if !results.is_empty() {
        println!("{}", formatter::format_documents(&results, format));
    }

    for err in &errors {
        eprintln!("{}", err.red());
    }

    Ok(())
}

fn handle_status() -> Result<()> {
    let store = Store::new(None)?;
    let status = store::get_status(&store)?;

    println!("{}\n", "QMD Status".bold());
    println!("Index: {}", store.db_path.display());

    let size = std::fs::metadata(&store.db_path)
        .map(|m| format_size(m.len() as usize))
        .unwrap_or_else(|_| "unknown".to_string());
    println!("Size:  {size}\n");

    println!("{}", "Documents".bold());
    println!("  Total:   {} files indexed", status.total_documents);
    if status.has_vector_index {
        let embedded = status.total_documents.saturating_sub(status.needs_embedding);
        println!("  Vectors: {} embedded", embedded);
    }
    if status.needs_embedding > 0 {
        println!(
            "  Pending: {} need embedding (run 'qmd embed')",
            status.needs_embedding.to_string().yellow()
        );
    }

    if !status.collections.is_empty() {
        println!("\n{}", "Collections".bold());
        for cs in &status.collections {
            println!(
                "  {} (qmd://{}/)",
                cs.name.cyan(),
                cs.name
            );
            println!("    Pattern: {}", cs.pattern);
            println!("    Files:   {}", cs.documents);
            if !cs.last_updated.is_empty() {
                println!("    Updated: {}", &cs.last_updated[..10.min(cs.last_updated.len())]);
            }
        }
    }

    Ok(())
}

fn handle_update(_pull: bool) -> Result<()> {
    let configs = collections::list_collections()?;
    if configs.is_empty() {
        println!("No collections configured.");
        return Ok(());
    }

    let store = Store::new(None)?;

    for nc in &configs {
        println!("Updating {}...", nc.name.cyan());

        // Run update command if configured
        if let Some(cmd) = &nc.collection.update {
            println!("  Running: {}", cmd.dimmed());
            let output = std::process::Command::new("bash")
                .args(["-c", cmd])
                .current_dir(&nc.collection.path)
                .output()?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("  {}", stderr.red());
                anyhow::bail!("Update command failed for {}", nc.name);
            }
        }

        let result = store::index_collection(&store, nc)?;
        println!(
            "  {} added, {} updated, {} unchanged, {} removed",
            result.added, result.updated, result.unchanged, result.removed
        );
    }

    Ok(())
}

fn handle_embed(_force: bool) -> Result<()> {
    println!(
        "{}",
        "Embedding not yet implemented in Rust version.".yellow()
    );
    println!("This will use llama-cpp-2 for local GGUF model inference.");
    println!("For now, use the TypeScript version: bun src/qmd.ts embed");
    Ok(())
}

fn handle_search(query: &str, opts: SearchOptions) -> Result<()> {
    let store = Store::new(None)?;
    let format = OutputFormat::from_flags(opts.json, opts.csv, opts.md, opts.xml, opts.files);
    let limit = if opts.all {
        100_000
    } else {
        opts.n.unwrap_or(format.default_limit())
    };

    let collection_filter = if opts.collection.is_empty() {
        None
    } else {
        Some(opts.collection.as_slice())
    };

    let results = search::search_fts(
        &store.conn,
        query,
        limit,
        collection_filter,
    )?;

    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.score >= opts.min_score.unwrap_or(0.0))
        .collect();

    let fmt_opts = FormatOptions {
        full: opts.full,
        query: query.to_string(),
        line_numbers: opts.line_numbers,
        intent: None,
    };

    println!(
        "{}",
        formatter::format_search_results(&results, format, &fmt_opts)
    );

    Ok(())
}

fn handle_vsearch(_query: &str, _opts: SearchOptions) -> Result<()> {
    println!(
        "{}",
        "Vector search not yet implemented in Rust version.".yellow()
    );
    println!("Requires llama-cpp-2 for embeddings. For now, use 'qmd search' for BM25 search.");
    Ok(())
}

fn handle_query(
    query: &str,
    opts: SearchOptions,
    candidate_limit: usize,
    intent: Option<String>,
) -> Result<()> {
    let store = Store::new(None)?;
    let format = OutputFormat::from_flags(opts.json, opts.csv, opts.md, opts.xml, opts.files);
    let limit = if opts.all {
        100_000
    } else {
        opts.n.unwrap_or(format.default_limit())
    };

    let collection_filter = if opts.collection.is_empty() {
        None
    } else {
        Some(opts.collection.clone())
    };

    let options = search::HybridQueryOptions {
        limit,
        min_score: opts.min_score.unwrap_or(0.0),
        collection_filter,
        candidate_limit,
        intent: intent.clone(),
    };

    let results = search::hybrid_query(&store.conn, query, &options)?;

    let fmt_opts = FormatOptions {
        full: opts.full,
        query: query.to_string(),
        line_numbers: opts.line_numbers,
        intent,
    };

    println!(
        "{}",
        formatter::format_search_results(&results, format, &fmt_opts)
    );

    Ok(())
}

fn handle_mcp(
    subcommand: Option<String>,
    _http: bool,
    _daemon: bool,
    _port: u16,
) -> Result<()> {
    if subcommand.as_deref() == Some("stop") {
        println!("MCP daemon stop not yet implemented.");
        return Ok(());
    }

    println!(
        "{}",
        "MCP server not yet implemented in Rust version.".yellow()
    );
    println!("This will use rmcp for MCP protocol support.");
    println!("For now, use the TypeScript version: bun src/qmd.ts mcp");
    Ok(())
}

fn handle_cleanup() -> Result<()> {
    let store = Store::new(None)?;

    let inactive = store::delete_inactive_documents(&store.conn)?;
    println!("Deleted {inactive} inactive documents");

    let orphaned = store::cleanup_orphaned_content(&store.conn)?;
    println!("Cleaned up {orphaned} orphaned content entries");

    store::vacuum_database(&store.conn)?;
    println!("Database vacuumed.");

    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}K", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}M", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}G", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
