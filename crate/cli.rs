//! cli.rs - CLI argument parsing and command dispatch
//!
//! Defines all QMD commands using clap derive macros.

use clap::{Parser, Subcommand, Args};

/// QMD - Query Markup Documents
///
/// On-device hybrid search for markdown files with BM25, vector search, and LLM reranking.
#[derive(Parser, Debug)]
#[command(name = "qmd", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Use a named index (default: "index")
    #[arg(long, global = true)]
    pub index: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Manage collections
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },

    /// List collections or files in a collection
    Ls {
        /// Collection name or path (e.g., "mynotes" or "mynotes/subfolder")
        path: Option<String>,
    },

    /// Add context to a path
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },

    /// Get a document by path or docid
    Get {
        /// File path, virtual path (qmd://...), or docid (#abc123)
        file: String,

        #[command(flatten)]
        opts: GetOptions,
    },

    /// Get multiple documents by glob or comma-separated list
    MultiGet {
        /// Glob pattern or comma-separated list of paths/docids
        pattern: String,

        #[command(flatten)]
        opts: MultiGetOptions,
    },

    /// Show index status and collections
    Status,

    /// Re-index all collections
    Update {
        /// Git pull before re-indexing
        #[arg(long)]
        pull: bool,
    },

    /// Generate vector embeddings
    Embed {
        /// Force re-embedding of existing vectors
        #[arg(short, long)]
        force: bool,
    },

    /// Full-text keyword search (BM25, no LLM)
    Search {
        /// Search query
        query: String,

        #[command(flatten)]
        opts: SearchOptions,
    },

    /// Vector similarity search (no reranking)
    Vsearch {
        /// Search query
        query: String,

        #[command(flatten)]
        opts: SearchOptions,
    },

    /// Search with query expansion + reranking (recommended)
    Query {
        /// Search query
        query: String,

        #[command(flatten)]
        opts: SearchOptions,

        /// Max candidates to rerank
        #[arg(short = 'C', long, default_value = "40")]
        candidate_limit: usize,

        /// Query intent for disambiguation
        #[arg(long)]
        intent: Option<String>,
    },

    /// Start MCP server
    Mcp {
        /// Stop background MCP daemon
        #[arg(value_name = "stop")]
        subcommand: Option<String>,

        /// Use HTTP transport instead of stdio
        #[arg(long)]
        http: bool,

        /// Run as background daemon
        #[arg(long)]
        daemon: bool,

        /// HTTP port (default: 8181)
        #[arg(long, default_value = "8181")]
        port: u16,
    },

    /// Clean up inactive documents and orphaned content
    Cleanup,
}

#[derive(Subcommand, Debug)]
pub enum CollectionAction {
    /// Add a new collection
    Add {
        /// Path to the directory to index
        path: String,

        /// Collection name
        #[arg(long)]
        name: Option<String>,

        /// Glob pattern for files to include (default: **/*.md)
        #[arg(long, default_value = "**/*.md")]
        mask: String,
    },

    /// List all collections
    List,

    /// Remove a collection
    #[command(alias = "rm")]
    Remove {
        /// Collection name to remove
        name: String,
    },

    /// Rename a collection
    #[command(alias = "mv")]
    Rename {
        /// Current name
        old_name: String,
        /// New name
        new_name: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ContextAction {
    /// Add context for a path
    Add {
        /// Path to add context to (defaults to current directory)
        path: Option<String>,
        /// Context description
        text: String,
    },

    /// List all contexts
    List,

    /// Check for collections/paths missing context
    Check,

    /// Remove context for a path
    #[command(alias = "remove")]
    Rm {
        /// Path to remove context from
        path: String,
    },
}

#[derive(Args, Debug)]
pub struct SearchOptions {
    /// Number of results
    #[arg(short)]
    pub n: Option<usize>,

    /// Minimum score threshold
    #[arg(long)]
    pub min_score: Option<f64>,

    /// Return all matches
    #[arg(long)]
    pub all: bool,

    /// Show full document content
    #[arg(long)]
    pub full: bool,

    /// Add line numbers to output
    #[arg(long)]
    pub line_numbers: bool,

    /// Restrict to collection(s)
    #[arg(short, long)]
    pub collection: Vec<String>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Output as CSV
    #[arg(long)]
    pub csv: bool,

    /// Output as Markdown
    #[arg(long)]
    pub md: bool,

    /// Output as XML
    #[arg(long)]
    pub xml: bool,

    /// Output as file paths only
    #[arg(long)]
    pub files: bool,

    /// Include retrieval traces
    #[arg(long)]
    pub explain: bool,
}

#[derive(Args, Debug)]
pub struct GetOptions {
    /// Maximum lines to return
    #[arg(short)]
    pub l: Option<usize>,

    /// Start from line number
    #[arg(long)]
    pub from: Option<usize>,

    /// Add line numbers to output
    #[arg(long)]
    pub line_numbers: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Output as Markdown
    #[arg(long)]
    pub md: bool,

    /// Output as XML
    #[arg(long)]
    pub xml: bool,
}

#[derive(Args, Debug)]
pub struct MultiGetOptions {
    /// Maximum lines per file
    #[arg(short)]
    pub l: Option<usize>,

    /// Maximum bytes per file (skip larger files)
    #[arg(long, default_value = "10240")]
    pub max_bytes: usize,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Output as CSV
    #[arg(long)]
    pub csv: bool,

    /// Output as Markdown
    #[arg(long)]
    pub md: bool,

    /// Output as XML
    #[arg(long)]
    pub xml: bool,

    /// Output as file paths only
    #[arg(long)]
    pub files: bool,
}
