//! llm.rs - LLM inference layer (placeholder)
//!
//! Will provide embeddings, reranking, and query expansion using local GGUF models
//! via llama-cpp-2. Currently a placeholder with type definitions and formatting
//! functions that don't require the LLM runtime.

// =============================================================================
// Constants
// =============================================================================

pub const DEFAULT_EMBED_MODEL: &str = "hf:ggml-org/embeddinggemma-300M-Q8_0/embeddinggemma-Q8_0.gguf";
pub const DEFAULT_RERANK_MODEL: &str = "hf:ggml-org/qwen3-reranker-0.6b-q8_0/qwen3-reranker-0.6b-q8_0.gguf";
pub const DEFAULT_QUERY_MODEL: &str = "hf:ahmedelgabri/qmd-query-expansion-1.7B-q4_k_m/qmd-query-expansion-1.7B-q4_k_m.gguf";

// =============================================================================
// Types
// =============================================================================

/// Embedding result
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub model: String,
}

/// Rerank result for a single document
#[derive(Debug, Clone)]
pub struct RerankDocumentResult {
    pub file: String,
    pub score: f64,
    pub index: usize,
}

/// Query expansion result
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    pub query_type: QueryType,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Lex,  // For BM25 keyword search
    Vec,  // For vector similarity search
    Hyde, // Hypothetical document embedding
}

// =============================================================================
// Embedding formatting functions (no LLM required)
// =============================================================================

/// Check if a model URI is a Qwen3-Embedding model.
pub fn is_qwen3_embedding_model(model_uri: &str) -> bool {
    let lower = model_uri.to_lowercase();
    (lower.contains("qwen") && lower.contains("embed"))
        || (lower.contains("embed") && lower.contains("qwen"))
}

/// Format a query for embedding (model-specific prompting).
pub fn format_query_for_embedding(query: &str, model_uri: Option<&str>) -> String {
    let uri = model_uri
        .or_else(|| std::env::var("QMD_EMBED_MODEL").ok().as_deref().map(|_| ""))
        .unwrap_or(DEFAULT_EMBED_MODEL);

    if is_qwen3_embedding_model(uri) {
        format!("Instruct: Retrieve relevant documents for the given query\nQuery: {query}")
    } else {
        format!("task: search result | query: {query}")
    }
}

/// Format a document for embedding.
pub fn format_doc_for_embedding(text: &str, title: Option<&str>, model_uri: Option<&str>) -> String {
    let uri = model_uri.unwrap_or(DEFAULT_EMBED_MODEL);

    if is_qwen3_embedding_model(uri) {
        match title {
            Some(t) => format!("{t}\n{text}"),
            None => text.to_string(),
        }
    } else {
        let t = title.unwrap_or("none");
        format!("title: {t} | text: {text}")
    }
}

// =============================================================================
// LLM Session (placeholder)
// =============================================================================

/// Placeholder for future LLM session management.
/// Will be implemented with llama-cpp-2 bindings.
pub struct LlmSession {
    _private: (),
}

impl LlmSession {
    /// Create a new LLM session (placeholder).
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!(
            "LLM inference not yet implemented. \
             This will use llama-cpp-2 for local GGUF model inference."
        );
    }
}

// =============================================================================
// Placeholder functions for future implementation
// =============================================================================

/// Generate embeddings for text (placeholder).
pub fn embed(_text: &str, _model: Option<&str>) -> anyhow::Result<EmbeddingResult> {
    anyhow::bail!("Embedding not yet implemented. Requires llama-cpp-2 integration.")
}

/// Rerank documents by relevance to a query (placeholder).
pub fn rerank(
    _query: &str,
    _documents: &[(String, String)],
    _model: Option<&str>,
) -> anyhow::Result<Vec<RerankDocumentResult>> {
    anyhow::bail!("Reranking not yet implemented. Requires llama-cpp-2 integration.")
}

/// Expand a query into multiple search variants (placeholder).
pub fn expand_query(
    _query: &str,
    _model: Option<&str>,
    _intent: Option<&str>,
) -> anyhow::Result<Vec<ExpandedQuery>> {
    anyhow::bail!("Query expansion not yet implemented. Requires llama-cpp-2 integration.")
}
