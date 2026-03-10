//! context.rs - Context management for collections and paths
//!
//! Allows users to attach descriptive context to collections, paths, and globally.
//! Context is included in search results and MCP responses to help LLMs understand content.

use anyhow::Result;

use crate::collections::{self, ContextMap};

// =============================================================================
// Context operations
// =============================================================================

/// A context entry with its full path info.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub collection: String,
    pub path: String,
    pub context: String,
    pub is_global: bool,
}

/// Add context to a path. If collection is None, detect from current directory.
pub fn add_context(
    collection_name: &str,
    path_prefix: &str,
    context_text: &str,
) -> Result<()> {
    let mut config = collections::load_config()?;

    if path_prefix == "/" {
        // Global context
        config.global_context = Some(context_text.to_string());
    } else {
        let coll = config
            .collections
            .get_mut(collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection '{collection_name}' not found"))?;

        let ctx_map = coll.context.get_or_insert_with(ContextMap::new);
        ctx_map.insert(path_prefix.to_string(), context_text.to_string());
    }

    collections::save_config(&config)
}

/// List all contexts across all collections.
pub fn list_all_contexts() -> Result<Vec<ContextEntry>> {
    let config = collections::load_config()?;
    let mut entries = Vec::new();

    // Global context
    if let Some(ctx) = &config.global_context {
        entries.push(ContextEntry {
            collection: String::new(),
            path: "/".to_string(),
            context: ctx.clone(),
            is_global: true,
        });
    }

    // Per-collection contexts
    for (name, coll) in &config.collections {
        if let Some(ctx_map) = &coll.context {
            for (path, ctx) in ctx_map {
                entries.push(ContextEntry {
                    collection: name.clone(),
                    path: path.clone(),
                    context: ctx.clone(),
                    is_global: false,
                });
            }
        }
    }

    Ok(entries)
}

/// Remove context for a specific path.
pub fn remove_context(collection_name: &str, path_prefix: &str) -> Result<bool> {
    let mut config = collections::load_config()?;

    if path_prefix == "/" {
        if config.global_context.is_some() {
            config.global_context = None;
            collections::save_config(&config)?;
            return Ok(true);
        }
        return Ok(false);
    }

    if let Some(coll) = config.collections.get_mut(collection_name) {
        if let Some(ctx_map) = &mut coll.context {
            if ctx_map.remove(path_prefix).is_some() {
                if ctx_map.is_empty() {
                    coll.context = None;
                }
                collections::save_config(&config)?;
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// Check for collections and paths that are missing context.
pub fn check_missing_contexts() -> Result<Vec<String>> {
    let config = collections::load_config()?;
    let mut missing = Vec::new();

    for (name, coll) in &config.collections {
        let has_root_context = coll
            .context
            .as_ref()
            .is_some_and(|m| m.contains_key("/"));

        if !has_root_context {
            missing.push(format!("qmd://{name}/ — no context set"));
        }
    }

    if config.global_context.is_none() && !config.collections.is_empty() {
        missing.push("/ — no global context set".to_string());
    }

    Ok(missing)
}
