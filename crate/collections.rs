//! collections.rs - YAML-based collection configuration management
//!
//! Manages the index.yml config file that defines which directories are indexed,
//! their glob patterns, context, and update commands.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Types
// =============================================================================

/// Context map: path prefix -> context description
pub type ContextMap = BTreeMap<String, String>;

/// A single collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub path: String,
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
    #[serde(default, skip_serializing_if = "is_default_include")]
    pub include_by_default: Option<bool>,
}

fn is_default_include(v: &Option<bool>) -> bool {
    v.is_none() || *v == Some(true)
}

/// A collection with its name attached
#[derive(Debug, Clone)]
pub struct NamedCollection {
    pub name: String,
    pub collection: Collection,
}

/// Top-level config file structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
    #[serde(default)]
    pub collections: BTreeMap<String, Collection>,
}

// =============================================================================
// Config path management
// =============================================================================

/// Current index name (defaults to "index")
static INDEX_NAME: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn get_index_name() -> String {
    let name = INDEX_NAME.lock().unwrap();
    if name.is_empty() {
        "index".to_string()
    } else {
        name.clone()
    }
}

/// Set the config index name (for --index flag)
pub fn set_config_index_name(name: &str) {
    let mut index_name = INDEX_NAME.lock().unwrap();
    *index_name = name.replace(['/', '\\'], "_");
}

/// Get the config directory: $XDG_CONFIG_HOME/qmd/ or ~/.config/qmd/
pub fn get_config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QMD_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join("qmd")
}

/// Get the full path to the config file
pub fn get_config_path() -> PathBuf {
    let name = get_index_name();
    get_config_dir().join(format!("{name}.yml"))
}

/// Check if the config file exists
pub fn config_exists() -> bool {
    get_config_path().exists()
}

// =============================================================================
// Config loading and saving
// =============================================================================

/// Load the collection config from YAML file, or return default if missing.
pub fn load_config() -> Result<CollectionConfig> {
    let path = get_config_path();
    if !path.exists() {
        return Ok(CollectionConfig::default());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    let config: CollectionConfig = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse config: {}", path.display()))?;
    Ok(config)
}

/// Save the collection config to YAML file.
pub fn save_config(config: &CollectionConfig) -> Result<()> {
    let path = get_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
    }
    let yaml = serde_yaml::to_string(config).context("Failed to serialize config")?;
    std::fs::write(&path, yaml)
        .with_context(|| format!("Failed to write config: {}", path.display()))?;
    Ok(())
}

// =============================================================================
// Collection operations
// =============================================================================

/// Validate that a collection name contains only alphanumeric, underscore, and hyphen.
pub fn is_valid_collection_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Add or update a collection in the config.
pub fn add_collection(name: &str, path: &str, pattern: &str) -> Result<()> {
    if !is_valid_collection_name(name) {
        bail!("Invalid collection name: {name}. Use only alphanumeric, underscore, and hyphen.");
    }
    let mut config = load_config()?;

    // Resolve path to absolute
    let abs_path = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        std::env::current_dir()?.join(path)
    };
    let abs_path = abs_path
        .canonicalize()
        .with_context(|| format!("Path does not exist: {path}"))?;

    // Preserve existing context when updating
    let existing_context = config
        .collections
        .get(name)
        .and_then(|c| c.context.clone());

    config.collections.insert(
        name.to_string(),
        Collection {
            path: abs_path.to_string_lossy().to_string(),
            pattern: pattern.to_string(),
            ignore: None,
            context: existing_context,
            update: None,
            include_by_default: None,
        },
    );

    save_config(&config)
}

/// Remove a collection from the config. Returns false if not found.
pub fn remove_collection(name: &str) -> Result<bool> {
    let mut config = load_config()?;
    if config.collections.remove(name).is_none() {
        return Ok(false);
    }
    save_config(&config)?;
    Ok(true)
}

/// Rename a collection. Returns false if old name not found.
pub fn rename_collection(old_name: &str, new_name: &str) -> Result<bool> {
    if !is_valid_collection_name(new_name) {
        bail!("Invalid collection name: {new_name}");
    }
    let mut config = load_config()?;
    if !config.collections.contains_key(old_name) {
        return Ok(false);
    }
    if config.collections.contains_key(new_name) {
        bail!("Collection '{new_name}' already exists");
    }
    let collection = config.collections.remove(old_name).unwrap();
    config.collections.insert(new_name.to_string(), collection);
    save_config(&config)?;
    Ok(true)
}

/// List all collections as NamedCollection.
pub fn list_collections() -> Result<Vec<NamedCollection>> {
    let config = load_config()?;
    Ok(config
        .collections
        .into_iter()
        .map(|(name, collection)| NamedCollection { name, collection })
        .collect())
}

/// Get a single collection by name.
pub fn get_collection(name: &str) -> Result<Option<NamedCollection>> {
    let config = load_config()?;
    Ok(config.collections.get(name).map(|c| NamedCollection {
        name: name.to_string(),
        collection: c.clone(),
    }))
}

/// Get collection names that have includeByDefault != false.
pub fn get_default_collection_names() -> Result<Vec<String>> {
    let config = load_config()?;
    Ok(config
        .collections
        .iter()
        .filter(|(_, c)| c.include_by_default != Some(false))
        .map(|(name, _)| name.clone())
        .collect())
}

// =============================================================================
// Global context
// =============================================================================

/// Get the global context string (applied to all collections).
pub fn get_global_context() -> Result<Option<String>> {
    let config = load_config()?;
    Ok(config.global_context)
}

/// Set the global context string.
pub fn set_global_context(context: &str) -> Result<()> {
    let mut config = load_config()?;
    config.global_context = Some(context.to_string());
    save_config(&config)
}

// =============================================================================
// File discovery
// =============================================================================

/// Discover files in a collection directory matching the glob pattern.
pub fn discover_files(collection: &Collection) -> Result<Vec<PathBuf>> {
    let base_path = PathBuf::from(&collection.path);
    if !base_path.exists() {
        bail!("Collection path does not exist: {}", collection.path);
    }

    let pattern = format!("{}/{}", collection.path, collection.pattern);
    let mut files: Vec<PathBuf> = glob::glob(&pattern)
        .with_context(|| format!("Invalid glob pattern: {}", collection.pattern))?
        .filter_map(|entry| entry.ok())
        .filter(|path| path.is_file())
        .collect();

    // Apply ignore patterns
    if let Some(ignore_patterns) = &collection.ignore {
        let mut builder = globset::GlobSetBuilder::new();
        for pattern in ignore_patterns {
            if let Ok(glob) = globset::Glob::new(pattern) {
                builder.add(glob);
            }
        }
        if let Ok(ignore_set) = builder.build() {
            files.retain(|path| {
                let rel = path.strip_prefix(&base_path).unwrap_or(path);
                !ignore_set.is_match(rel)
            });
        }
    }

    // Sort for deterministic output
    files.sort();
    Ok(files)
}

/// Get relative path of a file within a collection.
pub fn relative_path(collection: &Collection, file: &Path) -> String {
    let base = PathBuf::from(&collection.path);
    file.strip_prefix(&base)
        .unwrap_or(file)
        .to_string_lossy()
        .to_string()
}
