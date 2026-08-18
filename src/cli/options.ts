/**
 * Canonical CLI option definitions.
 *
 * Keep this object separate from qmd.ts so contextual help and tests can
 * validate documented flags without importing the full CLI entrypoint.
 */
export const CLI_OPTIONS = {
  // Global options
  index: { type: "string" },
  context: { type: "string" },
  help: { type: "boolean", short: "h" },
  version: { type: "boolean", short: "v" },
  skill: { type: "boolean" },
  global: { type: "boolean" },
  yes: { type: "boolean" },

  // Search and output options
  n: { type: "string" },
  "min-score": { type: "string" },
  all: { type: "boolean" },
  full: { type: "boolean" },
  format: { type: "string" },
  csv: { type: "boolean" },
  md: { type: "boolean" },
  xml: { type: "boolean" },
  files: { type: "boolean" },
  json: { type: "boolean" },
  explain: { type: "boolean" },
  collection: { type: "string", short: "c", multiple: true },

  // Collection options
  name: { type: "string" },
  mask: { type: "string" },
  glob: { type: "string" },

  // Embed options
  force: { type: "boolean", short: "f" },
  "max-docs-per-batch": { type: "string" },
  "max-batch-mb": { type: "string" },
  timeout: { type: "string" },

  // Update and maintenance options
  pull: { type: "boolean" },
  refresh: { type: "boolean" },
  progress: { type: "boolean" },
  "dry-run": { type: "boolean" },
  example: { type: "boolean" },

  // Document retrieval options
  l: { type: "string" },
  from: { type: "string" },
  "max-bytes": { type: "string" },
  "line-numbers": { type: "boolean" },
  "no-line-numbers": { type: "boolean" },
  "full-path": { type: "boolean" },

  // Query options
  "candidate-limit": { type: "string", short: "C" },
  "no-rerank": { type: "boolean", default: false },
  "no-gpu": { type: "boolean", default: false },
  intent: { type: "string" },
  "chunk-strategy": { type: "string" },

  // MCP HTTP transport options
  http: { type: "boolean" },
  daemon: { type: "boolean" },
  port: { type: "string" },
  host: { type: "string" },
} as const;

export type CliOptionName = keyof typeof CLI_OPTIONS;
