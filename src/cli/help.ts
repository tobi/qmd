import { CLI_OPTIONS, type CliOptionName } from "./options.js";

export type HelpRow = readonly [label: string, description: string];

export type HelpOption = {
  /** Parser option key, used to detect help/parser drift. */
  key: CliOptionName;
  /** User-facing spelling, including value placeholders. */
  flags: string;
  description: string;
};

export type HelpSection = {
  title: string;
  rows?: readonly HelpRow[];
  lines?: readonly string[];
};

export type HelpSpec = {
  usage: string;
  summary: string;
  commands?: readonly HelpRow[];
  options?: readonly HelpOption[];
  sections?: readonly HelpSection[];
  examples?: readonly string[];
};

const option = (key: CliOptionName, flags: string, description: string): HelpOption => ({
  key,
  flags,
  description,
});

const INDEX_OPTION = option("index", "--index <name>", "Use a named index (default: index)");
const COLLECTION_OPTION = option("collection", "-c, --collection <name>", "Restrict to a collection; repeat where supported");
const FORMAT_OPTION = option("format", "--format <kind>", "Output: cli | json | csv | md | xml | files");
const LIMIT_OPTION = option("n", "-n <num>", "Maximum results (default 5; 20 for files/json)");
const MIN_SCORE_OPTION = option("min-score", "--min-score <num>", "Minimum result score");
const ALL_OPTION = option("all", "--all", "Return all matches; usually pair with --min-score");
const FULL_OPTION = option("full", "--full", "Return full documents instead of snippets");
const FULL_PATH_OPTION = option("full-path", "--full-path", "Show on-disk paths instead of qmd:// URIs");
const NO_GPU_OPTION = option("no-gpu", "--no-gpu", "Force CPU mode for llama.cpp operations");
const CHUNK_OPTION = option("chunk-strategy", "--chunk-strategy <auto|regex>", "Choose AST-aware or regex chunking");
const HELP_OPTION = option("help", "-h, --help", "Show help for this command");

const SEARCH_OUTPUT_OPTIONS = [
  LIMIT_OPTION,
  ALL_OPTION,
  MIN_SCORE_OPTION,
  FULL_OPTION,
  FORMAT_OPTION,
  COLLECTION_OPTION,
  FULL_PATH_OPTION,
] as const;

const QUERY_GRAMMAR = [
  "query           = expand_query | query_document ;",
  "expand_query    = text | explicit_expand ;",
  'explicit_expand = "expand:" text ;',
  "query_document  = [ intent_line ] { typed_line } ;",
  'intent_line     = "intent:" text newline ;',
  'typed_line      = type ":" text newline ;',
  'type            = "lex" | "vec" | "hyde" ;',
  'text            = quoted_phrase | plain_text ;',
  'quoted_phrase   = \'"\' { character } \'"\' ;',
  "plain_text      = { character } ;",
  'newline         = "\\n" ;',
] as const;

/**
 * Declarative command help. Keys are command paths, allowing the same renderer
 * to support both top-level commands and nested commands such as collection add.
 */
export const HELP_SPECS: Readonly<Record<string, HelpSpec>> = {
  root: {
    usage: "qmd <command> [options]",
    summary: "Quick Markdown Search — local keyword, vector, and hybrid search.",
    sections: [
      {
        title: "Search and retrieval commands",
        rows: [
          ["query <query>", "Hybrid search with expansion and reranking (recommended)"],
          ["search <query>", "Full-text BM25 keyword search; no LLM"],
          ["vsearch <query>", "Vector similarity search"],
          ["get <file>", "Retrieve one document by path or docid"],
          ["multi-get <pattern>", "Retrieve documents by glob, list, or docids"],
        ],
      },
      {
        title: "Index and configuration commands",
        rows: [
          ["collection <command>", "Manage indexed folders"],
          ["context <command>", "Attach context to indexed paths"],
          ["ls [collection/path]", "List indexed files"],
          ["init", "Create a project-local .qmd index"],
          ["status", "Show index and collection health"],
          ["doctor", "Diagnose runtime, database, model, and device setup"],
          ["update", "Re-index configured collections"],
          ["embed", "Generate vector embeddings"],
          ["pull", "Download configured local models"],
          ["cleanup", "Remove stale data and compact the index"],
          ["trust <command>", "Approve or revoke project-local configuration"],
        ],
      },
      {
        title: "Integration and evaluation commands",
        rows: [
          ["mcp", "Start the MCP server over stdio or HTTP"],
          ["bench <fixture.json>", "Run search-quality benchmarks"],
          ["skills <command>", "Inspect bundled runtime skills"],
          ["skill <command>", "Show or install the QMD skill"],
        ],
      },
    ],
    options: [
      INDEX_OPTION,
      option("no-gpu", "--no-gpu", "Force CPU mode for model-backed commands"),
      option("help", "-h, --help", "Show help for qmd or a command"),
      option("version", "-v, --version", "Print the installed QMD version"),
    ],
    examples: [
      'qmd search "exact phrase"',
      'qmd query "how does authentication work"',
      "qmd collection --help",
      "qmd query --help",
    ],
  },

  query: {
    usage: "qmd query [options] <query>",
    summary: "Run hybrid retrieval with optional query expansion and LLM reranking.",
    options: [
      ...SEARCH_OUTPUT_OPTIONS,
      option("candidate-limit", "-C, --candidate-limit <n>", "Maximum candidates to rerank (default 40)"),
      option("no-rerank", "--no-rerank", "Skip reranking and use fused retrieval scores"),
      option("intent", "--intent <text>", "Disambiguation context for ranking and snippets"),
      option("explain", "--explain", "Include retrieval score traces"),
      option("line-numbers", "--line-numbers", "Include line numbers in output"),
      CHUNK_OPTION,
      NO_GPU_OPTION,
      INDEX_OPTION,
    ],
    sections: [
      { title: "Query grammar", lines: QUERY_GRAMMAR },
      {
        title: "Constraints",
        lines: [
          "A standalone expand query cannot be mixed with typed lines.",
          "Typed documents accept only lex:, vec:, hyde:, and one optional intent: line.",
          "Each typed line is single-line text with balanced quotes.",
        ],
      },
    ],
    examples: [
      'qmd query "how does auth work"',
      "qmd query $'lex: auth token\\nvec: authentication flow'",
      'qmd query --intent "web performance" "performance"',
      'qmd query --no-rerank -c docs "deployment"',
    ],
  },

  search: {
    usage: "qmd search [options] <query>",
    summary: "Search the SQLite FTS5 index with BM25 keyword ranking; no model required.",
    options: [
      ...SEARCH_OUTPUT_OPTIONS,
      option("line-numbers", "--line-numbers", "Include line numbers in output"),
      INDEX_OPTION,
    ],
    examples: [
      'qmd search "authentication"',
      'qmd search -c docs --min-score 0.3 "rate limiter"',
      'qmd search --format json -n 10 "database"',
    ],
  },

  vsearch: {
    usage: "qmd vsearch [options] <query>",
    summary: "Search embedded document chunks by semantic similarity.",
    options: [
      ...SEARCH_OUTPUT_OPTIONS,
      option("line-numbers", "--line-numbers", "Include line numbers in output"),
      CHUNK_OPTION,
      NO_GPU_OPTION,
      INDEX_OPTION,
    ],
    examples: [
      'qmd vsearch "how is user identity verified"',
      'qmd vsearch -c docs -n 10 "deployment process"',
    ],
  },

  get: {
    usage: "qmd get <file>[:from[:count]] [options]",
    summary: "Retrieve one indexed document by path, qmd:// URI, or short docid.",
    options: [
      option("from", "--from <line>", "Start at a 1-indexed line; overrides the path suffix"),
      option("l", "-l <lines>", "Maximum lines to return"),
      option("no-line-numbers", "--no-line-numbers", "Hide line numbers (shown by default)"),
      FULL_PATH_OPTION,
      INDEX_OPTION,
    ],
    examples: [
      'qmd get "docs/api.md"',
      'qmd get "#abc123:120:40"',
      'qmd get "docs/api.md" --from 50 -l 25',
    ],
  },

  "multi-get": {
    usage: "qmd multi-get <pattern> [options]",
    summary: "Retrieve multiple documents by glob, comma-separated list, or docids.",
    options: [
      option("l", "-l <lines>", "Maximum lines per document"),
      option("max-bytes", "--max-bytes <bytes>", "Skip documents larger than this limit (default 65536)"),
      option("no-line-numbers", "--no-line-numbers", "Hide line numbers (shown by default)"),
      FORMAT_OPTION,
      FULL_PATH_OPTION,
      INDEX_OPTION,
    ],
    examples: [
      'qmd multi-get "journals/2025-05*.md"',
      'qmd multi-get "#abc123,#def456" --format json',
    ],
  },

  ls: {
    usage: "qmd ls [collection[/path]]",
    summary: "List collections or indexed files below an optional virtual path.",
    options: [INDEX_OPTION],
    examples: ["qmd ls", "qmd ls notes", "qmd ls qmd://notes/journals"],
  },

  collection: {
    usage: "qmd collection <command> [options]",
    summary: "Manage indexed folders and their update behavior.",
    commands: [
      ["list", "List configured collections"],
      ["add <path>", "Add a folder as a collection"],
      ["remove <name>", "Remove a collection"],
      ["rename <old> <new>", "Rename a collection"],
      ["show <name>", "Show collection details"],
      ["update-cmd <name> [cmd]", "Set or clear a pre-update command"],
      ["include <name>", "Include a collection in unscoped searches"],
      ["exclude <name>", "Exclude a collection from unscoped searches"],
    ],
    options: [INDEX_OPTION],
    examples: ["qmd collection add --help", "qmd collection list", "qmd collection show notes"],
  },

  "collection add": {
    usage: "qmd collection add <path> [options]",
    summary: "Add an existing directory to the index configuration.",
    options: [
      option("name", "--name <name>", "Set the collection name explicitly"),
      option("mask", "--mask <glob>", "Set the file glob (default: **/*.md)"),
      option("glob", "--glob <glob>", "Alias for --mask"),
      INDEX_OPTION,
    ],
    examples: [
      "qmd collection add . --name project",
      "qmd collection add ~/notes --name notes --mask '**/*.md'",
      "qmd collection add ~/docs --glob 'README.md,docs/**/*.md'",
    ],
  },

  "collection list": {
    usage: "qmd collection list",
    summary: "List configured collections and their indexed-document counts.",
    options: [INDEX_OPTION],
  },

  "collection remove": {
    usage: "qmd collection remove <name>",
    summary: "Remove a collection and its indexed records.",
    options: [INDEX_OPTION],
    examples: ["qmd collection remove archive"],
  },

  "collection rename": {
    usage: "qmd collection rename <old-name> <new-name>",
    summary: "Rename a configured collection.",
    options: [INDEX_OPTION],
    examples: ["qmd collection rename work work-notes"],
  },

  "collection show": {
    usage: "qmd collection show <name>",
    summary: "Show one collection's path, glob, default-search state, update command, and context count.",
    options: [INDEX_OPTION],
  },

  "collection update-cmd": {
    usage: "qmd collection update-cmd <name> [command]",
    summary: "Set a command to run before re-indexing; omit command to clear it.",
    options: [INDEX_OPTION],
    examples: ["qmd collection update-cmd notes 'git pull --ff-only'", "qmd collection update-cmd notes"],
  },

  "collection include": {
    usage: "qmd collection include <name>",
    summary: "Include a collection in searches that do not specify -c.",
    options: [INDEX_OPTION],
  },

  "collection exclude": {
    usage: "qmd collection exclude <name>",
    summary: "Exclude a collection from searches that do not specify -c.",
    options: [INDEX_OPTION],
  },

  context: {
    usage: "qmd context <command>",
    summary: "Attach human-written context to collection paths.",
    commands: [
      ["add [path] <text>", "Add context; path defaults to the current directory"],
      ["list", "List configured contexts"],
      ["rm <path>", "Remove context from a path"],
    ],
    options: [INDEX_OPTION],
    examples: ["qmd context add --help", "qmd context list"],
  },

  "context add": {
    usage: "qmd context add [path] <text>",
    summary: "Add context to a physical or qmd:// path; omit path to use the current directory.",
    options: [INDEX_OPTION],
    examples: [
      'qmd context add "Notes from the platform team"',
      'qmd context add qmd://journals/2024 "Journal entries from 2024"',
      'qmd context add / "Global context for all collections"',
    ],
  },

  "context list": {
    usage: "qmd context list",
    summary: "List global and path-specific context entries.",
    options: [INDEX_OPTION],
  },

  "context rm": {
    usage: "qmd context rm <path>",
    summary: "Remove context from a physical or qmd:// path.",
    options: [INDEX_OPTION],
    examples: ["qmd context rm /", "qmd context rm qmd://journals/2024"],
  },

  init: {
    usage: "qmd init",
    summary: "Create a project-local .qmd/index.yml and SQLite index for the current directory.",
    examples: ["cd my-project && qmd init"],
  },

  status: {
    usage: "qmd status [options]",
    summary: "Show index health, collection state, embedding state, and MCP daemon status.",
    options: [INDEX_OPTION],
  },

  doctor: {
    usage: "qmd doctor [options]",
    summary: "Diagnose configuration, SQLite/vector support, models, and device acceleration.",
    options: [NO_GPU_OPTION, INDEX_OPTION],
  },

  update: {
    usage: "qmd update [options]",
    summary: "Re-scan every configured collection and refresh changed documents.",
    options: [
      option("pull", "--pull", "Deprecated compatibility flag; configured update commands run automatically when trusted"),
      INDEX_OPTION,
    ],
    examples: ["qmd update", "qmd collection update-cmd notes 'git pull --ff-only' && qmd update"],
  },

  trust: {
    usage: "qmd trust [list|revoke]",
    summary: "Approve the sensitive parts of a project-local .qmd configuration.",
    commands: [
      ["(no command)", "Review and trust the current project-local configuration"],
      ["list", "List trusted project-local configurations"],
      ["revoke", "Revoke trust for the current project-local configuration"],
    ],
    options: [INDEX_OPTION],
  },

  embed: {
    usage: "qmd embed [options]",
    summary: "Chunk indexed documents and generate local vector embeddings.",
    options: [
      option("force", "-f, --force", "Regenerate embeddings even when they are current"),
      COLLECTION_OPTION,
      option("max-docs-per-batch", "--max-docs-per-batch <n>", "Limit documents loaded per batch"),
      option("max-batch-mb", "--max-batch-mb <n>", "Limit UTF-8 megabytes loaded per batch"),
      option("timeout", "--timeout <minutes>", "Session limit; 0 disables it (default 30)"),
      CHUNK_OPTION,
      NO_GPU_OPTION,
      INDEX_OPTION,
    ],
    examples: ["qmd embed", "qmd embed -c notes --chunk-strategy auto", "qmd embed --force --timeout 0"],
  },

  pull: {
    usage: "qmd pull [options]",
    summary: "Download or verify the configured embedding, generation, and reranking models.",
    options: [
      option("refresh", "--refresh", "Refresh model files even when cached"),
      option("progress", "--progress", "Show node-llama-cpp download progress"),
      NO_GPU_OPTION,
      INDEX_OPTION,
    ],
    examples: ["qmd pull", "qmd pull --refresh --progress"],
  },

  cleanup: {
    usage: "qmd cleanup [options]",
    summary: "Clear caches and stale records, compact FTS, and vacuum SQLite.",
    options: [option("dry-run", "--dry-run", "Preview removals without changing the database"), INDEX_OPTION],
    examples: ["qmd cleanup --dry-run", "qmd cleanup"],
  },

  bench: {
    usage: "qmd bench <fixture.json> [options]",
    summary: "Measure search quality against a JSON fixture.",
    options: [
      option("example", "--example", "Print the bundled example fixture and exit"),
      option("json", "--json", "Emit structured JSON results"),
      COLLECTION_OPTION,
      INDEX_OPTION,
    ],
    examples: ["qmd bench --example > fixture.json", "qmd bench fixture.json -c docs --json"],
  },

  mcp: {
    usage: "qmd mcp [stop] [options]",
    summary: "Expose query, get, multi_get, and status tools to MCP clients.",
    options: [
      option("http", "--http", "Use Streamable HTTP instead of stdio"),
      option("daemon", "--daemon", "Run the HTTP server in the background"),
      option("port", "--port <number>", "HTTP port (default 8181)"),
      option("host", "--host <address>", "HTTP bind address (default localhost)"),
      NO_GPU_OPTION,
      INDEX_OPTION,
    ],
    sections: [{ title: "Subcommands", rows: [["stop", "Stop the background daemon for this index"]] }],
    examples: ["qmd mcp", "qmd mcp --http", "qmd mcp --http --daemon", "qmd mcp stop"],
  },

  skills: {
    usage: "qmd skills <list|get|path> [options]",
    summary: "Inspect the version-matched runtime skills bundled with QMD.",
    commands: [
      ["list", "List bundled runtime skills"],
      ["get <name>", "Print a bundled runtime skill"],
      ["path [name]", "Print runtime skill search paths"],
    ],
    options: [
      option("json", "--json", "Emit structured JSON"),
      option("full", "--full", "Include references, templates, and scripts with get"),
      option("all", "--all", "Print all runtime skills with get"),
    ],
    examples: ["qmd skills list", "qmd skills get qmd --full", "qmd skills path qmd"],
  },

  skill: {
    usage: "qmd skill <show|install> [options]",
    summary: "Show or install QMD's agent skill.",
    commands: [
      ["show", "Print the QMD skill"],
      ["install", "Install into ./.agents/skills/qmd"],
    ],
    options: [
      option("global", "--global", "Install into ~/.agents/skills/qmd"),
      option("yes", "--yes", "Also create the .claude/skills/qmd symlink"),
      option("force", "-f, --force", "Replace an existing install or symlink"),
    ],
    examples: ["qmd skill show", "qmd skill install", "qmd skill install --global --yes"],
  },
};

const TOPIC_ALIASES: Readonly<Record<string, string>> = {
  "deep-search": "query",
  "vector-search": "vsearch",
  "collection rm": "collection remove",
  "collection mv": "collection rename",
  "collection info": "collection show",
  "collection set-update": "collection update-cmd",
  "context remove": "context rm",
};

const NESTED_COMMANDS = new Set(["collection", "context"]);

export function resolveHelpTopic(command?: string, args: readonly string[] = []): string {
  if (!command) return "root";

  if (command === "help") {
    const requestedCommand = args[0];
    return resolveHelpTopic(requestedCommand, args.slice(1));
  }

  const nested = NESTED_COMMANDS.has(command) && args[0] && args[0] !== "help"
    ? `${command} ${args[0]}`
    : command;
  const canonical = TOPIC_ALIASES[nested] ?? nested;
  return HELP_SPECS[canonical] ? canonical : "root";
}

function printRows(rows: readonly HelpRow[], write: (line: string) => void): void {
  const width = Math.max(...rows.map(([label]) => label.length));
  for (const [label, description] of rows) {
    write(`  ${label.padEnd(width)}  ${description}`);
  }
}

export function renderHelp(topic: string, options: { indexPath?: string } = {}): string {
  const canonical = TOPIC_ALIASES[topic] ?? topic;
  // `root` is a required registry entry and is the safe fallback for unknown topics.
  const spec = HELP_SPECS[canonical] ?? HELP_SPECS.root!;
  const lines: string[] = [spec.summary, "", "Usage:", `  ${spec.usage}`];

  if (spec.commands?.length) {
    lines.push("", "Commands:");
    printRows(spec.commands, (line) => lines.push(line));
  }

  for (const section of spec.sections ?? []) {
    lines.push("", `${section.title}:`);
    if (section.rows?.length) printRows(section.rows, (line) => lines.push(line));
    for (const line of section.lines ?? []) lines.push(`  ${line}`);
  }

  const renderedOptions = canonical === "root"
    ? (spec.options ?? [])
    : [...(spec.options ?? []), HELP_OPTION];
  if (renderedOptions.length) {
    lines.push("", "Options:");
    printRows(renderedOptions.map(({ flags, description }) => [flags, description]), (line) => lines.push(line));
  }

  if (spec.examples?.length) {
    lines.push("", "Examples:");
    for (const example of spec.examples) lines.push(`  ${example}`);
  }

  if (canonical === "root" && options.indexPath) {
    lines.push("", `Index: ${options.indexPath}`);
  }

  return `${lines.join("\n")}\n`;
}

export function printHelp(topic: string, options: { indexPath?: string } = {}): void {
  process.stdout.write(renderHelp(topic, options));
}

export function validateHelpOptionKeys(): string[] {
  const errors: string[] = [];
  for (const [topic, spec] of Object.entries(HELP_SPECS)) {
    for (const documented of spec.options ?? []) {
      if (!(documented.key in CLI_OPTIONS)) {
        errors.push(`${topic}: unknown parser option ${documented.key}`);
      }
    }
  }
  return errors;
}
