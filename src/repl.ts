#!/usr/bin/env bun
/**
 * QMD REPL - Interactive shell with qmd collections mounted read-only
 *
 * Features:
 * - All qmd collections mounted under /qmd/<collection-name>/
 * - Home directory writable at /home
 * - Built-in search and query commands
 * - Tab completion for files and commands
 * - --prompt argument to initialize prompt.txt
 */

import * as readline from "node:readline";
import { parseArgs } from "util";
import {
  Bash,
  InMemoryFs,
  MountableFs,
  defineCommand,
  type IFileSystem,
  type FsStat,
  type MkdirOptions,
  type RmOptions,
  type CpOptions,
  type DirentEntry,
} from "just-bash";
import {
  createStore,
  enableProductionMode,
  searchFTS,
  searchVec,
  type Store,
  type SearchResult,
} from "./store.js";
import { listCollections as yamlListCollections, getCollection } from "./collections.js";

// Enable production mode for database access
enableProductionMode();

// Terminal colors
const useColor = !process.env.NO_COLOR && process.stdout.isTTY;
const c = {
  reset: useColor ? "\x1b[0m" : "",
  dim: useColor ? "\x1b[2m" : "",
  bold: useColor ? "\x1b[1m" : "",
  cyan: useColor ? "\x1b[36m" : "",
  yellow: useColor ? "\x1b[33m" : "",
  green: useColor ? "\x1b[32m" : "",
  magenta: useColor ? "\x1b[35m" : "",
  blue: useColor ? "\x1b[34m" : "",
  red: useColor ? "\x1b[31m" : "",
};

/**
 * Read-only filesystem that exposes qmd collections.
 * Files are loaded on-demand from the SQLite database.
 */
class QmdReadOnlyFs implements IFileSystem {
  private store: Store;
  private fileCache = new Map<string, { content: string; mtime: Date }>();
  private dirCache = new Map<string, string[]>();

  constructor(store: Store) {
    this.store = store;
    this.buildDirectoryCache();
  }

  private buildDirectoryCache(): void {
    const db = this.store.db;
    const collections = yamlListCollections();

    // Root directory contains collection names
    this.dirCache.set("/", collections.map(c => c.name));

    // For each collection, get all document paths and build directory structure
    for (const coll of collections) {
      const docs = db.prepare(`
        SELECT path FROM documents WHERE collection = ? AND active = 1
      `).all(coll.name) as { path: string }[];

      const collPath = `/${coll.name}`;
      const dirs = new Map<string, Set<string>>();
      dirs.set(collPath, new Set());

      for (const doc of docs) {
        const parts = doc.path.split("/");
        let currentPath = collPath;

        // Build directory hierarchy
        for (let i = 0; i < parts.length - 1; i++) {
          const part = parts[i]!;
          const parentPath = currentPath;
          currentPath = `${currentPath}/${part}`;

          if (!dirs.has(currentPath)) {
            dirs.set(currentPath, new Set());
          }
          dirs.get(parentPath)!.add(part);
        }

        // Add file to its parent directory
        const filename = parts[parts.length - 1]!;
        dirs.get(currentPath)!.add(filename);
      }

      // Convert Sets to arrays
      for (const [path, entries] of dirs) {
        this.dirCache.set(path, Array.from(entries).sort());
      }
    }
  }

  private normalizePath(path: string): string {
    const parts = path.split("/").filter(Boolean);
    const normalized: string[] = [];
    for (const part of parts) {
      if (part === "..") normalized.pop();
      else if (part !== ".") normalized.push(part);
    }
    return "/" + normalized.join("/");
  }

  private parseQmdPath(path: string): { collection: string; relativePath: string } | null {
    const normalized = this.normalizePath(path);
    const parts = normalized.split("/").filter(Boolean);
    if (parts.length < 1) return null;
    return {
      collection: parts[0]!,
      relativePath: parts.slice(1).join("/"),
    };
  }

  private throwReadOnly(): never {
    throw new Error("Read-only filesystem");
  }

  async readFile(path: string): Promise<string> {
    const normalized = this.normalizePath(path);

    // Check cache
    const cached = this.fileCache.get(normalized);
    if (cached) return cached.content;

    const parsed = this.parseQmdPath(normalized);
    if (!parsed || !parsed.relativePath) {
      throw new Error(`ENOENT: no such file: ${path}`);
    }

    const db = this.store.db;
    const row = db.prepare(`
      SELECT c.doc, d.modified_at
      FROM documents d
      JOIN content c ON d.hash = c.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collection, parsed.relativePath) as { doc: string; modified_at: string } | null;

    if (!row) {
      throw new Error(`ENOENT: no such file: ${path}`);
    }

    this.fileCache.set(normalized, {
      content: row.doc,
      mtime: new Date(row.modified_at),
    });

    return row.doc;
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const content = await this.readFile(path);
    return new TextEncoder().encode(content);
  }

  async writeFile(): Promise<void> {
    this.throwReadOnly();
  }

  async appendFile(): Promise<void> {
    this.throwReadOnly();
  }

  async exists(path: string): Promise<boolean> {
    const normalized = this.normalizePath(path);

    // Check if it's a directory
    if (this.dirCache.has(normalized)) return true;

    // Check if it's a file
    const parsed = this.parseQmdPath(normalized);
    if (!parsed || !parsed.relativePath) return false;

    const db = this.store.db;
    const row = db.prepare(`
      SELECT 1 FROM documents
      WHERE collection = ? AND path = ? AND active = 1
      LIMIT 1
    `).get(parsed.collection, parsed.relativePath);

    return !!row;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = this.normalizePath(path);

    // Check if it's a directory
    if (this.dirCache.has(normalized)) {
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 0o755,
        size: 0,
        mtime: new Date(),
      };
    }

    // Check if it's a file
    const parsed = this.parseQmdPath(normalized);
    if (!parsed || !parsed.relativePath) {
      throw new Error(`ENOENT: no such file or directory: ${path}`);
    }

    const db = this.store.db;
    const row = db.prepare(`
      SELECT LENGTH(c.doc) as size, d.modified_at
      FROM documents d
      JOIN content c ON d.hash = c.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collection, parsed.relativePath) as { size: number; modified_at: string } | null;

    if (!row) {
      throw new Error(`ENOENT: no such file or directory: ${path}`);
    }

    return {
      isFile: true,
      isDirectory: false,
      isSymbolicLink: false,
      mode: 0o644,
      size: row.size,
      mtime: new Date(row.modified_at),
    };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async mkdir(): Promise<void> {
    this.throwReadOnly();
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = this.normalizePath(path);
    const entries = this.dirCache.get(normalized);
    if (!entries) {
      throw new Error(`ENOENT: no such directory: ${path}`);
    }
    return [...entries];
  }

  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    const normalized = this.normalizePath(path);
    const entries = await this.readdir(path);

    return Promise.all(entries.map(async (name) => {
      const fullPath = `${normalized}/${name}`;
      const isDir = this.dirCache.has(fullPath);
      return {
        name,
        isFile: !isDir,
        isDirectory: isDir,
        isSymbolicLink: false,
      };
    }));
  }

  async rm(): Promise<void> {
    this.throwReadOnly();
  }

  async cp(): Promise<void> {
    this.throwReadOnly();
  }

  async mv(): Promise<void> {
    this.throwReadOnly();
  }

  resolvePath(base: string, path: string): string {
    if (path.startsWith("/")) return this.normalizePath(path);
    return this.normalizePath(`${base}/${path}`);
  }

  getAllPaths(): string[] {
    const paths: string[] = [];
    for (const dir of this.dirCache.keys()) {
      paths.push(dir);
    }
    // Add all files
    for (const [dir, entries] of this.dirCache) {
      for (const entry of entries) {
        const fullPath = `${dir}/${entry}`;
        if (!this.dirCache.has(fullPath)) {
          paths.push(fullPath);
        }
      }
    }
    return paths;
  }

  async chmod(): Promise<void> {
    this.throwReadOnly();
  }

  async symlink(): Promise<void> {
    this.throwReadOnly();
  }

  async link(): Promise<void> {
    this.throwReadOnly();
  }

  async readlink(): Promise<string> {
    throw new Error("EINVAL: not a symlink");
  }
}

/**
 * Format search results for terminal output
 */
function formatSearchResults(results: SearchResult[], query: string): string {
  if (results.length === 0) {
    return "No results found.\n";
  }

  const lines: string[] = [];
  for (const result of results) {
    const score = result.score.toFixed(2);
    lines.push(`${c.cyan}${result.displayPath}${c.reset} ${c.dim}(${score})${c.reset}`);
    lines.push(`  ${c.bold}${result.title}${c.reset}`);
    if (result.context) {
      lines.push(`  ${c.dim}Context: ${result.context.substring(0, 60)}...${c.reset}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

/**
 * Create custom commands for qmd
 */
function createQmdCommands(store: Store) {
  const searchCmd = defineCommand("search", async (args, ctx) => {
    const query = args.join(" ");
    if (!query) {
      return { stdout: "", stderr: "Usage: search <query>\n", exitCode: 1 };
    }

    const results = searchFTS(store.db, query, 10);
    const output = formatSearchResults(results, query);
    return { stdout: output, stderr: "", exitCode: 0 };
  });

  const queryCmd = defineCommand("query", async (args, ctx) => {
    const query = args.join(" ");
    if (!query) {
      return { stdout: "", stderr: "Usage: query <query>\n", exitCode: 1 };
    }

    // Check if vector index exists
    const tableExists = store.db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
    ).get();

    if (!tableExists) {
      // Fall back to FTS-only search
      const results = searchFTS(store.db, query, 10);
      const output = formatSearchResults(results, query);
      return {
        stdout: output,
        stderr: `${c.yellow}Note: Vector index not found. Using BM25 search only.${c.reset}\n`,
        exitCode: 0,
      };
    }

    // Perform both FTS and vector search
    const ftsResults = searchFTS(store.db, query, 20);
    const vecResults = await searchVec(store.db, query, "embeddinggemma", 20);

    // Simple RRF fusion
    const scores = new Map<string, { result: SearchResult; score: number }>();
    const k = 60;

    for (let i = 0; i < ftsResults.length; i++) {
      const r = ftsResults[i]!;
      const rrfScore = 1 / (k + i + 1);
      scores.set(r.filepath, { result: r, score: rrfScore });
    }

    for (let i = 0; i < vecResults.length; i++) {
      const r = vecResults[i]!;
      const rrfScore = 1 / (k + i + 1);
      const existing = scores.get(r.filepath);
      if (existing) {
        existing.score += rrfScore;
      } else {
        scores.set(r.filepath, { result: r, score: rrfScore });
      }
    }

    const merged = Array.from(scores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(({ result, score }) => ({ ...result, score }));

    const output = formatSearchResults(merged, query);
    return { stdout: output, stderr: "", exitCode: 0 };
  });

  const helpCmd = defineCommand("qmd-help", async () => {
    const help = `${c.bold}QMD REPL Commands${c.reset}

${c.cyan}search${c.reset} <query>   - BM25 full-text search
${c.cyan}query${c.reset} <query>    - Hybrid search (BM25 + vectors)

${c.bold}Filesystem${c.reset}
  /qmd/<collection>/  - Read-only qmd documents
  /home/              - Writable home directory

${c.bold}Standard Commands${c.reset}
  ls, cat, head, tail, grep, find, etc.

`;
    return { stdout: help, stderr: "", exitCode: 0 };
  });

  return [searchCmd, queryCmd, helpCmd];
}

/**
 * Get the shell prompt
 */
function getPrompt(cwd: string): string {
  // Simplify path for display
  let displayPath = cwd;
  if (cwd.startsWith("/home")) {
    displayPath = "~" + cwd.slice("/home".length);
  }

  const user = process.env.USER || "user";
  const hostname = process.env.HOSTNAME || "qmd";

  return `${c.green}${user}@${hostname}${c.reset}:${c.blue}${displayPath}${c.reset}$ `;
}

/**
 * Main REPL function
 */
async function startRepl(options: { prompt?: string }) {
  const store = createStore();

  // Create the QMD read-only filesystem
  const qmdFs = new QmdReadOnlyFs(store);

  // Create in-memory filesystem as base (for paths not under /qmd or /home)
  const baseFs = new InMemoryFs();

  // Create in-memory filesystem for /home with some initial structure
  const homeFs = new InMemoryFs({
    "/.bashrc": "# QMD REPL\nexport PS1='\\u@qmd:\\w$ '\n",
  });

  // Create mountable filesystem
  const fs = new MountableFs({ base: baseFs });
  fs.mount("/qmd", qmdFs);
  fs.mount("/home", homeFs);

  // Handle --prompt argument
  if (options.prompt) {
    await homeFs.writeFile("/prompt.txt", options.prompt);
    console.log(`${c.dim}Wrote prompt to /home/prompt.txt${c.reset}`);
  }

  // Create custom commands
  const customCommands = createQmdCommands(store);

  // Create bash environment
  const bash = new Bash({
    fs,
    cwd: "/home",
    env: {
      HOME: "/home",
      USER: process.env.USER || "user",
      PATH: "/usr/bin:/bin",
      TERM: process.env.TERM || "xterm-256color",
    },
    customCommands,
  });

  // Print welcome message
  console.log(`${c.bold}QMD REPL${c.reset} ${c.dim}(type 'qmd-help' for commands, 'exit' to quit)${c.reset}`);

  const collections = yamlListCollections();
  if (collections.length > 0) {
    console.log(`${c.dim}Collections mounted at /qmd/:${c.reset}`);
    for (const coll of collections) {
      const count = store.db.prepare(
        `SELECT COUNT(*) as c FROM documents WHERE collection = ? AND active = 1`
      ).get(coll.name) as { c: number };
      console.log(`  ${c.cyan}/qmd/${coll.name}/${c.reset} ${c.dim}(${count.c} files)${c.reset}`);
    }
  }
  console.log("");

  // Create readline interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
    completer: (line: string) => {
      // Basic tab completion
      const completions: string[] = [];
      const parts = line.split(/\s+/);
      const lastPart = parts[parts.length - 1] || "";

      // Command completion
      if (parts.length === 1) {
        const commands = [
          "search", "query", "qmd-help", "ls", "cat", "head", "tail",
          "grep", "find", "pwd", "cd", "echo", "exit", "help"
        ];
        completions.push(...commands.filter(cmd => cmd.startsWith(lastPart)));
      }

      // Path completion
      if (lastPart.includes("/") || parts.length > 1) {
        try {
          const cwd = bash.getCwd();
          let searchPath = lastPart;
          let prefix = "";

          if (!searchPath.startsWith("/")) {
            searchPath = `${cwd}/${searchPath}`;
          }

          // Find the directory to search
          const lastSlash = searchPath.lastIndexOf("/");
          const dir = lastSlash === 0 ? "/" : searchPath.substring(0, lastSlash);
          const partial = searchPath.substring(lastSlash + 1);
          prefix = lastPart.substring(0, lastPart.lastIndexOf("/") + 1);

          // Get directory contents (sync-ish via cache)
          const allPaths = fs.getAllPaths();
          const matching = allPaths
            .filter(p => p.startsWith(dir === "/" ? "/" : dir + "/"))
            .map(p => {
              const rel = p.substring(dir === "/" ? 1 : dir.length + 1);
              const firstPart = rel.split("/")[0];
              return firstPart;
            })
            .filter((v, i, a) => a.indexOf(v) === i) // unique
            .filter(name => name && name.startsWith(partial))
            .map(name => prefix + name);

          completions.push(...matching);
        } catch {
          // Ignore completion errors
        }
      }

      return [completions, lastPart];
    },
  });

  // Track cwd ourselves since just-bash doesn't persist cd across exec calls
  let cwd = "/home";

  // Resolve a path relative to cwd
  const resolvePath = (path: string): string => {
    if (path.startsWith("/")) return path;
    if (path === "~" || path.startsWith("~/")) {
      return "/home" + path.slice(1);
    }
    const parts = (cwd + "/" + path).split("/").filter(Boolean);
    const resolved: string[] = [];
    for (const part of parts) {
      if (part === "..") resolved.pop();
      else if (part !== ".") resolved.push(part);
    }
    return "/" + resolved.join("/");
  };

  const promptUser = () => {
    rl.question(getPrompt(cwd), async (line) => {
      const trimmed = line.trim();

      if (!trimmed) {
        promptUser();
        return;
      }

      if (trimmed === "exit" || trimmed === "quit") {
        console.log("Goodbye!");
        rl.close();
        store.close();
        process.exit(0);
      }

      try {
        // Handle cd specially to track cwd ourselves
        const cdMatch = trimmed.match(/^cd\s*(.*)$/);
        if (cdMatch) {
          const target = cdMatch[1]?.trim() || "/home";
          const newCwd = resolvePath(target);

          // Verify the directory exists
          const exists = await fs.exists(newCwd);
          if (!exists) {
            console.error(`${c.red}cd: ${target}: No such file or directory${c.reset}`);
          } else {
            const stat = await fs.stat(newCwd);
            if (!stat.isDirectory) {
              console.error(`${c.red}cd: ${target}: Not a directory${c.reset}`);
            } else {
              cwd = newCwd;
            }
          }
          promptUser();
          return;
        }

        // Execute other commands with our tracked cwd
        const result = await bash.exec(trimmed, { cwd });

        if (result.stdout) {
          process.stdout.write(result.stdout);
        }
        if (result.stderr) {
          process.stderr.write(`${c.red}${result.stderr}${c.reset}`);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`${c.red}Error: ${msg}${c.reset}`);
      }

      promptUser();
    });
  };

  promptUser();
}

/**
 * Parse CLI arguments and start REPL
 */
async function main() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      prompt: { type: "string", short: "p" },
      file: { type: "string", short: "f" },
      help: { type: "boolean", short: "h" },
    },
    allowPositionals: true,
  });

  if (values.help) {
    console.log(`Usage: qmd repl [options]

Options:
  -p, --prompt <text>   Initialize /home/prompt.txt with text
  -f, --file <path>     Initialize /home/prompt.txt with file contents
  -h, --help            Show this help message
`);
    process.exit(0);
  }

  let promptContent: string | undefined;

  if (values.file) {
    try {
      promptContent = await Bun.file(values.file).text();
    } catch (err) {
      console.error(`Error reading file: ${values.file}`);
      process.exit(1);
    }
  } else if (values.prompt) {
    promptContent = values.prompt;
  }
  // Note: positionals are not used for prompt - use --prompt or --file explicitly

  await startRepl({ prompt: promptContent });
}

// Export for CLI integration
export { main as startRepl };

// Run if called directly
if (import.meta.main) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
