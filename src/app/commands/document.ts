import type { OutputFormat } from "../../formatter.js";

export function handleGetCommand(
  args: string[],
  values: Record<string, unknown>,
  opts: { lineNumbers?: boolean },
  deps: {
    getDocument: (filename: string, fromLine?: number, maxLines?: number, lineNumbers?: boolean) => void;
  }
): void {
  if (!args[0]) {
    console.error("Usage: qmd get <filepath>[:line] [--from <line>] [-l <lines>] [--line-numbers]");
    process.exit(1);
  }
  const fromLine = values.from ? parseInt(String(values.from), 10) : undefined;
  const maxLines = values.l ? parseInt(String(values.l), 10) : undefined;
  deps.getDocument(args[0], fromLine, maxLines, opts.lineNumbers);
}

export function handleMultiGetCommand(
  args: string[],
  values: Record<string, unknown>,
  format: OutputFormat,
  defaultMaxBytes: number,
  deps: {
    multiGet: (pattern: string, maxLines?: number, maxBytes?: number, format?: OutputFormat) => void;
  }
): void {
  if (!args[0]) {
    console.error("Usage: qmd multi-get <pattern> [-l <lines>] [--max-bytes <bytes>] [--json|--csv|--md|--xml|--files]");
    console.error("  pattern: glob (e.g., 'journals/2025-05*.md') or comma-separated list");
    process.exit(1);
  }
  const maxLinesMulti = values.l ? parseInt(String(values.l), 10) : undefined;
  const maxBytes = values["max-bytes"] ? parseInt(String(values["max-bytes"]), 10) : defaultMaxBytes;
  deps.multiGet(args[0], maxLinesMulti, maxBytes, format);
}

export function handleLsCommand(
  args: string[],
  deps: {
    listFiles: (pathArg?: string) => void;
  }
): void {
  deps.listFiles(args[0]);
}
