import { spawn } from "node:child_process";
import { createHash } from "node:crypto";

export interface CommandProcessorSpec {
  command: string[];
  version: string;
  timeoutMs?: number;
}

export interface ProcessorInput {
  version: 1;
  collection: string;
  relativePath: string;
  hash: string;
  modifiedAt: string;
  bytes: Uint8Array;
}

export interface ProcessorDocument { key: string; markdown: string }

export interface DocumentProcessor {
  /** Must change whenever implementation or configuration changes. */
  fingerprint: string;
  process(input: ProcessorInput, signal: AbortSignal): Promise<readonly ProcessorDocument[]>;
}

export const PROCESSOR_INPUT_LIMIT = 10 * 1024 * 1024;
const OUTPUT_LIMIT = 24 * 1024 * 1024;
const MARKDOWN_LIMIT = 16 * 1024 * 1024;

export class ProcessorFailure extends Error {
  constructor(public readonly code: string) { super(code); }
}

function exactObject(value: unknown, keys: string[]): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).every(key => keys.includes(key));
}

export function copyCommandProcessorSpec(value: unknown): Readonly<CommandProcessorSpec> {
  if (!exactObject(value, ["command", "version", "timeoutMs"])
    || !Array.isArray(value.command) || value.command.length < 1 || value.command.length > 128
    || value.command.some(arg => typeof arg !== "string" || arg.includes("\0") || arg.length > 8192)
    || !value.command[0] || typeof value.version !== "string" || !value.version.trim() || value.version.length > 1024
    || (value.timeoutMs !== undefined && (!Number.isInteger(value.timeoutMs) || Number(value.timeoutMs) < 1 || Number(value.timeoutMs) > 60000))) {
    throw new ProcessorFailure("PROCESSOR_CONFIG_INVALID");
  }
  const command = [...value.command] as string[];
  Object.freeze(command);
  return Object.freeze({ command, version: value.version, timeoutMs: Number(value.timeoutMs ?? 10000) });
}

export function processorFingerprint(value: unknown): string {
  if (typeof value !== "string" || !value.trim() || value.length > 4096) throw new ProcessorFailure("PROCESSOR_FINGERPRINT_INVALID");
  return value;
}

export function snapshotProcessor(processor: DocumentProcessor): Readonly<DocumentProcessor> {
  const fingerprint = processorFingerprint(processor.fingerprint);
  const process = processor.process;
  if (typeof process !== "function") throw new ProcessorFailure("PROCESSOR_INVALID");
  return Object.freeze({ fingerprint, process: process.bind(processor) });
}

export function validateProcessorDocuments(value: unknown): readonly ProcessorDocument[] {
  if (!Array.isArray(value) || value.length > 1000) throw new ProcessorFailure("PROCESSOR_OUTPUT_INVALID");
  const seen = new Set<string>();
  let size = 0;
  return value.map(item => {
    if (!exactObject(item, ["key", "markdown"]) || typeof item.key !== "string" || typeof item.markdown !== "string") throw new ProcessorFailure("PROCESSOR_OUTPUT_INVALID");
    if (!item.markdown.isWellFormed()) throw new ProcessorFailure("PROCESSOR_OUTPUT_INVALID");
    const key = item.key;
    if (!key.isWellFormed() || !key.endsWith(".md") || Buffer.byteLength(key) > 256 || /[\\\x00-\x1f\x7f-\x9f:]/.test(key)
      || key.split("/").some(part => !part || part === "." || part === "..") || seen.has(key)) throw new ProcessorFailure("PROCESSOR_KEY_INVALID");
    seen.add(key);
    size += Buffer.byteLength(item.markdown);
    if (size > MARKDOWN_LIMIT) throw new ProcessorFailure("PROCESSOR_OUTPUT_LIMIT");
    return { key, markdown: item.markdown };
  });
}

/** Bounds asynchronous callbacks; synchronous CPU loops cannot be interrupted. */
export async function executeProcessor(processor: DocumentProcessor, input: ProcessorInput, signal?: AbortSignal): Promise<readonly ProcessorDocument[]> {
  const owned = snapshotProcessor(processor);
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let abort = (): void => {};
  try {
    return await new Promise((resolve, reject) => {
      abort = () => { controller.abort(); reject(new ProcessorFailure("PROCESSOR_ABORTED")); };
      signal?.addEventListener("abort", abort, { once: true });
      timer = setTimeout(() => { controller.abort(); reject(new ProcessorFailure("PROCESSOR_TIMEOUT")); }, 60000);
      if (signal?.aborted) { abort(); return; }
      Promise.resolve().then(() => owned.process({ ...input, bytes: new Uint8Array(input.bytes) }, controller.signal))
        .then(value => { if (!controller.signal.aborted) resolve(validateProcessorDocuments(value)); }, error => reject(error instanceof ProcessorFailure ? error : new ProcessorFailure("PROCESSOR_FAILED")))
        .catch(error => reject(error instanceof ProcessorFailure ? error : new ProcessorFailure("PROCESSOR_OUTPUT_INVALID")));
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

/** Explicit opt-in to executing an argv command; this is not an OS sandbox. */
export function createCommandProcessor(spec: CommandProcessorSpec, options: { cwd: string }): DocumentProcessor {
  const normalized = copyCommandProcessorSpec(spec);
  const cwd = options.cwd;
  return {
    fingerprint: createHash("sha256").update(JSON.stringify(normalized)).digest("hex"),
    process: (input, signal) => new Promise((resolve, reject) => {
      if (input.bytes.byteLength > PROCESSOR_INPUT_LIMIT) { reject(new ProcessorFailure("PROCESSOR_INPUT_LIMIT")); return; }
      if (signal.aborted) { reject(new ProcessorFailure("PROCESSOR_ABORTED")); return; }
      const child = spawn(normalized.command[0]!, normalized.command.slice(1), {
        cwd, shell: false, detached: process.platform !== "win32",
        env: { PATH: process.env.PATH, HOME: cwd, LANG: "C.UTF-8" }, stdio: ["pipe", "pipe", "pipe"],
      });
      const chunks: Buffer[] = [];
      let stdoutBytes = 0, stderrBytes = 0, settled = false;
      const dispose = (): void => {
        if (child.pid) {
          try { if (process.platform !== "win32") process.kill(-child.pid, "SIGKILL"); else child.kill("SIGKILL"); } catch { /* already exited */ }
        }
        child.stdin.destroy(); child.stdout.destroy(); child.stderr.destroy();
      };
      const finish = (error?: ProcessorFailure, documents?: readonly ProcessorDocument[]): void => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal.removeEventListener("abort", abort);
        dispose();
        if (error) reject(error); else resolve(documents!);
      };
      const abort = (): void => finish(new ProcessorFailure("PROCESSOR_ABORTED"));
      const timer = setTimeout(() => finish(new ProcessorFailure("PROCESSOR_TIMEOUT")), normalized.timeoutMs);
      signal.addEventListener("abort", abort, { once: true });
      child.on("error", () => finish(new ProcessorFailure("PROCESSOR_SPAWN_FAILED")));
      child.stdin.on("error", () => finish(new ProcessorFailure("PROCESSOR_INPUT_FAILED")));
      child.stdout.on("data", (chunk: Buffer) => {
        stdoutBytes += chunk.length;
        if (stdoutBytes > OUTPUT_LIMIT) finish(new ProcessorFailure("PROCESSOR_STDOUT_LIMIT")); else chunks.push(chunk);
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.length;
        if (stderrBytes > 64 * 1024) finish(new ProcessorFailure("PROCESSOR_STDERR_LIMIT"));
      });
      child.on("close", code => {
        if (settled) return;
        if (code !== 0) { finish(new ProcessorFailure("PROCESSOR_EXIT_FAILED")); return; }
        try {
          const parsed: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
          if (!exactObject(parsed, ["version", "documents"]) || parsed.version !== 1) throw new Error();
          finish(undefined, validateProcessorDocuments(parsed.documents));
        } catch (error) { finish(error instanceof ProcessorFailure ? error : new ProcessorFailure("PROCESSOR_OUTPUT_INVALID")); }
      });
      child.stdin.end(JSON.stringify({ version: 1, collection: input.collection, source: {
        path: input.relativePath, hash: input.hash, modifiedAt: input.modifiedAt,
        contentBase64: Buffer.from(input.bytes).toString("base64"),
      } }));
    }),
  };
}
