#!/usr/bin/env bun
/**
 * Check every formal model against test/formal/expectations.json.
 *
 * Each manifest entry pins one `everystate` run to an exact outcome:
 *
 *   { "model": "ServerShutdown.tla", "config": "ServerShutdown.cfg", "status": "ok" }
 *   { "model": "ServerShutdown.tla", "config": "ServerShutdownPreFix.cfg",
 *     "status": "invariant", "name": "SaneShutdownOrder" }
 *
 * `model` and `config` are resolved under test/formal (the extension is
 * optional); `status` and `name` are the JSON result fields everystate reports.
 * An empty or malformed manifest fails: a skipped check is not a pass.
 * Without an explicit manifest argument, the default manifest must list every
 * test/formal/*.tla and *.cfg, so a new model or negative control cannot
 * evade CI silently. Pass a manifest path for a targeted, partial run.
 *
 * Usage: bun scripts/check-formal.ts [manifest]
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const FORMAL_DIR = join(ROOT, "test", "formal");
const DEFAULT_MANIFEST = join(FORMAL_DIR, "expectations.json");
const EVERYSATE = "everystate@0.3.1";
const TIMEOUT_MS = Number(process.env.FORMAL_CHECK_TIMEOUT_MS ?? 600_000);

type Status =
  | "ok"
  | "assumption"
  | "deadlock"
  | "invariant"
  | "invariant-init"
  | "action-property"
  | "implied-init"
  | "liveness"
  | "error"
  | "init-error"
  | "incomplete"
  | "setup-error";

/** Documented everystate exit codes per JSON status. */
const STATUS_EXIT_CODES: Record<Status, readonly number[]> = {
  ok: [0],
  assumption: [10],
  deadlock: [11],
  invariant: [12],
  "invariant-init": [12],
  "action-property": [12],
  "implied-init": [12],
  liveness: [13],
  error: [1, 150, 151],
  "init-error": [1, 150, 151],
  incomplete: [1],
  "setup-error": [1, 150, 151],
};

/** Failures that must name the exact violated invariant/property. */
const NAMED_STATUSES: Record<string, true> = {
  assumption: true,
  invariant: true,
  "invariant-init": true,
  "action-property": true,
  "implied-init": true,
  liveness: true,
};

const ENTRY_KEYS: ReadonlySet<string> = new Set(["model", "config", "status", "name"]);

interface Expectation {
  model: string;
  config: string;
  status: Status;
  name?: string;
}

interface CheckedExpectation {
  expectation: Expectation;
  modelPath: string;
  configPath: string;
}

interface RunResult {
  exitCode: number;
  status: string;
  name?: string;
  distinct?: number;
  generated?: number;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  /** Set when the run could not be interpreted at all (no JSON, spawn failure). */
  parseError?: string;
}

function fatal(lines: string | string[]): never {
  for (const line of Array.isArray(lines) ? lines : [lines]) {
    console.error(`formal: ${line}`);
  }
  process.exit(1);
}

function isStatus(value: unknown): value is Status {
  return typeof value === "string" && Object.hasOwn(STATUS_EXIT_CODES, value);
}

function loadManifest(manifestPath: string): Expectation[] {
  const manifestLabel = relative(ROOT, manifestPath) || manifestPath;
  let text: string;
  try {
    text = readFileSync(manifestPath, "utf8");
  } catch (error) {
    fatal(`cannot read ${manifestLabel}: ${error instanceof Error ? error.message : String(error)}`);
  }

  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    fatal(`cannot parse ${manifestLabel}: ${error instanceof Error ? error.message : String(error)}`);
  }

  if (!Array.isArray(raw) || raw.length === 0) {
    fatal(`${manifestLabel} must be a non-empty array of {model, config, status, name?}`);
  }

  const problems: string[] = [];
  const expectations: Expectation[] = [];

  const items: unknown[] = raw;
  items.forEach((item, index) => {
    const where = `entry ${index + 1}`;
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      problems.push(`${where}: expected an object`);
      return;
    }
    const entry = item as Record<string, unknown>;
    for (const key of Object.keys(entry)) {
      if (!ENTRY_KEYS.has(key)) problems.push(`${where}: unknown key "${key}"`);
    }

    const { model, config, status, name } = entry;
    if (typeof model !== "string" || model.trim() === "") problems.push(`${where}: "model" must be a file name`);
    if (typeof config !== "string" || config.trim() === "") problems.push(`${where}: "config" must be a file name`);
    if (!isStatus(status)) {
      problems.push(`${where}: "status" must be one of ${Object.keys(STATUS_EXIT_CODES).join(", ")}`);
    }
    if (name !== undefined && (typeof name !== "string" || name.trim() === "")) {
      problems.push(`${where}: "name" must be a non-empty string when present`);
    }
    if (isStatus(status) && NAMED_STATUSES[status] === true && typeof name !== "string") {
      problems.push(`${where}: status "${status}" requires the exact invariant/property "name"`);
    }

    if (typeof model !== "string" || typeof config !== "string" || !isStatus(status)) return;

    expectations.push({
      model,
      config,
      status,
      ...(typeof name === "string" ? { name } : {}),
    });
  });

  if (problems.length > 0) fatal(problems);
  return expectations;
}

function resolveFormalFile(name: string, extension: string): string {
  const file = name.endsWith(extension) ? name : `${name}${extension}`;
  return isAbsolute(file) ? file : join(FORMAL_DIR, file);
}

function prepare(expectations: Expectation[]): CheckedExpectation[] {
  const problems: string[] = [];
  const seen = new Map<string, number>();
  const prepared = expectations.map((expectation, index) => {
    const modelPath = resolveFormalFile(expectation.model, ".tla");
    const configPath = resolveFormalFile(expectation.config, ".cfg");

    const key = `${modelPath}\u0000${configPath}`;
    const previous = seen.get(key);
    if (previous !== undefined) {
      problems.push(`entry ${index + 1}: duplicate of entry ${previous} (${expectation.model} + ${expectation.config})`);
    } else {
      seen.set(key, index + 1);
    }

    for (const [label, path] of [
      ["model", modelPath],
      ["config", configPath],
    ] as const) {
      const stats = statSync(path, { throwIfNoEntry: false });
      if (!stats?.isFile()) problems.push(`${label} ${relative(ROOT, path) || path}: missing or not a file`);
    }
    return { expectation, modelPath, configPath };
  });
  if (problems.length > 0) fatal(problems);
  return prepared;
}

/** The default manifest must cover every model and configuration on disk. */
function checkDefaultCoverage(checked: CheckedExpectation[]): void {
  const listedModels = new Set(checked.map(({ modelPath }) => modelPath));
  const listedConfigs = new Set(checked.map(({ configPath }) => configPath));
  const problems: string[] = [];

  for (const entry of readdirSync(FORMAL_DIR, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const path = join(FORMAL_DIR, entry.name);
    if (entry.name.endsWith(".tla") && !listedModels.has(path)) {
      problems.push(
        `model ${relative(ROOT, path)} has no configuration expectation in the default manifest`,
      );
    }
    if (entry.name.endsWith(".cfg") && !listedConfigs.has(path)) {
      problems.push(`config ${relative(ROOT, path)} is not listed in the default manifest`);
    }
  }

  if (problems.length > 0) fatal(problems);
}

interface JsonResult {
  status: string;
  name?: string;
  generated?: number;
  distinct?: number;
}

function parseJsonResult(stdout: string): JsonResult | string {
  // everystate -json prints one JSON object; tolerate Print output around it.
  const trimmed = stdout.trim();
  const candidates = [trimmed, ...trimmed.split("\n").reverse()];
  for (const candidate of candidates) {
    if (!candidate.startsWith("{")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(candidate);
    } catch {
      continue;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) continue;
    const result = parsed as Record<string, unknown>;
    if (typeof result.status !== "string") return "JSON result has no string \"status\"";
    return {
      status: result.status,
      ...(typeof result.name === "string" ? { name: result.name } : {}),
      ...(typeof result.generated === "number" ? { generated: result.generated } : {}),
      ...(typeof result.distinct === "number" ? { distinct: result.distinct } : {}),
    };
  }
  return "no JSON result on stdout";
}

async function runCheck(checked: CheckedExpectation): Promise<RunResult> {
  const proc = Bun.spawn(
    ["bunx", EVERYSATE, "-json", "-config", checked.configPath, checked.modelPath],
    { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
  );
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill();
  }, TIMEOUT_MS);

  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      Bun.readableStreamToText(proc.stdout),
      Bun.readableStreamToText(proc.stderr),
      proc.exited,
    ]);
    const parsed = parseJsonResult(stdout);
    if (typeof parsed === "string") {
      return { exitCode, status: "", stdout, stderr, timedOut, parseError: parsed };
    }
    return { exitCode, ...parsed, stdout, stderr, timedOut };
  } finally {
    clearTimeout(timer);
  }
}

function verify(expectation: Expectation, result: RunResult): string[] {
  const problems: string[] = [];
  if (result.timedOut) problems.push(`timed out after ${TIMEOUT_MS}ms`);
  if (result.parseError !== undefined) problems.push(result.parseError);

  const expectedCodes = STATUS_EXIT_CODES[expectation.status];
  if (!expectedCodes.includes(result.exitCode)) {
    problems.push(
      `expected exit ${expectedCodes.join(" or ")} for status "${expectation.status}", got ${result.exitCode}`,
    );
  }
  if (result.status !== expectation.status) {
    problems.push(
      `expected status "${expectation.status}", got ${result.status === "" ? "(none)" : `"${result.status}"`}`,
    );
  }
  if (expectation.name !== undefined && result.name !== expectation.name) {
    problems.push(
      `expected name "${expectation.name}", got ${result.name === undefined ? "(none)" : `"${result.name}"`}`,
    );
  }
  return problems;
}

function describe(expectation: Expectation, result: RunResult): string {
  const parts = [`status "${result.status || "(none)"}"`, `exit ${result.exitCode}`];
  if (result.name !== undefined) parts.push(`name "${result.name}"`);
  if (result.distinct !== undefined) parts.push(`${result.distinct} distinct states`);
  return `${expectation.model} (${expectation.config}) - ${parts.join(", ")}`;
}

function preview(text: string, max = 1500): string {
  const trimmed = text.trim();
  if (trimmed.length <= max) return trimmed;
  return `...${trimmed.slice(-max)}`;
}

async function main(): Promise<void> {
  const manifestArg = process.argv[2];
  const manifestPath = manifestArg === undefined ? DEFAULT_MANIFEST : resolve(manifestArg);
  const expectations = loadManifest(manifestPath);
  const prepared = prepare(expectations);
  if (manifestArg === undefined) checkDefaultCoverage(prepared);

  console.log(`formal: ${prepared.length} expectations from ${relative(ROOT, manifestPath) || manifestPath}`);

  const failures: string[] = [];
  for (const checked of prepared) {
    let result: RunResult;
    try {
      result = await runCheck(checked);
    } catch (error) {
      result = {
        exitCode: -1,
        status: "",
        stdout: "",
        stderr: "",
        timedOut: false,
        parseError: `could not run bunx ${EVERYSATE}: ${error instanceof Error ? error.message : String(error)}`,
      };
    }

    const problems = verify(checked.expectation, result);
    if (problems.length === 0) {
      console.log(`formal: PASS ${describe(checked.expectation, result)}`);
      continue;
    }

    failures.push(`${checked.expectation.model} (${checked.expectation.config})`);
    console.error(`formal: FAIL ${describe(checked.expectation, result)}`);
    for (const problem of problems) console.error(`       ${problem}`);
    for (const [channel, text] of [
      ["stdout", result.stdout],
      ["stderr", result.stderr],
    ] as const) {
      if (text.trim() === "") continue;
      console.error(`       ${channel}: ${preview(text).split("\n").join("\n       ")}`);
    }
  }

  if (failures.length > 0) {
    console.error(`formal: ${failures.length} of ${prepared.length} checks failed: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log(`formal: all ${prepared.length} checks passed`);
}

await main();
