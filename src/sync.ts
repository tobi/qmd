import { execFile } from "child_process";
import { mkdtemp, mkdir, rm, writeFile } from "fs/promises";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { homedir, hostname, tmpdir } from "os";
import YAML from "yaml";
import type { CollectionConfig } from "./collections.js";
import { getConfigPath, loadConfig } from "./collections.js";

export type SyncOptions = {
  host?: string;
  remoteUser?: string;
  remoteQmdUser?: string;
  remoteHome?: string;
  collection?: string[];
  dryRun?: boolean;
  delete?: boolean;
  update?: boolean;
  embed?: boolean;
  yes?: boolean;
  json?: boolean;
  localQmdCommand?: string[];
  runCommand?: CommandRunner;
};

export type CommandRunner = (
  command: string,
  args: string[],
  options?: { cwd?: string }
) => Promise<CommandResult>;

export type CommandResult = {
  stdout: string;
  stderr: string;
  exitCode: number;
};

export type SyncCollectionPlan = {
  name: string;
  direction: "bidirectional" | "download-mirror" | "upload-mirror";
  localPath: string;
  remotePath: string;
  pattern?: string;
  localConfigured: boolean;
  remoteConfigured: boolean;
};

export type SyncDependencyStatus = {
  qmdVersion?: string;
  rsync: boolean;
  flock: boolean;
  warnings: string[];
};

export type SyncRsyncResult = {
  label: string;
  phase: "preflight" | "apply";
  direction: "download" | "upload";
  source: string;
  destination: string;
  itemized: string[];
  skipped: boolean;
  reason?: string;
};

export type PostSyncResult = {
  side: "local" | "remote";
  action: "update" | "embed";
  command: string[];
  skipped: boolean;
  reason?: string;
  exitCode?: number;
  stdout?: string;
  stderr?: string;
};

export type SyncConflict = {
  collection: string;
  path: string;
  localConflictPath: string;
  remoteConflictPath: string;
};

export type SyncSummary = {
  dryRun: boolean;
  host: string;
  remoteQmdUser: string;
  localConfigPath: string;
  remoteConfigPath: string;
  collections: SyncCollectionPlan[];
  dependencies: SyncDependencyStatus;
  rsync: SyncRsyncResult[];
  conflicts: SyncConflict[];
  postSync: PostSyncResult[];
  failed: boolean;
  warnings: string[];
  nextSteps: string[];
};

const DEFAULT_HOST = "root@xworkmate-bridge.svc.plus";
const DEFAULT_REMOTE_QMD_USER = "ubuntu";
const DEFAULT_REMOTE_HOME = "/home/ubuntu";
const REMOTE_CONFIG_RELATIVE = ".config/qmd/index.yml";
const REMOTE_CACHE_RELATIVE = ".cache/qmd";

export const QMD_SYNC_EXCLUDES = [
  ".git/",
  "node_modules/",
  "vendor/",
  "dist/",
  "build/",
  ".cache/",
  ".qmd-rsync-partial/",
  ".qmd-rsync-tmp/",
  "*.sqlite",
  "*.sqlite-wal",
  "*.sqlite-shm",
  "models/",
];

export function getDefaultSyncOptions(options: SyncOptions = {}): Required<Omit<SyncOptions, "collection" | "runCommand" | "localQmdCommand">> & {
  collection?: string[];
  localQmdCommand: string[];
  runCommand: CommandRunner;
} {
  return {
    host: options.host || DEFAULT_HOST,
    remoteUser: options.remoteUser || "",
    remoteQmdUser: options.remoteQmdUser || DEFAULT_REMOTE_QMD_USER,
    remoteHome: options.remoteHome || DEFAULT_REMOTE_HOME,
    collection: options.collection,
    dryRun: Boolean(options.dryRun),
    delete: Boolean(options.delete),
    update: Boolean(options.update),
    embed: Boolean(options.embed),
    yes: Boolean(options.yes),
    json: Boolean(options.json),
    localQmdCommand: options.localQmdCommand?.length ? options.localQmdCommand : ["qmd"],
    runCommand: options.runCommand || defaultRunCommand,
  };
}

export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

export function remoteUserCommand(user: string, command: string): string {
  return `sudo -u ${shellQuote(user)} sh -lc ${shellQuote(command)}`;
}

export function remoteRsyncPath(user: string): string {
  return `sudo -u ${shellQuote(user)} rsync`;
}

export function getLocalSyncDataRoot(host: string): string {
  const base = process.env.XDG_DATA_HOME || join(homedir(), ".local", "share");
  return join(base, "qmd", "sync", sanitizePathSegment(host));
}

export function getRemoteSyncDataRoot(remoteHome: string, localHost: string = hostname()): string {
  return `${remoteHome}/.local/share/qmd/sync/${sanitizePathSegment(localHost)}`;
}

export function sanitizePathSegment(value: string): string {
  const sanitized = value.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return sanitized || "default";
}

export function parseConfigYaml(raw: string, label: string): CollectionConfig {
  try {
    const parsed = YAML.parse(raw || "collections: {}\n") as CollectionConfig | null;
    return { ...parsed, collections: parsed?.collections || {} };
  } catch (error) {
    throw new Error(`Failed to parse ${label}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export function buildCollectionPlans(params: {
  localConfig: CollectionConfig;
  remoteConfig: CollectionConfig;
  host: string;
  remoteHome: string;
  collectionNames?: string[];
}): SyncCollectionPlan[] {
  const localCollections = params.localConfig.collections || {};
  const remoteCollections = params.remoteConfig.collections || {};
  const names = params.collectionNames?.length
    ? params.collectionNames
    : Array.from(new Set([...Object.keys(localCollections), ...Object.keys(remoteCollections)])).sort();

  const localMirrorRoot = getLocalSyncDataRoot(params.host);
  const remoteMirrorRoot = getRemoteSyncDataRoot(params.remoteHome);

  return names.map((name) => {
    const local = localCollections[name];
    const remote = remoteCollections[name];
    if (local && remote) {
      return {
        name,
        direction: "bidirectional",
        localPath: local.path,
        remotePath: remote.path,
        pattern: local.pattern || remote.pattern,
        localConfigured: true,
        remoteConfigured: true,
      };
    }
    if (remote) {
      return {
        name,
        direction: "download-mirror",
        localPath: join(localMirrorRoot, name),
        remotePath: remote.path,
        pattern: remote.pattern,
        localConfigured: false,
        remoteConfigured: true,
      };
    }
    if (local) {
      return {
        name,
        direction: "upload-mirror",
        localPath: local.path,
        remotePath: `${remoteMirrorRoot}/${name}`,
        pattern: local.pattern,
        localConfigured: true,
        remoteConfigured: false,
      };
    }
    return {
      name,
      direction: "bidirectional",
      localPath: join(localMirrorRoot, name),
      remotePath: `${remoteMirrorRoot}/${name}`,
      pattern: undefined,
      localConfigured: false,
      remoteConfigured: false,
    };
  });
}

export function includePatternsForCollection(pattern?: string): string[] {
  if (!pattern) return [];
  if (pattern === "**/*.md") return ["*/", "*.md"];
  return [];
}

export function buildRsyncArgs(params: {
  source: string;
  destination: string;
  remoteQmdUser: string;
  dryRun?: boolean;
  delete?: boolean;
  excludes?: string[];
  includes?: string[];
  excludeFrom?: string;
  preserveFilePath?: boolean;
  tempDir?: string;
}): string[] {
  const args = [
    "-az",
    "--itemize-changes",
    "--partial",
    "--partial-dir=.qmd-rsync-partial",
    "--delay-updates",
    "--rsync-path",
    remoteRsyncPath(params.remoteQmdUser),
  ];

  if (!params.dryRun) {
    args.push("--temp-dir", params.tempDir || ".qmd-rsync-tmp");
  }
  if (params.dryRun) args.push("--dry-run");
  if (params.delete) args.push("--delete");
  for (const pattern of params.includes || []) {
    args.push("--include", pattern);
  }
  if (params.includes?.length) {
    args.push("--exclude", "*");
  }
  for (const pattern of params.excludes || QMD_SYNC_EXCLUDES) {
    args.push("--exclude", pattern);
  }
  if (params.excludeFrom) {
    args.push("--exclude-from", params.excludeFrom);
  }
  args.push(
    formatRsyncEndpoint(params.preserveFilePath ? params.source : ensureTrailingSlash(params.source)),
    formatRsyncEndpoint(params.preserveFilePath ? params.destination : ensureTrailingSlash(params.destination)),
  );
  return args;
}

export function parseRsyncItemized(stdout: string): string[] {
  return stdout.split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .filter(line => !line.endsWith("/"))
    .map(line => {
      const match = line.match(/^.{11}\s+(.+)$/);
      return match?.[1] || "";
    })
    .filter(path => path.length > 0)
    .filter(path => !path.startsWith(".qmd-rsync-"));
}

export function detectConflicts(collection: string, downloadPaths: string[], uploadPaths: string[], timestamp: string): SyncConflict[] {
  const uploads = new Set(uploadPaths);
  return downloadPaths
    .filter(path => uploads.has(path))
    .map(path => ({
      collection,
      path,
      localConflictPath: `${path}.conflict.remote.${timestamp}`,
      remoteConflictPath: `${path}.conflict.local.${timestamp}`,
    }));
}

export function formatSyncSummary(summary: SyncSummary): string {
  const lines: string[] = [];
  lines.push("QMD Sync");
  lines.push("");
  lines.push(`Host: ${summary.host}`);
  lines.push(`Remote user: ${summary.remoteQmdUser}`);
  lines.push(`Local config: ${summary.localConfigPath}`);
  lines.push(`Remote config: ${summary.remoteConfigPath}`);
  lines.push(`Mode: ${summary.dryRun ? "dry-run" : "apply"}`);
  lines.push("");
  lines.push("Collections:");
  for (const plan of summary.collections) {
    lines.push(`  ${plan.name}: ${plan.direction}`);
    lines.push(`    local:  ${plan.localPath}${plan.localConfigured ? "" : " (mirror)"}`);
    lines.push(`    remote: ${plan.remotePath}${plan.remoteConfigured ? "" : " (mirror)"}`);
  }
  if (summary.warnings.length > 0 || summary.dependencies.warnings.length > 0) {
    lines.push("");
    lines.push("Warnings:");
    for (const warning of [...summary.dependencies.warnings, ...summary.warnings]) {
      lines.push(`  ${warning}`);
    }
  }
  lines.push("");
  lines.push("Rsync:");
  for (const result of summary.rsync) {
    const count = result.itemized.length;
    const suffix = result.skipped ? ` skipped (${result.reason})` : `${count} item(s)`;
    lines.push(`  ${result.label} ${result.phase} ${result.direction}: ${suffix}`);
  }
  if (summary.postSync.length > 0) {
    lines.push("");
    lines.push("Post-sync:");
    for (const result of summary.postSync) {
      const command = result.command.join(" ");
      if (result.skipped) {
        lines.push(`  ${result.side} ${result.action}: skipped (${result.reason})`);
      } else {
        lines.push(`  ${result.side} ${result.action}: exit ${result.exitCode ?? 0} (${command})`);
      }
    }
  }
  lines.push("");
  lines.push(`Conflicts: ${summary.conflicts.length}`);
  for (const conflict of summary.conflicts) {
    lines.push(`  ${conflict.collection}/${conflict.path}`);
    lines.push(`    local copy:  ${conflict.localConflictPath}`);
    lines.push(`    remote copy: ${conflict.remoteConflictPath}`);
  }
  lines.push("");
  lines.push("Next steps:");
  for (const step of summary.nextSteps) {
    lines.push(`  ${step}`);
  }
  return `${lines.join("\n")}\n`;
}

export async function runQmdSync(options: SyncOptions = {}): Promise<SyncSummary> {
  const opts = getDefaultSyncOptions(options);
  const localConfig = loadConfig();
  const localConfigPath = getConfigPath();
  const remoteConfigPath = `${opts.remoteHome}/${REMOTE_CONFIG_RELATIVE}`;
  const remoteCachePath = `${opts.remoteHome}/${REMOTE_CACHE_RELATIVE}`;
  const warnings: string[] = [];

  const localLockDir = getLocalLockDir();
  acquireLocalLock(localLockDir);
  let tempDir: string | undefined;
  try {
    const dependencies = await probeRemote(opts.runCommand, opts.host, opts.remoteQmdUser, opts.remoteHome);
    const remoteConfigRaw = await readRemoteConfig(opts.runCommand, opts.host, opts.remoteQmdUser, remoteConfigPath);
    const remoteConfig = parseConfigYaml(remoteConfigRaw, remoteConfigPath);
    const plans = buildCollectionPlans({
      localConfig,
      remoteConfig,
      host: opts.host,
      remoteHome: opts.remoteHome,
      collectionNames: opts.collection,
    });

    const missing = plans.filter(plan => !plan.localConfigured && !plan.remoteConfigured).map(plan => plan.name);
    for (const name of missing) {
      warnings.push(`collection not found locally or remotely: ${name}`);
    }

    const summary: SyncSummary = {
      dryRun: opts.dryRun,
      host: opts.host,
      remoteQmdUser: opts.remoteQmdUser,
      localConfigPath,
      remoteConfigPath,
      collections: plans,
      dependencies,
      rsync: [],
      conflicts: [],
      postSync: [],
      failed: false,
      warnings,
      nextSteps: [],
    };

    tempDir = await mkdtemp(join(tmpdir(), "qmd-sync-"));
    const excludeFrom = join(tempDir, "conflicts.exclude");

    if (!opts.dryRun) {
      for (const plan of plans) {
        mkdirSync(plan.localPath, { recursive: true });
        mkdirSync(join(plan.localPath, ".qmd-rsync-tmp"), { recursive: true });
      }
      mkdirSync(join(dirname(localConfigPath), ".qmd-rsync-tmp"), { recursive: true });
      await ensureRemoteDirs(opts.runCommand, opts.host, opts.remoteQmdUser, [
        dirname(remoteConfigPath),
        remoteCachePath,
        ...plans.map(plan => plan.remotePath),
        `${dirname(remoteConfigPath)}/.qmd-rsync-tmp`,
        ...plans.map(plan => `${plan.remotePath}/.qmd-rsync-tmp`),
      ]);
      await runRemoteLockProbe(opts.runCommand, opts.host, opts.remoteQmdUser, `${remoteCachePath}/sync.lock`);
    }

    const configPlan: SyncCollectionPlan = {
      name: "config",
      direction: "bidirectional",
      localPath: dirname(localConfigPath),
      remotePath: dirname(remoteConfigPath),
      pattern: undefined,
      localConfigured: true,
      remoteConfigured: true,
    };
    const allPlans = [
      ...(opts.collection?.length ? [] : [configPlan]),
      ...plans.filter(plan => plan.localConfigured || plan.remoteConfigured),
    ];

    for (const plan of allPlans) {
      const preflight = await dryRunPair(opts, plan);
      summary.rsync.push(...preflight.results);
      const conflicts = detectConflicts(plan.name, preflight.downloadPaths, preflight.uploadPaths, timestampForConflict());
      summary.conflicts.push(...conflicts);
      await writeFile(excludeFrom, conflicts.map(c => c.path).join("\n"));

      if (!opts.dryRun && conflicts.length > 0) {
        await syncConflictCopies(opts, plan, conflicts);
      }
      if (!opts.dryRun) {
        const applyResults = await applyPair(opts, plan, excludeFrom);
        summary.rsync.push(...applyResults);
      }
    }

    const failedApply = summary.rsync.some(result => result.phase === "apply" && result.skipped);
    if (failedApply) {
      summary.failed = true;
      summary.postSync.push(...plannedPostSync(opts, "sync failed; update/embed not run"));
    } else {
      summary.postSync.push(...await runPostSync(opts));
      if (summary.postSync.some(result => !result.skipped && (result.exitCode ?? 0) !== 0)) {
        summary.failed = true;
      }
    }

    summary.nextSteps = buildNextSteps(opts, summary);
    return summary;
  } finally {
    releaseLocalLock(localLockDir);
    if (tempDir) await rm(tempDir, { recursive: true, force: true });
  }
}

async function dryRunPair(opts: ReturnType<typeof getDefaultSyncOptions>, plan: SyncCollectionPlan): Promise<{
  results: SyncRsyncResult[];
  downloadPaths: string[];
  uploadPaths: string[];
}> {
  const download = await runRsync(opts, {
    label: plan.name,
    phase: "preflight",
    direction: "download",
    source: `${opts.host}:${plan.remotePath}`,
    destination: plan.localPath,
    dryRun: true,
    pattern: plan.pattern,
  });
  const upload = await runRsync(opts, {
    label: plan.name,
    phase: "preflight",
    direction: "upload",
    source: plan.localPath,
    destination: `${opts.host}:${plan.remotePath}`,
    dryRun: true,
    pattern: plan.pattern,
  });
  return {
    results: [download, upload],
    downloadPaths: download.itemized,
    uploadPaths: upload.itemized,
  };
}

async function applyPair(opts: ReturnType<typeof getDefaultSyncOptions>, plan: SyncCollectionPlan, excludeFrom: string): Promise<SyncRsyncResult[]> {
  const download = await runRsync(opts, {
    label: plan.name,
    phase: "apply",
    direction: "download",
    source: `${opts.host}:${plan.remotePath}`,
    destination: plan.localPath,
    dryRun: false,
    excludeFrom,
    pattern: plan.pattern,
  });
  const upload = await runRsync(opts, {
    label: plan.name,
    phase: "apply",
    direction: "upload",
    source: plan.localPath,
    destination: `${opts.host}:${plan.remotePath}`,
    dryRun: false,
    excludeFrom,
    pattern: plan.pattern,
  });
  return [download, upload];
}

async function runRsync(opts: ReturnType<typeof getDefaultSyncOptions>, params: {
  label: string;
  phase: "preflight" | "apply";
  direction: "download" | "upload";
  source: string;
  destination: string;
  dryRun: boolean;
  excludeFrom?: string;
  pattern?: string;
}): Promise<SyncRsyncResult> {
  const args = buildRsyncArgs({
    source: params.source,
    destination: params.destination,
    remoteQmdUser: opts.remoteQmdUser,
    dryRun: params.dryRun,
    delete: opts.delete,
    excludeFrom: params.excludeFrom,
    tempDir: params.direction === "download"
      ? `${params.destination.replace(/\/$/, "")}/.qmd-rsync-tmp`
      : `${stripRemotePrefix(params.destination).replace(/\/$/, "")}/.qmd-rsync-tmp`,
    includes: includePatternsForCollection(params.label === "config" ? undefined : params.pattern),
  });
  if (isMissingLocalSource(params.source)) {
    return {
      label: params.label,
      phase: params.phase,
      direction: params.direction,
      source: params.source,
      destination: params.destination,
      itemized: [],
      skipped: true,
      reason: `local source does not exist: ${params.source}`,
    };
  }
  const result = await opts.runCommand("rsync", args);
  if (result.exitCode !== 0) {
    return {
      label: params.label,
      phase: params.phase,
      direction: params.direction,
      source: params.source,
      destination: params.destination,
      itemized: [],
      skipped: true,
      reason: (result.stderr || result.stdout || `rsync exited ${result.exitCode}`).trim(),
    };
  }
  return {
    label: params.label,
    phase: params.phase,
    direction: params.direction,
    source: params.source,
    destination: params.destination,
    itemized: parseRsyncItemized(result.stdout),
    skipped: false,
  };
}

async function syncConflictCopies(opts: ReturnType<typeof getDefaultSyncOptions>, plan: SyncCollectionPlan, conflicts: SyncConflict[]): Promise<void> {
  for (const conflict of conflicts) {
    const localSource = join(plan.localPath, conflict.path);
    const localConflictDestination = join(plan.localPath, conflict.localConflictPath);
    const remoteSource = `${opts.host}:${plan.remotePath}/${conflict.path}`;
    const remoteConflictDestination = `${opts.host}:${plan.remotePath}/${conflict.remoteConflictPath}`;
    mkdirSync(dirname(localConflictDestination), { recursive: true });
    mkdirSync(join(dirname(localConflictDestination), ".qmd-rsync-tmp"), { recursive: true });
    await opts.runCommand("rsync", buildRsyncArgs({
      source: remoteSource,
      destination: localConflictDestination,
      remoteQmdUser: opts.remoteQmdUser,
      dryRun: false,
      delete: false,
      preserveFilePath: true,
      tempDir: `${dirname(localConflictDestination)}/.qmd-rsync-tmp`,
    }));
    await ensureRemoteDirs(opts.runCommand, opts.host, opts.remoteQmdUser, [remoteDirname(`${plan.remotePath}/${conflict.remoteConflictPath}`)]);
    await ensureRemoteDirs(opts.runCommand, opts.host, opts.remoteQmdUser, [`${remoteDirname(`${plan.remotePath}/${conflict.remoteConflictPath}`)}/.qmd-rsync-tmp`]);
    await opts.runCommand("rsync", buildRsyncArgs({
      source: localSource,
      destination: remoteConflictDestination,
      remoteQmdUser: opts.remoteQmdUser,
      dryRun: false,
      delete: false,
      preserveFilePath: true,
      tempDir: `${remoteDirname(`${plan.remotePath}/${conflict.remoteConflictPath}`)}/.qmd-rsync-tmp`,
    }));
  }
}

export function buildPostSyncCommands(opts: ReturnType<typeof getDefaultSyncOptions>): PostSyncResult[] {
  if (!opts.update) {
    if (opts.embed) {
      return [{
        side: "local",
        action: "embed",
        command: [...opts.localQmdCommand, "embed"],
        skipped: true,
        reason: "--embed requires --update",
      }];
    }
    return [];
  }

  const localUpdate = [...opts.localQmdCommand, "update"];
  const remoteUpdateCommand = "qmd update";
  const results: PostSyncResult[] = [
    {
      side: "local",
      action: "update",
      command: localUpdate,
      skipped: opts.dryRun,
      ...(opts.dryRun ? { reason: "dry-run" } : {}),
    },
    {
      side: "remote",
      action: "update",
      command: ["ssh", opts.host, remoteUserCommand(opts.remoteQmdUser, remoteUpdateCommand)],
      skipped: opts.dryRun,
      ...(opts.dryRun ? { reason: "dry-run" } : {}),
    },
  ];

  if (opts.embed) {
    results.push(
      {
        side: "local",
        action: "embed",
        command: [...opts.localQmdCommand, "embed"],
        skipped: opts.dryRun,
        ...(opts.dryRun ? { reason: "dry-run" } : {}),
      },
      {
        side: "remote",
        action: "embed",
        command: ["ssh", opts.host, remoteUserCommand(opts.remoteQmdUser, "qmd embed")],
        skipped: opts.dryRun,
        ...(opts.dryRun ? { reason: "dry-run" } : {}),
      },
    );
  }

  return results;
}

function plannedPostSync(opts: ReturnType<typeof getDefaultSyncOptions>, reason: string): PostSyncResult[] {
  return buildPostSyncCommands(opts).map(result => ({
    ...result,
    skipped: true,
    reason: result.reason || reason,
  }));
}

async function runPostSync(opts: ReturnType<typeof getDefaultSyncOptions>): Promise<PostSyncResult[]> {
  const planned = buildPostSyncCommands(opts);
  const results: PostSyncResult[] = [];

  for (const step of planned) {
    if (step.skipped) {
      results.push(step);
      continue;
    }
    const [command, ...args] = step.command;
    if (!command) {
      results.push({ ...step, skipped: true, reason: "empty command" });
      continue;
    }
    const result = await opts.runCommand(command, args);
    const completed: PostSyncResult = {
      ...step,
      exitCode: result.exitCode,
      stdout: truncateOutput(result.stdout),
      stderr: truncateOutput(result.stderr),
    };
    results.push(completed);
    if (result.exitCode !== 0) {
      break;
    }
  }

  if (results.length < planned.length) {
    for (const step of planned.slice(results.length)) {
      results.push({ ...step, skipped: true, reason: "previous post-sync step failed" });
    }
  }

  return results;
}

function buildNextSteps(opts: ReturnType<typeof getDefaultSyncOptions>, summary: SyncSummary): string[] {
  if (summary.failed) {
    return ["Review failed rsync or post-sync steps before rerunning qmd sync."];
  }
  if (opts.dryRun && opts.update) {
    return ["Dry-run only; run qmd sync --update without --dry-run to refresh indexes on both sides."];
  }
  if (opts.update && opts.embed) {
    return ["Indexes and embeddings were refreshed on both sides."];
  }
  if (opts.update) {
    return ["Indexes were refreshed on both sides.", "Run qmd embed manually when vector embeddings should be refreshed."];
  }
  return [
    "Run qmd update manually on each side after reviewing synced files.",
    "Run qmd embed manually when vector embeddings should be refreshed.",
  ];
}

function truncateOutput(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length <= 4000) return trimmed;
  return `${trimmed.slice(0, 4000)}\n... truncated ...`;
}

async function probeRemote(runCommand: CommandRunner, host: string, remoteQmdUser: string, remoteHome: string): Promise<SyncDependencyStatus> {
  const command = [
    "set -eu",
    "command -v rsync >/dev/null 2>&1 && echo rsync=1 || echo rsync=0",
    "command -v flock >/dev/null 2>&1 && echo flock=1 || echo flock=0",
    "command -v qmd >/dev/null 2>&1 && qmd --version 2>/dev/null || true",
    `test -d ${shellQuote(remoteHome)} || echo missing_home=1`,
  ].join("; ");
  const result = await runCommand("ssh", [host, remoteUserCommand(remoteQmdUser, command)]);
  if (result.exitCode !== 0) {
    throw new Error(`Remote probe failed: ${(result.stderr || result.stdout).trim()}`);
  }
  const lines = result.stdout.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const warnings: string[] = [];
  const rsync = lines.includes("rsync=1");
  const flock = lines.includes("flock=1");
  const qmdVersion = lines.find(line => line.startsWith("qmd "));
  if (!rsync) warnings.push("remote rsync is missing");
  if (!flock) warnings.push("remote flock is missing");
  if (!qmdVersion) warnings.push("remote qmd version could not be detected");
  if (lines.includes("missing_home=1")) warnings.push(`remote home does not exist: ${remoteHome}`);
  return { qmdVersion, rsync, flock, warnings };
}

async function readRemoteConfig(runCommand: CommandRunner, host: string, remoteQmdUser: string, remoteConfigPath: string): Promise<string> {
  const command = `test -f ${shellQuote(remoteConfigPath)} && cat ${shellQuote(remoteConfigPath)} || printf 'collections: {}\\n'`;
  const result = await runCommand("ssh", [host, remoteUserCommand(remoteQmdUser, command)]);
  if (result.exitCode !== 0) {
    throw new Error(`Remote config read failed: ${(result.stderr || result.stdout).trim()}`);
  }
  return result.stdout;
}

async function ensureRemoteDirs(runCommand: CommandRunner, host: string, remoteQmdUser: string, paths: string[]): Promise<void> {
  const unique = Array.from(new Set(paths.filter(Boolean)));
  if (unique.length === 0) return;
  const command = `mkdir -p ${unique.map(shellQuote).join(" ")}`;
  const result = await runCommand("ssh", [host, remoteUserCommand(remoteQmdUser, command)]);
  if (result.exitCode !== 0) {
    throw new Error(`Remote directory creation failed: ${(result.stderr || result.stdout).trim()}`);
  }
}

async function runRemoteLockProbe(runCommand: CommandRunner, host: string, remoteQmdUser: string, lockPath: string): Promise<void> {
  const command = `mkdir -p ${shellQuote(dirname(lockPath))} && (flock -n 9 || exit 75) 9>${shellQuote(lockPath)}`;
  const result = await runCommand("ssh", [host, remoteUserCommand(remoteQmdUser, command)]);
  if (result.exitCode !== 0) {
    throw new Error(`Remote sync lock is busy or unavailable: ${(result.stderr || result.stdout).trim()}`);
  }
}

function getLocalLockDir(): string {
  const cacheRoot = process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
  return join(cacheRoot, "qmd", "sync.lock.d");
}

function acquireLocalLock(lockDir: string): void {
  mkdirSync(dirname(lockDir), { recursive: true });
  if (existsSync(lockDir)) {
    throw new Error(`Local sync lock is busy: ${lockDir}`);
  }
  mkdirSync(lockDir);
  writeFileSync(join(lockDir, "owner"), `${process.pid}\n`);
}

function releaseLocalLock(lockDir: string): void {
  rmSync(lockDir, { recursive: true, force: true });
}

function defaultRunCommand(command: string, args: string[], options?: { cwd?: string }): Promise<CommandResult> {
  return new Promise((resolve) => {
    execFile(command, args, { cwd: options?.cwd, maxBuffer: 50 * 1024 * 1024 }, (error, stdout, stderr) => {
      const code = typeof (error as NodeJS.ErrnoException | null)?.code === "number"
        ? Number((error as NodeJS.ErrnoException).code)
        : error
          ? 1
          : 0;
      resolve({ stdout: String(stdout || ""), stderr: String(stderr || ""), exitCode: code });
    });
  });
}

function ensureTrailingSlash(path: string): string {
  return path.endsWith("/") ? path : `${path}/`;
}

function formatRsyncEndpoint(endpoint: string): string {
  const colon = endpoint.indexOf(":");
  if (colon <= 0) return endpoint;
  const host = endpoint.slice(0, colon);
  const path = endpoint.slice(colon + 1);
  if (!path || path.startsWith("'")) return endpoint;
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(path)) return endpoint;
  return `${host}:${shellQuote(path)}`;
}

function timestampForConflict(): string {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\..+$/, "Z");
}

function remoteDirname(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx <= 0 ? "/" : path.slice(0, idx);
}

function stripRemotePrefix(path: string): string {
  const idx = path.indexOf(":");
  return idx >= 0 ? path.slice(idx + 1) : path;
}

function isMissingLocalSource(path: string): boolean {
  if (path.includes(":")) return false;
  return !existsSync(path);
}
