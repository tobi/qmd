import { expect, test } from "vitest";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { LlamaCpp, withLLMSessionForLlm } from "../src/llm.js";

const project = fileURLToPath(new URL("..", import.meta.url));
const script = join(project, "src", "cli", "qmd.ts");
const args = "Bun" in globalThis ? [script] : [join(project, "node_modules", "tsx", "dist", "cli.mjs"), script];

function run(root: string, command: string[]) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(process.execPath, [...args, ...command], {
      cwd: root,
      env: {
        PATH: process.env.PATH,
        HOME: root,
        XDG_CONFIG_HOME: join(root, "config"),
        XDG_CACHE_HOME: join(root, "cache"),
        QMD_CONFIG_DIR: join(root, "config"),
        INDEX_PATH: join(root, "index.sqlite"),
        CI: "true",
        NO_COLOR: "1",
        QMD_DOCTOR_DEVICE_PROBE: "0",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", data => { stdout += data; });
    child.stderr.on("data", data => { stderr += data; });
    child.on("error", reject);
    child.on("close", code => resolve({ code, stdout, stderr }));
  });
}

test("CLI update/embed/typed vector query uses explicit loopback embeddings; status/doctor do not probe", async () => {
  const root = mkdtempSync(join(tmpdir(), "qmd-remote-cli-"));
  let requests = 0;
  const unexpected: string[] = [];
  const server = createServer(async (request, response) => {
    requests++;
    if (request.url !== "/v1/embeddings" || request.method !== "POST" || request.headers.authorization !== undefined) {
      unexpected.push(`${request.method} ${request.url}`);
      response.writeHead(400).end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const body = JSON.parse(Buffer.concat(chunks).toString());
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify({ model: body.model, data: body.input.map((_text: string, index: number) => ({ index, embedding: [1, 0] })) }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No loopback port");
  mkdirSync(join(root, "docs"));
  mkdirSync(join(root, "config"));
  writeFileSync(join(root, "docs", "stars.md"), "# Nebula\nA cloud of distant stars.\n");
  writeFileSync(join(root, "config", "index.yml"), `collections:\n  docs:\n    path: ${JSON.stringify(join(root, "docs"))}\n    pattern: '**/*.md'\nmodels:\n  embed: 'http://127.0.0.1:${address.port}/v1#synthetic'\n  embedOptions:\n    dimensions: 2\n    revision: test-v1\n`);
  try {
    const update = await run(root, ["update"]);
    expect(update.code, update.stderr).toBe(0);
    const embed = await run(root, ["embed"]);
    expect(embed.code, embed.stderr).toBe(0);
    expect(embed.stdout).toContain("Embedded");
    const result = await run(root, ["query", "vec: unmentionedcosmos", "--no-rerank", "--json"]);
    expect(result.code, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)[0]?.title).toBe("Nebula");
    const before = requests;
    const status = await run(root, ["status"]);
    expect(status.code, status.stderr).toBe(0);
    expect(status.stdout).toContain("configured remote (not probed; no local download)");
    const doctor = await run(root, ["doctor"]);
    expect(doctor.code, doctor.stderr).toBe(0);
    expect(doctor.stdout).toContain("configured remote (not probed; no local download)");
    expect(requests).toBe(before);
    expect(unexpected).toEqual([]);
    expect(embed.stderr + result.stderr + doctor.stderr).not.toMatch(/Loading model|Downloading|LLM operations are disabled/);
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    rmSync(root, { recursive: true, force: true });
  }
});

test("native fetch aborts a held HTTP request when its session is cancelled", async () => {
  let received!: () => void;
  const started = new Promise<void>(resolve => { received = resolve; });
  let disconnected!: () => void;
  const closed = new Promise<void>(resolve => { disconnected = resolve; });
  const server = createServer((request, response) => {
    request.resume();
    response.on("close", disconnected);
    received(); // Hold response open until fetch aborts; no sleeps.
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No loopback port");
  const llm = new LlamaCpp({ embedModel: `http://127.0.0.1:${address.port}/v1#synthetic` });
  const controller = new AbortController();
  try {
    const work = withLLMSessionForLlm(llm, session => session.embed("held"), { signal: controller.signal });
    const settled = work.then(() => null, error => error);
    await started;
    controller.abort();
    const error = await settled;
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe("Remote model request aborted");
    await closed;
  } finally {
    await llm.dispose();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
