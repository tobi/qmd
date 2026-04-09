import { spawn, type ChildProcess } from "node:child_process";

export interface CSharpSidecarOptions {
  command?: string;
  timeoutMs?: number;
}

export interface CSharpSidecarResult {
  breakpoints: { pos: number; score: number; type: string }[];
  symbols: {
    name: string;
    kind: string;
    line: number;
    containerName?: string;
    signature?: string;
    modifiers?: string[];
  }[];
}

type RawSidecarResponse = {
  version?: unknown;
  breakpoints?: unknown;
  symbols?: unknown;
};

function parseSidecarResponse(value: unknown): CSharpSidecarResult | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const response = value as RawSidecarResponse;
  if (response.version !== 1 || !Array.isArray(response.breakpoints) || !Array.isArray(response.symbols)) {
    return null;
  }

  const breakpoints = response.breakpoints.filter(isBreakpoint);
  const symbols = response.symbols.filter(isSymbol);

  if (breakpoints.length !== response.breakpoints.length || symbols.length !== response.symbols.length) {
    return null;
  }

  return { breakpoints, symbols };
}

async function terminateChildProcess(child: ChildProcess): Promise<void> {
  const pid = child.pid;
  if (!pid) {
    return;
  }

  if (process.platform === "win32") {
    await new Promise<void>((resolve) => {
      const killer = spawn("taskkill", ["/pid", String(pid), "/t", "/f"], {
        stdio: "ignore",
        windowsHide: true,
      });

      killer.on("error", () => {
        resolve();
      });
      killer.on("close", () => {
        resolve();
      });
    });
    return;
  }

  try {
    process.kill(-pid, "SIGKILL");
  } catch {
    try {
      child.kill("SIGKILL");
    } catch {
      // Ignore cleanup failures; the wrapper still degrades to null.
    }
  }
}

function isBreakpoint(value: unknown): value is CSharpSidecarResult["breakpoints"][number] {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Record<string, unknown>;
  return typeof candidate.pos === "number"
    && typeof candidate.score === "number"
    && typeof candidate.type === "string";
}

function isSymbol(value: unknown): value is CSharpSidecarResult["symbols"][number] {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Record<string, unknown>;
  if (typeof candidate.name !== "string" || typeof candidate.kind !== "string" || typeof candidate.line !== "number") {
    return false;
  }
  if (candidate.containerName !== undefined && typeof candidate.containerName !== "string") {
    return false;
  }
  if (candidate.signature !== undefined && typeof candidate.signature !== "string") {
    return false;
  }
  if (candidate.modifiers !== undefined && (!Array.isArray(candidate.modifiers) || candidate.modifiers.some(item => typeof item !== "string"))) {
    return false;
  }

  return true;
}

export async function callCSharpSidecar(
  filePath: string,
  content: string,
  options: CSharpSidecarOptions = {},
): Promise<CSharpSidecarResult | null> {
  const command = options.command ?? process.env.QMD_CSHARP_SIDECAR;
  if (!command) {
    return null;
  }

  const timeoutMs = options.timeoutMs ?? 1500;
  const requestPayload = JSON.stringify({
    version: 1,
    language: "csharp",
    filePath,
    content,
    features: {
      breakpoints: true,
      symbols: true,
    },
  });

  return await new Promise<CSharpSidecarResult | null>((resolve) => {
    let settled = false;
    let stdout = "";

    const finish = (result: CSharpSidecarResult | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const child = spawn(command, [], {
      detached: process.platform !== "win32",
      shell: true,
      stdio: "pipe",
      windowsHide: true,
    });

    const timer = setTimeout(() => {
      void terminateChildProcess(child).finally(() => {
        finish(null);
      });
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.on("error", () => {
      finish(null);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        finish(null);
        return;
      }

      let response: unknown;
      try {
        response = JSON.parse(stdout) as unknown;
      } catch {
        finish(null);
        return;
      }

      const parsedResponse = parseSidecarResponse(response);
      if (!parsedResponse) {
        finish(null);
        return;
      }

      finish(parsedResponse);
    });

    child.stdin.on("error", () => {
      finish(null);
    });

    child.stdin.end(requestPayload);
  });
}
