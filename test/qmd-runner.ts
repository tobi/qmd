/** Select the source CLI with the runtime used by the native test suite. */
import { fileURLToPath } from "node:url";

export function qmdRunnerArgs(args: readonly string[]): { command: string; args: string[] } {
  const script = fileURLToPath(new URL("../src/cli/qmd.ts", import.meta.url));
  const entry = typeof process.versions.bun === "string"
    ? [script]
    : [fileURLToPath(new URL("../node_modules/tsx/dist/cli.mjs", import.meta.url)), script];
  return { command: process.execPath, args: [...entry, ...args] };
}
