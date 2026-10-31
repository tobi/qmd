/**
 * Child helper for embed-lock cross-process tests (#825).
 * Args: <lockPath> <holdMs> [ignored qmd tokens...]
 */
import { tryAcquireEmbedLock } from "../../src/cli/embed-lock.ts";

async function main(): Promise<void> {
  const lockPath = process.argv[2];
  const holdMs = Number(process.argv[3] ?? "1000");
  if (!lockPath || !Number.isFinite(holdMs)) {
    console.error("usage: embed-lock-holder.ts <lockPath> <holdMs>");
    process.exit(2);
  }

  const handle = tryAcquireEmbedLock(lockPath);
  if (!handle) {
    console.error("child failed to acquire");
    process.exit(2);
  }

  process.stdout.write("HOLD\n");
  await new Promise((r) => setTimeout(r, holdMs));
  handle.release();
  process.stdout.write("RELEASED\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
