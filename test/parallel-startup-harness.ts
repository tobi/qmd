import { createStore } from "../src/store.js";

const dbPath = process.argv[2];

if (!dbPath) {
  console.error("Usage: bun test/parallel-startup-harness.ts <dbPath>");
  process.exit(1);
}

let store: ReturnType<typeof createStore> | undefined;

try {
  store = createStore(dbPath);
  store.getStatus();
  console.log("startup-ok");
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
} finally {
  store?.close();
}
