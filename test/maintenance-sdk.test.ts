import { expect, test } from "vitest";
import * as sdk from "../src/index.js";

test("the public SDK exports opt-in maintenance without opening a store or starting a timer", async () => {
  expect(sdk).toHaveProperty("createMaintenanceJobs");
  expect(sdk).toHaveProperty("createEmbeddingScheduler");
  const jobs = sdk.createMaintenanceJobs();
  const owner = sdk.createEmbeddingScheduler({ maintenance: jobs,
    collections: () => [], pending: () => 0, activeQueries: () => 0,
    embed: async () => ({ chunksEmbedded: 0, errors: 0 }),
  });
  expect(owner.health().enabled).toBe(false);
  expect(owner.tick()).toBeNull();
  expect(jobs.activeJobId).toBeNull();
  await owner.stop();
  await jobs.stop();
});
