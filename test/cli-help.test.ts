import { describe, expect, test } from "vitest";
import {
  HELP_SPECS,
  renderHelp,
  resolveHelpTopic,
  validateHelpOptionKeys,
} from "../src/cli/help.js";

describe("contextual CLI help registry", () => {
  test("every documented option exists in the parser option registry", () => {
    expect(validateHelpOptionKeys()).toEqual([]);
  });

  test("every registered topic renders its own usage", () => {
    for (const [topic, spec] of Object.entries(HELP_SPECS)) {
      const help = renderHelp(topic);
      expect(help, topic).toContain(`  ${spec.usage}`);
      expect(help, topic).toContain("-h, --help");
    }
  });

  test("resolves command aliases and nested command topics", () => {
    expect(resolveHelpTopic("deep-search")).toBe("query");
    expect(resolveHelpTopic("vector-search")).toBe("vsearch");
    expect(resolveHelpTopic("collection", ["add"])).toBe("collection add");
    expect(resolveHelpTopic("collection", ["rm"])).toBe("collection remove");
    expect(resolveHelpTopic("context", ["remove"])).toBe("context rm");
    expect(resolveHelpTopic("help", ["embed"])).toBe("embed");
  });

  test("keeps query grammar out of unrelated help", () => {
    expect(renderHelp("query")).toContain("Query grammar:");
    expect(renderHelp("query")).toContain("typed_line");
    expect(renderHelp("root")).not.toContain("Query grammar:");
    expect(renderHelp("embed")).not.toContain("typed_line");
  });

  test("documents the formerly missing command-specific flags", () => {
    expect(renderHelp("query")).toContain("--intent <text>");
    expect(renderHelp("get")).toContain("--from <line>");
  });
});
