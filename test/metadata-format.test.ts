/**
 * Metadata discovery rendering: the plain-text shape the CLI and the MCP
 * `metadata` tool share, exercised directly where the CLI fixture cannot
 * reach (string identity, control characters, empty states under a filter).
 */

import * as childProcess from "node:child_process";

import { describe, test, expect } from "vitest";
import { formatMetadataKeySummaries, formatMetadataOverview, type FormatMetadataOptions } from "../src/metadata-format.js";
import type { ListMetadataResult, MetadataKeyTypeSummary } from "../src/metadata-store.js";
import type { MetadataScalar } from "../src/metadata.js";

const OPTIONS: FormatMetadataOptions = {
  valueWindowHint: "--value-limit <n>",
  keyWindowHint: "--key-limit <n>",
  keyOffset: 0,
  keyOffsetLabel: "--key-offset",
  emptyMessage: "No metadata matches.",
};

function stringTypeSummary(values: string[], remainingValues = 0): MetadataKeyTypeSummary {
  return {
    type: "string",
    multiValued: false,
    documents: values.length,
    distinctValues: values.length + remainingValues,
    values: values.map(value => ({ value, documents: 1 })),
    remainingValues,
    collections: ["notes"],
  };
}

function stringResult(values: string[]): ListMetadataResult {
  return { documents: values.length, totalKeys: 1, keys: [{ key: "label", documents: values.length, types: [stringTypeSummary(values)] }], remainingKeys: 0 };
}

/** A key held as a string by some documents and a number by one, which prints the compact `value (count)` list. */
function splitResult(values: string[], remainingValues = 0): ListMetadataResult {
  const numberSummary: MetadataKeyTypeSummary = {
    type: "number", multiValued: false, documents: 1, distinctValues: 1,
    values: [{ value: 7, documents: 1 }], remainingValues: 0, range: { min: 7, median: 7, max: 7 }, collections: ["notes"],
  };
  const documents = values.length + 1;
  return { documents, totalKeys: 1, keys: [{ key: "label", documents, types: [stringTypeSummary(values, remainingValues), numberSummary] }], remainingKeys: 0 };
}

/**
 * The items of the compact list on the string row of a type split, read back
 * the way a reader must: a quoted item is a JSON string (so it may contain the
 * delimiters), a bare item runs to its count, and the tail closes the list.
 */
function compactItems(output: string): { value: MetadataScalar; documents: number }[] {
  const row = output.split("\n").find(line => line.startsWith("  string  "))!;
  const list = row.replace(/^ {2}string {2}\d+ docs? {2}/u, "");
  const items: { value: MetadataScalar; documents: number }[] = [];
  let rest = list;
  while (rest !== "") {
    const tail = /^(\d+) more$/u.exec(rest);
    if (tail) {
      items.push({ value: `${tail[1]} more`, documents: -1 });
      break;
    }
    const item = rest.startsWith('"')
      ? /^("(?:[^"\\]|\\.)*") \((\d+)\)(?:, |$)/u.exec(rest)!
      : /^([^,()]*) \((\d+)\)(?:, |$)/u.exec(rest)!;
    const printed = item[1]!;
    items.push({ value: printed.startsWith('"') ? JSON.parse(printed) as MetadataScalar : printed, documents: Number(item[2]) });
    rest = rest.slice(item[0].length);
  }
  return items;
}

/** The printed value column of each row: everything before the two-space gap and the count. */
function printedValues(output: string): string[] {
  return output.split("\n").slice(1).filter(line => line !== "").map(line => line.slice(2).replace(/ {2,}\d+$/, ""));
}

test.skipIf(process.platform === "win32")("a type-conflict hint preserves apostrophes and shell syntax in the metadata key", () => {
  const result = splitResult(["a", "b"]);
  const key = "owner'$(printf injected)'s";
  result.keys[0]!.key = key;
  const output = formatMetadataOverview(result, { documentsWithMetadata: result.documents, pendingMetadata: 0, drillDownHint: "qmd collection metadata notes" });
  const command = output.split("types disagree, see: ")[1]!;

  // Stub qmd to capture its arguments. The fixture's substitution is harmless
  // if quoting regresses, but it must remain literal data in the JSON operand.
  const argumentsText = childProcess.execFileSync("sh", ["-c", `qmd() { printf '%s\\n' "$@"; }\n${command}`], { encoding: "utf8" });
  const args = argumentsText.trimEnd().split("\n");
  expect(args.slice(0, 4)).toEqual(["collection", "metadata", "notes", "--match"]);
  expect(JSON.parse(args[4]!)).toEqual({ field: "key", operator: "eq", value: key });
});

describe("formatMetadataKeySummaries values", () => {
  test("prints a plain string bare", () => {
    expect(printedValues(formatMetadataKeySummaries(stringResult(["published", "docs team", "v1.2-rc", "é"]), OPTIONS)))
      .toEqual(["published", "docs team", "v1.2-rc", "é"]);
  });

  test("quotes every string whose bare form could be read as another value or as layout", () => {
    const ambiguous = ["", '""', " padded", "padded ", "two  spaces", "tab\tin", "line\nbreak", "42", "-1.5", "true", "null", "[1]", "back\\slash"];

    const printed = printedValues(formatMetadataKeySummaries(stringResult(ambiguous), OPTIONS));

    expect(printed.every(value => value.startsWith('"') && value.endsWith('"'))).toBe(true);
    expect(printed.map(value => JSON.parse(value) as MetadataScalar)).toEqual(ambiguous);
  });

  test("escapes every control character, including the ones JSON.stringify leaves literal", () => {
    const controls = ["bell\u0007", "del\u007f", "csi\u009b", "line\u2028sep", "para\u2029sep", "nul\u0000"];

    const output = formatMetadataKeySummaries(stringResult(controls), OPTIONS);

    expect(output).not.toMatch(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u2028\u2029]/u);
    expect(printedValues(output).map(value => JSON.parse(value) as MetadataScalar)).toEqual(controls);
  });

  test("quotes a string that the compact list's delimiters or tails would otherwise absorb", () => {
    // "a (1), b" and the three items a, b, c would print identically without quotes.
    const values = ["a (1), b", "c", "3 more", "...", "(x)", "plain value"];

    const output = formatMetadataKeySummaries(splitResult(values, 2), OPTIONS);

    expect(output.split("\n")[1]).toBe('  string  6 docs  "a (1), b" (1), c (1), "3 more" (1), "..." (1), "(x)" (1), plain value (1), 2 more');
    expect(compactItems(output)).toEqual([
      ...values.map(value => ({ value, documents: 1 })),
      { value: "2 more", documents: -1 },
    ]);
    // The same value prints the same way in the vertical layout.
    expect(printedValues(formatMetadataKeySummaries(stringResult(values), OPTIONS))).toEqual(['"a (1), b"', "c", '"3 more"', '"..."', '"(x)"', "plain value"]);
  });

  test("prints numbers and booleans bare", () => {
    const typeSummary: MetadataKeyTypeSummary = {
      type: "number", multiValued: false, documents: 2, distinctValues: 2,
      values: [{ value: 1.5, documents: 1 }, { value: -2, documents: 1 }],
      remainingValues: 0, range: { min: -2, median: -0.25, max: 1.5 }, collections: ["notes"],
    };
    const result: ListMetadataResult = { documents: 2, totalKeys: 1, keys: [{ key: "n", documents: 2, types: [typeSummary] }], remainingKeys: 0 };

    expect(formatMetadataKeySummaries(result, OPTIONS)).toBe("n  number  2 of 2 documents  2 distinct\n  min -2  median -0.25  max 1.5\n  -2 (1)  1.5 (1)");
  });
});

describe("formatMetadataKeySummaries empty states", () => {
  test("keeps the filter line when no entry matched", () => {
    const result: ListMetadataResult = { documents: 3, filteredDocuments: 1, totalKeys: 0, keys: [], remainingKeys: 0 };

    expect(formatMetadataKeySummaries(result, OPTIONS)).toBe("filter: 1 of 3 documents\n\nNo metadata matches.");
  });

  test("keeps the filter line when the key window falls past the end", () => {
    const result: ListMetadataResult = { documents: 3, filteredDocuments: 2, totalKeys: 4, keys: [], remainingKeys: 0 };

    expect(formatMetadataKeySummaries(result, { ...OPTIONS, keyOffset: 10 })).toBe("filter: 2 of 3 documents\n\nNo keys at --key-offset 10, 4 keys in total.");
  });
});
