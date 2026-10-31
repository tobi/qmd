/**
 * metadata-filter.test-d.ts - Compile-time shape of the predicate grammar.
 * MetadataFilter and MetadataMatch share one recursive grammar and differ in
 * the conditions they admit. Checked by vitest's typecheck pass.
 */

import { describe, test, expectTypeOf } from "vitest";
import {
  compileMetadataMatch,
  parseMetadataFilter,
  parseMetadataMatch,
  type MetadataCondition,
  type MetadataEntryCondition,
  type MetadataFilter,
  type MetadataMatch,
  type MetadataPredicate,
} from "../src/metadata-filter.js";
import type { ListMetadataOptions } from "../src/metadata-store.js";

describe("MetadataPredicate", () => {
  test("filter and match are the one grammar over different conditions", () => {
    expectTypeOf<MetadataFilter>().toEqualTypeOf<MetadataPredicate<MetadataCondition>>();
    expectTypeOf<MetadataMatch>().toEqualTypeOf<MetadataPredicate<MetadataEntryCondition>>();
    expectTypeOf(parseMetadataFilter).returns.toEqualTypeOf<MetadataFilter>();
    expectTypeOf(parseMetadataMatch).returns.toEqualTypeOf<MetadataMatch>();
    expectTypeOf(compileMetadataMatch).parameter(0).toEqualTypeOf<MetadataMatch>();
    expectTypeOf<ListMetadataOptions["match"]>().toEqualTypeOf<MetadataMatch | undefined>();
  });

  test("a match composes with and, or, and not like a filter", () => {
    const match: MetadataMatch = {
      operator: "and",
      operands: [
        { field: "key", operator: "prefix", value: "mem-" },
        { operator: "not", operand: { field: "value", operator: "type", value: "boolean" } },
        { operator: "or", operands: [
          { field: "value", operator: "gte", value: 3 },
          { field: "value", operator: "in", value: ["a", "b"], caseInsensitive: true },
        ] },
      ],
    };
    expectTypeOf(match).toMatchTypeOf<MetadataMatch>();
  });

  test("a match names only the fields of a metadata entry", () => {
    // @ts-expect-error a document's metadata key is not a field of an entry
    const byMetadataKey: MetadataMatch = { field: "topics", operator: "eq", value: "typescript" };
    // @ts-expect-error nested in a group as well
    const nested: MetadataMatch = { operator: "and", operands: [{ field: "topics", operator: "eq", value: "x" }] };
    expectTypeOf(byMetadataKey).toEqualTypeOf<MetadataMatch>();
    expectTypeOf(nested).toEqualTypeOf<MetadataMatch>();
  });

  test("a match admits no condition about the set of values a document holds", () => {
    // @ts-expect-error `exists` has no meaning for a single entry
    const exists: MetadataMatch = { field: "value", operator: "exists", value: true };
    // @ts-expect-error `all` has no meaning for a single entry
    const all: MetadataMatch = { field: "value", operator: "all", value: ["a"] };
    expectTypeOf(exists).toEqualTypeOf<MetadataMatch>();
    expectTypeOf(all).toEqualTypeOf<MetadataMatch>();
  });

  test("a filter still admits every document condition", () => {
    const filter: MetadataFilter = {
      operator: "and",
      operands: [
        { field: "topics", operator: "all", value: ["typescript", "sqlite"] },
        { field: "reviewed", operator: "exists", value: true },
        { field: "key", operator: "eq", value: "a metadata key may be named key" },
      ],
    };
    expectTypeOf(filter).toMatchTypeOf<MetadataFilter>();
  });
});
