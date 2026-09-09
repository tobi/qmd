/**
 * store-fts-separator-queries.test.ts - lex queries whose terms contain a
 * separator, bare and quoted.
 *
 * The FTS index uses `tokenize='porter unicode61'`, which splits on every
 * non-alphanumeric character, so a document containing `PIO-1384` is stored as
 * the adjacent tokens `pio` and `1384`. A query has to produce that same token
 * sequence to match.
 *
 * Two query paths reach the index and they were fixed separately:
 *
 *   - The bare-term path in buildFTS5Query grew isHyphenatedToken /
 *     sanitizeHyphenatedTerm (upstream #463) and isDottedToken /
 *     sanitizeDottedTerm (upstream #696), so `PIO-1384` and `2026.4.10` split
 *     into phrase terms.
 *   - The quoted-phrase path, sanitizeFTS5Phrase, was introduced later for CJK
 *     and inherited only sanitizeFTS5Term, which DELETES every separator.
 *     `"PIO-1384"` therefore became the single token `pio1384`, which no
 *     document holds, and the query returned zero rows with no error. Upstream
 *     #757 added dotted-token splitting to this path and nothing else, so dots
 *     worked and every other separator did not.
 *
 * These tests pin both paths to the same rule: a query term is split on any run
 * of characters the tokenizer treats as a separator, and the parts are matched
 * as an adjacent phrase. Underscore and case are covered because they bound the
 * diagnosis: `_` already worked (sanitizeFTS5Term preserves it and FTS5
 * re-tokenizes the phrase symmetrically) and must not regress, and matching is
 * case-insensitive on both paths.
 *
 * PIO-3237, upstream https://github.com/tobi/qmd/issues/916.
 *
 * Run with: bun test test/store-fts-separator-queries.test.ts
 *        or: pnpm test:node test/store-fts-separator-queries.test.ts
 */

import { describe, test, expect, beforeAll, afterAll, afterEach } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import {
  createStore,
  hashContent,
  insertContent,
  insertDocument,
  syncConfigToDb,
  validateSemanticQuery,
  type Store,
} from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

// =============================================================================
// Fixtures / helpers
// =============================================================================

/**
 * The PIO-3237 probe corpus. The title is the real title of
 * twin-github/pionizer/gateway/pr-1.md, the document the four probes in that
 * issue were measured against, so the separators under test are the ones the
 * corpus actually contains rather than invented ones.
 */
const PROBE_TITLE =
  "gateway #1: feat: flash-and-forget gateway scaffold, compose stack + cloud-init provisioning (PIO-1384)";

const PROBE_BODY = [
  "The flash-and-forget gateway scaffold lands the compose stack and the",
  "cloud-init provisioning path together, under PIO-1384 and DEC-0066.",
  "It touches src/lib/provision.ts and docs/SYNTAX.md, pins version 2026.4.10,",
  "calls apply_secrets and __init__, and reaches qmd://Vault/wiki over zero-SSH.",
  "Package @tobilu/qmd is a dependency.",
].join("\n");

let testDir: string;
let testConfigDir: string;
let currentStore: Store | null = null;

async function createProbeStore(): Promise<{ store: Store; collection: string }> {
  testConfigDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = testConfigDir;

  const collection = "probes";
  const config: CollectionConfig = {
    collections: { [collection]: { path: "/test/probes", pattern: "**/*.md" } },
  };
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(config));

  const store = createStore(
    join(testDir, `probe-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`),
  );
  currentStore = store;
  syncConfigToDb(store.db, config);

  const now = new Date().toISOString();
  const hash = await hashContent(PROBE_BODY);
  insertContent(store.db, hash, PROBE_BODY, now);
  insertDocument(store.db, collection, "gateway/pr-1.md", PROBE_TITLE, hash, now, now);

  return { store, collection };
}

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-fts-separators-"));
});

afterEach(async () => {
  currentStore?.close();
  currentStore = null;
  delete process.env.QMD_CONFIG_DIR;
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

// =============================================================================
// The four PIO-3237 probes
// =============================================================================

describe("PIO-3237 probes: quoted and hyphenated is the combination that failed", () => {
  /**
   * The table as measured on 2026-09-03 against the live twin-github index.
   * Rows 2, 3 and 4 already passed and are kept as the positive controls that
   * make row 1's zero a defect rather than an absent document: the same store,
   * the same document, one variable changed per row.
   */
  const probes: ReadonlyArray<{ probe: number; query: string; why: string }> = [
    { probe: 1, query: '"flash-and-forget"', why: "quoted and hyphenated, returned 0" },
    { probe: 2, query: "flash-and-forget", why: "hyphenated, unquoted, already worked" },
    {
      probe: 3,
      query: '"flash and forget gateway scaffold"',
      why: "quoted, no hyphen, already worked",
    },
    { probe: 4, query: '"gateway scaffold"', why: "quoted, no hyphen, already worked" },
  ];

  for (const { probe, query, why} of probes) {
    test(`probe ${probe}: lex ${query} finds the document (${why})`, async () => {
      const { store } = await createProbeStore();
      expect(store.searchFTS(query, 10)).toHaveLength(1);
    });
  }
});

// =============================================================================
// The separator class, not one separator
// =============================================================================

describe("a quoted term splits on every separator the tokenizer splits on", () => {
  const quoted: ReadonlyArray<{ query: string; separator: string }> = [
    { query: '"PIO-1384"', separator: "hyphen, identifier" },
    { query: '"DEC-0066"', separator: "hyphen, identifier" },
    { query: '"cloud-init"', separator: "hyphen, compound" },
    { query: '"zero-SSH"', separator: "hyphen, mixed case" },
    { query: '"2026.4.10"', separator: "dot, already fixed upstream (#757)" },
    { query: '"src/lib/provision.ts"', separator: "slash, upstream #916" },
    { query: '"docs/SYNTAX.md"', separator: "slash, upstream #916" },
    { query: '"qmd://Vault/wiki"', separator: "scheme punctuation, upstream #916" },
    { query: '"@tobilu/qmd"', separator: "at sign, upstream #916" },
  ];

  for (const { query, separator } of quoted) {
    test(`lex ${query} finds the document (${separator})`, async () => {
      const { store } = await createProbeStore();
      expect(store.searchFTS(query, 10)).toHaveLength(1);
    });
  }

  test("a bare term with the same separators still matches", async () => {
    const { store } = await createProbeStore();
    for (const query of ["PIO-1384", "cloud-init", "2026.4.10", "src/lib/provision.ts"]) {
      expect(store.searchFTS(query, 10), `bare ${query}`).toHaveLength(1);
    }
  });

  test("a quoted phrase spanning several separated terms matches in order", async () => {
    const { store } = await createProbeStore();
    expect(store.searchFTS('"flash-and-forget gateway scaffold"', 10)).toHaveLength(1);
  });

  test("a quoted phrase whose words are not adjacent does not match", async () => {
    const { store } = await createProbeStore();
    // Both tokens are in the document, in the other order and far apart, so a
    // phrase query must still reject it. Without this the tests above would
    // pass on a sanitizer that dropped separated terms entirely.
    expect(store.searchFTS('"scaffold flash-and-forget"', 10)).toHaveLength(0);
  });

  test("a quoted term absent from the document does not match", async () => {
    const { store } = await createProbeStore();
    expect(store.searchFTS('"PIO-9999"', 10)).toHaveLength(0);
    expect(store.searchFTS('"never-indexed-term"', 10)).toHaveLength(0);
  });
});

// =============================================================================
// Underscores and case, which bound the diagnosis
// =============================================================================

describe("underscores and case behave the same on both paths", () => {
  test("underscored identifiers match bare and quoted", async () => {
    const { store } = await createProbeStore();
    for (const query of ["apply_secrets", '"apply_secrets"', "__init__", '"__init__"']) {
      expect(store.searchFTS(query, 10), query).toHaveLength(1);
    }
  });

  test("matching is case insensitive on both paths", async () => {
    const { store } = await createProbeStore();
    for (const query of ["pio-1384", "PIO-1384", '"pio-1384"', '"PIO-1384"', '"ZERO-ssh"']) {
      expect(store.searchFTS(query, 10), query).toHaveLength(1);
    }
  });
});

// =============================================================================
// Negation still parses, and vec/hyde still accept hyphens
// =============================================================================

describe("negation and semantic-query validation are unaffected", () => {
  test("a leading hyphen still negates, inside quotes and out", async () => {
    const { store } = await createProbeStore();
    expect(store.searchFTS("gateway -scaffold", 10)).toHaveLength(0);
    expect(store.searchFTS('gateway -"cloud-init"', 10)).toHaveLength(0);
    expect(store.searchFTS("gateway -kubernetes", 10)).toHaveLength(1);
  });

  test("vec and hyde accept mid-term separators and still reject negation", () => {
    expect(validateSemanticQuery("flash-and-forget gateway scaffold")).toBeNull();
    expect(validateSemanticQuery("src/lib/provision.ts and @tobilu/qmd")).toBeNull();
    expect(validateSemanticQuery('"cloud-init" provisioning')).toBeNull();
    expect(validateSemanticQuery("gateway -scaffold")).toContain("Negation");
    expect(validateSemanticQuery('gateway -"cloud-init"')).toContain("Negation");
  });
});
