/**
 * metadata.test.ts - Frontmatter metadata extraction and normalization.
 */

import { describe, test, expect } from "vitest";
import {
  extractDocumentMetadata,
  METADATA_EXTRACTION_VERSION,
  METADATA_LIMITS,
} from "../src/metadata.js";

function buildDoc(frontmatterYaml: string, body: string = "# Title\n\nBody text.\n"): string {
  return `---\n${frontmatterYaml}---\n\n${body}`;
}

describe("extractDocumentMetadata", () => {
  test("document without frontmatter yields empty metadata", () => {
    const extraction = extractDocumentMetadata("# Title\n\nBody.\n", "doc.md");
    expect(extraction).toEqual({ metadata: {}, extractionVersion: METADATA_EXTRACTION_VERSION });
  });

  test("frontmatter without qmd namespace yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("title: Hello\ntags: [a, b]\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("qmd namespace without metadata yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  other: true\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("empty qmd.metadata mapping yields empty metadata", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  metadata: {}\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toBeUndefined();
  });

  test("extracts string, number, and boolean scalars", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    status: published\n    priority: 3\n    reviewed: true\n"),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({ status: "published", priority: 3, reviewed: true });
    expect(extraction.error).toBeUndefined();
  });

  test("extracts homogeneous arrays of every scalar type", () => {
    const extraction = extractDocumentMetadata(
      buildDoc([
        "qmd:",
        "  metadata:",
        "    topics: [typescript, programming]",
        "    scores: [1, 2.5, 3]",
        "    flags: [true, false]",
        "",
      ].join("\n")),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({
      topics: ["typescript", "programming"],
      scores: [1, 2.5, 3],
      flags: [true, false],
    });
  });

  test("de-duplicates array values preserving first-seen order", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    topics: [b, a, b, c, a]\n"),
      "doc.md",
    );
    expect(extraction.metadata["topics"]).toEqual(["b", "a", "c"]);
  });

  test("tolerates BOM, CRLF, and '...' closing marker", () => {
    const bomDoc = "\uFEFF---\nqmd:\n  metadata:\n    status: ok\n---\nBody\n";
    expect(extractDocumentMetadata(bomDoc, "doc.md").metadata).toEqual({ status: "ok" });

    const crlfDoc = "---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nBody\r\n";
    expect(extractDocumentMetadata(crlfDoc, "doc.md").metadata).toEqual({ status: "ok" });

    const dotsDoc = "---\nqmd:\n  metadata:\n    status: ok\n...\nBody\n";
    expect(extractDocumentMetadata(dotsDoc, "doc.md").metadata).toEqual({ status: "ok" });
  });

  test("missing closing delimiter is treated as no frontmatter", () => {
    const extraction = extractDocumentMetadata("---\nqmd:\n  metadata:\n    status: ok\n", "doc.md");
    expect(extraction).toEqual({ metadata: {}, extractionVersion: METADATA_EXTRACTION_VERSION });
  });

  test("non-markdown extensions are not parsed as frontmatter", () => {
    const content = buildDoc("qmd:\n  metadata:\n    status: ok\n");
    expect(extractDocumentMetadata(content, "script.ts").metadata).toEqual({});
    expect(extractDocumentMetadata(content, "noext").metadata).toEqual({});
    expect(extractDocumentMetadata(content, "doc.markdown").metadata).toEqual({ status: "ok" });
    expect(extractDocumentMetadata(content, "doc.mdx").metadata).toEqual({ status: "ok" });
  });

  test("malformed YAML records an extraction error", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd: [unclosed\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/invalid frontmatter YAML/);
  });

  test("non-mapping qmd or qmd.metadata records an extraction error", () => {
    const qmdScalar = extractDocumentMetadata(buildDoc("qmd: hello\n"), "doc.md");
    expect(qmdScalar.error).toMatch(/'qmd' must be a mapping/);

    const metadataScalar = extractDocumentMetadata(buildDoc("qmd:\n  metadata: hello\n"), "doc.md");
    expect(metadataScalar.error).toMatch(/'qmd.metadata' must be a mapping/);
  });

  test("rejects nested objects, null, empty arrays, nested arrays, and mixed arrays", () => {
    const cases: [string, RegExp][] = [
      ["qmd:\n  metadata:\n    nested:\n      a: 1\n", /unsupported value type/],
      ["qmd:\n  metadata:\n    empty: null\n", /null is not supported/],
      ["qmd:\n  metadata:\n    empty: []\n", /empty arrays are not supported/],
      ["qmd:\n  metadata:\n    nested: [[1, 2]]\n", /nested arrays are not supported/],
      ["qmd:\n  metadata:\n    mixed: [1, two]\n", /mixed-type arrays are not supported/],
    ];

    for (const [frontmatterYaml, expected] of cases) {
      const extraction = extractDocumentMetadata(buildDoc(frontmatterYaml), "doc.md");
      expect(extraction.metadata).toEqual({});
      expect(extraction.error).toMatch(expected);
    }
  });

  test("rejects non-finite numbers", () => {
    const extraction = extractDocumentMetadata(buildDoc("qmd:\n  metadata:\n    bad: .inf\n"), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/finite/);
  });

  test("rejects oversized keys, strings, arrays, and key counts", () => {
    const longKey = "k".repeat(METADATA_LIMITS.maxKeyBytes + 1);
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    ${longKey}: 1\n`), "doc.md").error)
      .toMatch(/key exceeds/);

    const longString = "v".repeat(METADATA_LIMITS.maxStringLength + 1);
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    long: "${longString}"\n`), "doc.md").error)
      .toMatch(/string exceeds/);

    const bigArray = `[${Array.from({ length: METADATA_LIMITS.maxArrayLength + 1 }, (_, i) => i).join(", ")}]`;
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n    big: ${bigArray}\n`), "doc.md").error)
      .toMatch(/array exceeds/);

    const manyKeys = Array.from({ length: METADATA_LIMITS.maxKeys + 1 }, (_, i) => `    key${i}: 1`).join("\n");
    expect(extractDocumentMetadata(buildDoc(`qmd:\n  metadata:\n${manyKeys}\n`), "doc.md").error)
      .toMatch(/keys \(max/);
  });

  test("rejects oversized frontmatter blocks", () => {
    const filler = `filler: "${"x".repeat(METADATA_LIMITS.maxFrontmatterBytes)}"\n`;
    const extraction = extractDocumentMetadata(buildDoc(`${filler}qmd:\n  metadata:\n    status: ok\n`), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/frontmatter exceeds/);
  });

  test("bounds YAML alias expansion", () => {
    const aliasBomb = [
      "a: &a [x, x, x, x, x, x, x, x, x, x]",
      "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]",
      "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]",
      "qmd:",
      "  metadata:",
      "    status: ok",
      "",
    ].join("\n");
    const extraction = extractDocumentMetadata(buildDoc(aliasBomb), "doc.md");
    expect(extraction.metadata).toEqual({});
    expect(extraction.error).toMatch(/invalid frontmatter YAML/);
  });

  test("bounds extraction error message length", () => {
    const longKey = "k".repeat(600);
    const extraction = extractDocumentMetadata(
      buildDoc(`qmd:\n  metadata:\n    valid: 1\n    "${longKey}x": 1\n`),
      "doc.md",
    );
    expect(extraction.error).toBeDefined();
    expect(extraction.error!.length).toBeLessThanOrEqual(METADATA_LIMITS.maxErrorLength);
  });

  test("unquoted dates stay strings", () => {
    const extraction = extractDocumentMetadata(
      buildDoc("qmd:\n  metadata:\n    published: 2024-01-15\n"),
      "doc.md",
    );
    expect(extraction.metadata).toEqual({ published: "2024-01-15" });
  });
});
