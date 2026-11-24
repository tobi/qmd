/** Pure parsing and acceptance helpers for generated query expansions. */

export type GeneratedExpansionType = "lex" | "vec" | "hyde";
export type GeneratedExpansion = [GeneratedExpansionType, string];

export type GeneratedExpansionDiagnostic = {
  code:
    | "RUNTIME_EMPTY_OUTPUT"
    | "RUNTIME_INVALID_LINE"
    | "RUNTIME_EMPTY_TEXT";
  line: number;
  text: string;
};

export type ParsedGeneratedExpansionLines = {
  output: GeneratedExpansion[];
  diagnostics: GeneratedExpansionDiagnostic[];
};

const CANONICAL_LINE = /^(lex|vec|hyde): ([^\r\n]+)$/;

/**
 * Parse every line independently.  Production may keep valid lines, while a
 * stricter consumers can reject the response when diagnostics are non-empty.
 */
export function parseGeneratedExpansionLines(
  rawOutput: string,
): ParsedGeneratedExpansionLines {
  const trimmed = rawOutput.trim();
  if (trimmed.length === 0) {
    return {
      output: [],
      diagnostics: [{ code: "RUNTIME_EMPTY_OUTPUT", line: 0, text: "" }],
    };
  }

  const output: GeneratedExpansion[] = [];
  const diagnostics: GeneratedExpansionDiagnostic[] = [];
  const lines = trimmed.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!.replace(/\r$/, "");
    const canonicalMatch = CANONICAL_LINE.exec(line);

    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
      continue;
    }
    const type = line.slice(0, colonIndex).trim();
    if (type !== "lex" && type !== "vec" && type !== "hyde") {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
      continue;
    }
    const text = line.slice(colonIndex + 1).trim();
    if (text.length === 0) {
      diagnostics.push({
        code: "RUNTIME_EMPTY_TEXT",
        line: index + 1,
        text: line,
      });
      continue;
    }
    if (!canonicalMatch) {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
    }
    output.push([type, text]);
  }

  return { output, diagnostics };
}

// ---------------------------------------------------------------------------
// Strict V2-VH response validation (exactly 1 hyde first, then 1-2 vec lines)
// ---------------------------------------------------------------------------

export type VhExpansionDiagnosticCode =
  | "VH_EMPTY_OUTPUT"
  | "VH_INVALID_LINE"
  | "VH_EMPTY_TEXT"
  | "VH_MISSING_HYDE"
  | "VH_MULTIPLE_HYDE"
  | "VH_FORBIDDEN_TYPE"
  | "VH_MISSING_VEC"
  | "VH_TOO_MANY_LINES";

export type VhExpansionDiagnostic = {
  code: VhExpansionDiagnosticCode;
  line: number;
  text: string;
};

export type ParsedVhExpansionResponse = {
  /** Empty whenever diagnostics is non-empty: VH never keeps partial output. */
  output: GeneratedExpansion[];
  diagnostics: VhExpansionDiagnostic[];
};

const VH_HYDE_LINE = /^hyde: ([^\r\n]+)$/;
const VH_VEC_LINE = /^vec: ([^\r\n]+)$/;

/**
 * Validate a complete response against the V2-VH contract:
 * exactly one `hyde:` line first, followed by one or two `vec:` lines.
 * `lex:`, extra lines, missing hyde, empty text, wrong order, and
 * non-canonical whitespace are all rejected; nothing partial is kept.
 */
export function parseVhExpansionResponse(
  rawOutput: string,
): ParsedVhExpansionResponse {
  const trimmed = rawOutput.trim();
  if (trimmed.length === 0) {
    return {
      output: [],
      diagnostics: [{ code: "VH_EMPTY_OUTPUT", line: 0, text: "" }],
    };
  }

  const lines = trimmed.split("\n").map(line => line.replace(/\r$/, ""));
  const diagnostics: VhExpansionDiagnostic[] = [];
  const output: GeneratedExpansion[] = [];

  const hydeMatch = VH_HYDE_LINE.exec(lines[0] ?? "");
  if (!hydeMatch) {
    const emptyHyde = (lines[0] ?? "").trim() === "hyde:";
    diagnostics.push({
      code: emptyHyde ? "VH_EMPTY_TEXT" : "VH_MISSING_HYDE",
      line: 1,
      text: lines[0] ?? "",
    });
  } else {
    output.push(["hyde", hydeMatch[1]!]);
  }

  let vecCount = 0;
  for (let index = 1; index < lines.length; index++) {
    const line = lines[index]!;
    const vecMatch = VH_VEC_LINE.exec(line);
    if (vecMatch) {
      vecCount++;
      if (vecCount <= 2) output.push(["vec", vecMatch[1]!]);
      continue;
    }
    const emptyVec = line.trim() === "vec:";
    if (emptyVec) {
      diagnostics.push({ code: "VH_EMPTY_TEXT", line: index + 1, text: line });
      continue;
    }
    if (/^hyde:/.test(line)) {
      diagnostics.push({ code: "VH_MULTIPLE_HYDE", line: index + 1, text: line });
      continue;
    }
    if (/^(lex|vec|hyde):/.test(line)) {
      diagnostics.push({ code: "VH_FORBIDDEN_TYPE", line: index + 1, text: line });
      continue;
    }
    diagnostics.push({ code: "VH_INVALID_LINE", line: index + 1, text: line });
  }

  if (hydeMatch && vecCount === 0) {
    diagnostics.push({ code: "VH_MISSING_VEC", line: lines.length, text: "" });
  }
  if (vecCount > 2) {
    diagnostics.push({ code: "VH_TOO_MANY_LINES", line: 0, text: `${vecCount} vec lines` });
  }

  if (diagnostics.length > 0) return { output: [], diagnostics };
  return { output, diagnostics };
}
