import { describe, expect, test } from "vitest";
import { parseGeneratedExpansionLines } from "../src/query-expansion-parser.js";

describe("generated expansion line parser", () => {
  test("keeps usable lines and reports bad lines independently", () => {
    expect(parseGeneratedExpansionLines("lex: auth\nnot a typed line"))
      .toEqual({
        output: [["lex", "auth"]],
        diagnostics: [{
          code: "RUNTIME_INVALID_LINE",
          line: 2,
          text: "not a typed line",
        }],
      });
  });

  test("recognizes canonical syntax without imposing it on runtime", () => {
    expect(parseGeneratedExpansionLines("lex:auth")).toEqual({
      output: [["lex", "auth"]],
      diagnostics: [{
        code: "RUNTIME_INVALID_LINE",
        line: 1,
        text: "lex:auth",
      }],
    });
  });

  test("drops empty text", () => {
    expect(parseGeneratedExpansionLines("lex:")).toEqual({
      output: [],
      diagnostics: [{ code: "RUNTIME_EMPTY_TEXT", line: 1, text: "lex:" }],
    });
  });
});
