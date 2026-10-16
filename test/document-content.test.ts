import { test, expect } from "vitest";
import { documentReadContent } from "../src/mcp/document-content.js";

const doc = {
  uri: "qmd://wiki/readme.md",
  name: "wiki/readme.md",
  title: "Readme",
  text: "The document body",
};

test("QMD retrieval defaults to portable text content with source path", () => {
  expect(documentReadContent(doc)).toEqual({
    type: "text",
    text: "# wiki/readme.md\n\nThe document body",
  });
});

test("explicit resource exposure needs both opt-in and confirmed user approval", () => {
  expect(() => documentReadContent(doc, { exposeToUser: true })).toThrow(/explicit user approval/);
  expect(documentReadContent(doc, { confirmUserApprovedExposure: true }).type).toBe("text");
  expect(documentReadContent(doc, { exposeToUser: true, confirmUserApprovedExposure: true })).toEqual({
    type: "resource",
    resource: {
      uri: doc.uri,
      name: doc.name,
      title: doc.title,
      mimeType: "text/markdown",
      text: doc.text,
    },
  });
});
