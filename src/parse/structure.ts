import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import type { Root, Heading, Blockquote } from "mdast";

export interface TocEntry {
  level: number;
  heading: string;
  slug: string;
  seq: number;
  parent_seq: number | null;
  char_offset: number;
}

export interface ParsedSection {
  seq: number;
  parent_seq: number | null;
  level: number;
  heading: string | null;
  slug: string | null;
  body: string;
  body_no_callouts: string;
  word_count: number;
  char_offset: number;
  callouts: ParsedCallout[];
}

export interface ParsedCallout {
  kind: string;
  title: string | null;
  body: string;
}

export interface ParsedWikilink {
  target: string;
  anchor: string | null;
}

export interface ParsedStructure {
  toc: TocEntry[];
  sections: ParsedSection[];
  wikilinks: ParsedWikilink[];
}

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-").replace(/-+/g, "-").trim();
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function nodeText(node: { type: string; value?: string; children?: unknown[] }): string {
  if ("value" in node && typeof node.value === "string") return node.value;
  if (node.children) return (node.children as typeof node[]).map(nodeText).join("");
  return "";
}

function extractWikilinks(text: string): ParsedWikilink[] {
  const links: ParsedWikilink[] = [];
  const re = /\[\[([^\]|#]+)(?:#([^\]|]*))?(?:\|[^\]]*)?\]\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    links.push({ target: (m[1] ?? "").trim(), anchor: m[2]?.trim() ?? null });
  }
  return links;
}

function parseCallout(node: Blockquote, rawText: string): ParsedCallout | null {
  const firstChild = node.children[0];
  if (!firstChild || firstChild.type !== "paragraph") return null;
  const fullText = nodeText(firstChild as unknown as { type: string; children: unknown[] });
  const match = /^\[!([^\]]+)\](.*)/.exec(fullText);
  if (!match) return null;
  const kind = (match[1] ?? "note").toLowerCase().trim();
  const titlePart = match[2]?.trim() ?? "";
  const bodyStart = node.position?.start.offset ?? 0;
  const bodyEnd = node.position?.end.offset ?? rawText.length;
  const raw = rawText.slice(bodyStart, bodyEnd);
  const lines = raw.split("\n").map((l) => l.replace(/^>\s?/, ""));
  const bodyLines = lines.slice(1).join("\n").trim();
  return { kind, title: titlePart || null, body: bodyLines };
}

const parser = unified().use(remarkParse).use(remarkGfm);

export function parseStructure(rawMarkdown: string): ParsedStructure {
  const tree = parser.parse(rawMarkdown) as Root;
  const sections: ParsedSection[] = [];
  const wikilinks: ParsedWikilink[] = [];
  const toc: TocEntry[] = [];
  const parentStack: Map<number, number> = new Map();

  let currentSection: {
    seq: number;
    parent_seq: number | null;
    level: number;
    heading: string | null;
    slug: string | null;
    char_offset: number;
    bodyParts: string[];
    calloutParts: string[];
    callouts: ParsedCallout[];
  } = { seq: 0, parent_seq: null, level: 0, heading: null, slug: null, char_offset: 0, bodyParts: [], calloutParts: [], callouts: [] };

  function flushSection(): void {
    const body = currentSection.bodyParts.join("\n").trim();
    const body_no_callouts = currentSection.calloutParts.join("\n").trim();
    sections.push({
      seq: currentSection.seq,
      parent_seq: currentSection.parent_seq,
      level: currentSection.level,
      heading: currentSection.heading,
      slug: currentSection.slug,
      body,
      body_no_callouts,
      word_count: countWords(body),
      char_offset: currentSection.char_offset,
      callouts: currentSection.callouts,
    });
  }

  let seq = 0;

  for (const node of tree.children) {
    if (node.type === "heading") {
      flushSection();
      const h = node as Heading;
      seq++;
      const headingStr = nodeText(h as unknown as { type: string; children: unknown[] });
      const slug = slugify(headingStr);
      const level = h.depth;
      const offset = h.position?.start.offset ?? 0;

      // Update parent stack
      for (const [lvl] of [...parentStack.entries()]) {
        if (lvl >= level) parentStack.delete(lvl);
      }
      const parent_seq = parentStack.get(level - 1) ?? null;
      parentStack.set(level, seq);

      toc.push({ level, heading: headingStr, slug, seq, parent_seq, char_offset: offset });

      currentSection = { seq, parent_seq, level, heading: headingStr, slug, char_offset: offset, bodyParts: [], calloutParts: [], callouts: [] };
    } else if (node.type === "blockquote") {
      const bq = node as Blockquote;
      const callout = parseCallout(bq, rawMarkdown);
      const start = bq.position?.start.offset ?? 0;
      const end = bq.position?.end.offset ?? rawMarkdown.length;
      const raw = rawMarkdown.slice(start, end);
      currentSection.bodyParts.push(raw);
      if (callout) {
        currentSection.callouts.push(callout);
      } else {
        currentSection.calloutParts.push(raw);
      }
    } else {
      const start = node.position?.start.offset ?? 0;
      const end = node.position?.end.offset ?? rawMarkdown.length;
      const raw = rawMarkdown.slice(start, end);
      currentSection.bodyParts.push(raw);
      currentSection.calloutParts.push(raw);
    }
  }

  flushSection();
  wikilinks.push(...extractWikilinks(rawMarkdown));
  return { toc, sections, wikilinks };
}
