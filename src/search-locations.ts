import type { RerankTokenBudget } from "./llm.js";
import { containsCjk, parseLexicalQuery } from "./lexical-query.js";

export type DocumentLocationRef = {
  uri: string;
  contentHash: string;
};

export type Utf16Span = {
  startUtf16: number;
  endUtf16: number;
};

export type PassageBudget = {
  maxUtf8Bytes: number;
  maxUnicodeScalars?: number;
};

export type PassageWindow = Utf16Span & {
  text: string;
  utf8Bytes: number;
  unicodeScalars: number;
  anchorClipped: boolean;
};

export type PositiveLexicalAnchor = {
  kind: "literal" | "phrase";
  text: string;
  tokens: readonly string[];
  match: "prefix" | "phrase";
};

export type LexicalExactLocation = DocumentLocationRef & Utf16Span & {
  kind: "lexical_exact";
  anchorKind: "literal" | "phrase";
  matchedText: string;
};

export type LexicalApproximateLocation = DocumentLocationRef & Utf16Span & {
  kind: "lexical_approximate";
  reason: "stem_or_tokenizer";
  queryText: string;
};

export type VectorChunkStartLocation = DocumentLocationRef & {
  kind: "vector_chunk_start";
  startUtf16: number;
  endUtf16: null;
  chunkSeq: number | null;
};

export type SelectionWindowLocation = DocumentLocationRef & Utf16Span & {
  kind: "selection_window";
  origin: "keyword_intent";
};

export type LexicalLocation = LexicalExactLocation | LexicalApproximateLocation;

export type SearchLocation =
  | LexicalLocation
  | VectorChunkStartLocation
  | SelectionWindowLocation;

export type LocatedLexicalPassage = {
  location: LexicalLocation;
  passage: PassageWindow;
};

type ApproximateBodyToken = Utf16Span & {
  comparable: string;
};

const WORD_CHAR_PATTERN = /[\p{L}\p{N}]/u;

function tokensForPhrase(phrase: string): string[] {
  const tokens: string[] = [];
  for (const token of tokenizeApproximateBody(phrase)) {
    tokens.push(token.comparable);
  }
  return tokens;
}

/** Parse the positive anchors accepted by QMD's lexical query grammar. */
export function parsePositiveLexicalAnchors(query: string): PositiveLexicalAnchor[] {
  const anchors: PositiveLexicalAnchor[] = [];
  for (const clause of parseLexicalQuery(query)) {
    if (clause.negated) continue;
    const tokens = tokensForPhrase(clause.text);
    if (tokens.length === 0) continue;
    anchors.push({
      kind: clause.match === "phrase" ? "phrase" : "literal",
      text: clause.text,
      tokens,
      match: clause.match,
    });
  }
  return anchors;
}

/** Return a byte-bounded passage around a UTF-16 source span. */
export function boundedPassageWindow(
  body: string,
  anchor: Utf16Span,
  budget: PassageBudget,
): PassageWindow {
  assertBodySpan(body, anchor, "anchor");
  const validatedBudget = validatePassageBudget(budget);

  const maxScalars = validatedBudget.maxUnicodeScalars ?? Number.POSITIVE_INFINITY;
  const anchorSize = measureScalarRange(body, anchor.startUtf16, anchor.endUtf16);

  if (anchorSize.utf8Bytes > validatedBudget.maxUtf8Bytes || anchorSize.unicodeScalars > maxScalars) {
    let endUtf16 = anchor.startUtf16;
    let bytes = 0;
    let scalars = 0;

    while (endUtf16 < anchor.endUtf16) {
      const nextEnd = nextScalarEnd(body, endUtf16);
      const scalarBytes = scalarByteWidth(body.codePointAt(endUtf16)!);
      if (bytes + scalarBytes > validatedBudget.maxUtf8Bytes || scalars + 1 > maxScalars) break;
      bytes += scalarBytes;
      scalars++;
      endUtf16 = nextEnd;
    }

    return {
      startUtf16: anchor.startUtf16,
      endUtf16,
      text: body.slice(anchor.startUtf16, endUtf16),
      utf8Bytes: bytes,
      unicodeScalars: scalars,
      anchorClipped: endUtf16 < anchor.endUtf16,
    };
  }

  let startUtf16 = anchor.startUtf16;
  let endUtf16 = anchor.endUtf16;
  let bytes = anchorSize.utf8Bytes;
  let scalars = anchorSize.unicodeScalars;
  let nextSide: "left" | "right" = "left";
  let leftOpen = startUtf16 > 0;
  let rightOpen = endUtf16 < body.length;

  while (leftOpen || rightOpen) {
    let added = false;
    const preferredSide = nextSide;
    const alternateSide = preferredSide === "left" ? "right" : "left";

    for (let attempt = 0; attempt < 2; attempt++) {
      const side = attempt === 0 ? preferredSide : alternateSide;
      if (side === "left" && leftOpen) {
        const nextStart = previousScalarStart(body, startUtf16);
        const scalarBytes = scalarByteWidth(body.codePointAt(nextStart)!);
        if (bytes + scalarBytes <= validatedBudget.maxUtf8Bytes && scalars + 1 <= maxScalars) {
          startUtf16 = nextStart;
          bytes += scalarBytes;
          scalars++;
          leftOpen = startUtf16 > 0;
          nextSide = "right";
          added = true;
          break;
        }
        leftOpen = false;
      }

      if (side === "right" && rightOpen) {
        const nextEnd = nextScalarEnd(body, endUtf16);
        const scalarBytes = scalarByteWidth(body.codePointAt(endUtf16)!);
        if (bytes + scalarBytes <= validatedBudget.maxUtf8Bytes && scalars + 1 <= maxScalars) {
          endUtf16 = nextEnd;
          bytes += scalarBytes;
          scalars++;
          rightOpen = endUtf16 < body.length;
          nextSide = "left";
          added = true;
          break;
        }
        rightOpen = false;
      }
    }

    if (!added) break;
  }

  return {
    startUtf16,
    endUtf16,
    text: body.slice(startUtf16, endUtf16),
    utf8Bytes: bytes,
    unicodeScalars: scalars,
    anchorClipped: false,
  };
}

/** Fit a scalar-safe source passage within both caller and reranker limits. */
export function fitPassageWindow(
  body: string,
  anchor: Utf16Span,
  budget: PassageBudget,
  rerankBudget: RerankTokenBudget,
): PassageWindow {
  assertBodySpan(body, anchor, "anchor");
  const validatedBudget = validatePassageBudget(budget);
  assertOffset(rerankBudget.maxDocumentTokens, "maxDocumentTokens");

  const requestedPassage = boundedPassageWindow(body, anchor, validatedBudget);
  if (passageFitsReranker(requestedPassage, rerankBudget)) return requestedPassage;

  const emptyPassage = boundedPassageWindow(body, anchor, {
    ...validatedBudget,
    maxUtf8Bytes: 0,
    maxUnicodeScalars: 0,
  });
  if (!passageFitsReranker(emptyPassage, rerankBudget)) {
    throw new RangeError("rerank token budget must fit an empty passage");
  }

  let bestFit = emptyPassage;
  let minimumByteLimit = 1;
  let maximumByteLimit = requestedPassage.utf8Bytes - 1;
  const anchorSize = measureScalarRange(body, anchor.startUtf16, anchor.endUtf16);
  const callerBudgetFitsAnchor =
    anchorSize.utf8Bytes <= validatedBudget.maxUtf8Bytes
    && anchorSize.unicodeScalars <= (validatedBudget.maxUnicodeScalars ?? Number.POSITIVE_INFINITY);

  if (callerBudgetFitsAnchor) {
    const anchorPassage = boundedPassageWindow(body, anchor, {
      ...validatedBudget,
      maxUtf8Bytes: anchorSize.utf8Bytes,
      maxUnicodeScalars: anchorSize.unicodeScalars,
    });
    if (passageFitsReranker(anchorPassage, rerankBudget)) {
      bestFit = anchorPassage;
      minimumByteLimit = anchorPassage.utf8Bytes + 1;
    }
  }

  while (minimumByteLimit <= maximumByteLimit) {
    const maxUtf8Bytes = minimumByteLimit
      + Math.floor((maximumByteLimit - minimumByteLimit) / 2);
    const candidatePassage = boundedPassageWindow(body, anchor, {
      ...validatedBudget,
      maxUtf8Bytes,
    });

    if (passageFitsReranker(candidatePassage, rerankBudget)) {
      bestFit = candidatePassage;
      minimumByteLimit = maxUtf8Bytes + 1;
    } else {
      maximumByteLimit = maxUtf8Bytes - 1;
    }
  }

  if (passageFitsReranker(bestFit, rerankBudget)) return bestFit;
  if (passageFitsReranker(emptyPassage, rerankBudget)) return emptyPassage;
  throw new RangeError("rerank token budget changed while fitting the passage");
}

/** Locate the first strongest positive lexical anchor and bound its surrounding passage. */
export function locateLexical(
  body: string,
  query: string,
  ref: DocumentLocationRef,
  budget: PassageBudget,
): LocatedLexicalPassage | null {
  assertDocumentRef(ref);
  const validatedBudget = validatePassageBudget(budget);
  const location = findLexicalLocation(body, query, ref);
  if (!location) return null;
  return {
    location,
    passage: boundedPassageWindow(body, location, validatedBudget),
  };
}

/** Locate the first strongest positive lexical anchor in the document body. */
export function findLexicalLocation(
  body: string,
  query: string,
  ref: DocumentLocationRef,
): LexicalLocation | null {
  assertDocumentRef(ref);
  const anchors = parsePositiveLexicalAnchors(query);
  for (const anchor of anchors) {
    const span = findExactAnchor(body, anchor);
    if (!span) continue;

    return {
      ...ref,
      ...span,
      kind: "lexical_exact",
      anchorKind: anchor.kind,
      matchedText: body.slice(span.startUtf16, span.endUtf16),
    };
  }

  const bodyTokens = tokenizeApproximateBody(body);
  for (const anchor of anchors) {
    const span = findApproximateAnchor(bodyTokens, anchor);
    if (!span) continue;

    return {
      ...ref,
      ...span,
      kind: "lexical_approximate",
      reason: "stem_or_tokenizer",
      queryText: anchor.text,
    };
  }

  return null;
}

export function vectorChunkLocation(
  ref: DocumentLocationRef,
  startUtf16: number,
  chunkSeq: number | null,
): VectorChunkStartLocation {
  assertDocumentRef(ref);
  assertOffset(startUtf16, "startUtf16");
  if (chunkSeq !== null) assertOffset(chunkSeq, "chunkSeq");
  return { ...ref, kind: "vector_chunk_start", startUtf16, endUtf16: null, chunkSeq };
}

export function selectionWindowLocation(
  ref: DocumentLocationRef,
  span: Utf16Span,
): SelectionWindowLocation {
  assertDocumentRef(ref);
  assertSpan(span, "selection window");
  return { ...ref, ...span, kind: "selection_window", origin: "keyword_intent" };
}

function findExactAnchor(body: string, anchor: PositiveLexicalAnchor): Utf16Span | null {
  const pattern = new RegExp(escapeRegExp(anchor.text), "giu");
  for (const match of body.matchAll(pattern)) {
    const startUtf16 = match.index;
    const matchedText = match[0];
    if (startUtf16 === undefined || matchedText === undefined) continue;
    const endUtf16 = startUtf16 + matchedText.length;

    const firstAnchorScalar = anchor.text.slice(0, nextScalarEnd(anchor.text, 0));
    const lastAnchorStart = previousScalarStart(anchor.text, anchor.text.length);
    const lastAnchorScalar = anchor.text.slice(lastAnchorStart);
    const startsAtBoundary = containsCjk(firstAnchorScalar) || isTokenBoundary(body, startUtf16);
    const endsAtBoundary = containsCjk(lastAnchorScalar) || isTokenBoundary(body, endUtf16);
    if (startsAtBoundary && (anchor.match === "prefix" || endsAtBoundary)) {
      return { startUtf16, endUtf16 };
    }
  }
  return null;
}

function findApproximateAnchor(
  bodyTokens: readonly ApproximateBodyToken[],
  anchor: PositiveLexicalAnchor,
): Utf16Span | null {
  if (anchor.tokens.length === 0) return null;

  for (let bodyIndex = 0; bodyIndex + anchor.tokens.length <= bodyTokens.length; bodyIndex++) {
    let matches = true;
    for (let queryIndex = 0; queryIndex < anchor.tokens.length; queryIndex++) {
      const queryToken = anchor.tokens[queryIndex]!;
      const bodyToken = bodyTokens[bodyIndex + queryIndex]!;
      const prefixAllowed = anchor.match === "prefix" && queryIndex === anchor.tokens.length - 1;
      if (!tokensApproximate(queryToken, bodyToken.comparable, prefixAllowed)) {
        matches = false;
        break;
      }
    }

    if (matches) {
      return {
        startUtf16: bodyTokens[bodyIndex]!.startUtf16,
        endUtf16: bodyTokens[bodyIndex + anchor.tokens.length - 1]!.endUtf16,
      };
    }
  }

  return null;
}

function tokensApproximate(query: string, body: string, prefixAllowed: boolean): boolean {
  if (query === body) return true;
  if (prefixAllowed && body.startsWith(query)) return true;
  return approximateStem(query) === approximateStem(body);
}

function approximateStem(token: string): string {
  let stem = token;
  if (stem.length < 3) return stem;

  if (stem.endsWith("ies") && stem.length > 4) {
    stem = `${stem.slice(0, -3)}y`;
  } else if (stem.endsWith("sses")) {
    stem = stem.slice(0, -2);
  } else if (stem.endsWith("s") && !stem.endsWith("ss") && stem.length > 3) {
    stem = stem.slice(0, -1);
  }

  const suffix = stem.endsWith("ing") ? "ing" : stem.endsWith("ed") ? "ed" : null;
  if (suffix && stem.length - suffix.length >= 3) {
    stem = stem.slice(0, -suffix.length);
    if (/([^aeiou])\1$/u.test(stem) && !/[lsz]{2}$/u.test(stem)) {
      stem = stem.slice(0, -1);
    }
  }

  return stem;
}

function tokenizeApproximateBody(text: string): ApproximateBodyToken[] {
  const tokens: ApproximateBodyToken[] = [];
  let tokenStart: number | null = null;
  let index = 0;

  const flush = (endUtf16: number): void => {
    if (tokenStart === null) return;
    const tokenText = text.slice(tokenStart, endUtf16);
    tokens.push({
      startUtf16: tokenStart,
      endUtf16,
      comparable: comparableToken(tokenText),
    });
    tokenStart = null;
  };

  while (index < text.length) {
    const end = nextScalarEnd(text, index);
    const scalar = text.slice(index, end);

    if (containsCjk(scalar)) {
      flush(index);
      tokens.push({
        startUtf16: index,
        endUtf16: end,
        comparable: comparableToken(scalar),
      });
    } else if (WORD_CHAR_PATTERN.test(scalar)) {
      tokenStart ??= index;
    } else {
      flush(index);
    }

    index = end;
  }
  flush(text.length);
  return tokens;
}

function comparableToken(token: string): string {
  return token.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

function isTokenBoundary(body: string, offset: number): boolean {
  if (offset === 0 || offset === body.length) return true;
  const previousStart = previousScalarStart(body, offset);
  const previous = body.slice(previousStart, offset);
  const next = body.slice(offset, nextScalarEnd(body, offset));
  return !WORD_CHAR_PATTERN.test(previous) || !WORD_CHAR_PATTERN.test(next);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function previousScalarStart(text: string, offset: number): number {
  if (offset <= 0) return 0;
  const last = text.charCodeAt(offset - 1);
  if (offset >= 2 && last >= 0xdc00 && last <= 0xdfff) {
    const first = text.charCodeAt(offset - 2);
    if (first >= 0xd800 && first <= 0xdbff) return offset - 2;
  }
  return offset - 1;
}

function nextScalarEnd(text: string, offset: number): number {
  if (offset >= text.length) return text.length;
  const first = text.charCodeAt(offset);
  if (first >= 0xd800 && first <= 0xdbff && offset + 1 < text.length) {
    const second = text.charCodeAt(offset + 1);
    if (second >= 0xdc00 && second <= 0xdfff) return offset + 2;
  }
  return offset + 1;
}

/** Return whether an in-range UTF-16 offset falls between complete scalar values. */
export function isUtf16Boundary(text: string, offset: number): boolean {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > text.length) return false;
  if (offset === 0 || offset === text.length) return true;
  const previous = text.charCodeAt(offset - 1);
  const next = text.charCodeAt(offset);
  return !(previous >= 0xd800 && previous <= 0xdbff && next >= 0xdc00 && next <= 0xdfff);
}

function scalarByteWidth(codePoint: number): number {
  if (codePoint <= 0x7f) return 1;
  if (codePoint <= 0x7ff) return 2;
  if (codePoint <= 0xffff) return 3;
  return 4;
}

function measureScalarRange(body: string, startUtf16: number, endUtf16: number): {
  utf8Bytes: number;
  unicodeScalars: number;
} {
  let utf8Bytes = 0;
  let unicodeScalars = 0;
  for (let offset = startUtf16; offset < endUtf16; offset = nextScalarEnd(body, offset)) {
    utf8Bytes += scalarByteWidth(body.codePointAt(offset)!);
    unicodeScalars++;
  }
  return { utf8Bytes, unicodeScalars };
}

function passageFitsReranker(
  passage: PassageWindow,
  rerankBudget: RerankTokenBudget,
): boolean {
  const tokens = rerankBudget.countTokens(passage.text);
  assertOffset(tokens, "countTokens result");
  return tokens <= rerankBudget.maxDocumentTokens;
}

function assertDocumentRef(ref: DocumentLocationRef): void {
  if (ref.uri.length === 0) throw new RangeError("uri must contain at least one character");
  if (ref.contentHash.length === 0) {
    throw new RangeError("contentHash must contain at least one character");
  }
}

function assertOffset(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
}

function assertSpan(span: Utf16Span, name: string): void {
  assertOffset(span.startUtf16, `${name}.startUtf16`);
  assertOffset(span.endUtf16, `${name}.endUtf16`);
  if (span.endUtf16 < span.startUtf16) {
    throw new RangeError(`${name}.endUtf16 must be greater than or equal to ${name}.startUtf16`);
  }
}

function assertBodySpan(body: string, span: Utf16Span, name: string): void {
  assertSpan(span, name);
  if (span.endUtf16 > body.length) {
    throw new RangeError(`${name}.endUtf16 must be within the body`);
  }
  if (!isUtf16Boundary(body, span.startUtf16) || !isUtf16Boundary(body, span.endUtf16)) {
    throw new RangeError(`${name} boundaries must preserve Unicode scalar values`);
  }
}

/** Validate a passage budget and return an independent numeric copy. */
export function validatePassageBudget(budget: PassageBudget): PassageBudget {
  const { maxUtf8Bytes, maxUnicodeScalars } = budget;
  assertOffset(maxUtf8Bytes, "maxUtf8Bytes");
  if (maxUnicodeScalars !== undefined) {
    assertOffset(maxUnicodeScalars, "maxUnicodeScalars");
    return {
      maxUtf8Bytes,
      maxUnicodeScalars,
    };
  }
  return { maxUtf8Bytes };
}
