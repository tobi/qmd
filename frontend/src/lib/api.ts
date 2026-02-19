async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// Types matching store.ts
export interface SearchResult {
  filepath: string;
  displayPath: string;
  title: string;
  context: string | null;
  hash: string;
  docid: string;
  collectionName: string;
  modifiedAt: string;
  bodyLength: number;
  body?: string;
  score: number;
  source: "fts" | "vec";
  chunkPos?: number;
}

export interface HybridQueryResult {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  bestChunk: string;
  bestChunkPos: number;
  score: number;
  context: string | null;
  docid: string;
}

export interface DocumentResult {
  filepath: string;
  displayPath: string;
  title: string;
  context: string | null;
  hash: string;
  docid: string;
  collectionName: string;
  modifiedAt: string;
  bodyLength: number;
  body?: string;
}

export interface CollectionInfo {
  name: string;
  path: string;
  pattern: string;
  documents: number;
  lastUpdated: string;
}

export interface IndexStatus {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: CollectionInfo[];
}

export interface IndexHealth {
  needsEmbedding: number;
  totalDocs: number;
  daysStale: number | null;
}

export interface NamedCollection {
  name: string;
  path: string;
  pattern: string;
  context?: Record<string, string>;
}

export interface ContextEntry {
  collection: string;
  path: string;
  context: string;
}

export interface LsCollection {
  name: string;
  file_count: number;
}

export interface LsFile {
  displayPath: string;
  title: string;
  docid: string;
  modifiedAt: string;
  bodyLength: number;
}

export type LsResult =
  | { type: "collections"; collections: LsCollection[] }
  | { type: "files"; collection: string; path: string | null; files: LsFile[] };

// Search
export function search(q: string, n = 10, collection?: string) {
  const params = new URLSearchParams({ q, n: String(n) });
  if (collection) params.set("collection", collection);
  return fetchJSON<SearchResult[]>(`/api/search?${params}`);
}

export function vsearch(q: string, n = 10, collection?: string) {
  const params = new URLSearchParams({ q, n: String(n) });
  if (collection) params.set("collection", collection);
  return fetchJSON<SearchResult[]>(`/api/vsearch?${params}`);
}

export function deepSearch(
  query: string,
  opts?: { limit?: number; collection?: string; minScore?: number }
) {
  return fetchJSON<HybridQueryResult[]>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...opts }),
  });
}

// Documents
export function getDocument(path: string, opts?: { lines?: number; from?: number }) {
  const params = new URLSearchParams();
  if (opts?.lines) params.set("lines", String(opts.lines));
  if (opts?.from) params.set("from", String(opts.from));
  const qs = params.toString();
  return fetchJSON<DocumentResult & { body: string }>(
    `/api/doc/${encodeURIComponent(path)}${qs ? `?${qs}` : ""}`
  );
}

// Listing
export function listFiles(path?: string) {
  const params = path ? `?path=${encodeURIComponent(path)}` : "";
  return fetchJSON<LsResult>(`/api/ls${params}`);
}

// Status
export function getStatus() {
  return fetchJSON<IndexStatus & { health: IndexHealth }>("/api/status");
}

// Collections
export function getCollections() {
  return fetchJSON<NamedCollection[]>("/api/collections");
}

export function removeCollection(name: string) {
  return fetchJSON<{ ok: boolean }>(`/api/collections/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function renameCollection(oldName: string, newName: string) {
  return fetchJSON<{ ok: boolean }>(
    `/api/collections/${encodeURIComponent(oldName)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ newName }),
    }
  );
}

// Contexts
export function getContexts() {
  return fetchJSON<ContextEntry[]>("/api/contexts");
}

export function addContext(collection: string, path: string, context: string) {
  return fetchJSON<{ ok: boolean }>("/api/contexts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection, path, context }),
  });
}

export function removeContext(collection: string, path: string) {
  return fetchJSON<{ ok: boolean }>("/api/contexts", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection, path }),
  });
}
