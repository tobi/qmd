/**
 * QMD Web Server - REST API + static frontend
 *
 * Serves the React frontend from frontend/dist/ and provides
 * JSON API endpoints that call store.ts and collections.ts directly.
 */

import { existsSync } from "fs";
import { join, resolve } from "path";
import {
  createStore,
  hybridQuery,
  DEFAULT_EMBED_MODEL,
  type Store,
} from "./store";
import {
  listCollections,
  removeCollection,
  renameCollection,
  listAllContexts,
  addContext,
  removeContext,
  setGlobalContext,
  getCollection,
} from "./collections";
import { disposeDefaultLlamaCpp } from "./llm";

// =============================================================================
// Static file serving
// =============================================================================

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function getMimeType(path: string): string {
  const ext = path.slice(path.lastIndexOf("."));
  return MIME_TYPES[ext] || "application/octet-stream";
}

// =============================================================================
// API route handlers
// =============================================================================

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function errorResponse(message: string, status = 400): Response {
  return jsonResponse({ error: message }, status);
}

function corsHeaders(res: Response, origin?: string | null): Response {
  // In development, allow Vite dev server
  const allowOrigin = origin?.includes("localhost") ? origin : "*";
  res.headers.set("Access-Control-Allow-Origin", allowOrigin);
  res.headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS");
  res.headers.set("Access-Control-Allow-Headers", "Content-Type");
  return res;
}

// GET /api/search?q=...&n=10&collection=...
function handleSearch(store: Store, url: URL): Response {
  const q = url.searchParams.get("q");
  if (!q) return errorResponse("Missing query parameter 'q'");

  const n = parseInt(url.searchParams.get("n") || "10", 10);
  const collectionName = url.searchParams.get("collection") || undefined;

  // Resolve collection to ID if specified
  let collectionId: number | undefined;
  if (collectionName) {
    const coll = store.getCollectionByName(collectionName);
    if (!coll) return errorResponse(`Collection not found: ${collectionName}`, 404);
  }

  let results = store.searchFTS(q, n * 2); // Fetch extra for filtering
  if (collectionName) {
    results = results.filter((r) => r.collectionName === collectionName);
  }
  results = results.slice(0, n);

  return jsonResponse(results);
}

// GET /api/vsearch?q=...&n=10&collection=...
async function handleVSearch(store: Store, url: URL): Promise<Response> {
  const q = url.searchParams.get("q");
  if (!q) return errorResponse("Missing query parameter 'q'");

  const n = parseInt(url.searchParams.get("n") || "10", 10);
  const collectionName = url.searchParams.get("collection") || undefined;

  const results = await store.searchVec(q, DEFAULT_EMBED_MODEL, n, collectionName);
  return jsonResponse(results);
}

// POST /api/query { query, limit?, collection?, minScore? }
async function handleQuery(store: Store, req: Request): Promise<Response> {
  const body = await req.json();
  const query = body.query;
  if (!query) return errorResponse("Missing 'query' in request body");

  const results = await hybridQuery(store, query, {
    limit: body.limit ?? 10,
    collection: body.collection,
    minScore: body.minScore,
  });

  return jsonResponse(results);
}

// GET /api/doc/:path  (path can be displayPath or #docid)
function handleDoc(store: Store, pathname: string, url: URL): Response {
  // Extract the path after /api/doc/
  const docPath = decodeURIComponent(pathname.slice("/api/doc/".length));
  if (!docPath) return errorResponse("Missing document path");

  const lines = url.searchParams.get("lines")
    ? parseInt(url.searchParams.get("lines")!, 10)
    : undefined;
  const from = url.searchParams.get("from")
    ? parseInt(url.searchParams.get("from")!, 10)
    : undefined;

  const result = store.findDocument(docPath, { includeBody: false });
  if ("error" in result) {
    return jsonResponse(result, 404);
  }

  const body = store.getDocumentBody(result, from, lines);
  return jsonResponse({ ...result, body: body || "" });
}

// GET /api/multi-get?pattern=...&maxBytes=...
function handleMultiGet(store: Store, url: URL): Response {
  const pattern = url.searchParams.get("pattern");
  if (!pattern) return errorResponse("Missing 'pattern' parameter");

  const maxBytes = url.searchParams.get("maxBytes")
    ? parseInt(url.searchParams.get("maxBytes")!, 10)
    : undefined;

  const { docs, errors } = store.findDocuments(pattern, {
    includeBody: true,
    maxBytes,
  });

  return jsonResponse({ docs, errors });
}

// GET /api/ls?path=collection/subdir
function handleLs(store: Store, url: URL): Response {
  const pathArg = url.searchParams.get("path");

  if (!pathArg) {
    // List all collections with file counts
    const yamlCollections = listCollections();
    const collections = yamlCollections.map((coll) => {
      const stats = store.db
        .prepare(
          `SELECT COUNT(*) as file_count FROM documents WHERE collection = ? AND active = 1`
        )
        .get(coll.name) as { file_count: number } | null;
      return {
        name: coll.name,
        file_count: stats?.file_count || 0,
      };
    });
    return jsonResponse({ type: "collections", collections });
  }

  // Parse path: collection or collection/subpath
  const parts = pathArg.split("/");
  const collectionName = parts[0] || "";
  const pathPrefix = parts.length > 1 ? parts.slice(1).join("/") : null;

  const coll = getCollection(collectionName);
  if (!coll)
    return errorResponse(`Collection not found: ${collectionName}`, 404);

  let query: string;
  let params: unknown[];

  if (pathPrefix) {
    query = `
      SELECT d.path, d.title, d.hash, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name, `${pathPrefix}%`];
  } else {
    query = `
      SELECT d.path, d.title, d.hash, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name];
  }

  const files = store.db.prepare(query).all(...params) as {
    path: string;
    title: string;
    hash: string;
    modified_at: string;
    size: number;
  }[];

  return jsonResponse({
    type: "files",
    collection: collectionName,
    path: pathPrefix,
    files: files.map((f) => ({
      displayPath: `${collectionName}/${f.path}`,
      title: f.title,
      docid: f.hash.slice(0, 6),
      modifiedAt: f.modified_at,
      bodyLength: f.size,
    })),
  });
}

// GET /api/status
function handleStatus(store: Store): Response {
  const status = store.getStatus();
  const health = store.getIndexHealth();
  return jsonResponse({ ...status, health });
}

// GET /api/collections
function handleGetCollections(): Response {
  return jsonResponse(listCollections());
}

// DELETE /api/collections/:name
function handleDeleteCollection(pathname: string): Response {
  const name = decodeURIComponent(pathname.slice("/api/collections/".length));
  const removed = removeCollection(name);
  if (!removed) return errorResponse(`Collection not found: ${name}`, 404);
  return jsonResponse({ ok: true });
}

// PATCH /api/collections/:name  { newName }
async function handleRenameCollection(
  pathname: string,
  req: Request
): Promise<Response> {
  const oldName = decodeURIComponent(
    pathname.slice("/api/collections/".length)
  );
  const body = await req.json();
  const newName = body.newName;
  if (!newName) return errorResponse("Missing 'newName' in request body");

  try {
    const renamed = renameCollection(oldName, newName);
    if (!renamed) return errorResponse(`Collection not found: ${oldName}`, 404);
    return jsonResponse({ ok: true });
  } catch (e) {
    return errorResponse((e as Error).message);
  }
}

// GET /api/contexts
function handleGetContexts(): Response {
  return jsonResponse(listAllContexts());
}

// POST /api/contexts  { collection, path, context }
async function handleAddContext(req: Request): Promise<Response> {
  const body = await req.json();
  const { collection, path, context } = body;
  if (!context) return errorResponse("Missing 'context' in request body");

  // Global context
  if (collection === "*" || path === "/") {
    if (collection === "*") {
      setGlobalContext(context);
      return jsonResponse({ ok: true });
    }
  }

  if (!collection)
    return errorResponse("Missing 'collection' in request body");

  const ok = addContext(collection, path || "/", context);
  if (!ok) return errorResponse(`Collection not found: ${collection}`, 404);
  return jsonResponse({ ok: true });
}

// DELETE /api/contexts  { collection, path }
async function handleRemoveContext(req: Request): Promise<Response> {
  const body = await req.json();
  const { collection, path } = body;

  if (collection === "*" && path === "/") {
    setGlobalContext(undefined);
    return jsonResponse({ ok: true });
  }

  if (!collection)
    return errorResponse("Missing 'collection' in request body");
  if (!path) return errorResponse("Missing 'path' in request body");

  const ok = removeContext(collection, path);
  if (!ok) return errorResponse("Context not found", 404);
  return jsonResponse({ ok: true });
}

// =============================================================================
// Server
// =============================================================================

export async function startWebServer(port: number): Promise<void> {
  const store = createStore();
  const startTime = Date.now();

  // Resolve frontend dist directory
  const distDir = resolve(import.meta.dir, "../frontend/dist");
  const hasDistDir = existsSync(distDir);

  if (!hasDistDir) {
    console.error(
      "Warning: frontend/dist/ not found. Run 'cd frontend && bun run build' first."
    );
    console.error("API endpoints will still work.\n");
  }

  function ts(): string {
    return new Date().toISOString().slice(11, 23);
  }

  const httpServer = Bun.serve({
    port,
    hostname: "localhost",
    async fetch(req) {
      const reqStart = Date.now();
      const url = new URL(req.url);
      const pathname = url.pathname;
      const origin = req.headers.get("origin");

      // CORS preflight
      if (req.method === "OPTIONS") {
        return corsHeaders(new Response(null, { status: 204 }), origin);
      }

      let res: Response;

      try {
        // API routes
        if (pathname.startsWith("/api/")) {
          if (pathname === "/api/health" && req.method === "GET") {
            res = jsonResponse({
              status: "ok",
              uptime: Math.floor((Date.now() - startTime) / 1000),
            });
          } else if (pathname === "/api/search" && req.method === "GET") {
            res = handleSearch(store, url);
          } else if (pathname === "/api/vsearch" && req.method === "GET") {
            res = await handleVSearch(store, url);
          } else if (pathname === "/api/query" && req.method === "POST") {
            res = await handleQuery(store, req);
          } else if (
            pathname.startsWith("/api/doc/") &&
            req.method === "GET"
          ) {
            res = handleDoc(store, pathname, url);
          } else if (
            pathname === "/api/multi-get" &&
            req.method === "GET"
          ) {
            res = handleMultiGet(store, url);
          } else if (pathname === "/api/ls" && req.method === "GET") {
            res = handleLs(store, url);
          } else if (pathname === "/api/status" && req.method === "GET") {
            res = handleStatus(store);
          } else if (
            pathname === "/api/collections" &&
            req.method === "GET"
          ) {
            res = handleGetCollections();
          } else if (
            pathname.startsWith("/api/collections/") &&
            req.method === "DELETE"
          ) {
            res = handleDeleteCollection(pathname);
          } else if (
            pathname.startsWith("/api/collections/") &&
            req.method === "PATCH"
          ) {
            res = await handleRenameCollection(pathname, req);
          } else if (
            pathname === "/api/contexts" &&
            req.method === "GET"
          ) {
            res = handleGetContexts();
          } else if (
            pathname === "/api/contexts" &&
            req.method === "POST"
          ) {
            res = await handleAddContext(req);
          } else if (
            pathname === "/api/contexts" &&
            req.method === "DELETE"
          ) {
            res = await handleRemoveContext(req);
          } else {
            res = errorResponse("Not found", 404);
          }

          console.error(
            `${ts()} ${req.method} ${pathname} (${Date.now() - reqStart}ms)`
          );
          return corsHeaders(res, origin);
        }

        // Static files from frontend/dist/
        if (hasDistDir) {
          let filePath = join(distDir, pathname === "/" ? "index.html" : pathname);

          // Check if file exists
          const file = Bun.file(filePath);
          if (await file.exists()) {
            return corsHeaders(
              new Response(file, {
                headers: { "Content-Type": getMimeType(filePath) },
              }),
              origin
            );
          }

          // SPA fallback: return index.html for non-file paths
          const indexFile = Bun.file(join(distDir, "index.html"));
          if (await indexFile.exists()) {
            return corsHeaders(
              new Response(indexFile, {
                headers: { "Content-Type": "text/html" },
              }),
              origin
            );
          }
        }

        return corsHeaders(
          new Response("Not Found", { status: 404 }),
          origin
        );
      } catch (err) {
        console.error(`${ts()} ERROR ${pathname}:`, err);
        return corsHeaders(
          errorResponse((err as Error).message, 500),
          origin
        );
      }
    },
  });

  const actualPort = httpServer.port;

  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    httpServer.stop();
    store.close();
    try {
      await disposeDefaultLlamaCpp();
    } catch {
      // Ignore - may not be initialized
    }
  };

  process.on("SIGTERM", async () => {
    console.error("Shutting down (SIGTERM)...");
    await stop();
    process.exit(0);
  });
  process.on("SIGINT", async () => {
    console.error("Shutting down (SIGINT)...");
    await stop();
    process.exit(0);
  });

  console.error(`QMD web server listening on http://localhost:${actualPort}`);
  if (hasDistDir) {
    console.error(`Frontend: http://localhost:${actualPort}`);
  }
  console.error(`API: http://localhost:${actualPort}/api/`);
}
