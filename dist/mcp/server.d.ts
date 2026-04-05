/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Follows MCP spec 2025-06-18 for proper response types.
 */
export declare function startMcpServer(): Promise<void>;
export type HttpServerHandle = {
    httpServer: import("http").Server;
    port: number;
    stop: () => Promise<void>;
};
/**
 * Start MCP server over Streamable HTTP (JSON responses, no SSE).
 * Binds to localhost only. Returns a handle for shutdown and port discovery.
 */
export declare function startMcpHttpServer(port: number, options?: {
    quiet?: boolean;
}): Promise<HttpServerHandle>;
