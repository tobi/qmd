# ---- Stage 1: Build ----
FROM node:22 AS build

# Install build tools for native modules (better-sqlite3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Enable pnpm via corepack
RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY package.json pnpm-lock.yaml ./

# Install dependencies (including devDependencies for build)
RUN pnpm install --frozen-lockfile

# Copy source and build
COPY tsconfig.json tsconfig.build.json ./
COPY src/ src/
COPY bin/ bin/

RUN pnpm build

# ---- Stage 2: Runtime ----
FROM node:22-slim

# Install runtime libs needed by better-sqlite3 native binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built output and node_modules from build stage
COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./

# Use the built-in non-root user
USER node

# MCP HTTP port
EXPOSE 8181

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "fetch('http://localhost:8181/').catch(()=>process.exit(1))"

# Default command — can be overridden by Helm
CMD ["node", "dist/cli/qmd.js", "mcp", "--http", "--port", "8181"]
