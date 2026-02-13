# Stage 1: Builder
FROM oven/bun:1 AS builder

WORKDIR /app

# Install system dependencies required to compile native modules (node-llama-cpp)
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY package.json bun.lock tsconfig.json ./

# Install dependencies (including compiling native modules)
RUN bun install --frozen-lockfile

# Copy source code
COPY src ./src
COPY qmd ./qmd

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install dependencies required for runtime
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Bun in the runtime image
COPY --from=builder /usr/local/bin/bun /usr/local/bin/bun

# Copy node_modules and compiled code from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/tsconfig.json ./tsconfig.json
COPY --from=builder /app/qmd ./qmd

# Create directories for volumes
RUN mkdir -p /vault /data

# Environment variables for QMD
# Set HOME to /data so that models are saved in the volume
ENV XDG_CACHE_HOME=/data/.cache
ENV HOME=/data
ENV PATH="/app:${PATH}"

# Expose data and vault volumes
VOLUME ["/vault", "/data"]

# Keep the container alive waiting for commands
# The user is defined in docker-compose.yml
CMD ["tail", "-f", "/dev/null"]
