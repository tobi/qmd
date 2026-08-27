# GPU Acceleration

QMD runs its embedding, query-expansion, and reranking models locally through
[`node-llama-cpp`](https://node-llama-cpp.withcat.ai) (llama.cpp with GGUF
models). On CPU these models work but are slow; offloading them to a GPU makes
`qmd vsearch` and `qmd query` several times faster.

This guide explains how QMD picks a backend, how to build a GPU-enabled
`node-llama-cpp` binary when the packaged prebuilt is CPU-only, and how to enable
the Apple GPU on Linux (Asahi) via Vulkan.

## TL;DR

```sh
# 1. See what QMD is currently using
qmd doctor            # look for the "device probe" line

# 2. If it says "running on CPU", build a GPU-enabled llama.cpp binary
#    (pick the backend for your machine: metal | cuda | vulkan)
npx --no node-llama-cpp source download --gpu vulkan

# 3. Select the backend and verify
export QMD_LLAMA_GPU=vulkan
qmd doctor            # should now report "GPU <backend>; offloading enabled"
```

## How QMD selects a backend

Backend selection is controlled by two environment variables (also listed in the
main README):

| Variable | Default | Effect |
|----------|---------|--------|
| `QMD_LLAMA_GPU` | `auto` | Force a llama.cpp backend (`metal`, `cuda`, `vulkan`) or disable the GPU with `false`/`off`/`0`. |
| `QMD_FORCE_CPU` | unset | Set to `1`/`true` to force CPU before any GPU probing. Equivalent CLI flag: `--no-gpu`. |

In `auto` mode QMD asks `node-llama-cpp` for the best available backend (Metal on
Apple Silicon macOS, CUDA when fully available, then Vulkan, then CPU). If GPU
init fails at runtime, QMD logs a warning and **falls back to CPU** so search
keeps working.

> **Important:** auto-detection can only pick a GPU backend that is *compiled
> into the `node-llama-cpp` binary QMD loads*. The packaged prebuilt binaries are
> CPU-only on some platforms (notably Linux arm64). On those platforms `auto`
> will always resolve to CPU until you build a GPU-enabled binary — see below.

## Building a GPU-enabled binary

`node-llama-cpp` can download the llama.cpp source and compile it with a specific
compute backend. Run this once; the local build is reused afterwards:

```sh
# From anywhere; resolves the node-llama-cpp that QMD uses.
npx --no node-llama-cpp source download --gpu vulkan   # or: metal | cuda
```

If you installed QMD globally and `npx` does not resolve the right copy, run the
CLI from inside QMD's dependency directly:

```sh
QMD_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v qmd)")")")"
cd "$QMD_DIR/node_modules/node-llama-cpp"
node ./dist/cli/cli.js source download --gpu vulkan
```

Prerequisites and per-backend notes live in the node-llama-cpp docs:
[Metal](https://node-llama-cpp.withcat.ai/guide/Metal),
[CUDA](https://node-llama-cpp.withcat.ai/guide/CUDA),
[Vulkan](https://node-llama-cpp.withcat.ai/guide/Vulkan).

After the build finishes, set `QMD_LLAMA_GPU` to the backend you built and run
`qmd doctor` to confirm.

## Apple Silicon on Linux (Asahi) — Vulkan

On macOS, Apple Silicon uses the **Metal** backend automatically. Under **Asahi
Linux**, Metal / MLX / CoreML and the Neural Engine are not available, but the
Apple GPU is exposed to **Vulkan** through Mesa's Honeykrisp driver, and
llama.cpp has a Vulkan backend. That path works well for QMD's small models.

### 1. Install build + Vulkan dependencies

The packaged Linux arm64 prebuilt is CPU-only, so a source build is required.

**Arch Linux / Asahi (pacman):**

```sh
sudo pacman -S --needed cmake ninja vulkan-headers vulkan-icd-loader \
                        shaderc spirv-headers spirv-tools vulkan-tools
# The Apple GPU Vulkan driver (Honeykrisp) ships with mesa on Asahi.
```

> **Gotcha:** `spirv-headers` is easy to miss and is *not* the same package as
> `spirv-tools`. Without it, the Vulkan compile fails at configure time with:
> `Could not find a package configuration file provided by "SPIRV-Headers"`.

Other distributions: install the equivalent CMake, Ninja, glslc/shaderc, and
Vulkan headers/loader/SPIRV-Headers packages, or the LunarG Vulkan SDK as
described in the node-llama-cpp Vulkan guide.

### 2. Confirm the GPU is visible to Vulkan

```sh
vulkaninfo --summary | grep -iE 'deviceName|driverName|apiVersion'
# Example on an M2 Pro:
#   deviceName = Apple M2 Pro (G14S B1)
#   driverName = Honeykrisp
#   apiVersion = 1.4.x
```

### 3. Build the Vulkan binary and enable it

```sh
npx --no node-llama-cpp source download --gpu vulkan
export QMD_LLAMA_GPU=vulkan
qmd doctor
```

A working setup reports something like:

```
✓ device mode: vulkan
✓ device probe: GPU vulkan; offloading enabled; devices: Apple M2 Pro (G14S B1);
  VRAM 15.6 GB free / 15.6 GB total; 10 CPU math cores
```

To make it permanent, export `QMD_LLAMA_GPU=vulkan` from your shell profile (or,
if you launch QMD via a version manager such as mise, from its `[env]` block).

### Measured impact

On a MacBook Pro 14" (M2 Pro) running Asahi Linux, over a small notes collection,
switching from CPU to the Vulkan backend:

| Operation | CPU | Vulkan (Apple GPU) | Speedup |
|-----------|-----|--------------------|---------|
| `qmd vsearch` (embedding) | ~5.2 s | ~2.2 s | ~2.4× |
| `qmd query` (expand + embed + rerank) | ~14.1 s | ~2.4 s | ~5.9× |

Beyond wall-clock time, GPU offload frees the CPU: the deep pipeline dropped from
~64 s of CPU time (all cores saturated) to ~2.3 s, keeping the machine
responsive during search.

## Verifying and troubleshooting

- **Check the current backend:** `qmd doctor` and read the `device probe` line.
- **Force CPU** (to compare, or if a GPU driver is flaky): `QMD_FORCE_CPU=1 qmd doctor`.
- **`auto` still shows CPU:** the loaded `node-llama-cpp` binary has no GPU
  backend compiled in — build one with `node-llama-cpp source download --gpu ...`.
- **GPU init fails at runtime:** QMD logs a warning and falls back to CPU, so
  search still works; re-check driver/SDK install and the build backend.
- **Rebuild after upgrades:** the local llama.cpp build lives inside the
  `node-llama-cpp` install. If you reinstall QMD or change the Node.js version it
  runs under, re-run the `source download` step to rebuild the GPU binary.
