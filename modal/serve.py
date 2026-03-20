"""QMD Modal Inference Service.

Deploys three GGUF models (embedding, reranking, query expansion) on a Modal
GPU container using llama-server (the llama.cpp HTTP server) and exposes raw
inference methods via Modal's RPC.

Each model runs as a separate llama-server subprocess on its own port:
  - Port 8081: embeddinggemma  (embedding, --pooling mean)
  - Port 8082: qmd-query-expansion (completion)
  - Port 8083: qwen3-reranker (reranking, --pooling rank)

Manual test instructions:
    # 1. Authenticate with Modal (one-time setup)
    modal token set

    # 2. Deploy the service
    python modal/serve.py deploy --gpu T4 --scaledown-window 15

    # 3. Check status
    python modal/serve.py status

    # 4. Cleanup when done
    python modal/serve.py destroy
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import modal

# ---------------------------------------------------------------------------
# Section A: Image definition
# ---------------------------------------------------------------------------

MODELS_DIR = "/models"
LLAMA_SERVER_BIN = "/usr/local/bin/llama-server"

MODELS = [
    {
        "repo_id": "ggml-org/embeddinggemma-300M-GGUF",
        "filename": "embeddinggemma-300M-Q8_0.gguf",
    },
    {
        "repo_id": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "filename": "qwen3-reranker-0.6b-q8_0.gguf",
    },
    {
        "repo_id": "tobil/qmd-query-expansion-1.7B-gguf",
        "filename": "qmd-query-expansion-1.7B-q4_k_m.gguf",
    },
]


def download_models() -> None:
    """Download all GGUF models into /models/ during image build."""
    from huggingface_hub import hf_hub_download

    for model in MODELS:
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=MODELS_DIR,
        )


image = (
    modal.Image.from_registry(
        # Pre-built llama-server with CUDA 12.4 support.
        # Built from llama.cpp b8179 (same as node-llama-cpp v3.17.1).
        # Image built by .github/workflows/build-llama-server.yml
        "ghcr.io/ofekby/qmd-llama-server:b8179-sm75",
        add_python="3.11",
    )
    .pip_install("huggingface-hub", "requests")
    .run_function(download_models)
)

app = modal.App("qmd-inference", image=image)


# ---------------------------------------------------------------------------
# Section B: Server configuration
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Configuration for a single llama-server instance."""

    name: str
    port: int
    model_file: str
    extra_args: list[str]


EMBED_SERVER = ServerConfig(
    name="embed",
    port=8081,
    model_file="embeddinggemma-300M-Q8_0.gguf",
    extra_args=["--embedding", "--pooling", "mean", "--ubatch-size", "2048"],
)

EXPAND_SERVER = ServerConfig(
    name="expand",
    port=8082,
    model_file="qmd-query-expansion-1.7B-q4_k_m.gguf",
    extra_args=[],
)

RERANK_SERVER = ServerConfig(
    name="rerank",
    port=8083,
    model_file="qwen3-reranker-0.6b-q8_0.gguf",
    extra_args=["--embedding", "--pooling", "rank", "--ubatch-size", "2048"],
)

ALL_SERVERS = [EMBED_SERVER, EXPAND_SERVER, RERANK_SERVER]


# ---------------------------------------------------------------------------
# Section C: QMDInference class
# ---------------------------------------------------------------------------

gpu_config: str = os.environ.get("QMD_MODAL_GPU", "T4")
idle_timeout: int = int(os.environ.get("QMD_MODAL_SCALEDOWN", "15"))


@app.cls(
    gpu=gpu_config,
    scaledown_window=idle_timeout,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)
class QMDInference:
    """Raw inference service for QMD's three GGUF models.

    Runs three llama-server subprocesses (one per model) and proxies
    requests to them via HTTP.
    """

    @modal.enter(snap=True)
    def start_servers(self) -> None:
        """Start all llama-server instances and wait until healthy.

        Uses ``snap=True`` so the running subprocesses are captured in a
        memory snapshot.  On restore the servers are already running with
        models loaded.
        """
        import subprocess
        import time

        import requests

        self._processes: list[subprocess.Popen[bytes]] = []

        for server in ALL_SERVERS:
            cmd = [
                LLAMA_SERVER_BIN,
                "--model",
                f"{MODELS_DIR}/{server.model_file}",
                "--port",
                str(server.port),
                "--ctx-size",
                "2048",
                "--n-gpu-layers",
                "99",
                *server.extra_args,
            ]
            proc = subprocess.Popen(cmd)
            self._processes.append(proc)

        # Wait for all servers to become healthy
        for server in ALL_SERVERS:
            url = f"http://127.0.0.1:{server.port}/health"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    resp = requests.get(url, timeout=2)
                    if resp.status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"llama-server '{server.name}' on port {server.port} "
                    f"did not become healthy within 120s"
                )

        # Warmup passes to pre-fill caches before snapshot
        requests.post(
            f"http://127.0.0.1:{EMBED_SERVER.port}/embedding",
            json={"content": "warmup"},
            timeout=30,
        )
        requests.post(
            f"http://127.0.0.1:{EXPAND_SERVER.port}/completion",
            json={"prompt": "warmup", "n_predict": 1},
            timeout=30,
        )
        requests.post(
            f"http://127.0.0.1:{RERANK_SERVER.port}/reranking",
            json={
                "query": "warmup",
                "documents": ["warmup"],
            },
            timeout=30,
        )

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raw embedding -- no prompt formatting.

        Sends all texts in a single batch request to llama-server's
        /embedding endpoint for minimal round-trip overhead.
        Returns one embedding vector per input text.
        """
        import requests

        resp = requests.post(
            f"http://127.0.0.1:{EMBED_SERVER.port}/embedding",
            json={"content": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        result: list[list[float]] = []
        for entry in data:
            emb = entry["embedding"]
            # Flatten if nested [[...floats...]]
            if isinstance(emb[0], list):
                emb = emb[0]
            result.append(emb)
        return result

    @modal.method()
    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Tokenize texts using the embedding model's tokenizer.

        Proxies to llama-server's /tokenize endpoint on port 8081
        (the embeddinggemma model). Returns token IDs as nested int lists.
        """
        import requests

        resp = requests.post(
            f"http://127.0.0.1:{EMBED_SERVER.port}/tokenize",
            json={"content": texts},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @modal.method()
    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text using the embedding model's tokenizer.

        Proxies to llama-server's /detokenize endpoint on port 8081
        (the embeddinggemma model).
        """
        import requests

        resp = requests.post(
            f"http://127.0.0.1:{EMBED_SERVER.port}/detokenize",
            json={"tokens": tokens},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")

    @modal.method()
    def generate(
        self,
        prompt: str,
        grammar: str | None,
        max_tokens: int,
        model: str = "expand",
    ) -> str:
        """Raw completion with optional GBNF grammar constraint.

        The caller must construct the full prompt including any special
        tokens (``<|im_start|>``, ``<|im_end|>``, etc.).  This method does
        **not** apply any chat template.

        Args:
            prompt: Fully formatted prompt string.
            grammar: Optional GBNF grammar string for constrained generation.
            max_tokens: Maximum number of tokens to generate.
            model: Which model to use -- ``"expand"`` (default) for query
                   expansion, ``"rerank"`` for the reranker model.
        """
        import requests

        port = RERANK_SERVER.port if model == "rerank" else EXPAND_SERVER.port
        payload: dict = {"prompt": prompt, "n_predict": max_tokens}
        if grammar:
            payload["grammar"] = grammar
        resp = requests.post(
            f"http://127.0.0.1:{port}/completion",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"]

    @modal.method()
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder scoring using Qwen3-Reranker.

        Uses llama-server's native ``/reranking`` endpoint (the server runs
        with ``--pooling rank``), which applies the model's reranking logic
        directly.

        Returns a list of relevance scores, one per input text.
        """
        import requests

        resp = requests.post(
            f"http://127.0.0.1:{RERANK_SERVER.port}/reranking",
            json={"query": query, "documents": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response contains "results" sorted by index, each with
        # "index" and "relevance_score".  Return scores in the
        # original document order.
        results = sorted(data["results"], key=lambda r: r["index"])
        return [r["relevance_score"] for r in results]

    @modal.method()
    def ping(self) -> bool:
        """Health check -- verifies function is reachable and servers running."""
        import requests

        for server in ALL_SERVERS:
            resp = requests.get(
                f"http://127.0.0.1:{server.port}/health",
                timeout=5,
            )
            if resp.status_code != 200:
                return False
        return True


# ---------------------------------------------------------------------------
# Section D: CLI entry point
# ---------------------------------------------------------------------------


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy the QMD inference service to Modal."""
    import subprocess

    print(
        f"Deploying qmd-inference "
        f"(gpu={args.gpu}, scaledown_window={args.scaledown_window})..."
    )
    env = {
        **os.environ,
        "QMD_MODAL_GPU": args.gpu,
        "QMD_MODAL_SCALEDOWN": str(args.scaledown_window),
    }
    subprocess.run(
        [sys.executable, "-m", "modal", "deploy", __file__],
        check=True,
        env=env,
    )
    print(
        f"Deployed successfully. GPU: {args.gpu} "
        f"(~$0.59/hr, billed per second, scales to zero when idle)"
    )


def cmd_status(_args: argparse.Namespace) -> None:
    """Check if the qmd-inference app is deployed."""
    try:
        fn = modal.Function.from_name("qmd-inference", "QMDInference.ping")
        result = fn.remote()
        if result:
            print("qmd-inference is deployed and responding.")
        else:
            print("qmd-inference is deployed but ping returned unexpected result.")
    except modal.exception.NotFoundError:
        print("qmd-inference is not deployed. Run: python modal/serve.py deploy")
        sys.exit(1)
    except Exception as exc:
        print(f"Error checking status: {exc}")
        sys.exit(1)


def cmd_destroy(_args: argparse.Namespace) -> None:
    """Tear down the deployed qmd-inference app."""
    import subprocess

    print("Stopping qmd-inference...")
    subprocess.run(
        [sys.executable, "-m", "modal", "app", "stop", "qmd-inference"],
        check=True,
    )
    print("qmd-inference stopped successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QMD Modal Inference Service",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy the inference service")
    deploy_parser.add_argument(
        "--gpu",
        default="T4",
        help="GPU type for the Modal container (default: T4)",
    )
    deploy_parser.add_argument(
        "--scaledown-window",
        type=int,
        default=15,
        help="Seconds before idle container shuts down (default: 15)",
    )

    subparsers.add_parser("status", help="Check deployment status")
    subparsers.add_parser("destroy", help="Tear down the deployed service")

    args = parser.parse_args()

    commands = {
        "deploy": cmd_deploy,
        "status": cmd_status,
        "destroy": cmd_destroy,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
