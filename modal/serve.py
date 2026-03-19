"""QMD Modal Inference Service.

Deploys three GGUF models (embedding, reranking, query expansion) on a Modal
GPU container and exposes raw inference methods via Modal's RPC.

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
import math
import os
import sys

import modal

# ---------------------------------------------------------------------------
# Section A: Image definition
# ---------------------------------------------------------------------------

MODELS_DIR = "/models"

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
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    .pip_install("huggingface-hub")
    .run_function(download_models)
)

app = modal.App("qmd-inference", image=image)


# ---------------------------------------------------------------------------
# Section B: QMDInference class
# ---------------------------------------------------------------------------

# Read GPU/scaledown config from environment when the module is imported
# during ``modal deploy``.  The CLI deploy command passes these via env vars
# so the decorator picks up the configured values at import time.
gpu_config: str = os.environ.get("QMD_MODAL_GPU", "T4")
idle_timeout: int = int(os.environ.get("QMD_MODAL_SCALEDOWN", "15"))


@app.cls(
    gpu=gpu_config,
    scaledown_window=idle_timeout,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=4)
class QMDInference:
    """Raw inference service for QMD's three GGUF models."""

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load all 3 models and run warmup passes.

        The ``snap=True`` flag captures the loaded model state in a memory
        snapshot so subsequent cold starts restore from the snapshot instead
        of re-loading ~2 GB of weights.
        """
        from llama_cpp import Llama

        self.embed_model = Llama(
            model_path=f"{MODELS_DIR}/embeddinggemma-300M-Q8_0.gguf",
            embedding=True,
            n_ctx=2048,
        )
        self.rerank_model = Llama(
            model_path=f"{MODELS_DIR}/qwen3-reranker-0.6b-q8_0.gguf",
            n_ctx=2048,
        )
        self.expand_model = Llama(
            model_path=f"{MODELS_DIR}/qmd-query-expansion-1.7B-q4_k_m.gguf",
            n_ctx=2048,
        )

        # Warmup passes to pre-fill caches before snapshot
        self.embed_model.embed("warmup")
        self.rerank_model("warmup", max_tokens=1)
        self.expand_model("warmup", max_tokens=1)

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raw embedding -- no prompt formatting.

        Returns one embedding vector per input text.  llama-cpp-python's
        ``Llama.embed()`` returns ``list[list[float]]``; we normalise to
        always yield a single vector per input.
        """
        result: list[list[float]] = []
        for text in texts:
            vec = self.embed_model.embed(text)
            # Llama.embed() may return list[list[float]] or list[float]
            # depending on input; normalise to a single vector per text.
            if isinstance(vec[0], list):
                result.append(vec[0])
            else:
                result.append(vec)
        return result

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
        llm = self.rerank_model if model == "rerank" else self.expand_model
        kwargs: dict = {"prompt": prompt, "max_tokens": max_tokens}
        if grammar:
            from llama_cpp import LlamaGrammar

            kwargs["grammar"] = LlamaGrammar.from_string(grammar)
        result = llm(**kwargs)
        return result["choices"][0]["text"]

    @modal.method()
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder scoring using Qwen3-Reranker.

        Uses ``create_chat_completion()`` which applies the model's native
        chat template automatically (analogous to node-llama-cpp's
        ``rankAll()``).

        Returns a list of relevance scores (0--1), one per input text.
        """
        scores: list[float] = []
        for text in texts:
            response = self.rerank_model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Judge whether the Document meets the "
                            "requirements of the Query. Note that the "
                            'answer can only be "yes" or "no".'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"<Query>{query}</Query>\n"
                            f"<Document>{text}</Document>"
                        ),
                    },
                ],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
            )
            score = _extract_yes_probability(response)
            scores.append(score)
        return scores

    @modal.method()
    def ping(self) -> bool:
        """Health check -- verifies function is reachable and models loaded."""
        return True


def _extract_yes_probability(response: dict) -> float:
    """Extract the probability of the 'yes' token from chat completion logprobs.

    This is the standard Qwen3-Reranker scoring approach: the model is asked
    to judge relevance with a yes/no answer, and we use the probability of
    'yes' as the relevance score.
    """
    logprobs_data = response["choices"][0]["logprobs"]["content"][0][
        "top_logprobs"
    ]
    for lp in logprobs_data:
        if lp["token"].lower().strip() == "yes":
            return math.exp(lp["logprob"])
    return 0.0


# ---------------------------------------------------------------------------
# Section C: CLI entry point
# ---------------------------------------------------------------------------


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy the QMD inference service to Modal."""
    import subprocess

    print(
        f"Deploying qmd-inference "
        f"(gpu={args.gpu}, scaledown_window={args.scaledown_window})..."
    )
    # Shell out to ``modal deploy`` with env vars so the module re-imports
    # with the configured GPU and scaledown window values.
    env = {**os.environ, "QMD_MODAL_GPU": args.gpu, "QMD_MODAL_SCALEDOWN": str(args.scaledown_window)}
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
            print(
                "qmd-inference is deployed but ping returned unexpected result."
            )
    except modal.exception.NotFoundError:
        print(
            "qmd-inference is not deployed. "
            "Run: python modal/serve.py deploy"
        )
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

    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy the inference service"
    )
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
