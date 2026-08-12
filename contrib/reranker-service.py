#!/usr/bin/env python3
"""
QMD Reranker Service — Mac Studio
==================================

A lightweight HTTP service that loads Qwen3-Reranker-0.6B and serves
a /rerank endpoint for QMD's reranking step.

Runs on the Mac Studio, using Metal GPU acceleration via PyTorch MPS.
Patch QMD's store.js rerank() to call this service instead of local
llama.cpp — same pattern as the Ollama embedding hack.

Usage:
    python3 reranker-service.py [--port 8088] [--host 0.0.0.0]

Endpoints:
    GET  /health       — health check
    POST /rerank       — rerank documents against a query

POST /rerank body:
    {
        "query": "string",
        "documents": ["doc1 text", "doc2 text", ...],
        "top_n": 10  (optional, default: return all with scores)
    }

Response:
    {
        "results": [
            {"index": 0, "score": 0.95, "text": "doc1 text"},
            {"index": 1, "score": 0.72, "text": "doc2 text"},
            ...
        ]
    }

Dependencies:
    pip3 install torch transformers flask

First run downloads Qwen3-Reranker-0.6B (~600MB). Subsequent runs load from cache.

Scoring format follows the official Qwen3-Reranker model card:
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""

import json
import sys
import os
import argparse
import time
from pathlib import Path

# --- Model loading (lazy, on first request) ---
_model = None
_tokenizer = None
_device = None
_token_false_id = None
_token_true_id = None
_prefix_tokens = None
_suffix_tokens = None

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# Default instruction from the model card
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Fixed prefix/suffix as specified by the model card
# prefix: <|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n
# suffix: <|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n\n
# Note: the suffix uses the actual special tokens, not string literals
PREFIX_TEXT = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX_TEXT = '<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n\n'

MAX_LENGTH = 8192


def get_device():
    """Detect best available device: MPS (Apple Silicon) > CPU."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    """Load the reranker model and tokenizer, pre-resolve token IDs and prefix/suffix."""
    global _model, _tokenizer, _device, _token_false_id, _token_true_id, _prefix_tokens, _suffix_tokens
    if _model is not None:
        return _model, _tokenizer, _device

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = get_device()
    print(f"[reranker] Loading {MODEL_NAME} on {_device}...", flush=True)

    start = time.time()
    # CRITICAL: padding_side='left' so logits[:, -1, :] is the real last token
    # for every row in a padded batch, not a PAD token.
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )
    # Use eager attention + float32 on MPS to avoid a PyTorch MPS bug where
    # the default SDPA attention produces NaN logits for padded batch rows
    # when running in float16. Eager attention + float32 is slightly slower
    # but produces correct results for all batch sizes.
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    _model = _model.to(_device)
    _model.eval()

    # Resolve yes/no token IDs once at load time (lowercase, per model card)
    _token_false_id = _tokenizer.convert_tokens_to_ids("no")
    _token_true_id = _tokenizer.convert_tokens_to_ids("yes")

    # Pre-encode the fixed prefix and suffix tokens
    _prefix_tokens = _tokenizer.encode(PREFIX_TEXT, add_special_tokens=False)
    _suffix_tokens = _tokenizer.encode(SUFFIX_TEXT, add_special_tokens=False)

    elapsed = time.time() - start
    print(f"[reranker] Model loaded in {elapsed:.1f}s", flush=True)
    print(f"[reranker] token_false_id (no) = {_token_false_id}", flush=True)
    print(f"[reranker] token_true_id (yes) = {_token_true_id}", flush=True)
    print(f"[reranker] prefix_tokens length = {len(_prefix_tokens)}", flush=True)
    print(f"[reranker] suffix_tokens length = {len(_suffix_tokens)}", flush=True)
    return _model, _tokenizer, _device


def format_instruction(instruction, query, doc):
    """Format a (query, doc) pair per the model card template."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def process_inputs(pairs, tokenizer, model, prefix_tokens, suffix_tokens,
                   token_false_id, token_true_id, device, max_length=MAX_LENGTH):
    """
    Tokenize and prepare a batch of (query, doc) pairs for scoring.

    Follows the model card reference:
    1. Tokenize each pair (without padding) with truncation leaving room for prefix+suffix
    2. Prepend prefix_tokens and append suffix_tokens to each sequence
    3. Left-pad the batch to the longest sequence
    4. Move to device
    """
    import torch

    # Tokenize all pairs without padding, with truncation
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )

    # Prepend prefix and append suffix to each sequence
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens

    # Left-pad the batch
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

    # Zero out pad token positions in input_ids to avoid NaN on MPS.
    # The attention_mask already excludes these from attention, but some
    # embedding layers still produce NaN for the pad token id (151643) on
    # Apple Silicon MPS. Replacing pad positions with 0 is safe because
    # the attention mask ensures they're never attended to.
    if 'attention_mask' in inputs:
        pad_mask = inputs['attention_mask'] == 0
        if pad_mask.any():
            inputs['input_ids'][pad_mask] = 0

    # Move to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    return inputs


def compute_logits(inputs, model, token_false_id, token_true_id):
    """
    Compute relevance scores from model logits.

    Follows the model card reference:
    1. Get last-position logits: logits[:, -1, :] (correct with left padding)
    2. Extract yes and no logit vectors
    3. Stack [no, yes] and apply log_softmax
    4. Score = exp of the yes component (index 1)
    """
    import torch

    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def compute_scores_batch(query, documents, batch_size=8, instruction=None):
    """
    Compute relevance scores for (query, document) pairs using the
    Qwen3-Reranker cross-encoder, following the official model card format.

    Uses left padding so logits[:, -1, :] is the real last token for every
    row in a padded batch. Token IDs for "yes"/"no" are resolved once at
    model load time.

    Args:
        query: The search query string
        documents: List of document text strings
        batch_size: Number of documents per batch (default 8)
        instruction: Custom instruction string (default: model card default)

    Returns:
        List of float scores (0-1 range, higher = more relevant)
    """
    import torch

    model, tokenizer, device = load_model()

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    all_scores = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        pairs = [format_instruction(instruction, query, doc) for doc in batch]

        inputs = process_inputs(
            pairs, tokenizer, model, _prefix_tokens, _suffix_tokens,
            _token_false_id, _token_true_id, device
        )

        with torch.no_grad():
            scores = compute_logits(inputs, model, _token_false_id, _token_true_id)

        all_scores.extend(scores)

    return all_scores


def compute_scores(query, documents, instruction=None):
    """
    Compute scores one document at a time (batch_size=1).

    This is now a thin wrapper over compute_scores_batch with batch_size=1,
    ensuring there is exactly one scoring code path.
    """
    return compute_scores_batch(query, documents, batch_size=1, instruction=instruction)


# --- HTTP server ---
def create_app():
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model": MODEL_NAME,
            "loaded": _model is not None,
            "device": str(_device) if _device else "not loaded yet",
            "format": "qwen3-reranker-v2"
        })

    @app.route("/rerank", methods=["POST"])
    def rerank():
        try:
            data = request.get_json(force=True)
            query = data.get("query", "")
            documents = data.get("documents", [])
            top_n = data.get("top_n", len(documents))
            instruction = data.get("instruction", None)
            batch_size = data.get("batch_size", 8)

            if not query or not documents:
                return jsonify({"error": "query and documents are required"}), 400

            # Use batch scoring for efficiency
            scores = compute_scores_batch(query, documents, batch_size=batch_size,
                                          instruction=instruction)

            # Sort by score descending
            results = [
                {"index": i, "score": float(scores[i]), "text": documents[i][:200]}
                for i in range(len(documents))
            ]
            results.sort(key=lambda x: x["score"], reverse=True)

            # Apply top_n
            if top_n < len(results):
                results = results[:top_n]

            return jsonify({"results": results})

        except Exception as e:
            print(f"[reranker] Error: {e}", file=sys.stderr, flush=True)
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="QMD Reranker Service for Mac Studio")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on (default: 8088)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"[reranker] Starting QMD Reranker Service on {args.host}:{args.port}")
    print(f"[reranker] Model: {MODEL_NAME}")
    print(f"[reranker] Format: qwen3-reranker-v2 (official model card)")
    print(f"[reranker] Endpoints: GET /health, POST /rerank")
    print()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()