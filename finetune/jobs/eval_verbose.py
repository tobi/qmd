# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub>=0.20.0",
#     "accelerate",
# ]
# ///
"""
Verbose eval: prints the actual expansions for every query.

    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 30m jobs/eval_verbose.py
"""

import os
import re
import sys
from collections import Counter

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
GRPO_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"

QUERIES = [
    "how to configure authentication",
    "typescript async await",
    "docker compose networking",
    "git rebase vs merge",
    "react useEffect cleanup",
    "auth",
    "config",
    "setup",
    "api",
    "who is TDS motorsports",
    "React hooks tutorial",
    "Docker container networking",
    "Kubernetes pod deployment",
    "AWS Lambda functions",
    "meeting notes project kickoff",
    "ideas for new feature",
    "todo list app architecture",
    "what is dependency injection",
    "difference between sql and nosql",
    "kubernetes vs docker swarm",
    "connection timeout error",
    "memory leak debugging",
    "cors error fix",
    "recent news about Shopify",
    "latest AI developments",
    "best laptops right now",
    "what changed in kubernetes latest version",
    "how to implement caching with redis in nodejs",
    "best practices for api rate limiting",
    "setting up ci cd pipeline with github actions",
]


def load_model(base, sft=None, grpo=None):
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    if sft:
        model = PeftModel.from_pretrained(model, sft)
        model = model.merge_and_unload()
    if grpo:
        model = PeftModel.from_pretrained(model, grpo)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, query):
    messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "\nassistant\n" in text:
        text = text.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in text:
        text = text.split("assistant\n")[-1].strip()
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading GRPO model...", file=sys.stderr)
    model, tokenizer = load_model(BASE_MODEL, sft=SFT_MODEL, grpo=GRPO_MODEL)

    for i, query in enumerate(QUERIES, 1):
        expansion = generate(model, tokenizer, query)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(QUERIES)}] {query}")
        print(f"{'─'*60}")
        print(expansion)


if __name__ == "__main__":
    main()
