#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for MLX - Apple Silicon Edition

Usage:
    python grpo.py                    # Run GRPO training
    python grpo.py --steps 100        # Custom steps
    python grpo.py --eval-only        # Just evaluate current model
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from reward import score_expansion_detailed

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "base_model": "models/Qwen3-1.7B-mlx",
    "sft_adapter": "adapters/qwen3-3500",
    "output_adapter": "adapters/qwen3-grpo",
    
    "num_generations": 4,
    "max_tokens": 200,
    "beta": 0.04,
    "learning_rate": 5e-6,
    "max_steps": 200,
    
    "log_every": 5,
    "save_every": 50,
    "eval_every": 50,
}

TRAINING_QUERIES = [
    "auth config", "how to deploy", "rate limiting", "database connection",
    "api keys", "error handling", "caching strategy", "user permissions",
    "typescript async await", "docker compose networking", "git rebase vs merge",
    "react useEffect cleanup", "kubernetes pod deployment", "AWS Lambda functions",
    "memory leak debugging", "cors error fix", "connection timeout error",
    "dependency injection", "sql vs nosql", "ci cd pipeline",
]

PROMPT_TEMPLATE = """<|im_start|>user
/no_think Expand this search query: {query}<|im_end|>
<|im_start|>assistant
"""


def generate_completion(model, tokenizer, query, temp=0.7):
    """Generate a single completion."""
    prompt = PROMPT_TEMPLATE.format(query=query)
    sampler = make_sampler(temp=temp, top_p=0.9)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=200, sampler=sampler, verbose=False)
    return prompt, response.replace('<|im_end|>', '').strip()


def compute_reward(query, completion):
    """Score completion using reward function."""
    result = score_expansion_detailed(query, completion)
    return result['total'] / 140.0  # Normalize to [0, 1]


def compute_log_prob(model, tokenizer, prompt, completion):
    """Compute log probability of completion given prompt."""
    full_text = prompt + completion
    tokens = mx.array(tokenizer.encode(full_text))
    prompt_len = len(tokenizer.encode(prompt))
    
    # Forward pass
    logits = model(tokens[None, :-1])[0]  # [seq_len, vocab_size]
    
    # Get completion log probs
    log_probs = nn.log_softmax(logits[prompt_len-1:], axis=-1)
    target_tokens = tokens[prompt_len:]
    
    # Gather
    token_log_probs = mx.take_along_axis(
        log_probs[:len(target_tokens)],
        target_tokens[:, None],
        axis=-1
    ).squeeze(-1)
    
    return mx.sum(token_log_probs)


def grpo_step(policy_model, ref_model, tokenizer, query, config, optimizer):
    """Single GRPO training step."""
    prompt = PROMPT_TEMPLATE.format(query=query)
    
    # 1. Generate completions with varying temperatures (1.0 to 2.0 for diversity)
    completions = []
    for i in range(config["num_generations"]):
        temp = 1.0 + 1.0 * i / max(1, config["num_generations"] - 1)  # 1.0 to 2.0
        _, comp = generate_completion(policy_model, tokenizer, query, temp=temp)
        completions.append(comp)
    
    # 2. Compute rewards
    rewards = [compute_reward(query, c) for c in completions]
    mean_reward = sum(rewards) / len(rewards)
    
    # 3. Compute advantages (group relative)
    std_reward = math.sqrt(sum((r - mean_reward)**2 for r in rewards) / len(rewards) + 1e-8)
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    
    # 4. Compute reference log probs (frozen)
    ref_log_probs = []
    for comp in completions:
        ref_lp = compute_log_prob(ref_model, tokenizer, prompt, comp)
        mx.eval(ref_lp)
        ref_log_probs.append(ref_lp)
    
    # 5. Define loss function for this batch
    def loss_fn(model):
        total_loss = mx.array(0.0)
        total_kl = mx.array(0.0)
        
        for comp, adv, ref_lp in zip(completions, advantages, ref_log_probs):
            # Always compute even with small advantages
            policy_lp = compute_log_prob(model, tokenizer, prompt, comp)
            kl = policy_lp - ref_lp
            pg_loss = -mx.array(adv) * policy_lp
            
            total_loss = total_loss + pg_loss + config["beta"] * mx.abs(kl)
            total_kl = total_kl + kl
        
        n = len(completions)
        return total_loss / n, total_kl / n
    
    # 6. Compute gradients using nn.value_and_grad
    loss_grad_fn = nn.value_and_grad(policy_model, loss_fn)
    (loss, kl), grads = loss_grad_fn(policy_model)
    
    # 7. Update
    optimizer.update(policy_model, grads)
    mx.eval(policy_model.parameters())
    
    return float(loss), float(kl), mean_reward


def save_lora_weights(model, output_dir):
    """Save LoRA weights to directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / "adapters.safetensors"))


def evaluate(model, tokenizer, queries_file="evals/queries.txt"):
    """Run evaluation."""
    queries = []
    with open(queries_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)
    
    greedy = make_sampler(temp=0.0)
    scores = []
    for q in queries:
        prompt = PROMPT_TEMPLATE.format(query=q)
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=200, sampler=greedy, verbose=False)
        resp = resp.replace('<|im_end|>', '').strip()
        result = score_expansion_detailed(q, resp)
        scores.append(result['total'])
    
    return {
        "avg": sum(scores) / len(scores),
        "perfect": sum(1 for s in scores if s >= 100),
        "total": len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=CONFIG["max_steps"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter", type=str)
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config["max_steps"] = args.steps
    
    print("=" * 60)
    print("GRPO Training - QMD Query Expansion (MLX)")
    print("=" * 60)
    
    print(f"\n📥 Loading model...")
    policy_model, tokenizer = load(config["base_model"], adapter_path=args.adapter or config["sft_adapter"])
    
    if args.eval_only:
        print("\n📊 Evaluation:")
        r = evaluate(policy_model, tokenizer)
        print(f"Avg: {r['avg']:.1f}/120, Perfect: {r['perfect']}/{r['total']}, Range: [{r['min']}, {r['max']}]")
        return
    
    print(f"📥 Loading reference model (frozen)...")
    ref_model, _ = load(config["base_model"], adapter_path=config["sft_adapter"])
    ref_model.freeze()
    
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    
    output_dir = Path(config["output_adapter"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting GRPO ({config['max_steps']} steps)...\n")
    
    for step in range(1, config["max_steps"] + 1):
        query = random.choice(TRAINING_QUERIES)
        
        t0 = time.time()
        loss, kl, reward = grpo_step(policy_model, ref_model, tokenizer, query, config, optimizer)
        dt = time.time() - t0
        
        if step % config["log_every"] == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f} | KL: {kl:.4f} | Reward: {reward:.3f} | {dt:.1f}s")
        
        if step % config["save_every"] == 0:
            ckpt = output_dir / f"ckpt_{step:04d}"
            save_lora_weights(policy_model, ckpt)
            print(f"   💾 Saved: {ckpt}")
        
        if step % config["eval_every"] == 0:
            print(f"\n📊 Eval @ step {step}:")
            r = evaluate(policy_model, tokenizer)
            print(f"   Avg: {r['avg']:.1f}/120, Perfect: {r['perfect']}/{r['total']}\n")
    
    # Final save
    save_lora_weights(policy_model, output_dir)
    
    print("\n📊 Final evaluation:")
    r = evaluate(policy_model, tokenizer)
    print(f"Avg: {r['avg']:.1f}/120, Perfect: {r['perfect']}/{r['total']}, Range: [{r['min']}, {r['max']}]")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
