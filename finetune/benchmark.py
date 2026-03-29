#!/usr/bin/env python3
"""Benchmark QMD query expansion: LFM2.5 vs Qwen3 finetuned models."""

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

QUERIES = [
    "kubernetes pod networking",
    "best practices for React server components",
    "how to optimize PostgreSQL queries for large tables",
    "what is retrieval augmented generation",
    "python async await concurrency patterns",
    "nginx reverse proxy load balancing",
    "git rebase vs merge workflow",
    "rust ownership and borrowing explained",
    "docker compose multi-stage builds",
    "elasticsearch full text search performance",
    "shopify liquid template customization",
    "machine learning feature engineering techniques",
    "aws lambda cold start optimization",
    "typescript generics and utility types",
    "redis caching strategies for web apps",
]

def load_model(base_name, adapter_dir, device, trust_remote=False):
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=trust_remote)
    base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=trust_remote
    )
    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    model = model.merge_and_unload()
    model.eval()
    
    gen_config_path = Path(adapter_dir) / "generation_config.json"
    if gen_config_path.exists():
        gen_config = GenerationConfig.from_pretrained(adapter_dir)
    else:
        gen_config = GenerationConfig(
            temperature=0.1, top_k=50, top_p=0.1,
            repetition_penalty=1.05, do_sample=True, max_new_tokens=300,
        )
    return model, tokenizer, gen_config

def run_inference(model, tokenizer, gen_config, query, device):
    messages = [{"role": "user", "content": f"Expand this search query: {query}"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_config, max_new_tokens=300)
    elapsed = time.perf_counter() - start
    
    new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return result, elapsed, new_tokens

def score_output(output):
    """Simple quality scoring: check for lex/vec/hyde presence and specificity."""
    score = 0
    lines = output.strip().split("\n")
    has_lex = has_vec = has_hyde = False
    hyde_text = ""
    
    for line in lines:
        l = line.strip()
        if l.startswith("lex:"):
            has_lex = True
            score += 1
        elif l.startswith("vec:"):
            has_vec = True
            score += 1
        elif l.startswith("hyde:"):
            has_hyde = True
            hyde_text = l[5:].strip()
            score += 2  # hyde is worth more
    
    # Bonus for hyde length in sweet spot (80-200 chars)
    if hyde_text:
        hlen = len(hyde_text)
        if 80 <= hlen <= 200:
            score += 2
        elif 50 <= hlen <= 250:
            score += 1
    
    # Penalty for generic/template hyde
    generic_phrases = ["comprehensive guide", "everything you need to know", "beginners and advanced users"]
    for phrase in generic_phrases:
        if phrase in hyde_text.lower():
            score -= 1
    
    return score, {"has_lex": has_lex, "has_vec": has_vec, "has_hyde": has_hyde, "hyde_len": len(hyde_text)}

def main():
    device = "cuda:0"
    
    models = {
        "LFM2.5-1.2B (finetuned)": {
            "base": "LiquidAI/LFM2.5-1.2B-Instruct",
            "adapter": "outputs/sft-lfm2",
            "trust_remote": True,
        },
        "Qwen3-1.7B (finetuned)": {
            "base": "Qwen/Qwen3-1.7B",
            "adapter": "outputs/sft",
            "trust_remote": False,
        },
    }
    
    results = {}
    
    for name, cfg in models.items():
        print(f"\n{'='*60}")
        print(f"Loading {name}...")
        model, tokenizer, gen_config = load_model(
            cfg["base"], cfg["adapter"], device, cfg["trust_remote"]
        )
        
        model_results = []
        total_time = 0
        total_tokens = 0
        total_score = 0
        
        for query in QUERIES:
            output, elapsed, n_tokens = run_inference(model, tokenizer, gen_config, query, device)
            score, details = score_output(output)
            
            model_results.append({
                "query": query,
                "output": output,
                "time_s": round(elapsed, 3),
                "tokens": n_tokens,
                "score": score,
                "details": details,
            })
            total_time += elapsed
            total_tokens += n_tokens
            total_score += score
            
            tok_s = n_tokens / elapsed if elapsed > 0 else 0
            print(f"  [{score:2d}] {query[:40]:<40} {elapsed:.2f}s {n_tokens:3d}tok {tok_s:.0f}tok/s")
        
        avg_time = total_time / len(QUERIES)
        avg_score = total_score / len(QUERIES)
        avg_toks = total_tokens / total_time if total_time > 0 else 0
        
        results[name] = {
            "queries": model_results,
            "avg_time_s": round(avg_time, 3),
            "avg_score": round(avg_score, 2),
            "avg_tok_s": round(avg_toks, 1),
            "total_score": total_score,
        }
        
        print(f"\n  Summary: avg_score={avg_score:.2f} avg_time={avg_time:.2f}s avg_tok/s={avg_toks:.0f}")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Total Score: {r['total_score']} / {len(QUERIES) * 8}")  # max ~8 per query
        print(f"  Avg Score:   {r['avg_score']}")
        print(f"  Avg Time:    {r['avg_time_s']}s")
        print(f"  Throughput:  {r['avg_tok_s']} tok/s")
    
    # Save full results
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to outputs/benchmark_results.json")

if __name__ == "__main__":
    main()
