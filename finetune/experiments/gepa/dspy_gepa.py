#!/usr/bin/env python3
"""Run DSPy GEPA using reward.py as the metric."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _import_dspy():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    original_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p and str(p) != str(script_dir)]
        return importlib.import_module("dspy")
    finally:
        sys.path = original_sys_path


dspy = _import_dspy()

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dataset.schema import normalize_output_items, output_items_to_text, parse_output_text
from reward import score_expansion_detailed


class ExpandSignature(dspy.Signature):
    """Expand a search query into lex/vec/hyde lines."""

    query = dspy.InputField(desc="User search query")
    output = dspy.OutputField(
        desc=(
            "JSON array of [kind, text] pairs. kind is lex|vec|hyde. "
            "Return 2-3 lex, 2-3 vec, optional 0-1 hyde. "
            "Lex items are short keywords and must not echo the query. "
            "Vec items are natural language search phrases. "
            "Hyde is 50-200 chars, single line."
        )
    )


class Expander(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExpandSignature)

    def forward(self, query: str):
        return self.predict(query=query)


def reward_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    expansion = output_items_to_text(_coerce_output_items(pred))
    detail = score_expansion_detailed(gold.query, expansion)
    score = detail["percentage"] / 100.0
    feedback = "; ".join(detail.get("deductions", [])) or f"score={detail['percentage']:.1f}"
    return dspy.Prediction(score=score, feedback=feedback)


def load_queries(path: Path) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("query") or obj.get("input")
            if isinstance(query, str) and query.strip():
                queries.append(query.strip())
    return queries


def to_examples(queries: list[str]) -> list[dspy.Example]:
    return [dspy.Example(query=q).with_inputs("query") for q in queries]


def _coerce_output_items(pred) -> list[list[str]]:
    raw_output = getattr(pred, "output", None)
    if isinstance(raw_output, (list, tuple)):
        return normalize_output_items(raw_output)

    raw_text = str(raw_output or getattr(pred, "expansion", "") or "").strip()
    if not raw_text:
        return []

    if raw_text[0] in ("[", "{"):
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict) and "output" in obj:
                obj = obj["output"]
            if isinstance(obj, (list, tuple)):
                return normalize_output_items(obj)
        except Exception:
            pass

    return parse_output_text(raw_text)


def write_jsonl(path: Path, queries: list[str], outputs: list[list[list[str]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for query, output in zip(queries, outputs, strict=True):
            f.write(json.dumps({"query": query, "output": output}, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DSPy GEPA with reward.py")
    parser.add_argument("--input", type=str, required=True, help="Training JSONL path")
    parser.add_argument(
        "--model",
        type=str,
        default="grok-4-1-fast-reasoning",
        help="LM string in provider/model format (e.g., openai/gpt-4o)",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="grok-4-1-fast-reasoning",
        help="LM string in provider/model format (e.g., openai/gpt-4o)",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for student LM")
    parser.add_argument("--reflection-max-tokens", type=int, default=512, help="Max tokens for reflection LM")
    parser.add_argument("--auto", type=str, default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-full-evals", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--valset", type=str, default=None, help="Optional valset JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training queries")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit number of val queries")
    parser.add_argument("--emit", type=str, default=None, help="Write generated JSONL after compile")
    parser.add_argument("--save-prompt", type=str, default=None, help="Write best prompt text to file")
    args = parser.parse_args()

    if "/" not in args.model or "/" not in args.reflection_model:
        print("Error: DSPy expects provider/model format for LM strings (e.g., xai/grok-4-1-fast-reasoning).")
        return 1

    if args.max_full_evals is not None and args.max_metric_calls is not None:
        print("Provide only one of --max-full-evals or --max-metric-calls")
        return 1
    if args.max_full_evals is not None or args.max_metric_calls is not None:
        args.auto = None

    train_path = Path(args.input)
    queries = load_queries(train_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    trainset = to_examples(queries)
    valset = None
    if args.valset:
        val_queries = load_queries(Path(args.valset))
        if args.val_limit is not None:
            val_queries = val_queries[: args.val_limit]
        valset = to_examples(val_queries)

    lm = dspy.LM(model=args.model, max_tokens=args.max_tokens)
    reflection_lm = dspy.LM(model=args.reflection_model, max_tokens=args.reflection_max_tokens)

    student = Expander()
    student.set_lm(lm)

    compiler = dspy.GEPA(
        metric=reward_metric,
        reflection_lm=reflection_lm,
        auto=None if args.auto is None else args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
        failure_score=0.0,
        perfect_score=1.0,
    )

    optimized = compiler.compile(student=student, trainset=trainset, valset=valset)

    if args.save_prompt:
        prompt_text = getattr(optimized.predict.signature, "__doc__", "") or ""
        Path(args.save_prompt).write_text(prompt_text.strip() + "\n", encoding="utf-8")
        print(f"Wrote {args.save_prompt}")

    if args.emit:
        outputs = []
        for q in queries:
            pred = optimized(query=q)
            items = _coerce_output_items(pred)
            outputs.append(items)
        write_jsonl(Path(args.emit), queries, outputs)
        print(f"Wrote {args.emit}")

    if hasattr(optimized, "detailed_results"):
        best = getattr(optimized.detailed_results, "best_outputs_valset", None)
        if best:
            print(f"Best outputs tracked: {len(best)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
