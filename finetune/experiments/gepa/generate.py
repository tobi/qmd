#!/usr/bin/env python3
"""Generate expansions using a saved GEPA prompt."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _import_dspy():
    script_dir = Path(__file__).parent
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

from dataset.schema import parse_output_text


def load_topics(path: Path) -> list[str]:
    topics: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Allow JSONL {"topic": "..."} or plain lines.
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    topic = obj.get("topic") or obj.get("query") or obj.get("input")
                    if isinstance(topic, str) and topic.strip():
                        topics.append(topic.strip())
                        continue
                except json.JSONDecodeError:
                    pass
            topics.append(line)
    return topics


def write_jsonl_line(handle, query: str, output_text: str) -> None:
    output = parse_output_text(output_text)
    handle.write(json.dumps({"query": query, "output": output}, ensure_ascii=False) + "\n")


def parse_queries(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with saved GEPA prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Path to saved prompt text")
    parser.add_argument("--topics", type=str, required=True, help="Topics file (one per line or JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--model", type=str, required=True, help="LM string in provider/model format")
    parser.add_argument("--per-topic", type=int, default=3, help="Queries to generate per topic")
    args = parser.parse_args()

    prompt_text = Path(args.prompt).read_text(encoding="utf-8").strip()
    expansion_sig = dspy.Signature("query -> expansion", prompt_text)
    query_sig = dspy.Signature(
        "topic, count -> queries",
        (
            "Generate distinct user search queries for the given topic. "
            "Return exactly `count` queries, one per line, no numbering or extra text."
        ),
    )

    class Generator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(expansion_sig)

        def forward(self, query: str):
            return self.predict(query=query)

    class QueryGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(query_sig)

        def forward(self, topic: str, count: int):
            return self.predict(topic=topic, count=str(count))

    lm = dspy.LM(model=args.model)
    gen = Generator()
    gen.set_lm(lm)
    qgen = QueryGenerator()
    qgen.set_lm(lm)

    topics = load_topics(Path(args.topics))
    with Path(args.output).open("w", encoding="utf-8") as f_out:
        for topic in topics:
            qpred = qgen(topic=topic, count=args.per_topic)
            qtext = getattr(qpred, "queries", "") or ""
            generated = parse_queries(qtext)
            if not generated:
                generated = [topic]
            generated = generated[: args.per_topic]
            for query in generated:
                pred = gen(query=query)
                output_text = getattr(pred, "expansion", "") or ""
                write_jsonl_line(f_out, query, output_text)
                print(json.dumps({"query": query, "output": parse_output_text(output_text)}, ensure_ascii=False))

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
