#!/usr/bin/env python3
"""GEPA example schema for QMD training JSONL lines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class SearchType(str, Enum):
    LexSearch = "LexSearch"
    VecSearch = "VecSearch"
    HydeSearch = "HydeSearch"


SEARCH_TYPE_TO_PREFIX = {
    SearchType.LexSearch: "lex",
    SearchType.VecSearch: "vec",
    SearchType.HydeSearch: "hyde",
}


@dataclass
class OutputItem:
    """Single expansion line with validation hints."""

    kind: SearchType
    text: str

    # Validation hints (not strict rules).
    min_chars: int = 3
    max_chars: int | None = None

    def __post_init__(self) -> None:
        self.text = str(self.text).strip()
        if not self.text:
            raise ValueError("OutputItem.text must be non-empty")
        if "\n" in self.text:
            raise ValueError("OutputItem.text must be single-line")
        if len(self.text) < self.min_chars:
            raise ValueError("OutputItem.text is too short")
        if self.max_chars is not None and len(self.text) > self.max_chars:
            raise ValueError("OutputItem.text is too long")

    def to_pair(self) -> list[str]:
        return [SEARCH_TYPE_TO_PREFIX[self.kind], self.text]


@dataclass
class Example:
    """JSONL line schema for QMD training data."""

    query: str
    output: list[OutputItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.query = str(self.query).strip()
        if not self.query:
            raise ValueError("Example.query must be non-empty")
        if not self.output:
            raise ValueError("Example.output must not be empty")

    def to_json(self) -> dict:
        return {
            "query": self.query,
            "output": [item.to_pair() for item in self.output],
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)


def parse_output_items(raw_output: Iterable[Iterable[str]]) -> list[OutputItem]:
    items: list[OutputItem] = []
    for item in raw_output:
        if not item or len(item) < 2:
            continue
        kind_raw, text = item[0], item[1]
        kind_map = {
            "lex": SearchType.LexSearch,
            "vec": SearchType.VecSearch,
            "hyde": SearchType.HydeSearch,
        }
        kind = kind_map.get(str(kind_raw).strip().lower())
        if kind is None:
            continue
        max_chars = 200 if kind is SearchType.HydeSearch else None
        items.append(OutputItem(kind=kind, text=str(text), max_chars=max_chars))
    return items


def example_from_json(obj: dict) -> Example:
    query = obj.get("query") or obj.get("input") or ""
    output = obj.get("output") or []
    if isinstance(output, str):
        raise ValueError("String outputs are not supported in GEPA example schema")
    items = parse_output_items(output)
    return Example(query=query, output=items)


def load_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(example_from_json(obj))
            except Exception as exc:
                raise ValueError(f"Invalid line {line_num}: {exc}") from exc
    return examples

