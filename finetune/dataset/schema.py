#!/usr/bin/env python3
"""Schema helpers for QMD training JSONL data."""

from __future__ import annotations

from typing import Iterable

VALID_OUTPUT_TYPES = {"hyde", "lex", "vec"}


def parse_output_text(text: str) -> list[list[str]]:
    """Parse prefixed output text into list pairs.

    Returns: [["hyde", "..."], ["lex", "..."], ...]
    """
    items: list[list[str]] = []
    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            items.append(["lex", line[4:].strip()])
        elif line.startswith("vec:"):
            items.append(["vec", line[4:].strip()])
        elif line.startswith("hyde:"):
            items.append(["hyde", line[5:].strip()])
    return items


def output_items_to_text(items: Iterable[Iterable[str]]) -> str:
    """Render output list pairs to prefixed text lines."""
    lines = []
    for item in items:
        if not item:
            continue
        try:
            kind, text = item[0], item[1]
        except Exception:
            continue
        if kind not in VALID_OUTPUT_TYPES:
            continue
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        lines.append(f"{kind}: {text}")
    return "\n".join(lines)


def normalize_output_items(items: Iterable[Iterable[str]]) -> list[list[str]]:
    """Normalize output list pairs (filter invalid, trim whitespace)."""
    normalized: list[list[str]] = []
    for item in items:
        if not item:
            continue
        try:
            kind, text = item[0], item[1]
        except Exception:
            continue
        if kind not in VALID_OUTPUT_TYPES:
            continue
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        normalized.append([kind, text])
    return normalized


def has_type(items: Iterable[Iterable[str]], kind: str) -> bool:
    return any(item and item[0] == kind for item in items)
