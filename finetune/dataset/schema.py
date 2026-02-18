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


def reorder_hyde_first(items: list[list[str]]) -> list[list[str]]:
    """Reorder items to put hyde first, then lex, then vec."""
    hyde_items = [item for item in items if item and item[0] == "hyde"]
    lex_items = [item for item in items if item and item[0] == "lex"]
    vec_items = [item for item in items if item and item[0] == "vec"]
    return hyde_items + lex_items + vec_items


def output_items_to_text(items: Iterable[Iterable[str]], hyde_first: bool = True) -> str:
    """Render output list pairs to prefixed text lines.
    
    Args:
        items: Iterable of [type, text] pairs
        hyde_first: If True, reorder to put hyde first (default)
    """
    # First normalize to list
    normalized = []
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
    
    # Apply hyde-first ordering if requested
    if hyde_first:
        normalized = reorder_hyde_first(normalized)
    
    lines = [f"{kind}: {text}" for kind, text in normalized]
    return "\n".join(lines)


def normalize_output_items(items: Iterable[Iterable[str]], hyde_first: bool = True) -> list[list[str]]:
    """Normalize output list pairs (filter invalid, trim whitespace, reorder).
    
    Args:
        items: Iterable of [type, text] pairs
        hyde_first: If True, reorder to put hyde first (default)
    """
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
    
    # Apply hyde-first ordering if requested
    if hyde_first:
        normalized = reorder_hyde_first(normalized)
    
    return normalized


def has_type(items: Iterable[Iterable[str]], kind: str) -> bool:
    return any(item and item[0] == kind for item in items)
