#!/usr/bin/env python3
"""
Strict schema for QMD training data.

Every JSONL file in data/ MUST conform to this format:

    {"query": "auth config", "output": [["hyde", "..."], ["lex", "..."], ["vec", "..."]]}

- query: non-empty string
- output: list of [type, text] pairs where type is "lex", "vec", or "hyde"
- Extra fields (category, intent, is_short, etc.) are allowed but ignored

There is exactly ONE format. No alternatives, no legacy fallbacks.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterable

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    field_validator,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class OutputType(str, Enum):
    lex = "lex"
    vec = "vec"
    hyde = "hyde"


VALID_OUTPUT_TYPES = {t.value for t in OutputType}


class OutputPair(BaseModel):
    """A single expansion line: [type, text]."""

    type: OutputType
    text: str

    model_config = ConfigDict(frozen=True)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    def to_list(self) -> list[str]:
        return [self.type.value, self.text]


def _coerce_output_pairs(v: list) -> list[OutputPair]:
    """Accept [["lex", "..."], ...] from JSON and coerce to OutputPair list."""
    pairs = []
    for i, item in enumerate(v):
        if isinstance(item, OutputPair):
            pairs.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(OutputPair(type=item[0], text=item[1]))
        else:
            raise ValueError(
                f"output[{i}] must be [type, text], got {item!r}"
            )
    return pairs


# ---------------------------------------------------------------------------
# Pydantic model — single source of truth for the JSONL schema
# ---------------------------------------------------------------------------

class TrainingExample(BaseModel):
    """One training example in the canonical JSONL format."""

    query: str
    output: Annotated[list[OutputPair], BeforeValidator(_coerce_output_pairs)]

    # Optional metadata — present in some files, ignored during training.
    category: str | None = None
    intent: str | None = None
    is_short: bool | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v: list[OutputPair]) -> list[OutputPair]:
        if not v:
            raise ValueError("output must not be empty")
        return v

    def output_as_lists(self) -> list[list[str]]:
        """Return output as list-of-lists for JSON serialization."""
        return [p.to_list() for p in self.output]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_examples(path: str | Path) -> list[TrainingExample]:
    """Load and validate a JSONL file. Fails loudly on any bad line."""
    path = Path(path)
    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {e}") from e
            try:
                examples.append(TrainingExample.model_validate(obj))
            except Exception as e:
                raise ValueError(f"{path}:{line_num}: {e}") from e
    return examples


# ---------------------------------------------------------------------------
# Helpers (used by prepare_data.py, reward.py, and other tools)
# ---------------------------------------------------------------------------

def parse_output_text(text: str) -> list[list[str]]:
    """Parse prefixed output text into list pairs.

    >>> parse_output_text("lex: foo\\nvec: bar")
    [["lex", "foo"], ["vec", "bar"]]
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


def output_items_to_text(
    items: Iterable, hyde_first: bool = True
) -> str:
    """Render output pairs to prefixed text lines.

    Accepts list[OutputPair] or list[list[str]].
    """
    normalized = []
    for item in items:
        if isinstance(item, OutputPair):
            normalized.append([item.type.value, item.text.strip()])
            continue
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

    if hyde_first:
        normalized = reorder_hyde_first(normalized)

    lines = [f"{kind}: {text}" for kind, text in normalized]
    return "\n".join(lines)


def normalize_output_items(
    items: Iterable, hyde_first: bool = True
) -> list[list[str]]:
    """Normalize output pairs (filter invalid, trim whitespace, reorder).

    Accepts list[OutputPair] or list[list[str]].
    """
    normalized: list[list[str]] = []
    for item in items:
        if isinstance(item, OutputPair):
            normalized.append([item.type.value, item.text.strip()])
            continue
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

    if hyde_first:
        normalized = reorder_hyde_first(normalized)

    return normalized


def has_type(items: Iterable, kind: str) -> bool:
    for item in items:
        if isinstance(item, OutputPair):
            if item.type.value == kind:
                return True
        elif item and item[0] == kind:
            return True
    return False
