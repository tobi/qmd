#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich>=13.0.0",
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "prompt_toolkit>=3.0.0",
# ]
# ///
"""
QMD Query Expansion Model Tester
A cyberpunk-styled TUI for testing finetuned query expansion models.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "1": ("SFT (Hub)", "tobil/qmd-query-expansion-0.6B-v4", "v3"),
    "2": ("GRPO (Hub)", "tobil/qmd-query-expansion-0.6B-v4-grpo", "v3"),
    "3": ("SFT (local)", "./qmd-query-expansion-0.6B-v4", "v3"),
    "4": ("GRPO (local)", "./qmd-query-expansion-0.6B-v4-grpo", "v3"),
}

BASE_MODEL = "Qwen/Qwen3-0.6B"

# v1 used simple format (before proper chat template)
PROMPT_TEMPLATE_V1 = """Expand this search query:

{query}"""

# v3+ uses tokenizer.apply_chat_template() - see generate_expansion()

# Cyberpunk color palette
CYAN = "#00ffff"
MAGENTA = "#ff00ff"
PURPLE = "#bd93f9"
DIM = "#6272a4"
BG = "#1a0a2e"
GREEN = "#50fa7b"
YELLOW = "#f1fa8c"
RED = "#ff5555"

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING (from evaluate_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or',
             'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by'}


def parse_expansion(text: str) -> dict:
    lines = text.strip().split("\n")
    result = {"lex": [], "vec": [], "hyde": [], "invalid": []}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())
        else:
            result["invalid"].append(line)
    return result


def score_expansion(query: str, expansion: str) -> dict:
    """Score an expansion based on format, diversity, hyde, quality."""
    text = expansion.strip()
    if not text:
        return {"total": 0, "percentage": 0, "rating": "Failed", "format": 0,
                "diversity": 0, "hyde": 0, "quality": 0, "parsed": {"lex": [], "vec": [], "hyde": [], "invalid": []}}

    parsed = parse_expansion(expansion)

    # Check for continuation - but only if NO valid lines were found
    has_valid = parsed["lex"] or parsed["vec"] or parsed["hyde"]
    if not has_valid:
        return {"total": 0, "percentage": 0, "rating": "Failed", "format": 0,
                "diversity": 0, "hyde": 0, "quality": 0, "is_continuation": True,
                "parsed": parsed}

    # Format score (0-30)
    format_score = 0
    if parsed["lex"]:
        format_score += 10
    if parsed["vec"]:
        format_score += 10
    if not parsed["invalid"]:
        format_score += 10
    else:
        format_score += max(0, 10 - len(parsed["invalid"]) * 5)

    # Diversity score (0-30)
    diversity_score = 0
    types_present = sum(1 for t in ["lex", "vec"] if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    total_exp = len(parsed["lex"]) + len(parsed["vec"])
    if total_exp >= 2:
        diversity_score += 5
    diversity_score += 10  # Base diversity points
    diversity_score += 5   # Non-echo points

    # Hyde score (0-20)
    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5  # Present
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2
        if "\n" not in hyde_text:
            hyde_score += 5
        hyde_score += 5  # No repetition (simplified)

    # Quality score (0-20)
    quality_score = 5  # Base relevance (reduced to make room for key term check)

    # Lex must preserve key terms from query
    stopwords = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of', 'and', 'or', 'with', 'my'}
    key_terms = set(query.lower().split()) - stopwords
    if parsed["lex"] and key_terms:
        lex_with_terms = sum(1 for l in parsed["lex"] if key_terms & set(l.lower().split()))
        if lex_with_terms == len(parsed["lex"]):
            quality_score += 5
        elif lex_with_terms > 0:
            quality_score += 2

    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
    if parsed["vec"] and all(" " in v and len(v) > 15 for v in parsed["vec"]):
        quality_score += 5

    total = format_score + diversity_score + hyde_score + min(20, quality_score)
    max_possible = 100 if parsed["hyde"] else 80
    percentage = total / max_possible * 100

    if percentage >= 80:
        rating = "Excellent"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Acceptable"
    elif percentage >= 20:
        rating = "Poor"
    else:
        rating = "Failed"

    return {
        "format": format_score,
        "diversity": diversity_score,
        "hyde": hyde_score,
        "quality": quality_score,
        "total": total,
        "max_possible": max_possible,
        "percentage": percentage,
        "rating": rating,
        "parsed": parsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = """
[bold cyan]╔═══════════════════════════════════════════════════════════════════════════╗[/]
[bold cyan]║[/] [bold magenta]░▀▄░░░░░░░░░░░▄▀░[/] [bold white]Q M D   Q U E R Y   E X P A N D E R[/] [bold magenta]░▀▄░░░░░░░░░░░▄▀░[/] [bold cyan]║[/]
[bold cyan]╚═══════════════════════════════════════════════════════════════════════════╝[/]
"""

def show_banner():
    console.print(BANNER)


def show_model_menu(current: Optional[str] = None) -> str:
    """Display model selection menu."""
    console.print()
    console.print(f"[bold {CYAN}]◆ SELECT MODEL[/]")
    console.print(f"[{DIM}]{'─' * 50}[/]")

    for key, (name, path, version) in MODELS.items():
        marker = "[bold green]●[/]" if path == current else f"[{DIM}]○[/]"
        console.print(f"  {marker} [{CYAN}]{key}[/] │ {name} [{DIM}]({version})[/]")
        console.print(f"      [{DIM}]{path}[/]")

    console.print(f"[{DIM}]{'─' * 50}[/]")
    return prompt("  Enter choice (1-3): ", style=Style.from_dict({'': CYAN})).strip()


def render_expansion(expansion: str, scores: dict) -> Panel:
    """Render the expansion output with syntax highlighting."""
    parsed = scores.get("parsed", parse_expansion(expansion))

    content = Text()

    # Lex lines
    for lex in parsed["lex"]:
        content.append("lex: ", style=f"bold {CYAN}")
        content.append(f"{lex}\n", style="white")

    # Vec lines
    for vec in parsed["vec"]:
        content.append("vec: ", style=f"bold {MAGENTA}")
        content.append(f"{vec}\n", style="white")

    # Hyde lines
    for hyde in parsed["hyde"]:
        content.append("hyde: ", style=f"bold {PURPLE}")
        content.append(f"{hyde}\n", style=f"italic {DIM}")

    # Invalid lines
    for inv in parsed["invalid"]:
        content.append(f"[invalid] {inv}\n", style=f"dim {RED}")

    return Panel(
        content,
        title=f"[bold {CYAN}]◈ EXPANSION[/]",
        border_style=CYAN,
        padding=(0, 1),
    )


def render_scores(scores: dict) -> Panel:
    """Render score breakdown as a compact table."""
    rating = scores["rating"]
    rating_color = {
        "Excellent": GREEN,
        "Good": CYAN,
        "Acceptable": YELLOW,
        "Poor": RED,
        "Failed": RED,
    }.get(rating, DIM)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("metric", style=DIM)
    table.add_column("score", justify="right")
    table.add_column("bar", width=12)

    def score_bar(val: int, max_val: int) -> str:
        filled = int(val / max_val * 10)
        return f"[{CYAN}]{'█' * filled}[/][{DIM}]{'░' * (10 - filled)}[/]"

    table.add_row("Format", f"[white]{scores['format']}[/]/30", score_bar(scores['format'], 30))
    table.add_row("Diversity", f"[white]{scores['diversity']}[/]/30", score_bar(scores['diversity'], 30))
    table.add_row("Hyde", f"[white]{scores['hyde']}[/]/20", score_bar(scores['hyde'], 20))
    table.add_row("Quality", f"[white]{scores['quality']}[/]/20", score_bar(scores['quality'], 20))
    table.add_row("", "", "")
    table.add_row(
        f"[bold]TOTAL[/]",
        f"[bold white]{scores['total']}[/]/{scores.get('max_possible', 80)}",
        f"[bold {rating_color}]{rating}[/]"
    )

    return Panel(
        table,
        title=f"[bold {MAGENTA}]◈ SCORES[/]",
        border_style=MAGENTA,
        padding=(0, 1),
    )


def render_history(history: deque) -> Panel:
    """Render recent query history."""
    content = Text()
    for i, (query, rating) in enumerate(history):
        rating_color = {
            "Excellent": GREEN, "Good": CYAN, "Acceptable": YELLOW,
            "Poor": RED, "Failed": RED,
        }.get(rating, DIM)
        content.append(f"  [{DIM}]{i+1}.[/] {query[:40]}")
        if len(query) > 40:
            content.append(f"[{DIM}]...[/]")
        content.append(f" [{rating_color}]●[/]\n")

    if not history:
        content.append(f"  [{DIM}]No queries yet[/]")

    return Panel(
        content,
        title=f"[bold {PURPLE}]◈ HISTORY[/]",
        border_style=PURPLE,
        padding=(0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadedModel:
    model: any
    tokenizer: any
    name: str
    path: str
    version: str  # "v1" or "v3" - determines prompt template


def load_model(model_path: str, model_name: str, version: str) -> LoadedModel:
    """Load model with progress indicator."""
    with Progress(
        SpinnerColumn(spinner_name="dots", style=CYAN),
        TextColumn(f"[{CYAN}]Loading {{task.description}}...[/]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("tokenizer", total=None)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        progress.update(task, description="base model")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        progress.update(task, description="adapter")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, name=model_name, path=model_path, version=version)


DEBUG = False  # Set to True for debug output

def generate_expansion(loaded: LoadedModel, query: str) -> str:
    """Generate expansion using proper Qwen3 chat template."""
    if loaded.version == "v3":
        # Use tokenizer's chat template with /no_think to disable thinking mode
        messages = [{"role": "user", "content": f"/no_think Expand this search query: {query}"}]
        prompt_text = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # v1 used simple format
        prompt_text = PROMPT_TEMPLATE_V1.format(query=query)

    if DEBUG:
        console.print(f"[{DIM}]─── DEBUG: Prompt ───[/]")
        console.print(f"[{DIM}]{repr(prompt_text)}[/]")

    inputs = loaded.tokenizer(prompt_text, return_tensors="pt").to(loaded.model.device)

    with torch.no_grad():
        outputs = loaded.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )

    full_output = loaded.tokenizer.decode(outputs[0], skip_special_tokens=True)

    if DEBUG:
        console.print(f"[{DIM}]─── DEBUG: Full output ───[/]")
        console.print(f"[{DIM}]{repr(full_output[:500])}[/]")

    # Extract assistant response (skip_special_tokens leaves "user\n...\nassistant\n...")
    if "\nassistant\n" in full_output:
        expansion = full_output.split("\nassistant\n")[-1].strip()
    elif "assistant\n" in full_output:
        expansion = full_output.split("assistant\n")[-1].strip()
    else:
        expansion = full_output[len(prompt_text):].strip()

    # Remove any <think> tags that might remain
    if expansion.startswith("<think>"):
        think_end = expansion.find("</think>")
        if think_end != -1:
            expansion = expansion[think_end + 8:].strip()

    if DEBUG:
        console.print(f"[{DIM}]─── DEBUG: Expansion ───[/]")
        console.print(f"[{DIM}]{repr(expansion[:300])}[/]")

    return expansion


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    console.clear()
    show_banner()

    # Model selection
    choice = show_model_menu()
    if choice not in MODELS:
        console.print(f"[{RED}]Invalid choice. Exiting.[/]")
        return

    model_name, model_path, model_version = MODELS[choice]
    console.print()

    try:
        loaded = load_model(model_path, model_name, model_version)
    except Exception as e:
        console.print(f"[{RED}]Failed to load model: {e}[/]")
        return

    console.print(f"[{GREEN}]✓ Model loaded: {model_name}[/]")
    console.print()

    # Query history
    history: deque = deque(maxlen=5)
    input_history = InMemoryHistory()

    # Main loop
    console.print(f"[{DIM}]Enter queries to expand. Type 'quit' to exit, 'model' to switch models.[/]")
    console.print()

    while True:
        try:
            query = prompt(
                f"[{CYAN}]❯[/] ",
                history=input_history,
                style=Style.from_dict({'': 'ansicyan'}),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() == 'quit':
            break

        if query.lower() == 'model':
            console.clear()
            show_banner()
            choice = show_model_menu(loaded.path)
            if choice in MODELS:
                new_name, new_path, new_version = MODELS[choice]
                if new_path != loaded.path:
                    console.print()
                    loaded = load_model(new_path, new_name, new_version)
                    console.print(f"[{GREEN}]✓ Switched to: {new_name}[/]")
            console.print()
            continue

        if query.lower() == 'history':
            console.print(render_history(history))
            continue

        # Generate expansion
        with Progress(
            SpinnerColumn(spinner_name="dots", style=MAGENTA),
            TextColumn(f"[{MAGENTA}]Expanding...[/]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("gen", total=None)
            expansion = generate_expansion(loaded, query)

        # Score it
        scores = score_expansion(query, expansion)

        # Add to history
        history.appendleft((query, scores["rating"]))

        # Display results
        console.print()
        console.print(f"[{DIM}]Query: [/][bold white]{query}[/]")
        console.print()

        # Side-by-side layout
        console.print(Columns([
            render_expansion(expansion, scores),
            render_scores(scores),
        ], equal=True, expand=True))

        console.print()

    console.print(f"\n[{CYAN}]◆ Goodbye![/]\n")


if __name__ == "__main__":
    main()
