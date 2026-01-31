#!/usr/bin/env python3
"""
Generate diverse QMD training examples for underrepresented categories.

This script creates additional training examples focused on:
- Trivia, Geography, Philosophy, History (as requested)
- Temporal/Recency queries (important for evals)
- Named entity queries (critical for entity preservation scoring)
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

from dataset.schema import normalize_output_items, parse_output_text

# Additional diverse query categories
TRIVIA_QUERIES = [
    "world capitals quiz",
    "trivia facts about space",
    "did you know history",
    "random science facts",
    "famous inventions timeline",
    "world records list",
    "fun geography facts",
    "historical trivia questions",
    "animal trivia facts",
    "sports trivia records",
]

GEOGRAPHY_QUERIES = [
    "largest countries by area",
    "rivers that cross multiple countries",
    "highest mountain peaks",
    "desert climate zones",
    "island nations list",
    "capital cities europe",
    "population by continent",
    "time zones map",
    "latitude longitude coordinates",
    "borders between countries",
    "ocean currents patterns",
    "tectonic plate boundaries",
    "climate zones earth",
]

PHILOSOPHY_QUERIES = [
    "stoicism daily practice",
    "existentialism meaning life",
    "utilitarianism ethics explained",
    "kant categorical imperative",
    "free will determinism debate",
    "nietzsche will to power",
    "socrates method questioning",
    "plato theory forms",
    "aristotle virtue ethics",
    "descartes cogito ergo sum",
    "logic propositional calculus",
    "epistemology knowledge theory",
    "metaphysics existence reality",
]

HISTORY_QUERIES = [
    "ancient civilizations timeline",
    "roman empire fall reasons",
    "medieval period events",
    "renaissance art movement",
    "industrial revolution inventions",
    "world war i causes",
    "cold war key events",
    "french revolution timeline",
    "american civil war battles",
    "egyptian pharaohs dynasty",
    "bronze age collapse",
    "byzantine empire history",
    "vietnam war timeline",
]

SCIENCE_QUERIES = [
    "quantum mechanics basics",
    "theory of relativity explained",
    "dna structure discovery",
    "photosynthesis process steps",
    "black holes physics",
    "plate tectonics theory",
    "evolution natural selection",
    "periodic table elements",
    "cell biology fundamentals",
    "climate change evidence",
]

ARTS_CULTURE_QUERIES = [
    "impressionist painters list",
    "shakespeare plays summary",
    "classical music composers",
    "modern art movements",
    "film noir characteristics",
    "jazz history origins",
    "renaissance sculpture techniques",
    "photography composition rules",
    "poetry forms haiku",
    "baroque art characteristics",
    "street art graffiti history",
]

HEALTH_MEDICINE_QUERIES = [
    "symptoms of vitamin deficiency",
    "how vaccines work immune system",
    "blood pressure normal range",
    "sleep hygiene tips",
    "intermittent fasting benefits",
    "anxiety coping strategies",
    "stretching exercises back pain",
    "heart disease prevention",
    "diabetes type 2 management",
    "meditation mental health",
    "nutrition macros explained",
    "first aid basics",
]

BUSINESS_FINANCE_QUERIES = [
    "compound interest calculator",
    "stock market basics beginners",
    "startup funding stages",
    "tax deductions small business",
    "budgeting methods 50 30 20",
    "cryptocurrency explained simply",
    "inflation effects on savings",
    "retirement planning strategies",
    "passive income ideas",
    "venture capital vs angel investors",
    "balance sheet basics",
    "supply chain management",
]

SPORTS_QUERIES = [
    "marathon training schedule",
    "weightlifting proper form",
    "swimming stroke techniques",
    "tennis serve mechanics",
    "basketball dribbling drills",
    "soccer formations tactics",
    "golf swing fundamentals",
    "yoga poses beginners",
    "running injury prevention",
    "cycling gear ratios",
    "rock climbing grades",
    "surfing wave types",
]

TRAVEL_QUERIES = [
    "best time visit japan",
    "travel packing checklist",
    "budget backpacking europe",
    "visa requirements usa",
    "jet lag remedies",
    "road trip planning tips",
    "solo travel safety",
    "airport security rules",
    "travel insurance coverage",
    "language apps learning",
    "hostel vs hotel comparison",
    "travel photography tips",
]

FOOD_COOKING_QUERIES = [
    "bread baking techniques",
    "knife skills basics",
    "fermentation at home",
    "meal prep weekly",
    "spice combinations guide",
    "pasta making fresh",
    "coffee brewing methods",
    "wine pairing basics",
    "vegetarian protein sources",
    "food storage guidelines",
    "sourdough starter maintenance",
    "grilling temperature chart",
]

PSYCHOLOGY_QUERIES = [
    "cognitive biases list",
    "attachment theory styles",
    "maslow hierarchy needs",
    "growth mindset vs fixed",
    "emotional intelligence components",
    "memory techniques mnemonics",
    "habit formation science",
    "stress response fight flight",
    "personality types myers briggs",
    "motivation intrinsic extrinsic",
    "decision making psychology",
    "procrastination causes solutions",
]

ENVIRONMENT_NATURE_QUERIES = [
    "renewable energy types",
    "carbon footprint reduction",
    "composting basics home",
    "endangered species list",
    "recycling symbols meaning",
    "ocean plastic pollution",
    "deforestation effects",
    "sustainable living tips",
    "wildlife conservation efforts",
    "solar panel installation",
    "water conservation methods",
    "biodiversity importance",
]

MATH_QUERIES = [
    "calculus derivatives explained",
    "probability basics statistics",
    "linear algebra matrices",
    "geometry proofs theorems",
    "logarithms rules properties",
    "trigonometry identities",
    "set theory basics",
    "prime numbers properties",
    "fractions decimals conversion",
    "algebra equations solving",
    "graph theory fundamentals",
    "combinatorics permutations",
]

LANGUAGE_QUERIES = [
    "spanish verb conjugation",
    "japanese hiragana katakana",
    "french pronunciation rules",
    "german cases grammar",
    "mandarin tones guide",
    "latin phrases common",
    "arabic alphabet basics",
    "english idioms meanings",
    "sign language basics",
    "etymology word origins",
    "grammar punctuation rules",
    "writing style guides",
]

DIY_CRAFTS_QUERIES = [
    "woodworking joints types",
    "knitting patterns beginners",
    "home repair basics",
    "sewing machine threading",
    "painting techniques acrylic",
    "pottery wheel basics",
    "electronics soldering guide",
    "gardening soil preparation",
    "candle making supplies",
    "leather crafting tools",
    "origami folding instructions",
    "furniture restoration tips",
]

# Temporal/Recency queries (matches evals/queries.txt requirements)
TEMPORAL_TEMPLATES = [
    "latest {topic} updates",
    "recent {topic} changes {year}",
    "what changed in {topic} {year}",
    "{topic} changelog {year}",
    "{topic} new features {year}",
    "{topic} latest version release",
    "{topic} recent news {month}",
]

TEMPORAL_TOPICS = [
    "Shopify",
    "React",
    "Kubernetes",
    "Docker",
    "TypeScript",
    "Python",
    "AWS",
    "GitHub",
    "Next.js",
    "Vue",
    "AI",
    "machine learning",
    "climate tech",
    "space exploration",
]

# Named entity queries (critical for entity preservation testing)
NAMED_ENTITY_QUERIES = [
    "who is TDS motorsports",
    "React hooks tutorial",
    "Docker container networking",
    "Kubernetes pod deployment",
    "AWS Lambda functions setup",
    "Stripe payment integration",
    "GitHub Actions workflow",
    "Vercel deployment guide",
    "Supabase auth configuration",
    "Twilio SMS API",
    "Datadog monitoring setup",
    "Sentry error tracking",
    "Terraform AWS provider",
    "Ansible playbook examples",
]


# Generate temporal queries with recent dates
def generate_temporal_queries():
    queries = []
    current_year = datetime.now().year
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for template in TEMPORAL_TEMPLATES:
        for topic in TEMPORAL_TOPICS:
            if "{year}" in template:
                # Use current year and previous year
                for year in [current_year, current_year - 1]:
                    queries.append(template.format(topic=topic, year=year))
            elif "{month}" in template:
                # Use recent months
                for month in months[-3:]:  # Last 3 months
                    queries.append(template.format(topic=topic, month=month))
            else:
                queries.append(template.format(topic=topic))

    return list(set(queries))  # Remove duplicates


def generate_expansion(query: str) -> str:
    """Generate a realistic expansion for a query."""
    # This is a template-based generator - in production, use Claude API
    lex_variations = [
        f"{query} guide",
        f"{query} documentation",
        f"{query} tutorial",
        f"{query} examples",
        f"{query} best practices",
    ]

    vec_variations = [
        f"how to {query}",
        f"guide for {query}",
        f"learn about {query}",
        f"understanding {query}",
        f"complete {query} reference",
    ]

    # Select 2-3 lex and 2 vec variations
    selected_lex = random.sample(lex_variations, min(3, len(lex_variations)))
    selected_vec = random.sample(vec_variations, min(2, len(vec_variations)))

    # Generate hyde passage
    hyde = f"This comprehensive guide covers everything you need to know about {query}. It includes practical examples, best practices, and troubleshooting tips for beginners and advanced users alike."

    output_lines = []
    for lex in selected_lex:
        output_lines.append(f"lex: {lex}")
    for vec in selected_vec:
        output_lines.append(f"vec: {vec}")
    output_lines.append(f"hyde: {hyde}")

    return "\n".join(output_lines)


def main():
    """Generate diverse examples and append to training data."""
    output_file = Path("data/qmd_expansion_diverse_addon.jsonl")

    all_queries = (
        TRIVIA_QUERIES
        + GEOGRAPHY_QUERIES
        + PHILOSOPHY_QUERIES
        + HISTORY_QUERIES
        + SCIENCE_QUERIES
        + ARTS_CULTURE_QUERIES
        + HEALTH_MEDICINE_QUERIES
        + BUSINESS_FINANCE_QUERIES
        + SPORTS_QUERIES
        + TRAVEL_QUERIES
        + FOOD_COOKING_QUERIES
        + PSYCHOLOGY_QUERIES
        + ENVIRONMENT_NATURE_QUERIES
        + MATH_QUERIES
        + LANGUAGE_QUERIES
        + DIY_CRAFTS_QUERIES
        + generate_temporal_queries()
        + NAMED_ENTITY_QUERIES
    )

    print(f"Generating {len(all_queries)} diverse training examples...")
    print(f"  - Trivia: {len(TRIVIA_QUERIES)}")
    print(f"  - Geography: {len(GEOGRAPHY_QUERIES)}")
    print(f"  - Philosophy: {len(PHILOSOPHY_QUERIES)}")
    print(f"  - History: {len(HISTORY_QUERIES)}")
    print(f"  - Science: {len(SCIENCE_QUERIES)}")
    print(f"  - Arts/Culture: {len(ARTS_CULTURE_QUERIES)}")
    print(f"  - Health/Medicine: {len(HEALTH_MEDICINE_QUERIES)}")
    print(f"  - Business/Finance: {len(BUSINESS_FINANCE_QUERIES)}")
    print(f"  - Sports: {len(SPORTS_QUERIES)}")
    print(f"  - Travel: {len(TRAVEL_QUERIES)}")
    print(f"  - Food/Cooking: {len(FOOD_COOKING_QUERIES)}")
    print(f"  - Psychology: {len(PSYCHOLOGY_QUERIES)}")
    print(f"  - Environment: {len(ENVIRONMENT_NATURE_QUERIES)}")
    print(f"  - Math: {len(MATH_QUERIES)}")
    print(f"  - Language: {len(LANGUAGE_QUERIES)}")
    print(f"  - DIY/Crafts: {len(DIY_CRAFTS_QUERIES)}")
    print(f"  - Temporal: {len(generate_temporal_queries())}")
    print(f"  - Named Entities: {len(NAMED_ENTITY_QUERIES)}")

    examples = []
    for query in all_queries:
        expansion = generate_expansion(query)
        output_items = normalize_output_items(parse_output_text(expansion))
        examples.append(
            {"query": query, "output": output_items, "category": "diverse_addon"}
        )

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} diverse examples to {output_file}")
    print("\nTo use these examples:")
    print(f"  cat {output_file} >> data/qmd_expansion_v2.jsonl")
    print("  uv run dataset/prepare_data.py --add-short 2")


if __name__ == "__main__":
    main()
