#!/usr/bin/env python3
"""
Generate BALANCED QMD training examples - reduced tech focus, more life diversity.

Categories (non-tech heavy):
- Health & Wellness (15%)
- Personal Finance & Business (15%)
- Home, Garden & DIY (10%)
- Food & Cooking (10%)
- Travel & Geography (10%)
- Hobbies & Crafts (10%)
- Education & Learning (10%)
- Arts & Culture (10%)
- Lifestyle & Relationships (5%)
- Technology (5% - minimal)
"""

import json
import random
from pathlib import Path
from datetime import datetime

from dataset.schema import normalize_output_items, parse_output_text

# Category weights - balanced with reasonable tech representation
CATEGORY_WEIGHTS = {
    "health_wellness": 0.12,
    "finance_business": 0.12,
    "home_garden": 0.10,
    "food_cooking": 0.10,
    "travel_geography": 0.10,
    "hobbies_crafts": 0.10,
    "education_learning": 0.08,
    "arts_culture": 0.08,
    "lifestyle_relationships": 0.05,
    "technology": 0.15,  # 15% - reasonable for QMD use case
}

# === HEALTH & WELLNESS ===
HEALTH_TOPICS = [
    "meditation techniques",
    "sleep improvement",
    "stress management",
    "anxiety relief",
    "healthy eating",
    "meal planning",
    "weight loss",
    "muscle building",
    "yoga poses",
    "home workout",
    "running form",
    "swimming technique",
    "vitamin supplements",
    "hydration tips",
    "posture correction",
    "stretching routine",
    "mental health",
    "therapy types",
    "mindfulness practice",
    "breathing exercises",
    "first aid basics",
    "CPR technique",
    "common cold remedies",
    "allergy management",
    "chronic pain",
    "physical therapy",
    "massage techniques",
    "acupuncture",
    "eye health",
    "dental care",
    "skin care routine",
    "hair health",
]

HEALTH_ACTIVITIES = [
    "improve sleep",
    "reduce stress",
    "manage anxiety",
    "build muscle",
    "lose weight",
    "eat healthier",
    "start meditating",
    "practice yoga",
    "run faster",
    "swim better",
    "lift weights",
    "stretch properly",
    "boost immunity",
    "increase energy",
    "reduce inflammation",
    "detox body",
]

# === FINANCE & BUSINESS ===
FINANCE_TOPICS = [
    "budget planning",
    "emergency fund",
    "debt payoff",
    "credit score",
    "investment basics",
    "stock market",
    "retirement planning",
    "401k",
    "IRA",
    "tax deductions",
    "filing taxes",
    "side hustle",
    "freelance income",
    "mortgage rates",
    "home buying",
    "rent vs buy",
    "real estate investing",
    "credit cards",
    "rewards programs",
    "travel hacking",
    "points maximization",
    "small business",
    "LLC setup",
    "business plan",
    "marketing strategy",
    "negotiation skills",
    "salary raise",
    "job interview",
    "career change",
    "passive income",
    "dividend stocks",
    "index funds",
    "ETF investing",
    "financial independence",
    "FIRE movement",
    "frugal living",
    "minimalism",
]

BUSINESS_ACTIVITIES = [
    "create budget",
    "save money",
    "pay off debt",
    "improve credit",
    "start investing",
    "buy stocks",
    "plan retirement",
    "reduce taxes",
    "start business",
    "write business plan",
    "market product",
    "hire employees",
    "negotiate salary",
    "prepare interview",
    "switch careers",
    "build network",
    "negotiate price",
    "close deal",
    "manage team",
    "increase sales",
]

# === HOME & GARDEN ===
HOME_TOPICS = [
    "organize closet",
    "declutter home",
    "minimalist living",
    "storage solutions",
    "clean efficiently",
    "deep cleaning",
    "laundry tips",
    "stain removal",
    "home repair",
    "fix leaky faucet",
    "unclog drain",
    "patch drywall",
    "paint walls",
    "choose paint color",
    "interior design",
    "furniture arrangement",
    "small space",
    "apartment living",
    "studio setup",
    "home office",
    "garden planning",
    "vegetable garden",
    "herb growing",
    "indoor plants",
    "composting",
    "raised beds",
    "pest control",
    "organic gardening",
    "flower arranging",
    "landscaping",
    "lawn care",
    "pruning trees",
    "tool maintenance",
    "basic carpentry",
    "electrical basics",
    "plumbing 101",
]

# === FOOD & COOKING ===
FOOD_TOPICS = [
    "meal prep",
    "batch cooking",
    "knife skills",
    "cooking techniques",
    "baking bread",
    "sourdough starter",
    "cake decorating",
    "pastry making",
    "meal planning",
    "grocery shopping",
    "food storage",
    "meal ideas",
    "healthy recipes",
    "quick dinners",
    "breakfast ideas",
    "lunch prep",
    "dietary restrictions",
    "gluten free",
    "vegan cooking",
    "keto meals",
    "international cuisine",
    "Italian recipes",
    "Asian cooking",
    "Mexican food",
    "spice combinations",
    "flavor pairing",
    "wine pairing",
    "coffee brewing",
    "fermentation",
    "pickling vegetables",
    "making cheese",
    "home brewing",
    "grilling",
    "BBQ techniques",
    "smoking meat",
    "outdoor cooking",
]

# === TRAVEL & GEOGRAPHY ===
TRAVEL_TOPICS = [
    "budget travel",
    "solo travel",
    "backpacking",
    "road trip planning",
    "travel insurance",
    "passport renewal",
    "visa requirements",
    "travel documents",
    "packing light",
    "carry on essentials",
    "travel gear",
    "luggage selection",
    "travel photography",
    "scenic routes",
    "hidden gems",
    "local experiences",
    "cultural etiquette",
    "language basics",
    "travel phrases",
    "translation apps",
    "countries visited",
    "bucket list destinations",
    "seven wonders",
    "UNESCO sites",
    "capital cities",
    "world geography",
    "climate zones",
    "time zones",
    "currency exchange",
    "travel banking",
    "avoid fees",
    "travel rewards",
    "jet lag",
    "adjust timezone",
    "stay healthy",
    "travel safety",
]

# === HOBBIES & CRAFTS ===
HOBBY_TOPICS = [
    "photography basics",
    "camera settings",
    "photo editing",
    "composition rules",
    "drawing techniques",
    "sketching",
    "watercolor",
    "acrylic painting",
    "knitting",
    "crochet",
    "sewing",
    "embroidery",
    "cross stitch",
    "woodworking",
    "wood carving",
    "furniture making",
    "wood finishing",
    "pottery",
    "ceramics",
    "clay sculpting",
    "wheel throwing",
    "jewelry making",
    "beading",
    "wire wrapping",
    "metal smithing",
    "playing guitar",
    "piano lessons",
    "music theory",
    "songwriting",
    "bird watching",
    "stargazing",
    "astronomy",
    "telescope selection",
    "board games",
    "chess strategy",
    "puzzle solving",
    "escape rooms",
    "fishing",
    "fly tying",
    "bait selection",
    "fishing spots",
    "hiking",
    "trail finding",
    "orienteering",
    "survival skills",
    "camping",
    "tent setup",
    "campfire cooking",
    "outdoor gear",
    "calligraphy",
    "hand lettering",
    "font design",
    "typography",
    "DIY projects",
    "upcycling",
    "furniture restoration",
    "home decor",
]

# === EDUCATION & LEARNING ===
EDUCATION_TOPICS = [
    "study techniques",
    "memory techniques",
    "speed reading",
    "note taking",
    "learning languages",
    "Spanish basics",
    "French phrases",
    "Mandarin tones",
    "online courses",
    "MOOC platforms",
    "certification prep",
    "exam strategy",
    "math fundamentals",
    "calculus basics",
    "statistics",
    "probability",
    "physics concepts",
    "chemistry basics",
    "biology",
    "earth science",
    "history timeline",
    "ancient civilizations",
    "world wars",
    "modern history",
    "literature classics",
    "book recommendations",
    "reading list",
    "literary analysis",
    "public speaking",
    "presentation skills",
    "storytelling",
    "communication",
    "critical thinking",
    "logical reasoning",
    "problem solving",
    "decision making",
    "time management",
    "productivity systems",
    "focus techniques",
    "deep work",
    "learning styles",
    "visual learner",
    "auditory learner",
    "kinesthetic",
]

# === ARTS & CULTURE ===
ARTS_TOPICS = [
    "art history",
    "renaissance art",
    "impressionism",
    "modern art",
    "museum visits",
    "gallery etiquette",
    "art appreciation",
    "criticism",
    "classical music",
    "orchestra instruments",
    "opera",
    "ballet",
    "jazz history",
    "blues music",
    "rock music",
    "classical composers",
    "film analysis",
    "cinema history",
    "directors",
    "film genres",
    "theater",
    "plays",
    "Shakespeare",
    "acting techniques",
    "dance styles",
    "ballroom dancing",
    "salsa",
    "contemporary dance",
    "architecture styles",
    "Gothic",
    "Baroque",
    "Modern",
    "Art Deco",
    "fashion history",
    "style guide",
    "color theory",
    "design principles",
    "poetry forms",
    "haiku",
    "sonnet",
    "free verse",
    "spoken word",
    "cultural festivals",
    "holiday traditions",
    "world celebrations",
    "customs",
]

# === LIFESTYLE & RELATIONSHIPS ===
LIFESTYLE_TOPICS = [
    "work life balance",
    "morning routine",
    "evening routine",
    "habit formation",
    "digital detox",
    "screen time",
    "social media",
    "information diet",
    "minimalism",
    "simple living",
    "intentional living",
    "slow living",
    "dating advice",
    "relationship communication",
    "conflict resolution",
    "trust building",
    "parenting tips",
    "child development",
    "teenager advice",
    "empty nest",
    "friendship maintenance",
    "social skills",
    "networking",
    "introvert tips",
    "family dynamics",
    "sibling relationships",
    "in-laws",
    "family gatherings",
    "pet care",
    "dog training",
    "cat behavior",
    "pet health",
    "event planning",
    "dinner party",
    "birthday celebration",
    "wedding planning",
    "etiquette rules",
    "dining manners",
    "thank you notes",
    "gift giving",
    "personal style",
    "wardrobe basics",
    "capsule wardrobe",
    "seasonal fashion",
]

# === TECHNOLOGY (minimal 5%) ===
TECH_TOPICS = [
    "email etiquette",
    "video calls",
    "password manager",
    "backup data",
    "smartphone tips",
    "app recommendations",
    "digital organization",
    "file management",
    "basic troubleshooting",
    "wifi setup",
    "printer issues",
    "software updates",
    "online privacy",
    "social media privacy",
    "two factor auth",
    "secure browsing",
]

# === SHORT KEYWORD QUERIES (for all categories) ===
SHORT_KEYWORDS = [
    # Health
    "meditate",
    "hydrate",
    "stretch",
    "exercise",
    "sleep",
    "vitamins",
    "protein",
    # Finance
    "budget",
    "save",
    "invest",
    "taxes",
    "debt",
    "credit",
    "retirement",
    # Home
    "clean",
    "organize",
    "repair",
    "paint",
    "garden",
    "compost",
    "prune",
    # Food
    "cook",
    "bake",
    "recipe",
    "meal",
    "diet",
    "nutrition",
    "spices",
    # Travel
    "travel",
    "pack",
    "passport",
    "visa",
    "hotel",
    "flight",
    "itinerary",
    # Hobbies
    "photo",
    "draw",
    "paint",
    "knit",
    "woodwork",
    "guitar",
    "hike",
    "camp",
    # Education
    "study",
    "learn",
    "read",
    "course",
    "exam",
    "degree",
    "certificate",
    # Arts
    "art",
    "music",
    "film",
    "dance",
    "theater",
    "museum",
    "gallery",
    # Life
    "habit",
    "routine",
    "minimal",
    "organize",
    "relationship",
    "dating",
    "parent",
    # Tech (minimal)
    "email",
    "wifi",
    "backup",
    "password",
    "update",
]


def generate_query(category: str) -> str:
    """Generate a query for a specific category."""

    templates = {
        "health_wellness": [
            "how to {activity}",
            "best {topic}",
            "{topic} guide",
            "{topic} for beginners",
            "improve {topic}",
        ],
        "finance_business": [
            "how to {activity}",
            "{topic} basics",
            "{topic} strategy",
            "start {topic}",
            "{topic} tips",
        ],
        "home_garden": [
            "how to {topic}",
            "DIY {topic}",
            "{topic} ideas",
            "best {topic}",
            "{topic} tutorial",
        ],
        "food_cooking": [
            "how to {topic}",
            "{topic} recipe",
            "best {topic}",
            "{topic} techniques",
            "learn {topic}",
        ],
        "travel_geography": [
            "{topic} guide",
            "how to {topic}",
            "best {topic}",
            "{topic} tips",
            "plan {topic}",
        ],
        "hobbies_crafts": [
            "learn {topic}",
            "{topic} basics",
            "how to {topic}",
            "{topic} for beginners",
            "best {topic}",
        ],
        "education_learning": [
            "learn {topic}",
            "{topic} guide",
            "improve {topic}",
            "{topic} techniques",
            "study {topic}",
        ],
        "arts_culture": [
            "understand {topic}",
            "{topic} guide",
            "appreciate {topic}",
            "learn {topic}",
            "{topic} history",
        ],
        "lifestyle_relationships": [
            "how to {topic}",
            "improve {topic}",
            "{topic} advice",
            "{topic} tips",
            "best {topic}",
        ],
        "technology": [
            "how to {topic}",
            "{topic} setup",
            "fix {topic}",
            "{topic} tips",
        ],
    }

    word_lists = {
        "health_wellness": (HEALTH_ACTIVITIES, HEALTH_TOPICS),
        "finance_business": (BUSINESS_ACTIVITIES, FINANCE_TOPICS),
        "home_garden": (HOME_TOPICS, HOME_TOPICS),
        "food_cooking": (FOOD_TOPICS, FOOD_TOPICS),
        "travel_geography": (TRAVEL_TOPICS, TRAVEL_TOPICS),
        "hobbies_crafts": (HOBBY_TOPICS, HOBBY_TOPICS),
        "education_learning": (EDUCATION_TOPICS, EDUCATION_TOPICS),
        "arts_culture": (ARTS_TOPICS, ARTS_TOPICS),
        "lifestyle_relationships": (LIFESTYLE_TOPICS, LIFESTYLE_TOPICS),
        "technology": (TECH_TOPICS, TECH_TOPICS),
    }

    template = random.choice(templates[category])
    activities, topics = word_lists[category]

    if "{activity}" in template:
        return template.format(activity=random.choice(activities))
    else:
        return template.format(topic=random.choice(topics))


def generate_short_query() -> str:
    """Generate a short 1-2 word query."""
    return random.choice(SHORT_KEYWORDS)


def generate_expansion(query: str, category: str) -> str:
    """Generate a realistic expansion for a query."""
    # Generate contextually appropriate lex/vec/hyde based on category

    domain_hints = {
        "health_wellness": "health medical wellness fitness nutrition exercise",
        "finance_business": "finance money investing business career salary budget",
        "home_garden": "home repair DIY garden organization cleaning maintenance",
        "food_cooking": "food cooking recipe culinary kitchen meal nutrition diet",
        "travel_geography": "travel trip vacation destination geography tourism explore",
        "hobbies_crafts": "hobby craft creative DIY leisure recreation skill learn",
        "education_learning": "education learn study course academic knowledge skill",
        "arts_culture": "art culture creative music film theater literature history",
        "lifestyle_relationships": "life lifestyle relationship social personal development habits",
        "technology": "tech digital computer software internet online tool app",
    }

    domain = domain_hints.get(category, "general")

    lex_variations = [
        f"{query} guide",
        f"{query} tips",
        f"{query} how to",
        f"{query} tutorial",
        f"{query} advice",
    ]

    vec_variations = [
        f"how to {query} effectively",
        f"best way to {query}",
        f"complete guide to {query}",
        f"learn {query} step by step",
        f"tips for {query} success",
    ]

    # Select variations
    selected_lex = random.sample(lex_variations, min(3, len(lex_variations)))
    selected_vec = random.sample(vec_variations, min(2, len(vec_variations)))

    # Generate hyde passage
    hyde_templates = [
        f"This comprehensive guide to {query} covers all the essential information you need to get started. Follow the steps carefully for best results.",
        f"Learning {query} requires practice and patience. This resource provides detailed instructions, examples, and tips to help you master the basics quickly.",
        f"Whether you're a beginner or looking to improve, this guide to {query} offers practical advice, common pitfalls to avoid, and proven strategies for success.",
    ]
    hyde = random.choice(hyde_templates)

    output_lines = []
    for lex in selected_lex:
        output_lines.append(f"lex: {lex}")
    for vec in selected_vec:
        output_lines.append(f"vec: {vec}")
    output_lines.append(f"hyde: {hyde}")

    return "\n".join(output_lines)


def main():
    """Generate balanced diverse training examples."""
    output_file = Path("data/qmd_expansion_balanced.jsonl")

    # Generate 500 examples with balanced distribution
    target_count = 500
    print(f"Generating {target_count} balanced training examples...")
    print(f"Tech focus reduced to {CATEGORY_WEIGHTS['technology']:.0%}")
    print()

    # Show distribution
    for cat, weight in CATEGORY_WEIGHTS.items():
        count = int(target_count * weight)
        bar = "█" * int(weight * 40)
        print(f"  {cat:25} {count:3} ({weight:4.0%}) {bar}")
    print()

    examples = []
    category_counts = {cat: 0 for cat in CATEGORY_WEIGHTS.keys()}

    for i in range(target_count):
        # Select category based on weights
        categories = list(CATEGORY_WEIGHTS.keys())
        weights = list(CATEGORY_WEIGHTS.values())
        category = random.choices(categories, weights=weights, k=1)[0]

        # 20% of queries should be short (1-2 words)
        if random.random() < 0.20:
            query = generate_short_query()
            short_flag = True
        else:
            query = generate_query(category)
            short_flag = False

        expansion = generate_expansion(query, category)

        output_items = normalize_output_items(parse_output_text(expansion))
        examples.append(
            {
                "query": query,
                "output": output_items,
                "category": category,
                "is_short": short_flag,
            }
        )

        category_counts[category] += 1

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{target_count} examples...")

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Saved {len(examples)} balanced examples to {output_file}")
    print("\nActual distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(examples)
        bar = "█" * int(pct * 40)
        print(f"  {cat:25} {count:3} ({pct:4.1%}) {bar}")

    short_count = sum(1 for ex in examples if ex["is_short"])
    print(
        f"\n  {'Short queries (1-2 words)':25} {short_count:3} ({short_count / len(examples):4.1%})"
    )

    print("\n📋 Usage:")
    print(f"  cat {output_file} >> data/qmd_expansion_v2.jsonl")
    print("  uv run dataset/prepare_data.py")


if __name__ == "__main__":
    main()
