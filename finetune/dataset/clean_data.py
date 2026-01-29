#!/usr/bin/env python3
"""
Data Quality Reviewer for Query Expansion Training Dataset

This script identifies and flags/fixes semantic errors where technical terms
are misunderstood. For example:
- "gem find" expanded as "mineral hunt" instead of "ruby gem search"
- "yarn spin" expanded as "wool twist" instead of "yarn package manager"

The script uses contextual analysis to detect when technical terms
are likely being used in a programming context vs. their everyday meaning.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class TechnicalTerm:
    """Definition of a technical term that might be misunderstood."""
    term: str  # The ambiguous term (e.g., "liquid", "gem", "yarn")
    context_indicators: list[str]  # Words that suggest tech context
    wrong_expansions: list[str]  # Patterns that indicate wrong interpretation
    correct_domain: str  # What domain this belongs to when technical
    correct_lex: list[str]  # Correct lex expansions
    correct_vec: list[str]  # Correct vec expansions


# Known technical terms that are commonly misunderstood
KNOWN_TECHNICAL_TERMS = [
    TechnicalTerm(
        term="liquid",
        context_indicators=["shopify", "template", "filter", "tag", "theme", "jekyll"],
        wrong_expansions=["fluid", "water", "pour", "drink", "beverage", "h2o", "wet"],
        correct_domain="Shopify/Jekyll templating language",
        correct_lex=["shopify template syntax", "liquid template filter"],
        correct_vec=["shopify liquid templating language", "liquid template engine filters"],
    ),
    TechnicalTerm(
        term="gem",
        context_indicators=["ruby", "bundler", "install", "gemfile", "rails", "require"],
        wrong_expansions=["mineral", "crystal", "jewel", "stone", "diamond", "jewelry", "precious"],
        correct_domain="Ruby package manager",
        correct_lex=["ruby gem package", "gem install command"],
        correct_vec=["ruby gem package manager", "rubygems library installation"],
    ),
    TechnicalTerm(
        term="yarn",
        context_indicators=["npm", "package", "install", "node", "javascript", "react", "webpack"],
        wrong_expansions=["thread", "wool", "knit", "spin", "textile", "fabric", "sew", "twist"],
        correct_domain="JavaScript package manager",
        correct_lex=["yarn package manager", "yarn install dependencies"],
        correct_vec=["yarn javascript package manager", "yarn npm alternative"],
    ),
    TechnicalTerm(
        term="hook",
        context_indicators=["react", "use", "state", "effect", "component", "callback", "git"],
        wrong_expansions=["fish", "fishing", "bait", "catch", "hang", "pirate"],
        correct_domain="React hooks or Git hooks",
        correct_lex=["react hooks api", "usestate useeffect"],
        correct_vec=["react hooks state management", "react functional component hooks"],
    ),
    TechnicalTerm(
        term="container",
        context_indicators=["docker", "kubernetes", "k8s", "image", "orchestration", "pod"],
        wrong_expansions=["box", "storage", "shipping", "cargo", "tupperware", "jar", "vessel"],
        correct_domain="Docker/Kubernetes containers",
        correct_lex=["docker container", "container image"],
        correct_vec=["docker container virtualization", "container orchestration platform"],
    ),
    TechnicalTerm(
        term="branch",
        context_indicators=["git", "merge", "checkout", "commit", "main", "master", "repo"],
        wrong_expansions=["tree", "limb", "wood", "leaf", "twig", "forest"],
        correct_domain="Git version control",
        correct_lex=["git branch", "git checkout branch"],
        correct_vec=["git branch version control", "git branching workflow"],
    ),
    TechnicalTerm(
        term="decorator",
        context_indicators=["python", "@", "function", "wrapper", "class", "def"],
        wrong_expansions=["interior", "design", "paint", "furniture", "decor", "ornament"],
        correct_domain="Python decorators",
        correct_lex=["python decorator function", "@decorator syntax"],
        correct_vec=["python function decorators", "python decorator pattern"],
    ),
    TechnicalTerm(
        term="bean",
        context_indicators=["java", "spring", "injection", "dependency", "servlet", "ejb"],
        wrong_expansions=["coffee", "food", "vegetable", "legume", "plant", "soy"],
        correct_domain="Java Beans / Spring Beans",
        correct_lex=["java bean class", "spring bean injection"],
        correct_vec=["java enterprise beans", "spring dependency injection beans"],
    ),
    TechnicalTerm(
        term="shell",
        context_indicators=["bash", "script", "terminal", "command", "linux", "unix", "zsh"],
        wrong_expansions=["seashell", "ocean", "beach", "clam", "oyster", "egg"],
        correct_domain="Unix/Linux shell scripting",
        correct_lex=["bash shell script", "shell command"],
        correct_vec=["unix shell scripting", "bash command line shell"],
    ),
    TechnicalTerm(
        term="rust",
        context_indicators=["cargo", "crate", "ownership", "borrow", "lifetime", "unsafe"],
        wrong_expansions=["oxidation", "metal", "corrosion", "decay", "iron", "orange"],
        correct_domain="Rust programming language",
        correct_lex=["rust programming language", "rust cargo package"],
        correct_vec=["rust systems programming", "rust memory safety"],
    ),
    TechnicalTerm(
        term="go",
        context_indicators=["golang", "goroutine", "channel", "defer", "gofmt", "module"],
        wrong_expansions=["travel", "move", "walk", "game", "board game", "leave", "depart"],
        correct_domain="Go programming language",
        correct_lex=["golang programming", "go language syntax"],
        correct_vec=["go programming language", "golang concurrent programming"],
    ),
    TechnicalTerm(
        term="swift",
        context_indicators=["ios", "xcode", "apple", "uikit", "swiftui", "cocoa"],
        wrong_expansions=["fast", "quick", "bird", "speed", "rapid", "taylor"],
        correct_domain="Swift programming language",
        correct_lex=["swift ios development", "swift programming language"],
        correct_vec=["swift apple programming language", "swift ios app development"],
    ),
    TechnicalTerm(
        term="pod",
        context_indicators=["kubernetes", "k8s", "deployment", "service", "cluster", "node"],
        wrong_expansions=["pea", "seed", "plant", "vegetable", "legume", "whale"],
        correct_domain="Kubernetes pods",
        correct_lex=["kubernetes pod", "k8s pod deployment"],
        correct_vec=["kubernetes pod container group", "k8s pod orchestration"],
    ),
    TechnicalTerm(
        term="redis",
        context_indicators=["cache", "database", "key-value", "memory", "pub/sub", "queue"],
        wrong_expansions=[],  # "redis" doesn't have common wrong meanings
        correct_domain="Redis in-memory database",
        correct_lex=["redis cache", "redis database"],
        correct_vec=["redis in-memory data store", "redis caching solution"],
    ),
    TechnicalTerm(
        term="kafka",
        context_indicators=["message", "stream", "queue", "broker", "topic", "producer", "consumer"],
        wrong_expansions=["franz", "author", "writer", "novel", "metamorphosis", "literature"],
        correct_domain="Apache Kafka message queue",
        correct_lex=["apache kafka", "kafka message broker"],
        correct_vec=["apache kafka streaming platform", "kafka message queue"],
    ),
    TechnicalTerm(
        term="elastic",
        context_indicators=["elasticsearch", "search", "index", "kibana", "logstash", "query"],
        wrong_expansions=["stretch", "rubber", "flexible", "band", "bouncy"],
        correct_domain="Elasticsearch",
        correct_lex=["elasticsearch", "elastic search index"],
        correct_vec=["elasticsearch full-text search", "elastic stack"],
    ),
    TechnicalTerm(
        term="spark",
        context_indicators=["apache", "hadoop", "data", "rdd", "dataframe", "pyspark"],
        wrong_expansions=["fire", "ignite", "flame", "plug", "electricity"],
        correct_domain="Apache Spark",
        correct_lex=["apache spark", "spark data processing"],
        correct_vec=["apache spark big data processing", "spark cluster computing"],
    ),
    TechnicalTerm(
        term="flask",
        context_indicators=["python", "web", "route", "api", "jinja", "werkzeug"],
        wrong_expansions=["bottle", "container", "lab", "chemistry", "drink", "thermos"],
        correct_domain="Flask web framework",
        correct_lex=["flask python web framework", "flask api"],
        correct_vec=["flask python web development", "flask microframework"],
    ),
    TechnicalTerm(
        term="django",
        context_indicators=["python", "web", "orm", "model", "view", "template"],
        wrong_expansions=["jazz", "music", "reinhardt", "guitar", "movie", "western"],
        correct_domain="Django web framework",
        correct_lex=["django python framework", "django web development"],
        correct_vec=["django python web framework", "django orm models"],
    ),
    TechnicalTerm(
        term="rails",
        context_indicators=["ruby", "gem", "activerecord", "model", "controller", "migration"],
        wrong_expansions=["train", "track", "railroad", "railway", "metal"],
        correct_domain="Ruby on Rails",
        correct_lex=["ruby on rails", "rails web framework"],
        correct_vec=["ruby on rails framework", "rails mvc architecture"],
    ),
    TechnicalTerm(
        term="node",
        context_indicators=["javascript", "npm", "express", "async", "require", "module"],
        wrong_expansions=["lump", "knot", "bump", "growth", "junction"],
        correct_domain="Node.js",
        correct_lex=["node.js javascript", "nodejs runtime"],
        correct_vec=["node.js javascript runtime", "nodejs server-side javascript"],
    ),
    TechnicalTerm(
        term="maven",
        context_indicators=["java", "pom", "dependency", "build", "artifact", "repository"],
        wrong_expansions=["expert", "specialist", "connoisseur"],
        correct_domain="Apache Maven",
        correct_lex=["apache maven", "maven build tool"],
        correct_vec=["apache maven java build", "maven dependency management"],
    ),
    TechnicalTerm(
        term="gradle",
        context_indicators=["java", "kotlin", "android", "build", "groovy", "task"],
        wrong_expansions=["grade", "slope", "hill", "incline"],
        correct_domain="Gradle build tool",
        correct_lex=["gradle build tool", "gradle android"],
        correct_vec=["gradle java build automation", "gradle kotlin dsl"],
    ),
    TechnicalTerm(
        term="ant",
        context_indicators=["java", "build", "xml", "target", "task"],
        wrong_expansions=["insect", "bug", "colony", "hill", "picnic"],
        correct_domain="Apache Ant build tool",
        correct_lex=["apache ant", "ant build xml"],
        correct_vec=["apache ant java build", "ant build automation"],
    ),
]


@dataclass
class Issue:
    """Represents an issue found in a dataset example."""
    line_number: int
    input_text: str
    output_text: str
    issue_type: str
    technical_term: str
    wrong_expansion_found: str
    suggested_fix: Optional[str] = None


@dataclass
class AnalysisResult:
    """Results of analyzing the dataset."""
    total_examples: int = 0
    issues_found: list[Issue] = field(default_factory=list)
    examples_with_correct_tech_terms: list[tuple[int, str]] = field(default_factory=list)
    term_statistics: dict = field(default_factory=lambda: defaultdict(int))


def check_for_wrong_expansion(output_text: str, term: TechnicalTerm) -> Optional[str]:
    """Check if the output contains wrong expansions for a technical term."""
    output_lower = output_text.lower()
    for wrong in term.wrong_expansions:
        if wrong.lower() in output_lower:
            return wrong
    return None


def has_tech_context(input_text: str, term: TechnicalTerm) -> bool:
    """Check if the input has indicators of a technical context."""
    input_lower = input_text.lower()
    for indicator in term.context_indicators:
        if indicator.lower() in input_lower:
            return True
    return False


def is_likely_tech_query(input_text: str) -> bool:
    """
    Heuristic to determine if a short query is likely tech-related.
    Short queries like "gem find" or "yarn spin" are ambiguous.
    """
    tech_patterns = [
        r'\b(install|config|setup|build|run|debug|test|deploy|compile)\b',
        r'\b(api|cli|sdk|lib|pkg|npm|pip|cargo)\b',
        r'\b(func|class|method|var|const|let|def)\b',
        r'\b(http|https|url|port|host|server|client)\b',
        r'\b(json|xml|yaml|csv|sql|html|css|js)\b',
    ]
    input_lower = input_text.lower()
    for pattern in tech_patterns:
        if re.search(pattern, input_lower):
            return True
    return False


def has_non_tech_context(input_text: str, term: TechnicalTerm) -> bool:
    """
    Check if the input clearly indicates a non-technical context.
    This helps avoid false positives for words like "car rust", "yarn spin", etc.
    """
    input_lower = input_text.lower()
    term_lower = term.term.lower()

    # Define non-tech context indicators for each ambiguous term
    non_tech_contexts = {
        "rust": ["car", "metal", "iron", "steel", "corrosion", "prevention", "remove", "body"],
        "gem": ["gemstone", "jewelry", "jewel", "diamond", "precious", "stone", "cut", "shop", "buy", "wear"],
        "yarn": ["knit", "crochet", "spin", "wool", "thread", "textile", "fabric", "sew", "weave"],
        "hook": ["fishing", "crochet", "hang", "coat", "wall", "ceiling"],
        "container": ["storage", "plastic", "food", "shipping", "cargo", "kitchen", "box"],
        "branch": ["tree", "bank", "library", "store", "office", "organization"],
        "decorator": ["interior", "home", "room", "house", "design", "party", "cake", "wedding"],
        "bean": ["coffee", "soy", "kidney", "black", "green", "garden", "cooking", "food", "plant", "grow"],
        "shell": ["sea", "beach", "egg", "nut", "turtle", "snail", "crab", "clam", "oyster"],
        "spark": ["plug", "fire", "ignite", "car", "engine", "electric", "romance"],
        "go": ["travel", "vacation", "trip", "walk", "run", "leave", "visit", "tour"],
        "swift": ["taylor", "concert", "music", "singer", "speed", "fast", "bird"],
        "pod": ["pea", "whale", "orca", "dolphin", "vegetable", "seed", "plant"],
        "ant": ["insect", "colony", "fire", "carpenter", "pest", "bug", "picnic"],
        "node": ["lymph", "medical", "body", "tree", "network point"],
        "rails": ["train", "railroad", "railway", "track", "transit", "fence"],
        "flask": ["lab", "chemistry", "drink", "hip", "thermos", "bottle", "water"],
        "django": ["jazz", "music", "reinhardt", "guitar", "movie", "western", "unchained"],
        "maven": ["expert", "connoisseur", "specialist", "guru"],
        "gradle": ["grade", "school", "slope"],
        "kafka": ["franz", "author", "novel", "metamorphosis", "literature", "writer", "book"],
        "elastic": ["band", "rubber", "stretch", "flexible", "waist", "fabric"],
    }

    if term_lower in non_tech_contexts:
        for context_word in non_tech_contexts[term_lower]:
            if context_word.lower() in input_lower:
                return True

    return False


def analyze_example(line_num: int, input_text: str, output_text: str) -> list[Issue]:
    """Analyze a single example for potential issues."""
    issues = []
    input_lower = input_text.lower()

    for term in KNOWN_TECHNICAL_TERMS:
        term_lower = term.term.lower()

        # Check if the input contains this technical term
        if term_lower not in input_lower:
            continue

        # Check if output has wrong expansion
        wrong_expansion = check_for_wrong_expansion(output_text, term)
        if wrong_expansion is None:
            continue

        # Skip if the context clearly indicates non-technical usage
        if has_non_tech_context(input_text, term):
            continue

        # Determine if this is likely a technical context
        is_tech = has_tech_context(input_text, term) or is_likely_tech_query(input_text)

        # For very short inputs that contain ONLY the tech term (like "gem find"),
        # these are ambiguous and could be tech-related
        word_count = len(input_text.split())
        words = [w.lower() for w in input_text.split()]

        # Only flag if it's clearly a tech context OR a very short query
        # where the term appears prominently (e.g., "gem find", "yarn add")
        if is_tech:
            # Create suggested fix for definite tech issues
            suggested_output = f"lex: {term.correct_lex[0]}\nlex: {term.correct_lex[1] if len(term.correct_lex) > 1 else term.correct_lex[0]}\nvec: {term.correct_vec[0]}\nvec: {term.correct_vec[1] if len(term.correct_vec) > 1 else term.correct_vec[0]}\nhyde: {term.correct_domain} is a concept that provides functionality for software development."

            issue = Issue(
                line_number=line_num,
                input_text=input_text,
                output_text=output_text[:200] + "..." if len(output_text) > 200 else output_text,
                issue_type="wrong_tech_expansion",
                technical_term=term.term,
                wrong_expansion_found=wrong_expansion,
                suggested_fix=suggested_output
            )
            issues.append(issue)
        elif word_count <= 2 and term_lower in words:
            # Very short query with the term as a primary word - truly ambiguous
            issue = Issue(
                line_number=line_num,
                input_text=input_text,
                output_text=output_text[:200] + "..." if len(output_text) > 200 else output_text,
                issue_type="ambiguous_term",
                technical_term=term.term,
                wrong_expansion_found=wrong_expansion,
                suggested_fix=None
            )
            issues.append(issue)

    return issues


def analyze_dataset(filepath: Path) -> AnalysisResult:
    """Analyze the entire dataset for issues."""
    result = AnalysisResult()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                input_text = example.get('input', '')
                output_text = example.get('output', '')

                result.total_examples += 1

                # Analyze for issues
                issues = analyze_example(line_num, input_text, output_text)
                result.issues_found.extend(issues)

                # Track term statistics
                for term in KNOWN_TECHNICAL_TERMS:
                    if term.term.lower() in input_text.lower():
                        result.term_statistics[term.term] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")

    return result


def fix_example(example: dict, issues: list[Issue]) -> Optional[dict]:
    """
    Attempt to fix an example based on identified issues.
    Returns None if no fix is needed or possible.
    """
    # Only fix examples with definite tech context issues
    tech_issues = [i for i in issues if i.issue_type == "wrong_tech_expansion" and i.suggested_fix]

    if not tech_issues:
        return None

    # Use the first tech issue's fix (they should be similar)
    issue = tech_issues[0]

    fixed = example.copy()
    fixed['output'] = issue.suggested_fix
    fixed['_fixed'] = True
    fixed['_original_output'] = example['output']
    fixed['_fix_reason'] = f"Technical term '{issue.technical_term}' was incorrectly expanded as '{issue.wrong_expansion_found}'"

    return fixed


def generate_report(result: AnalysisResult) -> str:
    """Generate a human-readable report of the analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("QUERY EXPANSION DATASET QUALITY REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total examples analyzed: {result.total_examples}")
    lines.append(f"Issues found: {len(result.issues_found)}")
    lines.append("")

    # Group issues by type
    by_type = defaultdict(list)
    for issue in result.issues_found:
        by_type[issue.issue_type].append(issue)

    lines.append("-" * 70)
    lines.append("ISSUES BY TYPE:")
    lines.append("-" * 70)

    for issue_type, issues in by_type.items():
        lines.append(f"\n{issue_type.upper()}: {len(issues)} issues")
        lines.append("-" * 40)

        # Show up to 10 examples per type
        for issue in issues[:10]:
            lines.append(f"\n  Line {issue.line_number}:")
            lines.append(f"    Input: {issue.input_text}")
            lines.append(f"    Technical term: '{issue.technical_term}'")
            lines.append(f"    Wrong expansion found: '{issue.wrong_expansion_found}'")
            if issue.suggested_fix:
                lines.append(f"    Suggested fix available: Yes")

        if len(issues) > 10:
            lines.append(f"\n  ... and {len(issues) - 10} more")

    # Term statistics
    lines.append("\n" + "-" * 70)
    lines.append("TECHNICAL TERM OCCURRENCES IN DATASET:")
    lines.append("-" * 70)

    for term, count in sorted(result.term_statistics.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"  {term}: {count} occurrences")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def save_cleaned_dataset(filepath: Path, output_path: Path, result: AnalysisResult):
    """Save a cleaned version of the dataset."""
    issues_by_line = defaultdict(list)
    for issue in result.issues_found:
        issues_by_line[issue.line_number].append(issue)

    fixed_count = 0
    flagged_count = 0

    with open(filepath, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)

                if line_num in issues_by_line:
                    issues = issues_by_line[line_num]
                    fixed = fix_example(example, issues)

                    if fixed:
                        f_out.write(json.dumps(fixed) + '\n')
                        fixed_count += 1
                    else:
                        # Flag but don't fix ambiguous cases
                        example['_flagged'] = True
                        example['_flag_reason'] = f"Ambiguous term '{issues[0].technical_term}' may need review"
                        f_out.write(json.dumps(example) + '\n')
                        flagged_count += 1
                else:
                    f_out.write(json.dumps(example) + '\n')

            except json.JSONDecodeError:
                # Keep problematic lines as-is
                f_out.write(line + '\n')

    return fixed_count, flagged_count


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "data" / "qmd_expansion.jsonl"
    output_path = script_dir / "data" / "qmd_expansion_cleaned.jsonl"
    report_path = script_dir / "data" / "quality_report.txt"

    print(f"Analyzing dataset: {input_path}")
    print("-" * 50)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Analyze the dataset
    result = analyze_dataset(input_path)

    # Generate and print report
    report = generate_report(result)
    print(report)

    # Save report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save cleaned dataset
    fixed_count, flagged_count = save_cleaned_dataset(input_path, output_path, result)

    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"  - Examples fixed: {fixed_count}")
    print(f"  - Examples flagged for review: {flagged_count}")
    print(f"  - Examples unchanged: {result.total_examples - fixed_count - flagged_count}")

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total examples: {result.total_examples}")
    print(f"Total issues found: {len(result.issues_found)}")

    tech_issues = [i for i in result.issues_found if i.issue_type == "wrong_tech_expansion"]
    ambig_issues = [i for i in result.issues_found if i.issue_type == "ambiguous_term"]

    print(f"  - Definite tech term errors: {len(tech_issues)}")
    print(f"  - Ambiguous terms needing review: {len(ambig_issues)}")

    if len(result.issues_found) > 0:
        error_rate = len(result.issues_found) / result.total_examples * 100
        print(f"\nError rate: {error_rate:.2f}%")

    return 0


if __name__ == "__main__":
    exit(main())
