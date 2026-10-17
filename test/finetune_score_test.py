"""Regression tests for the standalone GEPA scoring command."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "finetune/experiments/gepa/score.py"


class ScoreCommandTests(unittest.TestCase):
    def run_score(self, contents, *, help_only=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            if contents is not None:
                path.write_text(contents, encoding="utf-8")
            args = ["--help"] if help_only else ["--input", str(path)]
            return subprocess.run(
                [sys.executable, str(SCRIPT), *args],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )

    def test_scoring_and_sorted_ratings(self):
        examples = [
            {
                "query": "sqlite search",
                "output": [["lex", "sqlite"], ["vec", "find text in sqlite"]],
            },
            {"query": "configure auth", "output": [["lex", "unrelated"]]},
        ]
        contents = "\n".join(json.dumps(example) for example in examples)
        result = self.run_score(contents + "\n\nnot json\n{}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "2 scored, 2 errors, avg 75.0, median 75.0, "
            "min 58.3, max 91.7, >=70 50.0%",
            result.stdout,
        )
        self.assertIn("  ratings: Acceptable:1, Excellent:1\n", result.stdout)

    def test_empty_input(self):
        result = self.run_score("\n  \n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 scored, 0 errors", result.stdout)
        self.assertNotIn("ratings:", result.stdout)

    def test_invalid_input(self):
        result = self.run_score("not json\n{}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 scored, 2 errors", result.stdout)
        self.assertNotIn("ratings:", result.stdout)

    def test_missing_input(self):
        result = self.run_score(None)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Input not found:", result.stdout)

    def test_help(self):
        result = self.run_score(None, help_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Score GEPA JSONL outputs", result.stdout)


if __name__ == "__main__":
    unittest.main()
