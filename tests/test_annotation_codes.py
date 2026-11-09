
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profile_annotations import parse_annotation_codes


class CodesTest(unittest.TestCase):

    def test_official(self):
        cases = [
            ("[F繁][C][BC，]", ["F", "C", "BC"]),
            ("{CJba}{CJbei}", ["CJba", "CJbei"]),
            ("{CC2甲}{W3abc}{CJ-zhuy人}{CJ+dy的}",
             ["CC2", "W3", "CJ-zhuy", "CJ+dy"]),
            ("[Pshang]{W2Email}", ["P", "W2"]),
            ("[#]{CJ？}", ["#", "CJ？"]),
            ("{CP正文[C]P}", ["CP", "C"]),
            ("{CP未完[C]", ["CP", "C"]),
            ("[F 樸[B 僕]]", ["F", "B"]),
            ("[C][C]", ["C", "C"]),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                codes, _ = parse_annotation_codes(text)
                self.assertEqual(codes, expected)

    def test_review(self):
        cases = [
            ("{CJ}", "pending_code"),
            ("{ZZ甲}", "unknown_code"),
            ("[C", "incomplete_or_mismatched_bracket"),
            ("{CC41940}", "ambiguous_numeric_boundary"),
            ("{CJbogus}", "ambiguous_code_boundary"),
        ]

        for text, reason in cases:
            with self.subTest(text=text):
                codes, review = parse_annotation_codes(text)
                self.assertEqual(codes, [])
                self.assertEqual(review[0]["reason"], reason)


if __name__ == "__main__":
    unittest.main()

# AI辅助生成：GPT-6, 2026-11-8