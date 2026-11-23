import json
import os
import unittest

os.environ["OPENAI_API_KEY"] = ""

from app.agents.relevance import RelevanceJudgeAgent
from app.agents.accuracy import AccuracyJudgeAgent
from app.agents.hallucination import HallucinationDetectionAgent


class Milestone2AgentConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(
            os.path.join(os.path.dirname(__file__), "milestone2_cases.json"),
            encoding="utf-8",
        ) as case_file:
            cls.cases = json.load(case_file)
        cls.relevance = RelevanceJudgeAgent()
        cls.accuracy = AccuracyJudgeAgent()
        cls.hallucination = HallucinationDetectionAgent()

    def test_representative_benchmark_categories(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                relevance = self.relevance.evaluate(
                    case["question"], case["response"], case["source_document"])
                accuracy = self.accuracy.evaluate(
                    case["question"], case["response"],
                    case["reference_answer"], case["source_document"])
                hallucination = self.hallucination.evaluate(
                    case["question"], case["response"],
                    case["reference_answer"], case["source_document"])

                expected = case["expected"]
                if "relevance_min" in expected:
                    self.assertGreaterEqual(relevance["score"], expected["relevance_min"])
                if "relevance_max" in expected:
                    self.assertLessEqual(relevance["score"], expected["relevance_max"])
                if "accuracy_min" in expected:
                    self.assertGreaterEqual(accuracy["score"], expected["accuracy_min"])
                if "accuracy_max" in expected:
                    self.assertLessEqual(accuracy["score"], expected["accuracy_max"])
                self.assertEqual(
                    hallucination["hallucination_status"],
                    expected["hallucination_status"],
                )

    def test_outputs_include_reasoning_and_evidence(self):
        case = self.cases[0]
        relevance = self.relevance.evaluate(case["question"], case["response"], case["source_document"])
        accuracy = self.accuracy.evaluate(
            case["question"], case["response"], case["reference_answer"], case["source_document"])
        hallucination = self.hallucination.evaluate(
            case["question"], case["response"], case["reference_answer"], case["source_document"])

        self.assertIn("explanation", relevance)
        self.assertIn("supporting_evidence", accuracy)
        self.assertIn("claim_assessment", accuracy)
        self.assertIn("flagged_claims", hallucination)
        self.assertIn("claim_assessment", hallucination)
        self.assertIn("supporting_evidence", hallucination)

    def test_supported_and_unsupported_claims_are_separate(self):
        result = self.hallucination.evaluate(
            "What is the capital of France?",
            "Paris is the capital of France and was founded by Napoleon in 1804.",
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        )
        statuses = [item["status"] for item in result["claim_assessment"]]
        self.assertIn("supported", statuses)
        self.assertTrue(result["flagged_claims"])
        self.assertTrue(result["supporting_evidence"])


if __name__ == "__main__":
    unittest.main()
