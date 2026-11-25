import asyncio
import importlib
import sys
import unittest
from unittest.mock import patch

from app.agents.completeness import CompletenessJudgeAgent
from app.agents.verdict import VerdictAgent
from app.models import BatchEvaluationRequest, EvaluationRequest
from app.utils.scoring import ScoringUtils
from ui.batch_utils import parse_evaluation_csv


class CompletenessJudgeTests(unittest.TestCase):
    def test_fallback_identifies_unaddressed_question_aspects(self):
        with patch("app.agents.completeness.config.is_llm_configured", return_value=False):
            result = CompletenessJudgeAgent().evaluate(
                "What are the causes and effects of climate change?",
                "Climate change changes temperatures.",
            )

        aspects = {item["aspect"] for item in result["omissions"]}
        self.assertIn("causes", aspects)
        self.assertIn("effects", aspects)
        self.assertTrue(all(item["reason"] for item in result["omissions"]))
        self.assertLess(result["score"], 8)

    def test_llm_omissions_are_parsed_as_aspect_and_reason(self):
        text = (
            "Score: 5\nExplanation: The answer omits key parts.\n"
            "Omissions:\n- Effects: the question also asks for consequences.\n"
        )
        agent = CompletenessJudgeAgent()

        self.assertEqual(agent._extract_score(text), 5.0)
        self.assertEqual(
            agent._extract_explanation(text), "The answer omits key parts."
        )
        self.assertEqual(
            agent._extract_omissions(text),
            [{
                "aspect": "Effects",
                "reason": "the question also asks for consequences.",
            }],
        )

    def test_empty_response_reports_missing_question_aspects(self):
        with patch("app.agents.completeness.config.is_llm_configured", return_value=False):
            result = CompletenessJudgeAgent().evaluate(
                "List causes and effects.", ""
            )

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(
            {item["aspect"] for item in result["omissions"]},
            {"causes", "effects"},
        )


class VerdictAgentTests(unittest.TestCase):
    def test_weighted_verdict_includes_consolidated_reasoning(self):
        results = {
            "relevance": {"score": 8, "explanation": "It addresses the question."},
            "accuracy": {"score": 6, "explanation": "Some details are unsupported."},
            "hallucination": {
                "score": 9,
                "explanation": "One claim is not grounded.",
                "flagged_claims": [{"claim": "Unsupported detail"}],
            },
            "completeness": {"score": 4, "explanation": "Two requested aspects are missing."},
        }

        verdict = VerdictAgent().generate_verdict(results)

        self.assertEqual(verdict["verdict"], "Needs Improvement")
        self.assertAlmostEqual(verdict["weighted_score"], 6.85)
        self.assertEqual(verdict["average_score"], verdict["weighted_score"])
        self.assertAlmostEqual(sum(verdict["weights"].values()), 1)
        self.assertIn("completeness", verdict["reasoning_summary"])
        self.assertIn("flagged 1 claim", verdict["reasoning_summary"])

    def test_verdict_rejects_scores_outside_the_scale(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 10"):
            ScoringUtils.weighted_score({"relevance": 11})


class BatchCsvTests(unittest.TestCase):
    def test_parser_accepts_answer_alias_and_optional_evidence(self):
        rows = parse_evaluation_csv(
            b"\xef\xbb\xbfquestion,answer,reference_answer,source_document\n"
            b'"What is 2 + 2?","4","The answer is 4.","Math reference"\n'
        )

        self.assertEqual(rows, [{
            "question": "What is 2 + 2?",
            "ai_response": "4",
            "reference_answer": "The answer is 4.",
            "source_document": "Math reference",
        }])

    def test_parser_reports_missing_columns_and_empty_rows(self):
        with self.assertRaisesRegex(ValueError, "missing required column"):
            parse_evaluation_csv(b"question,response\nQuestion?,Answer.\n")
        with self.assertRaisesRegex(ValueError, "row 2"):
            parse_evaluation_csv(b"question,ai_response\nQuestion?,\n")


class BatchEndpointTests(unittest.TestCase):
    def test_batch_endpoint_keeps_successes_when_a_row_fails(self):
        with patch("app.agents.orchestrator.EvaluationOrchestrator"):
            evaluation_api = importlib.import_module("app.api.evaluation")

        try:
            successful_evaluation = {
                "question": "Question one?",
                "ai_response": "Answer one.",
                "agent_results": {},
                "retrieved_context": "",
                "final_verdict": {
                    "scores": {"relevance": 8},
                    "verdict": "Pass",
                    "weighted_score": 8,
                    "average_score": 8,
                    "reasoning_summary": "Good response.",
                },
            }
            with patch.object(
                evaluation_api.orchestrator,
                "evaluate",
                side_effect=[successful_evaluation, RuntimeError("test failure")],
            ), self.assertLogs("app.api.evaluation", level="ERROR"):
                response = asyncio.run(evaluation_api.submit_batch_evaluation(
                    BatchEvaluationRequest(evaluations=[
                        EvaluationRequest(question="Question one?", ai_response="Answer one."),
                        EvaluationRequest(question="Question two?", ai_response="Answer two."),
                    ])
                ))
        finally:
            sys.modules.pop("app.api.evaluation", None)
            import app.api
            if hasattr(app.api, "evaluation"):
                del app.api.evaluation

        self.assertEqual(response["total"], 2)
        self.assertEqual(response["succeeded"], 1)
        self.assertEqual(response["failed"], 1)
        self.assertEqual(response["results"][0]["result"]["verdict"], "Pass")
        self.assertIn("error", response["results"][1])


if __name__ == "__main__":
    unittest.main()
