from typing import Dict, Any
from app.utils.scoring import ScoringUtils


class VerdictAgent:
    """Aggregates dimension scores into a weighted quality verdict."""

    def __init__(self):
        self.scoring_utils = ScoringUtils()

    def generate_verdict(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a Pass, Needs Improvement, or Fail verdict."""
        scores = {}
        details = {}

        for agent_name, result in agent_results.items():
            dimension = result.get("dimension", agent_name)
            score = result.get("score")
            explanation = result.get("explanation", "")
            if score is None:
                raise ValueError(f"Missing score for {dimension}")
            scores[dimension] = score
            details[f"{dimension}_explanation"] = explanation

        if not scores:
            return {
                "verdict": "Unable to evaluate",
                "average_score": 0.0,
                "weighted_score": 0.0,
                "scores": {},
                "weights": {},
                "details": details,
                "reasoning_summary": "No dimension scores were provided.",
            }

        weighted_score, weights = self.scoring_utils.weighted_score(scores)
        verdict = self.scoring_utils.get_verdict(scores)
        avg_score = round(weighted_score, 2)
        details["consolidated_reasoning"] = self._consolidated_reasoning(
            agent_results, scores, weighted_score
        )

        return {
            "verdict": verdict,
            "average_score": avg_score,
            "weighted_score": avg_score,
            "scores": scores,
            "weights": weights,
            "details": details,
            "reasoning_summary": details["consolidated_reasoning"],
        }

    @staticmethod
    def _consolidated_reasoning(agent_results, scores, weighted_score):
        weakest = sorted(scores.items(), key=lambda item: item[1])[:2]
        if weighted_score >= 8:
            summary = "The response meets the quality threshold, with a weighted score of "
            summary += f"{weighted_score:.2f}/10."
        else:
            summary = "The overall score is driven down by "
            summary += ", ".join(
                f"{dimension} ({score:.2f}/10)" for dimension, score in weakest
            )
            summary += "."

        explanations = []
        for dimension, score in sorted(scores.items(), key=lambda item: item[1]):
            if score >= 8:
                continue
            result = agent_results.get(dimension, {})
            explanation = result.get("explanation", "").strip()
            if explanation:
                explanations.append(f"{dimension.capitalize()}: {explanation}")

        if explanations:
            summary += " Areas to improve: " + " ".join(explanations)

        hallucination = agent_results.get("hallucination", {})
        flagged_claims = hallucination.get("flagged_claims", [])
        if flagged_claims:
            summary += f" The hallucination agent flagged {len(flagged_claims)} claim(s)."
        return summary
