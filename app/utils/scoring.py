import math
from typing import Dict, Any

class ScoringUtils:
    """Utilities for scoring and verdict generation"""

    DIMENSION_WEIGHTS = {
        "relevance": 0.25,
        "accuracy": 0.30,
        "hallucination": 0.25,
        "completeness": 0.20,
    }
    
    SCORE_RANGES = {
        "excellent": (9, 10),
        "good": (7, 8),
        "fair": (5, 6),
        "poor": (3, 4),
        "very_poor": (0, 2)
    }
    
    @staticmethod
    def get_verdict(scores: Dict[str, float]) -> str:
        """Return the milestone verdict for a weighted score."""
        if not scores:
            return "Unable to evaluate"
        weighted_score, _ = ScoringUtils.weighted_score(scores)
        if weighted_score >= 8:
            return "Pass"
        if weighted_score >= 5:
            return "Needs Improvement"
        return "Fail"

    @staticmethod
    def weighted_score(scores: Dict[str, float]):
        """Calculate a normalized weighted score for the supplied dimensions."""
        if not scores:
            return 0.0, {}

        weights = {}
        for dimension, score in scores.items():
            if dimension not in ScoringUtils.DIMENSION_WEIGHTS:
                raise ValueError(f"Unknown scoring dimension: {dimension}")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"Score for {dimension} must be a number")
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError(f"Score for {dimension} must be between 0 and 10")
            weights[dimension] = ScoringUtils.DIMENSION_WEIGHTS[dimension]

        total_weight = sum(weights.values())
        normalized_weights = {
            dimension: weight / total_weight
            for dimension, weight in weights.items()
        }
        result = sum(scores[dimension] * weight for dimension, weight in normalized_weights.items())
        return round(result, 2), normalized_weights
    
    @staticmethod
    def get_score_category(score: float) -> str:
        """Get category for a single score"""
        for category, (min_score, max_score) in ScoringUtils.SCORE_RANGES.items():
            if min_score <= score <= max_score:
                return category
        return "unknown"
    
    @staticmethod
    def normalize_score(score: float, min_val: float = 0, max_val: float = 10) -> float:
        """Normalize score to 0-10 range"""
        normalized = (score - min_val) / (max_val - min_val) * 10
        return max(0, min(10, normalized))
