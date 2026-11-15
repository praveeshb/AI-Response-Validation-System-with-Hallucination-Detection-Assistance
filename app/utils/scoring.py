from typing import Dict, Any

class ScoringUtils:
    """Utilities for scoring and verdict generation"""
    
    SCORE_RANGES = {
        "excellent": (9, 10),
        "good": (7, 8),
        "fair": (5, 6),
        "poor": (3, 4),
        "very_poor": (0, 2)
    }
    
    @staticmethod
    def get_verdict(scores: Dict[str, float]) -> str:
        """Generate overall verdict based on individual scores"""
        if not scores:
            return "Unable to evaluate"
        
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 9:
            return "Excellent"
        elif avg_score >= 7:
            return "Good"
        elif avg_score >= 5:
            return "Fair"
        elif avg_score >= 3:
            return "Poor"
        else:
            return "Very Poor"
    
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
