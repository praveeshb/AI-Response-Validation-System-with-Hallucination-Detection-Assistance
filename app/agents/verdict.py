from typing import Dict, Any
from app.utils.scoring import ScoringUtils

class VerdictAgent:
    """Aggregates scores from all agents and provides final verdict"""
    
    def __init__(self):
        self.scoring_utils = ScoringUtils()
    
    def generate_verdict(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final verdict from all agent results"""
        scores = {}
        details = {}
        
        for agent_name, result in agent_results.items():
            dimension = result.get("dimension", agent_name)
            score = result.get("score", 5.0)
            explanation = result.get("explanation", "")
            
            scores[dimension] = score
            details[f"{dimension}_explanation"] = explanation
        
        # Generate overall verdict
        verdict = self.scoring_utils.get_verdict(scores)
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "verdict": verdict,
            "average_score": avg_score,
            "scores": scores,
            "details": details
        }
