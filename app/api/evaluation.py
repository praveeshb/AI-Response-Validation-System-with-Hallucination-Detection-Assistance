from uuid import uuid4
from fastapi import APIRouter, HTTPException
from app.models import EvaluationRequest
from app.agents.orchestrator import EvaluationOrchestrator
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
orchestrator = EvaluationOrchestrator()


@router.post("/submit", response_model=Dict[str, Any])
async def submit_evaluation(request: EvaluationRequest):
    """Submit a question and AI response for evaluation."""
    try:
        logger.info(f"Received evaluation request for question: {request.question[:50]}...")

        result = orchestrator.evaluate(
            question=request.question,
            ai_response=request.ai_response,
            reference_answer=request.reference_answer,
            source_document=request.source_document
        )

        final_verdict = result.get("final_verdict", {})
        response = {
            "evaluation_id": str(uuid4()),
            "question": result["question"],
            "ai_response": result["ai_response"],
            "scores": final_verdict.get("scores", {}),
            "verdict": final_verdict.get("verdict", "Unknown"),
            "average_score": final_verdict.get("average_score", 0.0),
            "details": final_verdict.get("details", {}),
            "agent_results": result.get("agent_results", {}),
            "retrieved_context": result.get("retrieved_context", "")
        }

        logger.info(f"Evaluation completed. Verdict: {response['verdict']}")
        return response

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "evaluation-api"}
