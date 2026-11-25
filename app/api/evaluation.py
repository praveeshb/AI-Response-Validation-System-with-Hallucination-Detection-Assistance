from uuid import uuid4
from fastapi import APIRouter, HTTPException
from app.models import BatchEvaluationRequest, EvaluationRequest
from app.agents.orchestrator import EvaluationOrchestrator
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
orchestrator = EvaluationOrchestrator()


def _format_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    final_verdict = result.get("final_verdict", {})
    return {
        "evaluation_id": str(uuid4()),
        "question": result["question"],
        "ai_response": result["ai_response"],
        "scores": final_verdict.get("scores", {}),
        "verdict": final_verdict.get("verdict", "Unknown"),
        "average_score": final_verdict.get("average_score", 0.0),
        "weighted_score": final_verdict.get("weighted_score", 0.0),
        "weights": final_verdict.get("weights", {}),
        "reasoning_summary": final_verdict.get("reasoning_summary", ""),
        "details": final_verdict.get("details", {}),
        "agent_results": result.get("agent_results", {}),
        "retrieved_context": result.get("retrieved_context", ""),
    }


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

        response = _format_evaluation(result)

        logger.info(f"Evaluation completed. Verdict: {response['verdict']}")
        return response

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=Dict[str, Any])
async def submit_batch_evaluation(request: BatchEvaluationRequest):
    """Evaluate multiple question/answer submissions and report row-level failures."""
    results = []
    for row_number, evaluation in enumerate(request.evaluations, start=1):
        try:
            result = orchestrator.evaluate(
                question=evaluation.question,
                ai_response=evaluation.ai_response,
                reference_answer=evaluation.reference_answer,
                source_document=evaluation.source_document,
            )
            results.append({
                "row_number": row_number,
                "result": _format_evaluation(result),
            })
        except Exception:
            logger.exception("Batch evaluation failed for row %s", row_number)
            results.append({
                "row_number": row_number,
                "question": evaluation.question,
                "error": "Evaluation failed for this row; check the API logs for details.",
            })

    succeeded = sum(1 for item in results if "result" in item)
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "evaluation-api"}
