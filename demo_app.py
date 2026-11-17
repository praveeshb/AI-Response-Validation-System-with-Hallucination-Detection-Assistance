"""
Simple demo version of the AI Response Validation System
This version uses mock evaluation without requiring LLM API keys or heavy dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI(title="AI Response Validation System - Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    question: str
    ai_response: str
    reference_answer: Optional[str] = None
    source_document: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "AI Response Validation System - Demo Version",
        "version": "1.0.0-demo",
        "note": "This is a demo with mock evaluation. Full version requires LLM API keys."
    }

@app.post("/api/evaluation/submit")
async def submit_evaluation(request: EvaluationRequest):
    """
    Demo evaluation endpoint with mock scores
    """
    # Generate mock scores for demo purposes
    scores = {
        "relevance": round(random.uniform(7, 10), 2),
        "accuracy": round(random.uniform(6, 10), 2),
        "hallucination": round(random.uniform(7, 10), 2),
        "completeness": round(random.uniform(6, 9), 2)
    }
    
    avg_score = sum(scores.values()) / len(scores)
    
    if avg_score >= 9:
        verdict = "Excellent"
    elif avg_score >= 7:
        verdict = "Good"
    elif avg_score >= 5:
        verdict = "Fair"
    else:
        verdict = "Poor"
    
    return {
        "evaluation_id": "demo-" + str(random.randint(1000, 9999)),
        "question": request.question,
        "ai_response": request.ai_response,
        "scores": scores,
        "verdict": verdict,
        "average_score": round(avg_score, 2),
        "details": {
            "relevance_explanation": "The response directly addresses the question asked.",
            "accuracy_explanation": "The information provided appears to be factually correct based on general knowledge.",
            "hallucination_explanation": "No significant hallucinations detected in the response.",
            "completeness_explanation": "The response covers the main aspects of the question."
        },
        "retrieved_context": "Demo mode - knowledge base not initialized. This is a mock evaluation.",
        "note": "This is a demo with mock scores. Enable full version with LLM API for real evaluation."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
