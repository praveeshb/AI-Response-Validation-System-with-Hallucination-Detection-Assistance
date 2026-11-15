from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import uuid4

class EvaluationRequest(BaseModel):
    question: str = Field(..., description="The question to evaluate")
    ai_response: str = Field(..., description="The AI-generated response to evaluate")
    reference_answer: Optional[str] = Field(None, description="Optional reference answer for comparison")
    source_document: Optional[str] = Field(None, description="Optional source document for context")

class EvaluationScore(BaseModel):
    dimension: str
    score: float
    explanation: str

class EvaluationResult(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    ai_response: str
    scores: Dict[str, float]
    verdict: str
    details: Dict[str, str]
    retrieved_context: Optional[Dict[str, Any]] = None

class ChunkMetadata(BaseModel):
    source: str
    chunk_id: str
    question_id: Optional[str] = None
    dataset: Optional[str] = None

class DocumentChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
