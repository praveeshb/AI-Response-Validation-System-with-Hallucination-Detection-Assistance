import re
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    ChatOpenAI = None
    ChatPromptTemplate = None

from app.utils.config import config


class AccuracyJudgeAgent:
    """Evaluates the factual accuracy of an AI response."""

    def __init__(self):
        self.llm = None
        self.prompt = None
        if config.is_llm_configured() and ChatOpenAI is not None and ChatPromptTemplate is not None:
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a factual accuracy evaluation expert. Your task is to evaluate the factual accuracy of an AI-generated response.

Score the accuracy on a scale of 0-10:
- 9-10: Excellent - The response is completely factually accurate
- 7-8: Good - The response is mostly accurate with minor factual errors
- 5-6: Fair - The response has some factual errors but is partially correct
- 3-4: Poor - The response has significant factual errors
- 0-2: Very Poor - The response is mostly or completely factually incorrect

Use the reference information and retrieved context to verify facts.

Provide your evaluation in the following format:
Score: [0-10]
Explanation: [detailed explanation of why you gave this score]"""),
                ("user", """Question: {question}
AI Response: {ai_response}
Reference Answer: {reference_answer}
Retrieved Context: {retrieved_context}""")
            ])

    def evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Evaluate accuracy of the response."""
        if self.llm is not None and self.prompt is not None:
            chain = self.prompt | self.llm
            input_data = {
                "question": question,
                "ai_response": ai_response,
                "reference_answer": reference_answer or "Not provided",
                "retrieved_context": retrieved_context or "Not provided"
            }
            result = chain.invoke(input_data)
            response_text = result.content
            return {
                "dimension": "accuracy",
                "score": self._extract_score(response_text),
                "explanation": self._extract_explanation(response_text),
            }

        return self._fallback_evaluate(question, ai_response, reference_answer, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based accuracy evaluation when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {"dimension": "accuracy", "score": 0.0, "explanation": "No answer was supplied, so the accuracy cannot be confirmed."}

        evidence_sources = [part for part in [reference_answer, retrieved_context] if part and part.strip() and "not provided" not in part.lower()]
        if not evidence_sources:
            score = 6.0
            explanation = "There is no reference answer or retrieved context for strict factual validation, so the system uses a cautious neutral score."
            return {"dimension": "accuracy", "score": round(score, 2), "explanation": explanation}

        reference_text = " ".join(evidence_sources).lower()
        response_text = ai_response.lower()
        reference_tokens = set(re.findall(r"\b\w+\b", reference_text))
        response_tokens = set(re.findall(r"\b\w+\b", response_text))
        overlap = len(reference_tokens & response_tokens)
        overlap_ratio = overlap / max(len(reference_tokens), 1)

        score = 8.0 if overlap_ratio >= 0.5 else 6.0 if overlap_ratio >= 0.25 else 3.0 if overlap_ratio > 0 else 1.0
        if reference_answer and ai_response.lower().strip() == reference_answer.lower().strip():
            score = 10.0

        explanation = (
            "The factual content in the response matches " + str(overlap) + " key evidence terms from the provided reference material, "
            + ("which supports the claim." if score >= 6 else "which leaves some factual gaps or unsupported details.")
        )
        return {"dimension": "accuracy", "score": round(score, 2), "explanation": explanation}

    def _extract_score(self, text: str) -> float:
        """Extract score from the response."""
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
        return 5.0

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from the response."""
        match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
