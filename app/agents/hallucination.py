import re
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    ChatOpenAI = None
    ChatPromptTemplate = None

from app.utils.config import config


class HallucinationDetectionAgent:
    """Detects hallucinations in AI responses."""

    def __init__(self):
        self.llm = None
        self.prompt = None
        if config.is_llm_configured() and ChatOpenAI is not None and ChatPromptTemplate is not None:
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a hallucination detection expert. Your task is to detect hallucinations (fabricated or incorrect information) in an AI-generated response.

Score the absence of hallucination on a scale of 0-10:
- 9-10: Excellent - No hallucinations detected, all information is factual
- 7-8: Good - Minor potential hallucinations or slight inaccuracies
- 5-6: Fair - Some hallucinations detected but response is mostly grounded
- 3-4: Poor - Significant hallucinations present
- 0-2: Very Poor - Response is mostly or completely hallucinated

Use the reference information and retrieved context to verify facts.

Provide your evaluation in the following format:
Score: [0-10]
Explanation: [detailed explanation of any hallucinations detected]"""),
                ("user", """Question: {question}
AI Response: {ai_response}
Reference Answer: {reference_answer}
Retrieved Context: {retrieved_context}""")
            ])

    def evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Evaluate hallucinations in the response."""
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
                "dimension": "hallucination",
                "score": self._extract_score(response_text),
                "explanation": self._extract_explanation(response_text),
            }

        return self._fallback_evaluate(question, ai_response, reference_answer, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based hallucination detection when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {"dimension": "hallucination", "score": 0.0, "explanation": "The response is empty, so no claim can be validated against the reference material."}

        evidence_sources = [part for part in [reference_answer, retrieved_context] if part and part.strip() and "not provided" not in part.lower()]
        if not evidence_sources:
            score = 6.0
            explanation = "No external reference was provided to confirm or contradict the statement, so the result is neutral rather than fully confident."
            return {"dimension": "hallucination", "score": round(score, 2), "explanation": explanation}

        evidence_text = " ".join(evidence_sources).lower()
        response_text = ai_response.lower()
        evidence_tokens = set(re.findall(r"\b\w+\b", evidence_text))
        response_tokens = set(re.findall(r"\b\w+\b", response_text))
        unsupported_terms = response_tokens - evidence_tokens
        unsupported_ratio = len(unsupported_terms) / max(len(response_tokens), 1)

        score = 9.0 if unsupported_ratio < 0.1 else 7.0 if unsupported_ratio < 0.3 else 4.0 if unsupported_ratio < 0.6 else 1.0
        explanation = (
            "The response contains " + str(len(unsupported_terms)) + " terms that are not covered by the available evidence, "
            + ("which suggests a low hallucination risk." if score >= 7 else "which indicates some unsupported or fabricated content.")
        )
        return {"dimension": "hallucination", "score": round(score, 2), "explanation": explanation}

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
