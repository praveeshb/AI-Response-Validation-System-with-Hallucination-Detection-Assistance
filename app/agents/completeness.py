import re
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    ChatOpenAI = None
    ChatPromptTemplate = None

from app.utils.config import config


class CompletenessJudgeAgent:
    """Evaluates the completeness of an AI response."""

    def __init__(self):
        self.llm = None
        self.prompt = None
        if config.is_llm_configured() and ChatOpenAI is not None and ChatPromptTemplate is not None:
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a completeness evaluation expert. Your task is to evaluate how complete and thorough an AI-generated response is.

Score the completeness on a scale of 0-10:
- 9-10: Excellent - The response is comprehensive and covers all aspects of the question
- 7-8: Good - The response is mostly complete with minor omissions
- 5-6: Fair - The response addresses the main points but misses important details
- 3-4: Poor - The response is incomplete and misses significant aspects
- 0-2: Very Poor - The response is very incomplete or superficial

Consider whether the response provides sufficient detail, context, and addresses all parts of the question.

Provide your evaluation in the following format:
Score: [0-10]
Explanation: [detailed explanation of why you gave this score]"""),
                ("user", "Question: {question}\n\nAI Response: {ai_response}")
            ])

    def evaluate(self, question: str, ai_response: str, retrieved_context: str = None) -> Dict[str, Any]:
        """Evaluate completeness of the response."""
        if self.llm is not None and self.prompt is not None:
            chain = self.prompt | self.llm
            input_data = {"question": question, "ai_response": ai_response}
            if retrieved_context:
                input_data["retrieved_context"] = retrieved_context
            result = chain.invoke(input_data)
            response_text = result.content
            return {
                "dimension": "completeness",
                "score": self._extract_score(response_text),
                "explanation": self._extract_explanation(response_text),
            }

        return self._fallback_evaluate(question, ai_response, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based completeness evaluation when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {"dimension": "completeness", "score": 0.0, "explanation": "The response is empty, so it does not provide a complete answer."}

        question_words = set(re.findall(r"\b\w+\b", question.lower()))
        answer_words = set(re.findall(r"\b\w+\b", ai_response.lower()))
        overlap = len(question_words & answer_words)
        coverage = overlap / max(len(question_words), 1)
        length_score = min(1.0, len(re.findall(r"\b\w+\b", ai_response)) / 40.0)

        score = (coverage * 7.5) + (length_score * 2.5)
        score = max(0.0, min(10.0, score))
        explanation = (
            "The response addresses " + str(overlap) + " major question terms and has "
            + ("sufficient detail to be considered fairly complete." if score >= 6 else "limited detail and misses important parts of the question.")
        )
        return {"dimension": "completeness", "score": round(score, 2), "explanation": explanation}

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
