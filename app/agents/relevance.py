import re
from typing import Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    ChatOpenAI = None
    ChatPromptTemplate = None

from app.utils.config import config


class RelevanceJudgeAgent:
    """Evaluates the relevance of an AI response to the question."""

    def __init__(self):
        self.llm = None
        self.prompt = None
        if config.is_llm_configured() and ChatOpenAI is not None and ChatPromptTemplate is not None:
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a relevance evaluation expert. Your task is to evaluate how relevant an AI-generated response is to a given question.

Score the relevance on a scale of 0-10:
- 9-10: Excellent - The response directly and completely addresses the question
- 7-8: Good - The response addresses the question well with minor gaps
- 5-6: Fair - The response partially addresses the question but has significant gaps
- 3-4: Poor - The response barely addresses the question
- 0-2: Very Poor - The response is irrelevant to the question

Provide your evaluation in the following format:
Score: [0-10]
Explanation: [detailed explanation of why you gave this score]"""),
                ("user", "Question: {question}\n\nAI Response: {ai_response}")
            ])

    def evaluate(self, question: str, ai_response: str, retrieved_context: str = None) -> Dict[str, Any]:
        """Evaluate relevance of the response."""
        if self.llm is not None and self.prompt is not None:
            chain = self.prompt | self.llm
            input_data = {"question": question, "ai_response": ai_response}
            if retrieved_context:
                input_data["retrieved_context"] = retrieved_context
            result = chain.invoke(input_data)
            response_text = result.content
            score = self._extract_score(response_text)
            return {
                "dimension": "relevance",
                "score": score,
                "category": self._category(score),
                "scale": "0-10",
                "explanation": self._extract_explanation(response_text),
            }

        return self._fallback_evaluate(question, ai_response, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based fallback used when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {
                "dimension": "relevance",
                "score": 0.0,
                "category": "very poor",
                "scale": "0-10",
                "explanation": "The response is empty, so it does not answer the question.",
            }

        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "of", "to", "and",
            "or", "for", "in", "on", "at", "by", "it", "what", "which", "how",
            "why", "when", "where", "who",
        }
        q_tokens = {
            token.lower() for token in re.findall(r"\b\w+\b", question)
            if token.lower() not in stop_words
        }
        r_tokens = {
            token.lower() for token in re.findall(r"\b\w+\b", ai_response)
            if token.lower() not in stop_words
        }
        stem_set = lambda tokens: {re.sub(r"(ing|ed|s)$", "", token) for token in tokens}
        q_stems = stem_set(q_tokens)
        r_stems = stem_set(r_tokens)
        stem_overlap = len(q_stems & r_stems)
        overlap = len(q_tokens & r_tokens)
        overlap_ratio = max(overlap, stem_overlap) / max(len(q_tokens), 1)
        response_length = len(re.findall(r"\b\w+\b", ai_response))

        score = 0.0
        if overlap_ratio >= 0.5 or response_length < 5:
            score = 8.0
        elif overlap_ratio >= 0.25:
            score = 6.0
        elif overlap_ratio > 0:
            score = 4.0
        if q_tokens and r_tokens and q_tokens.issubset(r_tokens):
            score = 10.0
        if overlap_ratio >= 0.5:
            score = max(score, 9.0)
        if overlap > 0 or stem_overlap > 0:
            score = max(score, 6.0)

        explanation = (
            "The response is "
            + ("highly relevant" if score >= 8 else "partially relevant" if score >= 5 else "weakly relevant")
            + " to the question because it shares " + str(overlap) + " matching content terms with the prompt."
        )
        return {
            "dimension": "relevance",
            "score": round(score, 2),
            "category": self._category(score),
            "scale": "0-10",
            "explanation": explanation,
        }

    @staticmethod
    def _category(score: float) -> str:
        if score >= 9:
            return "excellent"
        if score >= 7:
            return "good"
        if score >= 5:
            return "fair"
        if score >= 3:
            return "poor"
        return "very poor"

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
