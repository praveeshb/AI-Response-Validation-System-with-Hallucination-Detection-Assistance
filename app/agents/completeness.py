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

    QUESTION_STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "what", "which", "who", "whom", "when", "where",
        "why", "how", "can", "could", "would", "should", "will", "please",
        "explain", "describe", "discuss", "list", "name", "give", "tell",
        "of", "to", "and", "or", "for", "in", "on", "at", "by", "with",
        "from", "about", "as", "into", "than", "that", "this", "it",
    }

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

Identify the distinct requirements or aspects in the question and check whether the response addresses each one. Do not penalize a response for irrelevant detail that the question did not request. List concrete omissions and explain why each is required by the question. If nothing is omitted, write "None".

Provide your evaluation in the following format:
Score: [0-10]
Explanation: [detailed explanation of why you gave this score]
Omissions:
- [specific omitted aspect and why it is required, or None]"""),
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
            score = self._extract_score(response_text)
            return {
                "dimension": "completeness",
                "score": score,
                "category": self._category(score),
                "scale": "0-10",
                "explanation": self._extract_explanation(response_text),
                "omissions": self._extract_omissions(response_text),
            }

        return self._fallback_evaluate(question, ai_response, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based completeness evaluation when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            missing_terms = self._missing_terms(question, "")
            return {
                "dimension": "completeness",
                "score": 0.0,
                "category": self._category(0.0),
                "scale": "0-10",
                "explanation": "The response is empty, so it does not provide a complete answer.",
                "omissions": [
                    {
                        "aspect": term,
                        "reason": f"The question asks about '{term}', but no response was provided.",
                    }
                    for term in missing_terms
                ],
            }

        question_terms = list(dict.fromkeys(
            token.lower()
            for token in re.findall(r"\b\w+\b", question)
            if token.lower() not in self.QUESTION_STOP_WORDS
        ))
        missing_terms = self._missing_terms(question, ai_response)

        coverage = (len(question_terms) - len(missing_terms)) / max(len(question_terms), 1)
        length_score = min(1.0, len(re.findall(r"\b\w+\b", ai_response)) / 40.0)
        score = max(0.0, min(10.0, coverage * 8.0 + length_score * 2.0))
        omissions = [
            {
                "aspect": term,
                "reason": f"The question asks about '{term}', but the response does not mention it.",
            }
            for term in missing_terms
        ]
        if omissions:
            explanation = (
                "The response does not cover all identifiable question aspects. "
                "The listed omissions are terms explicitly requested by the question; "
                "this rule-based check may not recognize equivalent wording."
            )
        else:
            explanation = (
                "The response covers the identifiable content terms in the question. "
                "This is a rule-based estimate and may not recognize every multi-part requirement."
            )
        return {
            "dimension": "completeness",
            "score": round(score, 2),
            "category": self._category(score),
            "scale": "0-10",
            "explanation": explanation,
            "omissions": omissions,
        }

    @staticmethod
    def _stem(token: str) -> str:
        return re.sub(r"(ing|ed|s)$", "", token)

    @classmethod
    def _missing_terms(cls, question: str, ai_response: str):
        question_terms = list(dict.fromkeys(
            token.lower()
            for token in re.findall(r"\b\w+\b", question)
            if token.lower() not in cls.QUESTION_STOP_WORDS
        ))
        answer_terms = {
            cls._stem(token.lower())
            for token in re.findall(r"\b\w+\b", ai_response)
        }
        missing_terms = []
        for term in question_terms:
            if cls._stem(term) not in answer_terms and term not in missing_terms:
                missing_terms.append(term)
        return missing_terms

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
        match = re.search(
            r"Explanation:\s*(.*?)(?=\n\s*Omissions:|\Z)", text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return text

    def _extract_omissions(self, text: str):
        """Parse the concrete omissions listed by the judge."""
        match = re.search(r"Omissions:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        omissions = []
        for line in match.group(1).splitlines():
            item = re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*)", "", line).strip()
            if not item or item.lower() in {"none", "none identified", "n/a"}:
                continue
            aspect, separator, reason = item.partition(":")
            omissions.append({
                "aspect": aspect.strip(),
                "reason": reason.strip() if separator else item,
            })
        return omissions
