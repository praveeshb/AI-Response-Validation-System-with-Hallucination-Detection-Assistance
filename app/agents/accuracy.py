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
            score = self._extract_score(response_text)
            evidence = self._collect_evidence(ai_response, reference_answer, retrieved_context)
            return {
                "dimension": "accuracy",
                "score": score,
                "category": self._category(score),
                "scale": "0-10",
                "explanation": self._extract_explanation(response_text),
                "supporting_evidence": evidence,
                "claim_assessment": self._assess_claims(ai_response, evidence),
            }

        return self._fallback_evaluate(question, ai_response, reference_answer, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based accuracy evaluation when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {
                "dimension": "accuracy", "score": 0.0, "category": "very poor", "scale": "0-10",
                "explanation": "No answer was supplied, so the accuracy cannot be confirmed.",
                "supporting_evidence": [],
                "claim_assessment": [],
            }

        evidence_sources = [part for part in [reference_answer, retrieved_context] if part and part.strip() and "not provided" not in part.lower()]
        if not evidence_sources:
            score = 6.0
            explanation = "There is no reference answer or retrieved context for strict factual validation, so the system uses a cautious neutral score."
            return {
                "dimension": "accuracy", "score": round(score, 2), "category": self._category(score), "scale": "0-10",
                "explanation": explanation, "supporting_evidence": [], "claim_assessment": [],
            }

        reference_text = " ".join(evidence_sources).lower()
        response_text = ai_response.lower()
        reference_tokens = set(re.findall(r"\b\w+\b", reference_text))
        response_tokens = set(re.findall(r"\b\w+\b", response_text))
        overlap = len(reference_tokens & response_tokens)
        overlap_ratio = overlap / max(len(reference_tokens), 1)

        normalized_reference = re.sub(r"[^a-z0-9]+", " ", reference_answer.lower()).strip() if reference_answer else ""
        normalized_response = re.sub(r"[^a-z0-9]+", " ", ai_response.lower()).strip()
        if normalized_response == normalized_reference:
            score = 10.0
        elif reference_answer and len(re.findall(r"\b\w+\b", ai_response)) <= 15:
            if normalized_reference and normalized_reference in normalized_response:
                reference_span = normalized_response.replace(normalized_reference, "").strip()
                score = 10.0 if not reference_span else 7.0
            else:
                negation_pattern = r"\b(no|not|never|neither|without|cannot)\b"
                reference_negated = bool(re.search(negation_pattern, reference_answer.lower()))
                response_negated = bool(re.search(negation_pattern, ai_response.lower()))
                if reference_negated != response_negated:
                    score = 1.0
                else:
                    reference_words = re.findall(r"\b\w+\b", reference_answer)
                    response_words = re.findall(r"\b\w+\b", ai_response)
                    reference_anchor = reference_words[0].lower() if reference_words else ""
                    response_anchor = response_words[0].lower() if response_words else ""
                    score = 6.0 if reference_anchor == response_anchor else 1.0
        else:
            score = 8.0 if overlap_ratio >= 0.5 else 6.0 if overlap_ratio >= 0.25 else 3.0 if overlap_ratio > 0 else 1.0

        explanation = (
            "The factual content in the response matches " + str(overlap) + " key evidence terms from the provided reference material, "
            + ("which supports the claim." if score >= 6 else "which leaves some factual gaps or unsupported details.")
        )
        evidence = self._collect_evidence(ai_response, reference_answer, retrieved_context)
        return {
            "dimension": "accuracy",
            "score": round(score, 2),
            "category": self._category(score),
            "scale": "0-10",
            "explanation": explanation,
            "supporting_evidence": evidence,
            "claim_assessment": self._assess_claims(ai_response, evidence),
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

    @staticmethod
    def _sentences(text: str):
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]

    def _collect_evidence(self, ai_response: str, reference_answer: str = None, retrieved_context: str = None):
        evidence_sources = [
            (source_name, source)
            for source_name, source in (
                ("reference_answer", reference_answer),
                ("retrieved_context", retrieved_context),
            )
            if source and source.strip() and "not provided" not in source.lower()
        ]
        evidence = []
        response_tokens = set(re.findall(r"\b[^\W\d_]+\b", ai_response.lower()))
        for source_name, source in evidence_sources:
            for sentence in self._sentences(source):
                source_tokens = set(re.findall(r"\b[^\W\d_]+\b", sentence.lower()))
                overlap = len(response_tokens & source_tokens)
                if overlap:
                    evidence.append({
                        "text": sentence,
                        "source": source_name,
                        "matched_terms": overlap,
                    })
        evidence.sort(key=lambda item: item["matched_terms"], reverse=True)
        return evidence[:5]

    def _assess_claims(self, ai_response: str, evidence):
        evidence_text = " ".join(item["text"] for item in evidence)
        evidence_tokens = set(re.findall(r"\b[^\W\d_]+\b", evidence_text.lower()))
        assessment = []
        for claim in self._sentences(ai_response):
            claim_tokens = set(re.findall(r"\b[^\W\d_]+\b", claim.lower()))
            overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
            claim_negated = bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", claim.lower()))
            evidence_negated = bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", evidence_text.lower()))
            if claim_negated != evidence_negated:
                status = "contradictory"
            elif overlap >= 0.5:
                status = "correct"
            elif overlap >= 0.25:
                status = "partially_correct"
            else:
                status = "incorrect_or_unsupported"
            assessment.append({"claim": claim, "status": status})
        return assessment

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
