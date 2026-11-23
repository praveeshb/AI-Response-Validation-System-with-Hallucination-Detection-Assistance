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
            evidence = self._collect_evidence(reference_answer, retrieved_context)
            claim_results = self._assess_claims(ai_response, evidence)
            normalized_reference = re.sub(r"[^a-z0-9]+", " ", (reference_answer or "").lower()).strip()
            normalized_response = re.sub(r"[^a-z0-9]+", " ", ai_response.lower()).strip()
            concise_mismatch = (
                bool(normalized_reference)
                and bool(normalized_response)
                and len(re.findall(r"\b\w+\b", ai_response)) <= 15
                and (
                    bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", (reference_answer or "").lower()))
                    != bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", ai_response.lower()))
                    or re.findall(r"\b\w+\b", reference_answer)[0].lower()
                    != re.findall(r"\b\w+\b", ai_response)[0].lower()
                )
            )
            if concise_mismatch:
                score = 1.0
                claim_results = [{
                    "claim": claim,
                    "status": "contradictory",
                    "evidence": evidence[0]["text"] if evidence else None,
                    "overlap": 0.0,
                    "reason": "The response contradicts or substitutes a different answer for the reference.",
                } for claim in self._sentences(ai_response)]
            flagged = [claim for claim in claim_results if claim["status"] in ("unsupported", "contradictory")]
            score = self._extract_score(response_text)
            return {
                "dimension": "hallucination",
                "score": score,
                "category": self._category(score),
                "scale": "0-10 (absence of hallucination)",
                "explanation": self._extract_explanation(response_text),
                "hallucination_status": "detected" if flagged else "not_detected",
                "flagged_claims": flagged,
                "claim_assessment": claim_results,
                "supporting_evidence": evidence,
            }

        return self._fallback_evaluate(question, ai_response, reference_answer, retrieved_context)

    def _fallback_evaluate(self, question: str, ai_response: str, reference_answer: str = None, retrieved_context: str = None) -> Dict[str, Any]:
        """Rule-based hallucination detection when no LLM is configured."""
        if not ai_response or not ai_response.strip():
            return {
                "dimension": "hallucination", "score": 0.0, "category": "very poor",
                "scale": "0-10 (absence of hallucination)",
                "explanation": "The response is empty, so no claim can be validated against the reference material.",
                "hallucination_status": "undetermined",
                "flagged_claims": [],
                "claim_assessment": [],
                "supporting_evidence": [],
            }

        evidence_sources = [part for part in [reference_answer, retrieved_context] if part and part.strip() and "not provided" not in part.lower()]
        if not evidence_sources:
            score = 6.0
            explanation = "No external reference was provided to confirm or contradict the statement, so the result is neutral rather than fully confident."
            return {
                "dimension": "hallucination", "score": round(score, 2), "category": self._category(score),
                "scale": "0-10 (absence of hallucination)", "explanation": explanation,
                "hallucination_status": "undetermined", "flagged_claims": [],
                "claim_assessment": [], "supporting_evidence": [],
            }

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
        evidence = self._collect_evidence(reference_answer, retrieved_context)
        claim_results = self._assess_claims(ai_response, evidence)
        normalized_reference = re.sub(r"[^a-z0-9]+", " ", (reference_answer or "").lower()).strip()
        normalized_response = re.sub(r"[^a-z0-9]+", " ", ai_response.lower()).strip()
        concise_mismatch = (
            bool(normalized_reference)
            and bool(normalized_response)
            and len(re.findall(r"\b\w+\b", ai_response)) <= 15
            and (
                bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", (reference_answer or "").lower()))
                != bool(re.search(r"\b(no|not|never|neither|without|cannot)\b", ai_response.lower()))
                or re.findall(r"\b\w+\b", reference_answer)[0].lower()
                != re.findall(r"\b\w+\b", ai_response)[0].lower()
            )
        )
        if concise_mismatch:
            score = 1.0
            claim_results = [{
                "claim": claim,
                "status": "contradictory",
                "evidence": evidence[0]["text"] if evidence else None,
                "overlap": 0.0,
                "reason": "The response contradicts or substitutes a different answer for the reference.",
            } for claim in self._sentences(ai_response)]
        flagged = [claim for claim in claim_results if claim["status"] in ("unsupported", "contradictory")]
        return {
            "dimension": "hallucination",
            "score": round(score, 2),
            "category": self._category(score),
            "scale": "0-10 (absence of hallucination)",
            "explanation": explanation,
            "hallucination_status": "detected" if flagged else "not_detected",
            "flagged_claims": flagged,
            "claim_assessment": claim_results,
            "supporting_evidence": evidence,
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
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|[;]|\s+\b(?:and)\b\s+", text or "")
            if part.strip()
        ]

    def _collect_evidence(self, reference_answer: str = None, retrieved_context: str = None):
        evidence = []
        for source_name, source in (("reference_answer", reference_answer), ("retrieved_context", retrieved_context)):
            if not source or not source.strip() or "not provided" in source.lower():
                continue
            for sentence in self._sentences(source):
                evidence.append({"text": sentence, "source": source_name})
        return evidence

    @staticmethod
    def _tokens(text: str):
        stop_words = {"a", "an", "the", "is", "was", "are", "were", "of", "to", "and", "or", "for", "in", "on", "at", "by", "it"}
        return set(re.findall(r"\b[^\W\d_]+\b", text.lower())) - stop_words

    def _assess_claims(self, ai_response: str, evidence):
        results = []
        for claim in self._sentences(ai_response):
            claim_tokens = self._tokens(claim)
            best_status = "unsupported"
            best_evidence = None
            best_overlap = 0.0
            for item in evidence:
                evidence_tokens = self._tokens(item["text"])
                overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_evidence = item["text"]

                claim_negation = bool(re.search(r"\b(no|not|never|neither|without|cannot|isn't|aren't)\b", claim.lower()))
                evidence_negation = bool(re.search(r"\b(no|not|never|neither|without|cannot|isn't|aren't)\b", item["text"].lower()))
                if overlap >= 0.5 and claim_negation != evidence_negation:
                    best_status = "contradictory"
                elif overlap >= 0.5:
                    claim_entities = set(re.findall(r"\b[A-Z][a-z]+\b", claim))
                    evidence_entities = set(re.findall(r"\b[A-Z][a-z]+\b", item["text"]))
                    if claim_entities and evidence_entities and claim_entities.isdisjoint(evidence_entities):
                        best_status = "contradictory"
                    else:
                        best_status = "supported"
            results.append({
                "claim": claim,
                "status": best_status,
                "evidence": best_evidence,
                "overlap": round(best_overlap, 2),
                "reason": (
                    "The claim is grounded in the available evidence."
                    if best_status == "supported"
                    else "The claim conflicts with the polarity or facts in the available evidence."
                    if best_status == "contradictory"
                    else "No sufficiently matching evidence was found for this claim."
                ),
            })
        return results

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
