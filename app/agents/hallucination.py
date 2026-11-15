from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.utils.config import config

class HallucinationDetectionAgent:
    """Detects hallucinations in AI responses"""
    
    def __init__(self):
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
        """Evaluate hallucinations in the response"""
        chain = self.prompt | self.llm
        
        input_data = {
            "question": question,
            "ai_response": ai_response,
            "reference_answer": reference_answer or "Not provided",
            "retrieved_context": retrieved_context or "Not provided"
        }
        
        result = chain.invoke(input_data)
        response_text = result.content
        
        # Parse the response
        score = self._extract_score(response_text)
        explanation = self._extract_explanation(response_text)
        
        return {
            "dimension": "hallucination",
            "score": score,
            "explanation": explanation
        }
    
    def _extract_score(self, text: str) -> float:
        """Extract score from the response"""
        import re
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        return 5.0  # Default to middle score
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from the response"""
        import re
        match = re.search(r'Explanation:\s*(.+)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
