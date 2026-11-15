from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.utils.config import config

class AccuracyJudgeAgent:
    """Evaluates the factual accuracy of an AI response"""
    
    def __init__(self):
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
        """Evaluate accuracy of the response"""
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
            "dimension": "accuracy",
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
