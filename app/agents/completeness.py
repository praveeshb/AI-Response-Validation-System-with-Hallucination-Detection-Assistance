from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.utils.config import config

class CompletenessJudgeAgent:
    """Evaluates the completeness of an AI response"""
    
    def __init__(self):
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
        """Evaluate completeness of the response"""
        chain = self.prompt | self.llm
        
        input_data = {
            "question": question,
            "ai_response": ai_response
        }
        
        if retrieved_context:
            input_data["retrieved_context"] = retrieved_context
        
        result = chain.invoke(input_data)
        response_text = result.content
        
        # Parse the response
        score = self._extract_score(response_text)
        explanation = self._extract_explanation(response_text)
        
        return {
            "dimension": "completeness",
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
