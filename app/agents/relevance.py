from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.utils.config import config

class RelevanceJudgeAgent:
    """Evaluates the relevance of an AI response to the question"""
    
    def __init__(self):
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
        """Evaluate relevance of the response"""
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
            "dimension": "relevance",
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
