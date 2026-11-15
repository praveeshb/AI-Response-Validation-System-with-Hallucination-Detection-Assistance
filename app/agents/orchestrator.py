from typing import Dict, Any, Optional
from app.agents.relevance import RelevanceJudgeAgent
from app.agents.accuracy import AccuracyJudgeAgent
from app.agents.hallucination import HallucinationDetectionAgent
from app.agents.completeness import CompletenessJudgeAgent
from app.agents.verdict import VerdictAgent
from app.knowledge_base.retrieval import VectorStore

class EvaluationOrchestrator:
    """Orchestrates the evaluation pipeline across all agents"""
    
    def __init__(self):
        self.relevance_agent = RelevanceJudgeAgent()
        self.accuracy_agent = AccuracyJudgeAgent()
        self.hallucination_agent = HallucinationDetectionAgent()
        self.completeness_agent = CompletenessJudgeAgent()
        self.verdict_agent = VerdictAgent()
        self.vector_store = VectorStore()
    
    def evaluate(self, question: str, ai_response: str, 
                 reference_answer: Optional[str] = None,
                 source_document: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        
        # Step 1: Retrieve relevant context from knowledge base
        retrieved_docs = self.vector_store.search(question, n_results=3)
        retrieved_context = self._format_retrieved_context(retrieved_docs)
        
        # Step 2: Run all evaluation agents
        agent_results = {}
        
        # Relevance evaluation
        relevance_result = self.relevance_agent.evaluate(
            question=question,
            ai_response=ai_response,
            retrieved_context=retrieved_context
        )
        agent_results["relevance"] = relevance_result
        
        # Accuracy evaluation
        accuracy_result = self.accuracy_agent.evaluate(
            question=question,
            ai_response=ai_response,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context
        )
        agent_results["accuracy"] = accuracy_result
        
        # Hallucination detection
        hallucination_result = self.hallucination_agent.evaluate(
            question=question,
            ai_response=ai_response,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context
        )
        agent_results["hallucination"] = hallucination_result
        
        # Completeness evaluation
        completeness_result = self.completeness_agent.evaluate(
            question=question,
            ai_response=ai_response,
            retrieved_context=retrieved_context
        )
        agent_results["completeness"] = completeness_result
        
        # Step 3: Generate final verdict
        verdict_result = self.verdict_agent.generate_verdict(agent_results)
        
        # Step 4: Compile final result
        final_result = {
            "question": question,
            "ai_response": ai_response,
            "retrieved_context": retrieved_context,
            "agent_results": agent_results,
            "final_verdict": verdict_result
        }
        
        return final_result
    
    def _format_retrieved_context(self, retrieved_docs: list) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            return "No relevant context found in knowledge base."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Context {i}: {doc['content']}")
        
        return "\n\n".join(context_parts)
