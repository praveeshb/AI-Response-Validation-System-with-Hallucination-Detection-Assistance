from typing import List
from datasets import load_dataset
from app.knowledge_base.chunking import ChunkingStrategy
from app.knowledge_base.embeddings import EmbeddingGenerator
from app.knowledge_base.retrieval import VectorStore
from app.models import DocumentChunk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseIngestion:
    """Ingest public QA datasets into the knowledge base"""
    
    def __init__(self):
        self.chunking_strategy = ChunkingStrategy()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
    
    def ingest_truthfulqa(self, sample_size: int = 1000):
        """Ingest TruthfulQA dataset from HuggingFace"""
        logger.info("Loading TruthfulQA dataset...")
        try:
            dataset = load_dataset("truthful_qa", "generation", split="validation")
            
            if sample_size:
                dataset = dataset.select(range(min(sample_size, len(dataset))))
            
            all_chunks = []
            for item in dataset:
                question = item.get('question', '')
                correct_answers = item.get('correct_answers', [])
                
                if question and correct_answers:
                    # Combine all correct answers
                    answer_text = " | ".join(correct_answers)
                    chunks = self.chunking_strategy.chunk_qa_pair(
                        question=question,
                        answer=answer_text,
                        source="TruthfulQA",
                        dataset="truthful_qa",
                        question_id=str(hash(question))
                    )
                    all_chunks.extend(chunks)
            
            logger.info(f"Generated {len(all_chunks)} chunks from TruthfulQA")
            self._process_and_store_chunks(all_chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting TruthfulQA: {e}")
    
    def ingest_squad(self, sample_size: int = 1000):
        """Ingest SQuAD dataset from HuggingFace"""
        logger.info("Loading SQuAD dataset...")
        try:
            dataset = load_dataset("squad", split="validation")
            
            if sample_size:
                dataset = dataset.select(range(min(sample_size, len(dataset))))
            
            all_chunks = []
            for item in dataset:
                question = item.get('question', '')
                context = item.get('context', '')
                answers = item.get('answers', {})
                answer_text = answers.get('text', [''])[0] if answers.get('text') else ''
                
                if question and context:
                    # Chunk the context with the question
                    combined_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer_text}"
                    chunks = self.chunking_strategy.chunk_document(
                        text=combined_text,
                        source="SQuAD",
                        dataset="squad",
                        question_id=str(hash(question))
                    )
                    all_chunks.extend(chunks)
            
            logger.info(f"Generated {len(all_chunks)} chunks from SQuAD")
            self._process_and_store_chunks(all_chunks)
            
        except Exception as e:
            logger.error(f"Error ingesting SQuAD: {e}")
    
    def _process_and_store_chunks(self, chunks: List[DocumentChunk]):
        """Generate embeddings and store chunks in vector database"""
        logger.info("Generating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        logger.info("Storing chunks in vector database...")
        self.vector_store.add_chunks(chunks, embeddings)
        
        stats =self.vector_store.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
    
    def ingest_all(self, truthfulqa_sample: int = 1000, squad_sample: int = 1000):
        """Ingest all datasets"""
        logger.info("Starting knowledge base ingestion...")
        self.ingest_truthfulqa(sample_size=truthfulqa_sample)
        self.ingest_squad(sample_size=squad_sample)
        logger.info("Knowledge base ingestion complete!")

if __name__ == "__main__":
    ingestion = KnowledgeBaseIngestion()
    ingestion.ingest_all(truthfulqa_sample=500, squad_sample=500)
