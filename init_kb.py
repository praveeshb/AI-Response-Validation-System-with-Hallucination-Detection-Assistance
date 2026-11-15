"""
Initialization script for the knowledge base.
Run this script to populate the vector database with benchmark datasets.
"""

from app.knowledge_base.ingestion import KnowledgeBaseIngestion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting knowledge base initialization...")
    
    ingestion = KnowledgeBaseIngestion()
    
    # Ingest datasets with sample sizes
    # Adjust sample sizes based on your needs and API rate limits
    ingestion.ingest_all(
        truthfulqa_sample=500,  # Number of TruthfulQA samples to ingest
        squad_sample=500         # Number of SQuAD samples to ingest
    )
    
    logger.info("Knowledge base initialization complete!")

if __name__ == "__main__":
    main()
