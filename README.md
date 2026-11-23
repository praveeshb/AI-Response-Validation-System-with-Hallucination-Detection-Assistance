# AI Response Validation System with Hallucination Detection Assistance

## Project Overview
This system evaluates AI-generated responses for quality, factuality, and hallucination detection using multiple evaluation agents and RAG-grounded reference knowledge.

## Tech Stack
- **Backend Framework**: FastAPI
- **LLM Integration**: OpenAI GPT-4, LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers / OpenAI Embeddings
- **Evaluation Frameworks**: RAGAS, TruLens
- **Data Processing**: Pandas, NumPy, HuggingFace Datasets
- **UI**: Streamlit (optional dashboard)

## System Architecture

### Components
1. **Evaluation Input Module**: Single submission interface for questions, AI responses, and optional reference materials
2. **Reference Knowledge Base**: Vector store seeded with TruthfulQA and SQuAD datasets
3. **Evaluation Agent Layer**: Multiple specialized agents for different evaluation dimensions
4. **Agent Orchestrator**: Coordinates evaluation pipeline
5. **Scoring & Verdict Module**: Aggregates scores and generates final verdict

### Evaluation Agents
- **Evaluation Orchestrator**: Manages the overall evaluation workflow
- **Relevance Judge Agent**: Evaluates response relevance to the question
- **Accuracy Judge Agent**: Assesses factual accuracy
- **Hallucination Detection Agent**: Detects hallucinated content
- **Completeness Judge Agent**: Evaluates response completeness
- **Verdict Agent**: Aggregates scores and provides final verdict

### Scoring Dimensions
- **Relevance**: How well the response addresses the question (0-10 scale)
- **Accuracy**: Factual correctness of the response (0-10 scale)
- **Hallucination**: Presence of fabricated information (0-10 scale, inverted)
- **Completeness**: Thoroughness of the response (0-10 scale)

## Project Structure
```
project1/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── models.py               # Pydantic models for data validation
│   ├── api/
│   │   └── evaluation.py      # Evaluation API endpoints
│   ├── agents/
│   │   ├── orchestrator.py    # Evaluation orchestrator
│   │   ├── relevance.py       # Relevance judge agent
│   │   ├── accuracy.py        # Accuracy judge agent
│   │   ├── hallucination.py   # Hallucination detection agent
│   │   ├── completeness.py    # Completeness judge agent
│   │   └── verdict.py         # Verdict aggregation agent
│   ├── knowledge_base/
│   │   ├── ingestion.py       # Dataset ingestion
│   │   ├── chunking.py        # Text chunking strategies
│   │   ├── embeddings.py      # Embedding generation
│   │   └── retrieval.py       # Semantic retrieval
│   └── utils/
│       ├── config.py          # Configuration management
│       └── scoring.py         # Scoring utilities
├── data/
│   └── chroma/                # Vector database storage
├── ui/
│   └── streamlit_app.py       # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

## Milestone 1 Deliverables
- [x] Project structure and dependencies
- [x] System architecture documentation
- [x] Evaluation Input Module
- [x] Reference Knowledge Base with dataset ingestion
- [x] Chunking and embedding pipeline
- [x] Vector store indexing and semantic retrieval

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Initialize the knowledge base**
   ```bash
   python -m app.knowledge_base.ingestion
   ```

6. **Run the application**
   ```bash
   # FastAPI backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

   # Streamlit UI (optional)
   streamlit run ui/streamlit_app.py
   ```

## API Endpoints

### POST /api/evaluation/submit
Submit a question and AI response for evaluation.

**Request Body:**
```json
{
  "question": "What is the capital of France?",
  "ai_response": "The capital of France is Paris.",
  "reference_answer": "Paris is the capital city of France.",
  "source_document": "Optional source text..."
}
```

**Response:**
```json
{
  "evaluation_id": "uuid",
  "scores": {
    "relevance": 9.5,
    "accuracy": 10.0,
    "hallucination": 10.0,
    "completeness": 8.0
  },
  "verdict": "Excellent",
  "details": {
    "relevance_explanation": "...",
    "accuracy_explanation": "...",
    "hallucination_explanation": "...",
    "completeness_explanation": "..."
  }
}
```

## Evaluation Methodology

### LLM-as-a-Judge Approach
The system uses GPT-4 as a judge to evaluate responses across multiple dimensions. Each agent is specialized for a specific evaluation criterion and uses prompt engineering to ensure consistent, high-quality evaluations.

### RAG-Grounded Evaluation
The reference knowledge base provides contextual information for evaluation. When evaluating accuracy and hallucination, the system retrieves relevant reference content from the vector store to ground the evaluation in factual information.

### Scoring Scale
- **9-10**: Excellent
- **7-8**: Good
- **5-6**: Fair
- **3-4**: Poor
- **0-2**: Very Poor

## Datasets Used
- **TruthfulQA**: Benchmark for measuring truthfulness of language models
- **SQuAD**: Stanford Question Answering Dataset for reading comprehension

## Milestone 2 Validation

The Relevance, Accuracy, and Hallucination agents expose structured scores and reasoning:

- Relevance: `score`, `category`, `scale`, and `explanation`
- Accuracy: `score`, `category`, `supporting_evidence`, `claim_assessment`, and `explanation`
- Hallucination: `hallucination_status`, `flagged_claims`, `claim_assessment`, `supporting_evidence`, and `explanation`

Run the representative M2 consistency suite from the project root:

```bash
python -m unittest tests/test_milestone2.py
```

The suite covers correct, partial, incorrect, irrelevant, unsupported, and contradictory responses using local examples from TruthfulQA/SQuAD-style categories.

## Future Enhancements
- Batch evaluation module
- Evaluation report generation
- Advanced visualization dashboard
- Custom dataset upload support
- Multi-model comparison
