# System Architecture

## Overview
The AI Response Validation System uses a multi-agent architecture to evaluate AI-generated responses across multiple dimensions. The system leverages RAG (Retrieval-Augmented Generation) to ground evaluations in factual information from a knowledge base seeded with benchmark datasets.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Input Module                      │
│  (FastAPI API + Streamlit UI)                                   │
│  - Accepts question, AI response, optional reference materials  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Evaluation Orchestrator                         │
│  - Coordinates evaluation pipeline                               │
│  - Manages agent execution flow                                  │
└──────┬──────────────────────────────────────────────┬───────────┘
       │                                              │
       ▼                                              ▼
┌─────────────────────┐                  ┌─────────────────────────┐
│ Reference Knowledge │                  │   Evaluation Agents     │
│       Base          │                  │                         │
│                     │                  │ ┌─────────────────────┐ │
│ ┌─────────────────┐│                  │ │ Relevance Judge     │ │
│ │ Vector Database ││◄─────────────────┼─│ Agent               │ │
│ │ (ChromaDB)      ││  Semantic Search │ └─────────────────────┘ │
│ └─────────────────┘│                  │ ┌─────────────────────┐ │
│                     │                  │ │ Accuracy Judge      │ │
│ ┌─────────────────┐│                  │ │ Agent               │ │
│ │ TruthfulQA Data ││                  │ └─────────────────────┘ │
│ └─────────────────┘│                  │ ┌─────────────────────┐ │
│                     │                  │ │ Hallucination       │ │
│ ┌─────────────────┐│                  │ │ Detection Agent     │ │
│ │ SQuAD Data      ││                  │ └─────────────────────┘ │
│ └─────────────────┘│                  │ ┌─────────────────────┐ │
│                     │                  │ │ Completeness Judge  │ │
│ ┌─────────────────┐│                  │ │ Agent               │ │
│ │ Chunking        ││                  │ └─────────────────────┘ │
│ │ Pipeline        ││                  │                         │
│ └─────────────────┘│                  └───────────┬─────────────┘
│                     │                              │
│ ┌─────────────────┐│                              ▼
│ │ Embedding       ││              ┌─────────────────────────┐
│ │ Generation      ││              │   Verdict Agent         │
│ └─────────────────┘│              │ - Aggregates scores     │
│                     │              │ - Generates final verdict│
└─────────────────────┘              └───────────┬─────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────────┐
                                      │  Evaluation Results     │
                                      │  - Scores per dimension │
                                      │  - Overall verdict      │
                                      │  - Detailed explanations│
                                      └─────────────────────────┘
```

## Component Details

### 1. Evaluation Input Module
**Purpose**: Single submission interface for evaluation requests

**Components**:
- FastAPI REST API endpoint (`/api/evaluation/submit`)
- Streamlit UI dashboard for interactive evaluation
- Input validation and error handling
- Support for optional reference materials

**Data Flow**:
```
User Input → Validation → Orchestrator → Agent Pipeline → Results
```

### 2. Reference Knowledge Base
**Purpose**: Store and retrieve factual information for RAG-grounded evaluation

**Components**:
- **Dataset Ingestion Module**: Downloads and processes TruthfulQA and SQuAD datasets
- **Data Cleaning & Chunking Pipeline**: Splits documents into searchable chunks
- **Embedding Generation Module**: Creates vector embeddings using Sentence Transformers
- **Vector Database (ChromaDB)**: Stores embeddings with metadata for semantic search
- **RAG Retrieval Pipeline**: Retrieves relevant context for evaluation

**Data Flow**:
```
Datasets → Cleaning → Chunking → Embedding → Vector Store → Semantic Search
```

### 3. Evaluation Agent Layer
**Purpose**: Specialized agents for evaluating different response dimensions

**Agents**:

#### a. Evaluation Orchestrator
- Manages the overall evaluation workflow
- Coordinates agent execution
- Handles data flow between components

#### b. Relevance Judge Agent
- Evaluates how well the response addresses the question
- Uses LLM-as-a-Judge approach with GPT-4
- Scores on 0-10 scale

#### c. Accuracy Judge Agent
- Assesses factual correctness of the response
- Uses retrieved context and reference answers for verification
- Scores on 0-10 scale

#### d. Hallucination Detection Agent
- Detects fabricated or incorrect information
- Cross-checks against knowledge base
- Scores on 0-10 scale (higher = less hallucination)

#### e. Completeness Judge Agent
- Evaluates thoroughness of the response
- Checks if all aspects of the question are addressed
- Scores on 0-10 scale

#### f. Verdict Agent
- Aggregates scores from all agents
- Calculates average score
- Generates overall verdict (Excellent/Good/Fair/Poor/Very Poor)

### 4. Scoring & Verdict Module
**Purpose**: Aggregate and present evaluation results

**Scoring Scale**:
- 9-10: Excellent
- 7-8: Good
- 5-6: Fair
- 3-4: Poor
- 0-2: Very Poor

**Output Format**:
```json
{
  "scores": {
    "relevance": 9.5,
    "accuracy": 10.0,
    "hallucination": 10.0,
    "completeness": 8.0
  },
  "verdict": "Excellent",
  "average_score": 9.375,
  "details": {
    "relevance_explanation": "...",
    "accuracy_explanation": "...",
    "hallucination_explanation": "...",
    "completeness_explanation": "..."
  }
}
```

## Technology Stack

### Backend
- **FastAPI**: High-performance web framework for API
- **Python 3.9+**: Core programming language
- **Pydantic**: Data validation and settings management

### AI/ML
- **OpenAI GPT-4**: LLM for LLM-as-a-Judge evaluation
- **LangChain**: Framework for LLM application development
- **Sentence Transformers**: Embedding generation
- **RAGAS**: Evaluation framework for RAG systems
- **TruLens**: Evaluation framework for AI applications

### Data & Storage
- **ChromaDB**: Vector database for semantic search
- **HuggingFace Datasets**: Source for benchmark datasets
- **Pandas/NumPy**: Data processing

### Frontend
- **Streamlit**: Interactive UI dashboard
- **HTML/CSS**: Custom styling

## Data Flow

1. **Input Submission**: User submits question and AI response via API or UI
2. **Context Retrieval**: System retrieves relevant context from knowledge base using semantic search
3. **Agent Evaluation**: Each specialized agent evaluates the response:
   - Relevance Judge checks question-response alignment
   - Accuracy Judge verifies factual correctness
   - Hallucination Detector identifies fabricated content
   - Completeness Judge assesses thoroughness
4. **Verdict Generation**: Verdict Agent aggregates scores and generates final verdict
5. **Result Presentation**: Results are returned with scores, verdict, and detailed explanations

## Scalability Considerations

### Current Design
- Single-process evaluation
- In-memory vector store (ChromaDB with persistent storage)
- Sequential agent execution

### Future Enhancements
- Batch evaluation module for processing multiple requests
- Async agent execution for parallel processing
- Distributed vector store for larger datasets
- Caching layer for frequently asked questions
- Custom dataset upload support
- Multi-model comparison capabilities

## Security Considerations

- API key management via environment variables
- Input validation to prevent injection attacks
- Rate limiting for API endpoints
- CORS configuration for frontend access
- No sensitive data storage in knowledge base

## Monitoring & Logging

- Structured logging for all components
- Error tracking and reporting
- Performance metrics for evaluation pipeline
- Knowledge base statistics and health checks
