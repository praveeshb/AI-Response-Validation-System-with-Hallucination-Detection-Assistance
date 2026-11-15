import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Response Validation System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .score-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .excellent { background-color: #d4edda; border-left: 5px solid #28a745; }
    .good { background-color: #d1ecf1; border-left: 5px solid #17a2b8; }
    .fair { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .poor { background-color: #f8d7da; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Response Validation System</h1>', unsafe_allow_html=True)
    st.markdown("Evaluate AI-generated responses for relevance, accuracy, hallucination, and completeness.")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This system uses multiple evaluation agents to assess AI responses:
    
    - **Relevance Judge**: Evaluates how well the response addresses the question
    - **Accuracy Judge**: Assesses factual correctness
    - **Hallucination Detector**: Identifies fabricated information
    - **Completeness Judge**: Evaluates thoroughness
    
    The system uses RAG-grounded evaluation with a knowledge base seeded with TruthfulQA and SQuAD datasets.
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        question = st.text_area(
            "Question",
            placeholder="Enter your question here...",
            height=100
        )
        
        ai_response = st.text_area(
            "AI Response",
            placeholder="Enter the AI-generated response to evaluate...",
            height=150
        )
        
        reference_answer = st.text_area(
            "Reference Answer (Optional)",
            placeholder="Enter a reference answer for comparison...",
            height=100
        )
        
        source_document = st.text_area(
            "Source Document (Optional)",
            placeholder="Enter source document for context...",
            height=100
        )
        
        submit_button = st.button("Evaluate Response", type="primary", use_container_width=True)
    
    with col2:
        st.header("Results")
        
        if submit_button:
            if not question or not ai_response:
                st.error("Please provide both a question and an AI response.")
            else:
                with st.spinner("Evaluating response..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/api/evaluation/submit",
                            json={
                                "question": question,
                                "ai_response": ai_response,
                                "reference_answer": reference_answer if reference_answer else None,
                                "source_document": source_document if source_document else None
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            display_results(result)
                        else:
                            st.error(f"Evaluation failed: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to API: {str(e)}")
                        st.info("Make sure the FastAPI server is running on http://localhost:8000")
        else:
            st.info("Submit a question and AI response to see evaluation results.")

def display_results(result):
    """Display evaluation results"""
    
    # Overall verdict
    verdict = result.get("verdict", "Unknown")
    avg_score = result.get("average_score", 0)
    
    verdict_class = get_verdict_class(verdict)
    
    st.markdown(f"""
    <div class="score-card {verdict_class}">
        <h2>Overall Verdict: {verdict}</h2>
        <h3>Average Score: {avg_score:.2f}/10</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual scores
    st.subheader("Individual Scores")
    scores = result.get("scores", {})
    
    for dimension, score in scores.items():
        score_class = get_score_class(score)
        st.markdown(f"""
        <div class="score-card {score_class}">
            <strong>{dimension.capitalize()}:</strong> {score:.2f}/10
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed explanations
    with st.expander("View Detailed Explanations"):
        details = result.get("details", {})
        for key, explanation in details.items():
            st.subheader(key.replace("_", " ").title())
            st.write(explanation)
    
    # Retrieved context
    with st.expander("View Retrieved Context"):
        context = result.get("retrieved_context", "")
        st.text(context)

def get_verdict_class(verdict):
    """Get CSS class based on verdict"""
    verdict_lower = verdict.lower()
    if verdict_lower == "excellent":
        return "excellent"
    elif verdict_lower == "good":
        return "good"
    elif verdict_lower == "fair":
        return "fair"
    else:
        return "poor"

def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 9:
        return "excellent"
    elif score >= 7:
        return "good"
    elif score >= 5:
        return "fair"
    else:
        return "poor"

if __name__ == "__main__":
    main()
