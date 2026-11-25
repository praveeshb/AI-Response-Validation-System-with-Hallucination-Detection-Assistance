import os

import requests
import streamlit as st
from dotenv import load_dotenv

from batch_utils import parse_evaluation_csv

load_dotenv()

st.set_page_config(
    page_title="AI response validation system",
    page_icon="🤖",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
DIMENSIONS = ("relevance", "accuracy", "hallucination", "completeness")


def main():
    st.title("AI response validation system")
    st.write(
        "Evaluate AI-generated responses for relevance, accuracy, hallucinations, "
        "and completeness."
    )

    st.sidebar.header("About")
    st.sidebar.info(
        "Four specialized evaluation agents assess each response. "
        "A weighted verdict combines the dimension scores."
    )

    single_tab, batch_tab = st.tabs(["Single evaluation", "Batch evaluation"])
    with single_tab:
        show_single_evaluation()
    with batch_tab:
        show_batch_evaluation()


def show_single_evaluation():
    with st.form("single_evaluation"):
        question = st.text_area("Question", placeholder="Enter the question...")
        ai_response = st.text_area(
            "AI response", placeholder="Enter the response to evaluate..."
        )
        reference_answer = st.text_area(
            "Reference answer (optional)",
            placeholder="Enter an answer for comparison...",
        )
        source_document = st.text_area(
            "Source document (optional)",
            placeholder="Enter supporting source text...",
        )
        submitted = st.form_submit_button("Evaluate response", type="primary")

    if not submitted:
        st.info("Submit a question and response to see evaluation results.")
        return
    if not question.strip() or not ai_response.strip():
        st.error("Please provide both a question and an AI response.")
        return

    with st.spinner("Evaluating response..."):
        try:
            response = requests.post(
                f"{API_URL}/api/evaluation/submit",
                json={
                    "question": question,
                    "ai_response": ai_response,
                    "reference_answer": reference_answer or None,
                    "source_document": source_document or None,
                },
                timeout=60,
            )
            response.raise_for_status()
            display_results(response.json())
        except requests.exceptions.RequestException as error:
            st.error(f"Could not complete the evaluation: {error}")
            st.info(f"Confirm the evaluation API is running at {API_URL}.")
        except ValueError as error:
            st.error(f"The evaluation API returned an invalid response: {error}")


def show_batch_evaluation():
    st.subheader("Evaluate a CSV file")
    st.write(
        "Required columns: `question` and `ai_response` (or `answer`). "
        "Optional columns: `reference_answer` and `source_document`."
    )
    st.download_button(
        "Download CSV example",
        data=(
            "question,ai_response,reference_answer,source_document\n"
            '"What is the capital of France?","Paris is the capital of France.",'
            '"Paris is the capital of France.","Paris is the capital of France."\n'
        ),
        file_name="evaluation-example.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload question and answer pairs", type="csv", key="batch_csv"
    )
    rows = None
    if uploaded_file is not None:
        try:
            rows = parse_evaluation_csv(uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))
        else:
            st.caption(f"{len(rows)} row(s) ready for evaluation.")
            st.dataframe(
                [
                    {
                        "question": row["question"],
                        "ai_response": row["ai_response"],
                    }
                    for row in rows
                ],
                hide_index=True,
                alt="Question and response rows queued for batch evaluation",
            )

    if st.button(
        "Evaluate all rows",
        type="primary",
        disabled=rows is None,
        key="evaluate_batch",
    ):
        with st.spinner(f"Evaluating {len(rows)} row(s)..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/evaluation/batch",
                    json={"evaluations": rows},
                    timeout=600,
                )
                response.raise_for_status()
                display_batch_results(response.json())
            except requests.exceptions.RequestException as error:
                st.error(f"Could not complete the batch evaluation: {error}")
            except ValueError as error:
                st.error(f"The evaluation API returned an invalid response: {error}")


def display_batch_results(batch):
    results = batch.get("results", [])
    st.subheader("Batch results")
    total, succeeded, failed = st.columns(3)
    total.metric("Submissions", batch.get("total", len(results)))
    succeeded.metric("Evaluated", batch.get("succeeded", 0))
    failed.metric("Failed", batch.get("failed", 0))

    summary_rows = []
    for item in results:
        result = item.get("result")
        if result is None:
            summary_rows.append({
                "row": item.get("row_number"),
                "verdict": "Error",
                "weighted score": None,
                "error": item.get("error", "Evaluation failed."),
            })
            continue
        scores = result.get("scores", {})
        summary_rows.append({
            "row": item.get("row_number"),
            "verdict": result.get("verdict", "Unknown"),
            "weighted score": result.get("weighted_score", result.get("average_score")),
            **{dimension: scores.get(dimension) for dimension in DIMENSIONS},
            "error": "",
        })

    if summary_rows:
        st.dataframe(
            summary_rows,
            hide_index=True,
            alt="Batch evaluation verdict and dimension score summary",
        )

    for item in results:
        result = item.get("result")
        if result is None:
            st.error(
                f"Row {item.get('row_number')}: "
                f"{item.get('error', 'Evaluation failed.')}"
            )
            continue
        with st.expander(
            f"Row {item.get('row_number')}: {result.get('verdict', 'Unknown')}"
        ):
            display_results(result, compact=True)


def display_results(result, compact=False):
    verdict = result.get("verdict", "Unknown")
    weighted_score = result.get(
        "weighted_score", result.get("average_score", 0.0)
    )
    if verdict == "Pass":
        st.success(f"Overall verdict: {verdict}")
    elif verdict == "Needs Improvement":
        st.warning(f"Overall verdict: {verdict}")
    elif verdict == "Fail":
        st.error(f"Overall verdict: {verdict}")
    else:
        st.info(f"Overall verdict: {verdict}")
    st.metric("Weighted score", f"{weighted_score:.2f} / 10")

    reasoning = result.get("reasoning_summary")
    if reasoning:
        st.write(reasoning)

    scores = result.get("scores", {})
    agent_results = result.get("agent_results", {})
    dimensions = list(DIMENSIONS) + [
        name for name in scores if name not in DIMENSIONS
    ]
    for start in range(0, len(dimensions), 2):
        columns = st.columns(2)
        for column, dimension in zip(columns, dimensions[start:start + 2]):
            with column:
                agent_result = agent_results.get(dimension, {})
                score = scores.get(dimension, agent_result.get("score"))
                with st.container(border=True):
                    st.subheader(dimension.replace("_", " ").capitalize())
                    if isinstance(score, (int, float)):
                        st.metric("Score", f"{score:.2f} / 10")

                    explanation = agent_result.get("explanation")
                    if not explanation:
                        explanation = result.get("details", {}).get(
                            f"{dimension}_explanation"
                        )
                    if explanation:
                        st.write(explanation)

                    if dimension == "completeness":
                        omissions = agent_result.get("omissions", [])
                        if omissions:
                            st.markdown("**Identified omissions**")
                            for omission in omissions:
                                if isinstance(omission, dict):
                                    aspect = omission.get("aspect", "Omitted aspect")
                                    reason = omission.get("reason", "")
                                    st.write(f"- **{aspect}**: {reason}")
                                else:
                                    st.write(f"- {omission}")
                        elif agent_result:
                            st.caption("No specific omissions were identified.")

                    if dimension == "hallucination":
                        status = agent_result.get("hallucination_status")
                        if status:
                            st.write(f"Hallucination status: **{status.replace('_', ' ')}**")
                        for claim in agent_result.get("flagged_claims", []):
                            if isinstance(claim, dict):
                                st.warning(
                                    f"{claim.get('claim', 'Flagged claim')} "
                                    f"- {claim.get('reason', claim.get('status', 'flagged'))}"
                                )
                            else:
                                st.warning(str(claim))

                    evidence = agent_result.get("supporting_evidence", [])
                    if evidence and dimension in {"accuracy", "hallucination"}:
                        if compact:
                            for item in evidence:
                                text = item.get("text", str(item))
                                source = item.get("source")
                                st.write(f"**{source}:** {text}" if source else text)
                        else:
                            with st.expander("Supporting evidence"):
                                for item in evidence:
                                    text = item.get("text", str(item))
                                    source = item.get("source")
                                    st.write(f"**{source}:** {text}" if source else text)

    if compact:
        st.caption("Retrieved context is available in the full evaluation response.")
    else:
        with st.expander("Retrieved context"):
            st.text(result.get("retrieved_context") or "No retrieved context.")


if __name__ == "__main__":
    main()
