import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.evaluation.datasets import load_fixture
from src.ingestion.loaders import MAX_TOTAL_BYTES, load_bytes
from src.pipeline import Config, RAGPipeline

st.set_page_config(page_title="RAGForge | Evidence & Evaluation", page_icon="R", layout="wide")
st.title("RAGForge")
st.caption("Ask your documents. Inspect the evidence. Measure retrieval quality.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    retrieval = st.selectbox("Retrieval", ["bm25", "tfidf", "dense", "hybrid", "hybrid_rerank"])
    generation = st.selectbox("Answers", ["extractive", "openai"])
    st.caption(
        "Extractive mode returns evidence excerpts without API calls. Neural retrieval "
        "downloads local models. OpenAI mode sends the question and retrieved text to OpenAI."
    )
    config = Config(retrieval=retrieval, generation=generation)
    uploaded = st.file_uploader(
        "Documents", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True
    )

    def replace_index(documents, chunk):
        pipeline = RAGPipeline(config)
        count = pipeline.index_records(documents, chunk=chunk)
        st.session_state.pipeline = pipeline
        st.session_state.messages = []
        st.success(f"Indexed {count} chunks. Previous knowledge base replaced.")

    if st.button("Load sample handbook", use_container_width=True):
        try:
            with st.spinner("Building index..."):
                docs, _, _ = load_fixture(ROOT / "data/demo.json")
                replace_index(docs, False)
        except Exception as exc:
            st.error(f"Could not initialize: {exc}")
    if st.button("Index uploaded documents", disabled=not uploaded, use_container_width=True):
        try:
            if sum(file.size for file in uploaded) > MAX_TOTAL_BYTES:
                raise ValueError("Total uploads must be at most 30 MiB.")
            with st.spinner("Building index..."):
                docs = [doc for file in uploaded for doc in load_bytes(file.name, file.getvalue())]
                replace_index(docs, True)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
    if st.button("Clear knowledge base and history"):
        st.session_state.pipeline = RAGPipeline()
        st.session_state.messages = []
        st.rerun()
    active = st.session_state.pipeline.config
    st.caption(f"Active: {active.retrieval} / {active.generation}")
    if active != config:
        st.info("Load the sample or index documents to apply these settings.")

ask_tab, eval_tab = st.tabs(["Ask & inspect", "Evaluation reports"])
with ask_tab:
    pipeline = st.session_state.pipeline
    if pipeline.index is None:
        st.info("Load the sample handbook or upload documents to begin.")
    st.caption(
        "Ask a self-contained question. This version does not rewrite conversational follow-ups."
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "result" in message:
                result = message["result"]
                for source in result["sources"]:
                    cited = " | cited" if source["id"] in result["citations"] else ""
                    with st.expander(f"{source['metadata'].get('source', source['id'])}{cited}"):
                        st.write(source["content"])
                        st.caption(f"ID: {source['id']} | rank score: {source['score']:.4f}")
                with st.expander("Query trace"):
                    st.json(result["trace"])
    question = st.chat_input("How often are API keys rotated?", disabled=pipeline.index is None)
    if question:
        try:
            with st.spinner("Retrieving evidence..."):
                result = pipeline.query(question)
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"], "result": result},
                ]
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Query failed: {exc}")

with eval_tab:
    st.subheader("Reproducible retrieval measurements")
    reports = sorted((ROOT / "reports").glob("*/retrieval.json"))
    if not reports:
        st.info("Run python -m src.evaluation.run to produce a report.")
    else:
        selected = st.selectbox("Benchmark", reports, format_func=lambda p: p.parent.name)
        report = json.loads(selected.read_text(encoding="utf-8"))
        st.write(f"{report['query_count']} labeled queries | {report['document_count']} documents")
        rows = [
            {
                "mode": mode,
                **{
                    key: value
                    for key, value in data["summary"].items()
                    if not isinstance(value, list)
                },
            }
            for mode, data in report["results"].items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.bar_chart({row["mode"]: row[f"ndcg@{report['config']['k']}"] for row in rows})
        st.caption(
            "Retrieval quality does not establish answer correctness. Latency excludes "
            "index construction and model loading; see report provenance."
        )
        st.download_button(
            "Download full benchmark JSON",
            selected.read_bytes(),
            "retrieval.json",
            "application/json",
        )
        with st.expander("Run conditions"):
            st.json({"config": report["config"], "provenance": report["provenance"]})
        modes = list(report["results"])
        mode = st.selectbox("Inspect failures", modes)
        failures = [
            r
            for r in report["results"][mode]["queries"]
            if r["metrics"][f"recall@{report['config']['k']}"] < 1
        ]
        st.write(f"{len(failures)} queries with incomplete recall")
        st.dataframe(failures[:50], use_container_width=True)
