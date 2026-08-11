"""
streamlit_app.py - Retrieval-only Demo (No LLM)
Safe for Streamlit Cloud Free Tier.
Shows Source, Confidence, and Refusal logic.
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="Pandas QA — Anti-Hallucination Assistant",
    page_icon="📊",
    layout="wide",
)

# Cache the model + index loading
@st.cache_resource
def load_system():
    try:
        index = faiss.read_index("data/index_400/passages.faiss")
        
        meta = []
        with open("data/index_400/passages.meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        return index, meta, model
    except Exception as e:
        st.error("Failed to load system: " + str(e))
        return None, None, None

# UI Layout
st.title("📊 Pandas QA — Anti-Hallucination Documentation Assistant")

st.markdown(
    "Ask any question about the **pandas library**. This system searches **only** from "
    "the official pandas 3.0.5 API documentation. When docs don't contain a "
    "good match, it **refuses** instead of hallucinating."
)

# Load system (cached)
with st.spinner("📚 Loading retrieval index (first load ~30s)..."):
    index, meta, model = load_system()

if index is None:
    st.stop()

st.success("Loaded " + str(index.ntotal) + " passages from pandas docs")

# Refusal threshold
REFUSAL_THRESHOLD = 0.82

# Question input
question = st.text_input(
    "Your Question:",
    placeholder="e.g., What does the 'how' parameter do in DataFrame.merge?"
)

k = st.slider("Number of passages to retrieve:", 1, 10, 5)

# Handle example click via query param
query_params = st.query_params
if "q" in query_params and not question:
    question = query_params["q"]

if st.button("Ask", type="primary") or question:
    if question:
        # Embed the question
        q_embedding = model.encode(
            [question],
            normalize_embeddings=True
        ).astype(np.float32)

        # Search
        scores, indices = index.search(q_embedding, k)

        top_score = float(scores[0][0])

        # Show refusal or results
        if top_score < REFUSAL_THRESHOLD:
            st.error(
                "**REFUSED** — Answer not found in pandas documentation.\n\n"
                "**Reason:** Top retrieval score (" + str(round(top_score, 3)) +
                ") is below our confidence threshold (" + str(REFUSAL_THRESHOLD) + ").\n\n"
                "The docs likely do not contain a direct answer to this question. "
                "The system refuses instead of making up an answer."
            )
        else:
            top_passage = meta[indices[0][0]]

            st.success(
                "**Answer Found!**\n\n"
                "**Source:** `" + top_passage["doc_name"] + "` (" + top_passage["section"] + ")\n\n"
                "**Confidence:** " + str(round(top_score, 3))
            )

            st.markdown("### 📖 Relevant documentation:")
            st.code(top_passage["text"], language=None)

        # Show all retrieved passages
        with st.expander("🔍 View all " + str(k) + " retrieved passages", expanded=False):
            for i in range(k):
                p = meta[indices[0][i]]
                s = float(scores[0][i])
                header = "**" + str(i+1) + ". `" + p["doc_name"] + "` (" + p["section"] + ") — score: " + str(round(s, 3)) + "**"
                st.markdown(header)
                st.code(p["text"][:500], language=None)
                st.markdown("---")

# Example questions
st.markdown("---")
st.markdown("### Try these examples:")

examples = [
    "What does the 'how' parameter do in DataFrame.merge?",
    "What is the default sorting algorithm in DataFrame.sort_values?",
    "How do I use pandas to compute matrix eigenvalues?",
    "What does the 'optimize' parameter do in DataFrame.merge?",
    "What is read_csv in pandas?",
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key="ex_" + str(i)):
            st.query_params["q"] = ex
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "### About This System\n"
    "- **Docs indexed:** 2,087 pandas API pages → 11,855 passages\n"
    "- **Retriever:** BAAI/bge-small-en-v1.5\n"
    "- **Refusal threshold:** 0.82 (chosen on dev set)\n"
    "- **Goal:** Zero hallucinations — refuses if answer not found in docs.\n\n"
    "Built for CAID internship at Namal University Mianwali by Muhammad Asad Riaz.\n\n"
    "[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)"
)
