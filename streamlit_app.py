"""
streamlit_app.py - Retrieval-only demo (shows top-3 for full context)
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Pandas Docs Assistant",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_system():
    index = faiss.read_index("data/index_400/passages.faiss")
    meta = []
    with open("data/index_400/passages.meta.jsonl") as f:
        for line in f:
            meta.append(json.loads(line))
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return index, meta, model


st.title("📊 Pandas Docs Assistant")

st.markdown(
    "Ask any question about the **pandas library**. This system searches only from "
    "the official pandas 3.0.5 API documentation. When docs do not contain a "
    "good match, it **refuses** instead of hallucinating."
)

with st.spinner("Loading retrieval index (first load ~30s)..."):
    index, meta, model = load_system()

st.success("Loaded " + str(index.ntotal) + " passages from pandas docs")

REFUSAL_THRESHOLD = 0.82

question = st.text_input(
    "Your Question:",
    placeholder="e.g., What does the how parameter do in DataFrame.merge?"
)

k = 5  # fixed for simplicity

# Handle example click
query_params = st.query_params
if "q" in query_params and not question:
    question = query_params["q"]

if st.button("Ask", type="primary") or question:
    if question:
        q_embedding = model.encode(
            [question], 
            normalize_embeddings=True
        ).astype(np.float32)
        
        scores, indices = index.search(q_embedding, k)
        top_score = float(scores[0][0])
        
        if top_score < REFUSAL_THRESHOLD:
            st.error(
                "**REFUSED** — Answer not found in pandas documentation.\n\n"
                "**Reason:** Top retrieval score (" + str(round(top_score, 3)) + 
                ") is below our confidence threshold (" + str(REFUSAL_THRESHOLD) + ").\n\n"
                "The docs do not contain a direct answer to this question. "
                "The system refuses instead of making up an answer."
            )
        else:
            # Group top passages by doc_name
            doc_passages = {}
            for i in range(k):
                p = meta[indices[0][i]]
                s = float(scores[0][i])
                doc = p["doc_name"]
                if doc not in doc_passages:
                    doc_passages[doc] = {
                        "top_score": s,
                        "sections": []
                    }
                doc_passages[doc]["sections"].append({
                    "section": p["section"],
                    "text": p["text"],
                    "score": s
                })
            
            # Show top doc(s) with all their retrieved sections
            st.success(
                "**Answer found!** Confidence: " + str(round(top_score, 3))
            )
            
            for doc_name, info in list(doc_passages.items())[:2]:
                st.markdown("---")
                st.markdown("## 📖 " + doc_name)
                
                # Sort sections in logical order: description, signature, parameters, examples
                order = {"description": 0, "signature": 1, "parameters": 2, "examples": 3}
                sorted_sections = sorted(
                    info["sections"], 
                    key=lambda x: order.get(x["section"], 99)
                )
                
                for sec in sorted_sections:
                    st.markdown("**" + sec["section"].upper() + "** (relevance: " + str(round(sec["score"], 3)) + ")")
                    st.code(sec["text"], language=None)
        
        # Show all retrieved passages
        with st.expander("🔍 Raw retrieval details (all " + str(k) + " passages)", expanded=False):
            for i in range(k):
                p = meta[indices[0][i]]
                s = float(scores[0][i])
                header = "**" + str(i+1) + ". " + p["doc_name"] + " (" + p["section"] + ") — score: " + str(round(s, 3)) + "**"
                st.markdown(header)
                st.code(p["text"][:400], language=None)
                st.markdown("---")

# Example questions
st.markdown("---")
st.markdown("### Try these examples:")

examples = [
    "What does the 'how' parameter do in DataFrame.merge?",
    "What does DataFrame.melt do?",
    "What is the default sorting algorithm in DataFrame.sort_values?",
    "How do I use pandas to compute matrix eigenvalues?",
    "What does the 'optimize' parameter do in DataFrame.merge?",
    "How do you use DataFrame.append() to add rows?",
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
    "- **Heldback results:** 100% correct refusal on 20 unanswerable questions\n\n"
    "*Note: This is the retrieval-only demo. The full pipeline includes "
    "Qwen 2.5 3B LLM for natural-language answer generation.*\n\n"
    "Built for CAID internship at Namal University Mianwali by Muhammad Asad.\n\n"
    "[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)"
)
