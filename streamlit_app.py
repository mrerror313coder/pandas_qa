"""
streamlit_app.py — Retrieval-only demo for Streamlit Cloud
No LLM needed. Just shows top-k passages with refusal logic.
"""

import streamlit as st
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pandas Docs Assistant",
    page_icon="📊",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# Cache the model + index loading
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_system():
    # Load index
    index = faiss.read_index("data/index_400/passages.faiss")
    
    # Load metadata
    meta = []
    with open("data/index_400/passages.meta.jsonl") as f:
        for line in f:
            meta.append(json.loads(line))
    
    # Load embedding model
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    return index, meta, model


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

st.title("📊 Pandas QA — Anti-Hallucination Documentation Assistant")

st.markdown("""
Ask any question about the **pandas library**. This system searches only from
the official pandas 3.0.5 API documentation. When docs don't contain a 
good match, it **refuses** instead of hallucinating.
""")

# Load system (cached)
with st.spinner("Loading retrieval index..."):
    index, meta, model = load_system()

st.success(f"✓ Loaded {index.ntotal:,} passages from pandas docs")

# Refusal threshold
REFUSAL_THRESHOLD = 0.82

# Question input
question = st.text_input(
    "Your Question:",
    placeholder="e.g., What does the how parameter do in DataFrame.merge?"
)

k = st.slider("Number of passages to retrieve:", 1, 10, 5)

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
            st.error(f"""
            ❌ **REFUSED** — Answer not found in pandas documentation.
            
            **Reason:** Top retrieval score ({top_score:.3f}) is below our 
            confidence threshold ({REFUSAL_THRESHOLD}).
            
            The docs likely don't contain a direct answer to this question. 
            The system refuses instead of making up an answer.
            """)
        else:
            top_passage = meta[indices[0][0]]
            
            st.success(f"""
            ✅ **Answer found!**
            
            **Source:** `{top_passage['doc_name']}` ({top_passage['section']})
            
            **Confidence:** {top_score:.3f}
            """)
            
            st.markdown("### 📖 Relevant documentation:")
            st.markdown(f"```
{top_passage['text']}
```")
        
        # Always show all retrieved passages
        with st.expander(f"🔍 View all {k} retrieved passages", expanded=False):
            for i in range(k):
                p = meta[indices[0][i]]
                s = float(scores[0][i])
                st.markdown(f"**{i+1}. `{p['doc_name']}` ({p['section']}) — score: {s:.3f}**")
                st.markdown(f"```
{p['text'][:400]}
```")
                st.markdown("---")

# Example questions
st.markdown("---")
st.markdown("### Try these examples:")
examples = [
    "What does the 'how' parameter do in DataFrame.merge?",
    "What is the default sorting algorithm in DataFrame.sort_values?",
    "How do I use pandas to compute matrix eigenvalues?",  # should refuse
    "What does the 'optimize' parameter do in DataFrame.merge?",  # should refuse
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}"):
            st.experimental_set_query_params(q=ex)
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
### About This System
- **Docs indexed:** 2,087 pandas API pages → 11,855 passages
- **Retriever:** BAAI/bge-small-en-v1.5
- **Refusal threshold:** 0.82 (chosen on dev set)
- **Heldback results:** 100% correct refusal on 20 unanswerable questions

Built for CAID internship at Namal University Mianwali by Muhammad Asad.

[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)
""")
