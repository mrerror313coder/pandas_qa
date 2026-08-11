"""
streamlit_app.py - Full RAG Pipeline (Retrieval + Generation)
Runs on CPU with HuggingFace SmolLM2-360M-Instruct
Shows direct answer + source like Gradio
"""

import streamlit as st
import json
import numpy as np
import faiss
import torch
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Pandas QA — Anti-Hallucination Assistant",
    page_icon="📊",
    layout="wide",
)

# Configuration
GEN_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
REFUSAL_THRESHOLD = 0.82
K = 5

# Simple prompt for small model
PROMPT_TEMPLATE = """Answer the question using ONLY the text below.
If you don't know, say "NOT_FOUND".

Text:
{passages}

Question: {question}

Answer:"""

@st.cache_resource
def load_retrieval_system():
    """Loads FAISS index and embedding model."""
    try:
        if not os.path.exists("data/index_400/passages.faiss"):
            st.error("❌ Data files missing! Ensure 'data/index_400' is in your repo.")
            return None, None, None
        
        index = faiss.read_index("data/index_400/passages.faiss")
        
        meta = []
        with open("data/index_400/passages.meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        return index, meta, model
    except Exception as e:
        st.error("Failed to load retrieval system: " + str(e))
        return None, None, None

@st.cache_resource
def load_generator():
    """Loads the SmolLM2 model."""
    try:
        st.info("⏳ Loading AI model (SmolLM2-360M)... this takes ~15s on CPU.")
        
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error("Failed to load generator: " + str(e))
        return None, None

def retrieve(question, index, meta, embed_model, k=K):
    if index is None or embed_model is None:
        return []
    
    q_embedding = embed_model.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_embedding, k)
    
    passages = []
    for i in range(k):
        if indices[0][i] < len(meta):
            p = meta[indices[0][i]]
            passages.append({
                "doc_name": p["doc_name"],
                "section": p["section"],
                "text": p["text"],
                "score": float(scores[0][i]),
            })
    return passages

def format_passages(passages):
    return "\n\n".join([
        f"[{p['doc_name']}] {p['text']}"
        for p in passages
    ])

def generate(question, passages, tokenizer, model):
    passages_str = format_passages(passages)
    prompt = PROMPT_TEMPLATE.format(passages=passages_str, question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = output[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Parse answer
    if "NOT_FOUND" in raw_answer.upper():
        return "NOT_FOUND", None
    
    # Try to extract source if present
    source_match = re.search(r"Source:\s*(\S+)", raw_answer, re.IGNORECASE)
    source_text = source_match.group(1) if source_match else None
    
    # Clean answer text
    answer_text = re.sub(r"\s*Source:\s*\S+", "", raw_answer, flags=re.IGNORECASE).strip()
    if not answer_text:
        answer_text = raw_answer
        
    return answer_text, source_text

# --- UI ---

st.title("📊 Pandas QA — Anti-Hallucination Documentation Assistant")

st.markdown(
    "Ask any question about the **pandas library**. This system answers **only** from "
    "the official pandas 3.0.5 API documentation. When docs don't contain the answer, "
    "it **refuses** instead of hallucinating."
)

# Load retrieval
with st.spinner("📚 Loading retrieval index..."):
    index, meta, embed_model = load_retrieval_system()

if index is None:
    st.stop()

st.success("Loaded " + str(index.ntotal) + " passages from pandas docs")

# Question input
question = st.text_input("Your Question:", placeholder="e.g., What does the 'how' parameter do in DataFrame.merge?")

k = st.slider("Number of passages to retrieve:", 1, 10, 5)

if st.button("Ask", type="primary"):
    if question:
        # Retrieve
        passages = retrieve(question, index, meta, embed_model, k)
        
        if not passages:
            st.error("No relevant documents found.")
        else:
            top_score = passages[0]["score"]
            
            if top_score < REFUSAL_THRESHOLD:
                st.error(
                    "**REFUSED** — Answer not found in pandas documentation.\n\n"
                    "**Reason:** Top retrieval score (" + str(round(top_score, 3)) +
                    ") is below our confidence threshold (" + str(REFUSAL_THRESHOLD) + ").\n\n"
                    "The docs likely do not contain a direct answer. The system refuses instead of hallucinating."
                )
            else:
                # Load generator
                tokenizer, gen_model = load_generator()
                
                if tokenizer:
                    with st.spinner("🤖 Generating answer..."):
                        answer_text, source_text = generate(question, passages, tokenizer, gen_model)
                        
                        if answer_text.upper() == "NOT_FOUND":
                            st.error(
                                "**REFUSED** — The model determined the retrieved passages do not contain the answer."
                            )
                        else:
                            # Display Direct Answer (Green Box)
                            st.success("**Answer:** " + answer_text)
                            
                            # Display Source (Blue Box)
                            if source_text:
                                st.info("**Source:** `" + source_text + "`")
                            
                            st.caption("🎯 Retrieval Confidence: " + str(round(top_score, 3)) + " (above " + str(REFUSAL_THRESHOLD) + ")")
                else:
                    st.error("❌ Model failed to load.")

    # Show Retrieved Passages
    with st.expander("🔍 View Retrieved Passages (click to expand)"):
        for i, p in enumerate(passages, 1):
            st.markdown("**" + str(i) + ". `" + p["doc_name"] + "` (" + p["section"] + ") — score: " + str(round(p["score"], 3)) + "**")
            st.code(p["text"][:500], language=None)
            st.markdown("---")

# Example questions
st.markdown("---")
st.markdown("### Try these examples:")

examples = [
    "What does the 'how' parameter do in DataFrame.merge?",
    "What is read_csv in pandas?",
    "What does DataFrame.melt do?",
    "How do I filter rows in a DataFrame?",
    "What is the difference between loc and iloc?",
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
    "- **Generator:** HuggingFace **SmolLM2-360M-Instruct** (CPU-optimized)\n"
    "- **Goal:** Zero hallucinations — refuses if answer not found in docs.\n\n"
    "Built for CAID internship at Namal University Mianwali by Muhammad Asad Riaz.\n\n"
    "[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)"
)
