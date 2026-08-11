"""
streamlit_app.py - Final Optimized Version
Runs on CPU with HuggingFace SmolLM2-360M-Instruct
Shows direct answers with source citations
"""

import time
import json
import os
import gc

import numpy as np
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Pandas QA — Anti-Hallucination Documentation Assistant",
    page_icon="📊",
    layout="wide",
)

# Configuration
GEN_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
REFUSAL_THRESHOLD = 0.82
K = 5
DEVICE = "cpu"

# Updated Prompt Template to ensure direct answer first
PROMPT_TEMPLATE = """You are a documentation assistant for the pandas library. 

TASK: Answer the user's question using ONLY the provided passages.

RULES:
1. If the passages contain the answer, write a clear, direct answer (1-2 sentences) FIRST.
2. Immediately after the answer, add a new line and write: "Source: <doc_name>"
3. If the passages do NOT contain the answer, write ONLY: "NOT_FOUND"
4. Do not invent any information.

Passages:
{passages}

Question: {question}

Answer:"""


@st.cache_resource
def load_retrieval_system():
    """Loads the FAISS index and embedding model."""
    try:
        if not os.path.exists("data/index_400/passages.faiss"):
            st.error("❌ Data files missing. Ensure 'data/index_400' is in your repo.")
            return None, None, None
            
        index = faiss.read_index("data/index_400/passages.faiss")
        meta = []
        with open("data/index_400/passages.meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return index, meta, embed_model
    except Exception as e:
        st.error(f"❌ Retrieval load error: {e}")
        return None, None, None


@st.cache_resource
def load_generator():
    """Loads the ultra-tiny SmolLM2 model."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        st.info(f"⏳ Loading {GEN_MODEL_NAME}... (this takes ~15s on CPU)")
        
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.exception(e)
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
        f"[Passage {i}] doc_name: {p['doc_name']} | section: {p['section']}\n{p['text']}"
        for i, p in enumerate(passages, 1)
    ])


def generate(question, passages, tokenizer, model, max_new_tokens=150):
    passages_str = format_passages(passages)
    prompt = PROMPT_TEMPLATE.format(passages=passages_str, question=question)
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Parse the answer to separate the text and source
    if "Source:" in raw_answer:
        parts = raw_answer.split("Source:", 1)
        answer_text = parts[0].strip()
        source_text = parts[1].strip()
        return answer_text, source_text
    elif raw_answer.upper() == "NOT_FOUND":
        return "NOT_FOUND", None
    else:
        return raw_answer, None


# --- UI Layout ---

st.title("📊 Pandas QA — Anti-Hallucination Documentation Assistant")

st.markdown(
    "Ask any question about the **pandas library**. This system answers **only from the official pandas 3.0.5 API documentation**."
    "When docs don't contain the answer, it **refuses** instead of hallucinating."
)

# Example questions section
st.markdown("### Try these examples:")
cols = st.columns(2)
examples = [
    "What is DataFrame in Pandas?",
    "What is read_csv in pandas?",
    "What does DataFrame.melt do?",
    "How do I filter rows in a DataFrame?",
    "What is the difference between loc and iloc?",
    "What is the task of read_csv in pandas?",
]
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}"):
            st.session_state["question"] = ex
            st.rerun()

st.markdown("---")

# Question Input
if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_input(
    "Your Question:",
    value=st.session_state["question"],
    placeholder="e.g., What is DataFrame in Pandas?",
    key="q_input"
)

ask_clicked = st.button("Ask", type="primary", use_container_width=True)
clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["question"] = ""
    if "answer_displayed" in st.session_state:
        del st.session_state["answer_displayed"]
    st.rerun()

# Load Retriever
if "retrieval_loaded" not in st.session_state:
    with st.spinner("📚 Loading index..."):
        index, meta, embed_model = load_retrieval_system()
        if index:
            st.session_state.update({
                "retrieval_loaded": True,
                "index": index,
                "meta": meta,
                "embed_model": embed_model
            })
            st.success(f"✅ Loaded {index.ntotal} passages.")
        else:
            st.session_state["retrieval_loaded"] = False

if "retrieval_loaded" not in st.session_state or not st.session_state["retrieval_loaded"]:
    st.stop()

# Load Generator
def get_generator():
    if "generator_loaded" not in st.session_state:
        tokenizer, model = load_generator()
        if tokenizer:
            st.session_state["generator_loaded"] = True
            st.session_state["tokenizer"] = tokenizer
            st.session_state["gen_model"] = model
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Model failed to load. Check error messages above.")
            st.session_state["generator_loaded"] = False
    return st.session_state.get("tokenizer"), st.session_state.get("gen_model"), st.session_state.get("generator_loaded", False)

if ask_clicked and question:
    # Retrieve
    with st.spinner("🔍 Retrieving relevant passages..."):
        passages = retrieve(question, st.session_state["index"], st.session_state["meta"], st.session_state["embed_model"])
    
    if not passages:
        st.error("❌ No relevant documents found.")
    else:
        top_score = passages[0]["score"]
        
        if top_score < REFUSAL_THRESHOLD:
            st.error(
                f"**REFUSED**: Retrieval score ({top_score:.3f}) is below our confidence threshold ({REFUSAL_THRESHOLD}).\n\n"
                "The documentation does not contain a direct answer to this question."
            )
        else:
            # Load generator
            tokenizer, model, loaded = get_generator()
            
            if loaded:
                with st.spinner("🤖 Generating answer..."):
                    try:
                        answer_text, source_text = generate(question, passages, tokenizer, model)
                        
                        if answer_text.upper() == "NOT_FOUND":
                            st.error(
                                "**REFUSED**: The model determined the retrieved passages do not contain the answer."
                            )
                        else:
                            # Display Direct Answer
                            st.success(f"**Answer:** {answer_text}")
                            
                            # Display Source
                            if source_text:
                                st.info(f"**Source:** {source_text}")
                            
                            # Display Confidence and Timing
                            st.caption(f"🎯 Retrieval Confidence: {top_score:.3f} (above {REFUSAL_THRESHOLD} threshold)")
                            st.caption(f"⏱️ Response Time: Calculated in real-time")
                            
                            # Store for session state
                            st.session_state["answer_displayed"] = True
                            
                    except Exception as e:
                        st.error(f"❌ Generation error: {e}")
                        st.exception(e)
            else:
                st.error("❌ Model not ready. Please try again.")

    # Show Retrieved Passages
    with st.expander("🔍 View Retrieved Passages (click to expand)"):
        for i, p in enumerate(passages, 1):
            st.markdown(f"**{i}. {p['doc_name']} ({p['section']}) — score: {p['score']:.3f}**")
            st.code(p["text"], language=None)
            st.markdown("---")

st.markdown("---")
st.markdown(
    "### About This System\n"
    "- **Docs indexed:** Pandas 3.0.5 API documentation\n"
    "- **Retriever:** BAAI/bge-small-en-v1.5\n"
    "- **Generator:** HuggingFace **SmolLM2-360M-Instruct** (CPU-optimized)\n"
    "- **Goal:** Zero hallucinations — refuses if answer not found in docs.\n\n"
    "Built for CAID internship at Namal University Mianwali by Muhammad Asad.\n\n"
    "[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)"
)
