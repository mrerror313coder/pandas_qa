"""
streamlit_app.py - Ultra-Lightweight CPU Version
Uses SmolLM2-360M-Instruct (360MB, fits in 1GB RAM)
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
    page_title="Pandas Docs Assistant",
    page_icon="📊",
    layout="wide",
)

# ULTRA-TINY MODEL: 360MB, fits easily in free tier RAM
GEN_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
REFUSAL_THRESHOLD = 0.82
K = 5
DEVICE = "cpu"

PROMPT_TEMPLATE = """You are a documentation assistant for the pandas library. Answer the question using ONLY the information in the passages below.

STRICT RULES:
1. If the passages contain the answer, give a concise answer (1-2 sentences) followed by "Source: <doc_name>".
2. If the passages do not contain the answer, respond with just: NOT_FOUND
3. Do not invent information.

Passages:
{passages}

Question: {question}

Answer:"""


@st.cache_resource
def load_retrieval_system():
    """Loads the FAISS index and embedding model."""
    try:
        if not os.path.exists("data/index_400/passages.faiss"):
            st.error("Data files missing. Ensure 'data/index_400' is in your repo.")
            return None, None, None
            
        index = faiss.read_index("data/index_400/passages.faiss")
        meta = []
        with open("data/index_400/passages.meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return index, meta, embed_model
    except Exception as e:
        st.error(f"Retrieval load error: {e}")
        return None, None, None


@st.cache_resource
def load_generator():
    """Loads the ultra-tiny SmolLM2 model."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        st.info(f"Loading {GEN_MODEL_NAME}... (this takes ~15s on CPU)")
        
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        
        # Force CPU, float32, and low memory usage
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.exception(e)  # Show the full error
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
    
    # SmolLM2 uses standard chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


st.title("📊 Pandas Docs Assistant")
st.markdown("Ask questions about the pandas library. (Running on CPU with SmolLM2-360M)")

# Load Retriever
if "retrieval_loaded" not in st.session_state:
    with st.spinner("Loading index..."):
        index, meta, embed_model = load_retrieval_system()
        if index:
            st.session_state.update({
                "retrieval_loaded": True,
                "index": index,
                "meta": meta,
                "embed_model": embed_model
            })
            st.success(f"Loaded {index.ntotal} passages.")
        else:
            st.session_state["retrieval_loaded"] = False

if not st.session_state.get("retrieval_loaded"):
    st.stop()

# Load Generator
def get_generator():
    if "generator_loaded" not in st.session_state:
        tokenizer, model = load_generator()
        if tokenizer:
            st.session_state["generator_loaded"] = True
            st.session_state["tokenizer"] = tokenizer
            st.session_state["gen_model"] = model
            st.success("Model loaded successfully!")
        else:
            st.error("Model failed to load. Check logs above.")
            st.session_state["generator_loaded"] = False
    return st.session_state.get("tokenizer"), st.session_state.get("gen_model"), st.session_state.get("generator_loaded", False)

question = st.text_input("Your Question:", placeholder="e.g., What is the use of read_csv?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    passages = retrieve(question, st.session_state["index"], st.session_state["meta"], st.session_state["embed_model"])
    
    if not passages:
        st.error("No relevant documents found.")
    else:
        top_score = passages[0]["score"]
        if top_score < REFUSAL_THRESHOLD:
            st.error(f"**REFUSED**: Retrieval score ({top_score:.3f}) too low.")
        else:
            tokenizer, model, loaded = get_generator()
            
            if loaded:
                with st.spinner("Generating answer..."):
                    try:
                        answer = generate(question, passages, tokenizer, model)
                        if "NOT_FOUND" in answer.upper():
                            st.error("**REFUSED**: Model could not find answer in context.")
                        else:
                            st.success(f"**Answer:**\n\n{answer}")
                            st.caption(f"Confidence: {top_score:.3f}")
                    except Exception as e:
                        st.error(f"Generation error: {e}")
                        st.exception(e)
            else:
                st.error("Model not ready. Check error messages above.")

    with st.expander("🔍 Retrieved Passages"):
        for p in passages:
            st.code(f"Source: {p['doc_name']}\nScore: {p['score']:.3f}\n\n{p['text']}")

st.markdown("---")
st.caption("Built with Streamlit. Running on CPU with HuggingFace SmolLM2-360M-Instruct.")
