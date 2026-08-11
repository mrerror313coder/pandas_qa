"""
streamlit_app.py - Optimized for Streamlit Cloud Free Tier (CPU Only)
"""

import time
import json
import os
import sys

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

# Configuration
GEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
REFUSAL_THRESHOLD = 0.82
K = 5

# Force CPU usage to prevent CUDA errors on free tier
DEVICE = "cpu"

PROMPT_TEMPLATE = """You are a documentation assistant for the pandas library. Answer the question using ONLY the information in the passages below.

STRICT RULES:
1. If the passages contain the answer, give a concise answer (1-2 sentences) followed by "Source: <doc_name>".
2. If and ONLY IF the passages truly do not contain the answer, respond with just: NOT_FOUND
3. Never write both an answer AND NOT_FOUND. Pick ONE.
4. Do not invent information not in the passages.

Passages:
{passages}

Question: {question}

Answer:"""


@st.cache_resource
def load_retrieval_system():
    """Loads the FAISS index and embedding model."""
    try:
        # Check if data exists
        if not os.path.exists("data/index_400/passages.faiss"):
            st.error("Data files not found. Please ensure 'data/index_400' is uploaded to your repo.")
            return None, None, None
            
        index = faiss.read_index("data/index_400/passages.faiss")
        meta = []
        with open("data/index_400/passages.meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        
        # Use CPU-only embedding model
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        return index, meta, embed_model
    except Exception as e:
        st.error(f"Failed to load retrieval system: {e}")
        return None, None, None


@st.cache_resource
def load_generator():
    """Loads the LLM. Optimized for CPU."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        
        # Explicitly force CPU and float32 for stability on free tier
        # device_map="auto" handles layer distribution if memory allows
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.float32, # CPU works best with float32
            device_map="auto",         # Automatically splits layers if possible
            low_cpu_mem_usage=True,    # Critical for free tier
        )
        
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load generator model: {e}")
        st.warning("The model is too large for the free tier resources. Consider using a smaller model like 'Qwen/Qwen2.5-1.5B-Instruct' or upgrading to a paid tier with GPU.")
        return None, None


def retrieve(question, index, meta, embed_model, k=K):
    if index is None or embed_model is None:
        return []
        
    q_embedding = embed_model.encode(
        [question],
        normalize_embeddings=True,
    ).astype(np.float32)
    
    scores, indices = index.search(q_embedding, k)
    passages = []
    for i in range(k):
        # Safety check for index bounds
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
    formatted = []
    for i, p in enumerate(passages, 1):
        formatted.append(
            f"[Passage {i}] doc_name: {p['doc_name']} | section: {p['section']}\n"
            f"{p['text']}"
        )
    return "\n\n".join(formatted)


def generate(question, passages, tokenizer, model, max_new_tokens=200):
    passages_str = format_passages(passages)
    prompt = PROMPT_TEMPLATE.format(passages=passages_str, question=question)

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    )
    
    # Ensure inputs are on CPU
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
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


st.title("📊 Pandas Docs Assistant")

st.markdown(
    "Ask any question about the **pandas library**. This system answers only from "
    "the official pandas 3.0.5 API documentation and cites its source."
)

# Initialize session state
if "retrieval_loaded" not in st.session_state:
    with st.spinner("Loading retrieval index (this may take a moment)..."):
        index, meta, embed_model = load_retrieval_system()
        if index is not None:
            st.session_state["retrieval_loaded"] = True
            st.session_state["index"] = index
            st.session_state["meta"] = meta
            st.session_state["embed_model"] = embed_model
            st.success(f"Loaded {index.ntotal} passages from pandas docs")
        else:
            st.session_state["retrieval_loaded"] = False

# Check if retrieval loaded successfully
if not st.session_state.get("retrieval_loaded", False):
    st.error("Retrieval system failed to load. Check logs for details.")
    st.stop()

question = st.text_input(
    "Your Question:",
    value=st.session_state.get("pending_q", ""),
    placeholder="e.g., What does the how parameter do in DataFrame.merge?",
)

examples = [
    "What does the 'how' parameter do in DataFrame.merge?",
    "What does DataFrame.melt do?",
    "What is the default sorting algorithm in DataFrame.sort_values?",
    "How do I use pandas to compute matrix eigenvalues?",
]

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    # Retrieve first (fast)
    passages = retrieve(
        question, 
        st.session_state["index"], 
        st.session_state["meta"], 
        st.session_state["embed_model"], 
        k=K
    )
    
    if not passages:
        st.error("No passages found.")
        st.stop()

    top_score = passages[0]["score"]

    if top_score < REFUSAL_THRESHOLD:
        st.error(
            f"**REFUSED** — Answer not found in pandas documentation.\n\n"
            f"**Reason:** Top retrieval score ({round(top_score, 3)}) is below our confidence threshold ({REFUSAL_THRESHOLD})."
        )
    else:
        # Load generator only when needed
        if "generator_loaded" not in st.session_state:
            with st.spinner("Loading AI model (this takes ~30s on CPU)..."):
                tokenizer, gen_model = load_generator()
                if tokenizer and gen_model:
                    st.session_state["generator_loaded"] = True
                    st.session_state["tokenizer"] = tokenizer
                    st.session_state["gen_model"] = gen_model
                else:
                    st.error("Model failed to load due to resource limits.")
                    st.stop()

        if st.session_state["generator_loaded"]:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate(
                        question, 
                        passages, 
                        st.session_state["tokenizer"], 
                        st.session_state["gen_model"]
                    )
                    
                    if answer.strip().upper().startswith("NOT_FOUND"):
                        st.error("**REFUSED** — The model determined the retrieved passages do not answer this question.")
                    else:
                        st.success("**Answer:**\n\n" + answer)
                        st.markdown(f"**Retrieval confidence:** {round(top_score, 3)}")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.info("This might be due to memory limits. Try a simpler question or upgrade resources.")
        else:
            st.error("Generator model not loaded.")

    # Show passages
    with st.expander("🔍 Retrieved passages (all " + str(K) + ")", expanded=False):
        for i, p in enumerate(passages, 1):
            header = f"**{i}. {p['doc_name']} ({p['section']}) — score: {round(p['score'], 3)}**"
            st.markdown(header)
            st.code(p["text"], language=None)
            st.markdown("---")

elif ask_clicked and not question:
    st.warning("Please enter a question first.")

st.markdown("---")
st.markdown("### Try these examples:")

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key="ex_" + str(i)):
            st.session_state["pending_q"] = ex
            st.rerun()

st.markdown("---")
st.markdown(
    "### About This System\n"
    "- **Docs indexed:** 2,087 pandas API pages → 11,855 passages\n"
    "- **Retriever:** BAAI/bge-small-en-v1.5\n"
    "- **Generator:** Qwen 2.5 3B Instruct (CPU Mode)\n"
    "- **Note:** Running on free tier resources. Response times may be slow."
)
