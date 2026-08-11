"""
streamlit_app.py - Full pipeline (retrieval + grounded generation)
Matches src/retrieve.py + src/generate.py logic.
"""

import time
import json

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

GEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
REFUSAL_THRESHOLD = 0.82
K = 5

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
    index = faiss.read_index("data/index_400/passages.faiss")
    meta = []
    with open("data/index_400/passages.meta.jsonl") as f:
        for line in f:
            meta.append(json.loads(line))
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return index, meta, embed_model


@st.cache_resource
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    return tokenizer, model, device


def retrieve(question, index, meta, embed_model, k=K):
    q_embedding = embed_model.encode(
        [question],
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, indices = index.search(q_embedding, k)
    passages = []
    for i in range(k):
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


def generate(question, passages, tokenizer, model, device, max_new_tokens=200):
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
    ).to(device)

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
    "the official pandas 3.0.5 API documentation and cites its source. When the "
    "docs do not contain a good match, it **refuses** instead of hallucinating."
)

with st.spinner("Loading retrieval index (first load ~30s)..."):
    index, meta, embed_model = load_retrieval_system()

st.success("Loaded " + str(index.ntotal) + " passages from pandas docs")

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
    "What does the 'optimize' parameter do in DataFrame.merge?",
    "How do you use DataFrame.append() to add rows?",
]

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    start = time.time()
    passages = retrieve(question, index, meta, embed_model, k=K)
    top_score = passages[0]["score"]

    if top_score < REFUSAL_THRESHOLD:
        st.error(
            "**REFUSED** — Answer not found in pandas documentation.\n\n"
            "**Reason:** Top retrieval score (" + str(round(top_score, 3)) +
            ") is below our confidence threshold (" + str(REFUSAL_THRESHOLD) + ").\n\n"
            "The docs do not contain a direct answer to this question. "
            "The system refuses instead of making up an answer."
        )
    else:
        with st.spinner("Generating answer..."):
            tokenizer, gen_model, device = load_generator()
            answer = generate(question, passages, tokenizer, gen_model, device)
        elapsed = time.time() - start

        if answer.strip().upper().startswith("NOT_FOUND"):
            st.error(
                "**REFUSED** — The model determined the retrieved passages "
                "do not answer this question, even though retrieval confidence "
                "was above threshold (" + str(round(top_score, 3)) + ")."
            )
        else:
            st.success("**Answer:**\n\n" + answer)
            st.markdown(
                "**Retrieval confidence:** " + str(round(top_score, 3)) +
                " (above " + str(REFUSAL_THRESHOLD) + " threshold)"
            )
            st.markdown("**Response time:** " + str(round(elapsed, 2)) + "s")

        with st.expander("🔍 Retrieved passages (all " + str(K) + ")", expanded=False):
            for i, p in enumerate(passages, 1):
                header = (
                    "**" + str(i) + ". " + p["doc_name"] +
                    " (" + p["section"] + ") — score: " + str(round(p["score"], 3)) + "**"
                )
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
    "- **Generator:** Qwen 2.5 3B Instruct\n"
    "- **Refusal threshold:** 0.82 (chosen on dev set)\n"
    "- **Heldback results:** 100% correct refusal on 20 unanswerable questions\n\n"
    "Built for CAID internship at Namal University Mianwali by Muhammad Asad.\n\n"
    "[GitHub Repository](https://github.com/mrerror313coder/pandas_qa)"
)
