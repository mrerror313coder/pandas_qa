#!/usr/bin/env python3
"""
embed_index.py
==============
Loads all passages from passages.jsonl, embeds them using a
sentence-transformer model, and builds a FAISS index for fast
similarity search.

Usage:
    python src/embed_index.py

Output:
    data/index/passages.faiss    - FAISS index file
    data/index/passages.meta.jsonl - metadata (doc_name, section, text)

Runtime:
    ~5-10 minutes on Colab T4 GPU
    ~30-60 minutes on CPU
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def main(passages_path, index_dir, model_name, batch_size):
    print("=" * 55)
    print("  Building embedding index")
    print("=" * 55)
    
    # ── Load passages ────────────────────────────────────────
    print(f"\nLoading passages from {passages_path}...")
    passages = []
    with open(passages_path) as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    print(f"  Loaded {len(passages):,} passages")
    
    # ── Load model ───────────────────────────────────────────
    print(f"\nLoading embedding model: {model_name}")
    print(f"  (First run downloads model - ~100-500 MB)")
    model = SentenceTransformer(model_name)
    print(f"  Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # ── Prepare texts for embedding ──────────────────────────
    # For better retrieval, prepend doc_name and section as context
    texts = []
    for p in passages:
        context = f"[{p['doc_name']}] [{p['section']}] {p['text']}"
        texts.append(context)
    
    # ── Embed all passages ───────────────────────────────────
    print(f"\nEmbedding {len(texts):,} passages...")
    print(f"  Batch size: {batch_size}")
    start = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # for cosine similarity via dot product
    )
    
    elapsed = time.time() - start
    print(f"  Embedding done in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.1f} passages/sec)")
    print(f"  Shape: {embeddings.shape}")
    
    # ── Build FAISS index ────────────────────────────────────
    print(f"\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    
    # IndexFlatIP = exact inner product (cosine sim with normalized vectors)
    # For 18k passages, exact search is fast enough (no need for approximate)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  Index size: {index.ntotal:,} vectors")
    
    # ── Save index + metadata ────────────────────────────────
    os.makedirs(index_dir, exist_ok=True)
    
    index_path = os.path.join(index_dir, "passages.faiss")
    faiss.write_index(index, index_path)
    print(f"\n  Saved FAISS index: {index_path}")
    
    # Save metadata (parallel to index rows)
    meta_path = os.path.join(index_dir, "passages.meta.jsonl")
    with open(meta_path, "w") as f:
        for p in passages:
            # Only keep what we need at retrieval time
            meta = {
                "doc_name": p["doc_name"],
                "section":  p["section"],
                "position": p["position"],
                "text":     p["text"],
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"  Saved metadata:    {meta_path}")
    
    # ── Save build config ────────────────────────────────────
    config = {
        "model_name":        model_name,
        "n_passages":        len(passages),
        "embedding_dim":     dim,
        "index_type":        "IndexFlatIP",
        "normalized":        True,
        "batch_size":        batch_size,
        "build_time_sec":    round(elapsed, 1),
    }
    config_path = os.path.join(index_dir, "build_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config:      {config_path}")
    
    print()
    print("=" * 55)
    print("  Index build complete!")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from passages")
    parser.add_argument("--passages", default="data/passages.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                        help="sentence-transformer model name")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    
    main(args.passages, args.index_dir, args.model, args.batch_size)
