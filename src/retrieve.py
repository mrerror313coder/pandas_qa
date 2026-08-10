#!/usr/bin/env python3
"""
retrieve.py
===========
Retrieves top-k most relevant passages for a given question.

Can be used as:
  1. A CLI: python src/retrieve.py --question "..." --k 5
  2. A module: from retrieve import Retriever
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    """
    Loads the FAISS index and metadata, exposes a `retrieve` method.
    """
    
    def __init__(self, index_dir="data/index",
                 model_name="BAAI/bge-small-en-v1.5"):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        
        # Load FAISS index
        index_path = self.index_dir / "passages.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        self.meta = []
        meta_path = self.index_dir / "passages.meta.jsonl"
        with open(meta_path) as f:
            for line in f:
                self.meta.append(json.loads(line))
        
        # Load embedding model (same one used to build index)
        self.model = SentenceTransformer(model_name)
    
    def retrieve(self, question, k=5):
        """
        Return top-k passages for a question.
        
        Returns list of dicts:
            [{"score": 0.85, "doc_name": ..., "section": ..., "text": ...}, ...]
        """
        # Embed the question
        q_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(q_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            passage = self.meta[idx].copy()
            passage["score"] = float(score)
            results.append(passage)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-k passages")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()
    
    retriever = Retriever(index_dir=args.index_dir)
    results = retriever.retrieve(args.question, k=args.k)
    
    print(f"\nQuestion: {args.question}\n")
    print("=" * 60)
    for rank, r in enumerate(results, 1):
        print(f"[Rank {rank}] Score: {r['score']:.4f}")
        print(f"  Doc: {r['doc_name']} | Section: {r['section']}")
        print(f"  Text: {r['text'][:250]}...")
        print()


if __name__ == "__main__":
    main()
