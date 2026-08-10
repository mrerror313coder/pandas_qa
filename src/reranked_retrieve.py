#!/usr/bin/env python3
"""
reranked_retrieve.py
====================
Two-stage retrieval:
  1. Fast bi-encoder retrieves top-N candidates (e.g. N=20)
  2. Cross-encoder reranks them and returns top-k (e.g. k=5)

Cross-encoders look at (question, passage) pairs together, so they
are more accurate than bi-encoders but too slow to run on all
18,000 passages. Two-stage retrieval combines fast + accurate.
"""

import numpy as np
import faiss
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder


class RerankedRetriever:
    """
    Two-stage retriever: bi-encoder → cross-encoder rerank.
    """
    
    def __init__(self, 
                 index_dir="data/index",
                 bi_encoder_name="BAAI/bge-small-en-v1.5",
                 cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 candidates=20):
        """
        Args:
            candidates: number of passages retrieved before reranking.
                       Larger = more thorough but slower.
        """
        self.index_dir = Path(index_dir)
        self.candidates = candidates
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_dir / "passages.faiss"))
        
        # Load metadata
        self.meta = []
        with open(self.index_dir / "passages.meta.jsonl") as f:
            for line in f:
                self.meta.append(json.loads(line))
        
        # Load bi-encoder (for stage 1)
        print(f"  Loading bi-encoder: {bi_encoder_name}")
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        
        # Load cross-encoder (for stage 2)
        print(f"  Loading cross-encoder: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
    
    def retrieve(self, question, k=5):
        """
        Retrieve top-k passages using two-stage retrieval.
        """
        # ── Stage 1: bi-encoder retrieves top-N candidates ──────
        q_embedding = self.bi_encoder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        
        _, indices = self.index.search(q_embedding, self.candidates)
        candidate_passages = [self.meta[idx] for idx in indices[0]]
        
        # ── Stage 2: cross-encoder reranks ──────────────────────
        # Score each (question, passage) pair
        pairs = [(question, p["text"]) for p in candidate_passages]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score descending
        ranked = sorted(
            zip(scores, candidate_passages),
            key=lambda x: -x[0]
        )
        
        # Return top-k
        results = []
        for score, passage in ranked[:k]:
            p = passage.copy()
            p["score"] = float(score)
            results.append(p)
        
        return results
