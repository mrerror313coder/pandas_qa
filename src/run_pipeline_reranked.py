#!/usr/bin/env python3
"""
run_pipeline_reranked.py
========================
Same as run_pipeline.py but uses RerankedRetriever (2-stage retrieval).
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reranked_retrieve import RerankedRetriever
from generate import Generator
from run_pipeline import parse_answer  # reuse existing parser


def main(questions_path, output_path, k, index_dir, 
         bi_encoder, cross_encoder, candidates):
    print("=" * 60)
    print("  Running QA Pipeline (with reranker)")
    print("=" * 60)
    
    questions = []
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"\nLoaded {len(questions)} questions")
    
    print(f"\nLoading reranked retriever (candidates={candidates}, k={k})...")
    retriever = RerankedRetriever(
        index_dir=index_dir,
        bi_encoder_name=bi_encoder,
        cross_encoder_name=cross_encoder,
        candidates=candidates,
    )
    
    print("\nLoading generator...")
    generator = Generator()
    
    print(f"\nRunning pipeline...")
    predictions = []
    
    for i, q in enumerate(questions, 1):
        start = time.time()
        
        passages = retriever.retrieve(q["question"], k=k)
        raw_answer = generator.generate(q["question"], passages)
        predicted_answer, cited_doc, refused = parse_answer(raw_answer)
        
        if refused:
            cited_passage = None
        else:
            cited_passage = " ||| ".join(p["text"] for p in passages)
        
        elapsed = time.time() - start
        
        prediction = {
            "id": q["id"],
            "predicted_answer": predicted_answer,
            "cited_passage": cited_passage,
            "retrieved_passages": [p["text"] for p in passages],
            "refused": refused,
            "response_time": round(elapsed, 3),
            "raw_answer": raw_answer,
            "top_score": passages[0]["score"] if passages else 0.0,
        }
        predictions.append(prediction)
        
        status = "REFUSED" if refused else "answered"
        print(f"  [{i:3d}/{len(questions)}] {status:8s} "
              f"({elapsed:.1f}s)  {q['question'][:55]}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n  Saved to: {output_path}")
    
    refused_count = sum(1 for p in predictions if p["refused"])
    avg_time = sum(p["response_time"] for p in predictions) / len(predictions)
    print(f"\n  Refused: {refused_count} | Answered: {len(predictions)-refused_count} | "
          f"Avg time: {avg_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/index_400")
    parser.add_argument("--bi-encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--cross-encoder", 
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidates", type=int, default=20)
    args = parser.parse_args()
    
    main(args.questions, args.output, args.k, args.index_dir,
         args.bi_encoder, args.cross_encoder, args.candidates)
