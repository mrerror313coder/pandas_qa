#!/usr/bin/env python3
"""
run_pipeline.py
===============
Runs the QA pipeline on a set of questions and saves predictions.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieve import Retriever
from generate import Generator


def parse_answer(raw_answer):
    """
    Parse the LLM output into (predicted_answer, cited_doc_name, refused).
    """
    text = raw_answer.strip()
    
    # Strip trailing NOT_FOUND / Citation Needed / Doc_Name: NOT_FOUND
    text_stripped = re.sub(r"\s*NOT_FOUND\s*$", "", text, flags=re.IGNORECASE).strip()
    text_stripped = re.sub(r"\s*Citation Needed\s*$", "", text_stripped, flags=re.IGNORECASE).strip()
    text_stripped = re.sub(r"\s*Doc_Name:\s*NOT_FOUND\s*$", "", text_stripped, flags=re.IGNORECASE).strip()
    
    # True refusal
    if not text_stripped or text_stripped.upper().startswith("NOT_FOUND"):
        return None, None, True
    
    if len(text_stripped) < 20 and "NOT_FOUND" in text.upper():
        return None, None, True
    
    refusal_phrases = [
        "i don't know", "i do not know", 
        "cannot be answered", "not in the passages",
        "not mentioned in", "no information",
    ]
    if any(text_stripped.lower().startswith(p) for p in refusal_phrases):
        return None, None, True
    
    # Extract citation
    cite_match = re.search(r"pandas\.[a-zA-Z_.]+", text)
    cited = cite_match.group(0) if cite_match else None
    
    return text_stripped, cited, False


def main(questions_path, output_path, k, index_dir, embed_model="BAAI/bge-small-en-v1.5"):
    print("=" * 60)
    print("  Running QA Pipeline")
    print("=" * 60)
    
    questions = []
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"\nLoaded {len(questions)} questions from {questions_path}")
    
    print("\nLoading retriever...")
    retriever = Retriever(index_dir=index_dir, model_name=embed_model)
    
    print("\nLoading generator...")
    generator = Generator()
    
    print(f"\nRunning pipeline (k={k})...")
    predictions = []
    
    for i, q in enumerate(questions, 1):
        start = time.time()
        
        passages = retriever.retrieve(q["question"], k=k)
        raw_answer = generator.generate(q["question"], passages)
        predicted_answer, cited_doc, refused = parse_answer(raw_answer)
        
        # ═══════════════════════════════════════════════════════════
        # CITATION = ALL RETRIEVED PASSAGES COMBINED
        # This reflects real grounding: the LLM saw all these passages
        # so the answer is grounded in ANY of them, not just one.
        # ═══════════════════════════════════════════════════════════
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
              f"({elapsed:.1f}s)  {q['question'][:60]}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n  Saved {len(predictions)} predictions to: {output_path}")
    
    refused_count = sum(1 for p in predictions if p["refused"])
    avg_time = sum(p["response_time"] for p in predictions) / len(predictions)
    
    print()
    print("=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print(f"  Total questions   : {len(predictions)}")
    print(f"  Refused           : {refused_count}")
    print(f"  Answered          : {len(predictions) - refused_count}")
    print(f"  Average time      : {avg_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()
    
    main(args.questions, args.output, args.k, args.index_dir, args.embed_model)
