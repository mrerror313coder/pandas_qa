#!/usr/bin/env python3
"""
sweep.py
========
Runs the pipeline across 9 (chunk_size, k) combinations and
produces a comparison report.
"""

import json
import os
import subprocess


CONFIGS = [
    (400, 3),
    (400, 5),
    (400, 10),
    (800, 3),
    (800, 5),   # baseline
    (800, 10),
    (1200, 3),
    (1200, 5),
    (1200, 10),
]


def run_config(chunk_size, k):
    config_id = f"chunk{chunk_size}_k{k}"
    index_dir = f"data/index_{chunk_size}"
    pred_file = f"results/sweep_{config_id}_preds.jsonl"
    score_file = f"results/sweep_{config_id}_scores.json"
    
    # Skip if already done
    if os.path.exists(score_file):
        print(f"[SKIP] {config_id} - already done")
        with open(score_file) as f:
            return json.load(f)
    
    print(f"\n{'=' * 60}")
    print(f"  Running: {config_id}")
    print(f"{'=' * 60}")
    
    # Run pipeline
    result = subprocess.run([
        "python", "src/run_pipeline.py",
        "--questions", "data/questions/dev.jsonl",
        "--output", pred_file,
        "--k", str(k),
        "--index-dir", index_dir,
    ])
    
    if result.returncode != 0:
        print(f"  FAILED: {config_id}")
        return None
    
    # Evaluate
    result = subprocess.run([
        "python", "src/evaluate.py",
        "--questions", "data/questions/dev.jsonl",
        "--predictions", pred_file,
        "--output", score_file,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  EVAL FAILED: {config_id}")
        return None
    
    with open(score_file) as f:
        return json.load(f)


def main():
    results = {}
    
    for chunk_size, k in CONFIGS:
        config_id = f"chunk{chunk_size}_k{k}"
        scores = run_config(chunk_size, k)
        if scores:
            results[config_id] = {
                "chunk_size": chunk_size,
                "k": k,
                **scores,
            }
    
    # Save summary
    with open("results/sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print("\n" + "=" * 100)
    print("  CONFIGURATION SWEEP RESULTS")
    print("=" * 100)
    print(f"  {'config':<20}{'recall':<10}{'correct':<10}"
          f"{'UNSUPP':<10}{'ref-abs':<10}{'ref-pres':<10}{'time':<8}")
    print("-" * 100)
    
    for config_id, r in results.items():
        marker = "  <-- baseline" if config_id == "chunk800_k5" else ""
        print(
            f"  {config_id:<20}"
            f"{r['retrieval_recall']:>7.1%}   "
            f"{r['answer_correctness']:>7.1%}   "
            f"{r['unsupported_rate']:>7.1%}   "
            f"{r['refusal_when_absent']:>7.1%}   "
            f"{r['refusal_when_present']:>7.1%}   "
            f"{r['median_response_time']:>5.2f}s"
            f"{marker}"
        )
    
    print("=" * 100)
    
    # Find best
    best = min(results.values(), key=lambda r: r["unsupported_rate"])
    print(f"\n  BEST by unsupported rate:")
    print(f"    chunk={best['chunk_size']}, k={best['k']}")
    print(f"    unsupported={best['unsupported_rate']:.1%}")
    print(f"    recall={best['retrieval_recall']:.1%}")
    print(f"    correctness={best['answer_correctness']:.1%}")


if __name__ == "__main__":
    main()
