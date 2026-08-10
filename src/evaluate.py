#!/usr/bin/env python3
"""
evaluate.py
===========
Computes all 6 metrics from the project brief for a set of
pipeline predictions against the question ground truth.

Usage:
    python src/evaluate.py --questions data/questions/dev.jsonl \
                            --predictions results/dev_predictions.jsonl \
                            --output      results/dev_scores.json

Predictions file format (JSONL, one prediction per line):
    {
        "id":                "q001",
        "predicted_answer":  "..."          or None if refused,
        "cited_passage":     "..."          or None if refused,
        "retrieved_passages": [str, str],   top-k passages (for recall)
        "refused":           false,
        "response_time":     0.87           seconds
    }
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize(text):
    """
    Normalize text for comparison.
    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_wording(passage_text, supporting_wording):
    """
    Check if supporting_wording appears in passage_text.
    Both are normalized before comparison.

    Returns True if the wording is found (even approximately).
    """
    if not passage_text or not supporting_wording:
        return False

    passage_norm = normalize(passage_text)
    wording_norm = normalize(supporting_wording)

    # Check substring first (fastest)
    if wording_norm in passage_norm:
        return True

    # If not exact substring, check if all significant words appear
    # (handles minor formatting differences)
    wording_words = set(wording_norm.split())
    passage_words = set(passage_norm.split())

    # If 90%+ of words appear, count as match
    if len(wording_words) >= 4:
        overlap = len(wording_words & passage_words)
        if overlap / len(wording_words) >= 0.9:
            return True

    return False


# =============================================================================
# METRIC 1: RETRIEVAL RECALL @ K
# =============================================================================

def compute_retrieval_recall(questions, predictions_by_id):
    """
    For answerable questions: what fraction have the supporting_wording
    in ANY of the retrieved passages?
    """
    answerable = [q for q in questions if q.get("answerable")]
    if not answerable:
        return None

    hits = 0
    for q in answerable:
        pred = predictions_by_id.get(q["id"])
        if pred is None:
            continue

        retrieved = pred.get("retrieved_passages", [])
        supporting = q.get("supporting_wording", "")

        # Check if any retrieved passage contains the supporting wording
        for passage in retrieved:
            if contains_wording(passage, supporting):
                hits += 1
                break

    return hits / len(answerable)


# =============================================================================
# METRIC 2: ANSWER CORRECTNESS
# =============================================================================

def compute_answer_correctness(questions, predictions_by_id):
    """
    For answerable questions that were NOT refused: does the predicted
    answer match the recorded answer?
    """
    answerable = [q for q in questions if q.get("answerable")]
    if not answerable:
        return None

    correct = 0
    attempted = 0

    for q in answerable:
        pred = predictions_by_id.get(q["id"])
        if pred is None:
            continue

        # Skip refusals — they're counted elsewhere
        if pred.get("refused"):
            continue

        attempted += 1

        predicted = normalize(pred.get("predicted_answer", ""))
        expected  = normalize(q.get("answer", ""))

        # Consider match if the expected answer text appears in the prediction
        # (handles cases where prediction is longer but contains the answer)
        if expected and (expected in predicted or predicted in expected):
            correct += 1
            continue

        # Also check word overlap (looser matching for longer answers)
        expected_words = set(expected.split())
        predicted_words = set(predicted.split())

        if len(expected_words) >= 3:
            overlap = len(expected_words & predicted_words)
            if overlap / len(expected_words) >= 0.7:
                correct += 1

    if attempted == 0:
        return 0.0

    return correct / attempted


# =============================================================================
# METRIC 3: UNSUPPORTED ANSWERS (THE MAIN METRIC)
# =============================================================================

def compute_unsupported_rate(questions, predictions_by_id):
    """
    For every non-refused answer: does the cited passage actually contain
    something supporting the answer?

    An "unsupported answer" is one where the model gave an answer but
    the cited passage doesn't back it up.

    This is the MOST IMPORTANT metric per the project brief.
    """
    non_refused_count = 0
    unsupported_count = 0

    for q in questions:
        pred = predictions_by_id.get(q["id"])
        if pred is None:
            continue

        if pred.get("refused"):
            continue

        non_refused_count += 1

        cited = pred.get("cited_passage", "")
        answer = pred.get("predicted_answer", "")

        if not cited:
            # Answered but cited nothing = unsupported
            unsupported_count += 1
            continue

        # For answerable questions: check if cited passage contains
        # the ground-truth supporting wording
        if q.get("answerable"):
            supporting = q.get("supporting_wording", "")
            if not contains_wording(cited, supporting):
                unsupported_count += 1
        else:
            # For unanswerable questions: ANY confident answer is unsupported
            # (because there's no correct passage in the docs)
            unsupported_count += 1

    if non_refused_count == 0:
        return 0.0

    return unsupported_count / non_refused_count


# =============================================================================
# METRIC 4: REFUSAL RATES
# =============================================================================

def compute_refusal_when_absent(questions, predictions_by_id):
    """
    Of the UNANSWERABLE questions, what fraction were correctly refused?
    Higher is better.
    """
    unanswerable = [q for q in questions if not q.get("answerable")]
    if not unanswerable:
        return None

    refused = 0
    for q in unanswerable:
        pred = predictions_by_id.get(q["id"])
        if pred is None:
            continue
        if pred.get("refused"):
            refused += 1

    return refused / len(unanswerable)


def compute_refusal_when_present(questions, predictions_by_id):
    """
    Of the ANSWERABLE questions, what fraction were wrongly refused?
    Lower is better.
    """
    answerable = [q for q in questions if q.get("answerable")]
    if not answerable:
        return None

    refused = 0
    for q in answerable:
        pred = predictions_by_id.get(q["id"])
        if pred is None:
            continue
        if pred.get("refused"):
            refused += 1

    return refused / len(answerable)


# =============================================================================
# METRIC 5: RESPONSE TIME
# =============================================================================

def compute_median_response_time(predictions):
    """
    Median response time across all predictions.
    """
    times = [
        p.get("response_time")
        for p in predictions
        if p.get("response_time") is not None
    ]
    if not times:
        return None

    times.sort()
    n = len(times)
    if n % 2 == 1:
        return times[n // 2]
    return (times[n // 2 - 1] + times[n // 2]) / 2


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(questions_path, predictions_path, output_path=None):
    """
    Run all metrics and print results.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Safety: block scoring against heldback if it is still locked
    # ─────────────────────────────────────────────────────────────────────
    questions_dir = Path(questions_path).parent
    lock_file = questions_dir / "HELDBACK_LOCKED"
    if "heldback" in str(questions_path) and lock_file.exists():
        print("=" * 60)
        print("  REFUSING TO SCORE — HELDBACK SET IS LOCKED")
        print("=" * 60)
        print(f"\n  The heldback set may not be scored while the lock file exists:")
        print(f"    {lock_file}")
        print(f"\n  This is a safeguard against accidentally using the held-back")
        print(f"  set during development. It may be scored ONLY on Day 9,")
        print(f"  and only once, to produce the final unbiased result.")
        print(f"\n  To unlock (Day 9 only):")
        print(f"    rm {lock_file}")
        print("=" * 60)
        sys.exit(1)

    # Load questions
    questions = []
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Load predictions
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    print(f"Loaded {len(predictions)} predictions from {predictions_path}")

    # Build lookup
    predictions_by_id = {p["id"]: p for p in predictions}

    # ─────────────────────────────────────────────────────────────────────
    # Compute all metrics
    # ─────────────────────────────────────────────────────────────────────
    recall             = compute_retrieval_recall(questions, predictions_by_id)
    correctness        = compute_answer_correctness(questions, predictions_by_id)
    unsupported_rate   = compute_unsupported_rate(questions, predictions_by_id)
    refuse_when_absent = compute_refusal_when_absent(questions, predictions_by_id)
    refuse_when_present = compute_refusal_when_present(questions, predictions_by_id)
    median_time        = compute_median_response_time(predictions)

    # ─────────────────────────────────────────────────────────────────────
    # Print report
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Question set : {questions_path}")
    print(f"  Predictions  : {predictions_path}")
    print()
    print(f"  {'Retrieval recall@k':<30} : {recall:.1%}"
          if recall is not None else "  Retrieval recall@k             : N/A")
    print(f"  {'Answer correctness':<30} : {correctness:.1%}"
          if correctness is not None else "  Answer correctness             : N/A")
    print(f"  {'Unsupported answer rate':<30} : {unsupported_rate:.1%}  <-- MAIN METRIC"
          if unsupported_rate is not None else "  Unsupported answer rate        : N/A")
    print(f"  {'Refusal when absent':<30} : {refuse_when_absent:.1%}"
          if refuse_when_absent is not None else "  Refusal when absent            : N/A")
    print(f"  {'Refusal when present':<30} : {refuse_when_present:.1%}"
          if refuse_when_present is not None else "  Refusal when present           : N/A")
    print(f"  {'Median response time':<30} : {median_time:.2f}s"
          if median_time is not None else "  Median response time           : N/A")
    print()
    print("=" * 60)
    print("  Note: Unsupported answer rate is the MAIN metric per project brief.")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────
    # Save scores as JSON
    # ─────────────────────────────────────────────────────────────────────
    scores = {
        "questions_file"       : str(questions_path),
        "predictions_file"     : str(predictions_path),
        "n_questions"          : len(questions),
        "n_predictions"        : len(predictions),
        "retrieval_recall"     : recall,
        "answer_correctness"   : correctness,
        "unsupported_rate"     : unsupported_rate,
        "refusal_when_absent"  : refuse_when_absent,
        "refusal_when_present" : refuse_when_present,
        "median_response_time" : median_time,
    }

    if output_path:
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nScores saved to: {output_path}")

    return scores


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline predictions against ground truth"
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSONL file (dev.jsonl or heldback.jsonl)"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: save scores as JSON to this path"
    )
    args = parser.parse_args()

    evaluate(args.questions, args.predictions, args.output)
