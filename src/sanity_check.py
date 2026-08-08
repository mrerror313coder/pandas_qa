#!/usr/bin/env python3
"""
sanity_check.py
===============
Validates the passages.jsonl file produced by ingest.py.
Runs six checks and prints PASS, FAIL, or WARN for each.

Usage:
    python src/sanity_check.py
    python src/sanity_check.py --passages data/passages.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def run_sanity_check(filepath):
    """Load passages.jsonl and run all validation checks."""

    print("=" * 50)
    print("  SANITY CHECK - passages.jsonl")
    print("=" * 50)

    path = Path(filepath)

    if not path.exists():
        print(f"FAIL: File not found: {filepath}")
        sys.exit(1)

    # Load all passages
    passages = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                passage = json.loads(line)
                passages.append(passage)
            except json.JSONDecodeError as error:
                print(f"FAIL: Invalid JSON on line {line_number}: {error}")
                sys.exit(1)

    print(f"\nFile loaded: {len(passages)} passages found")
    print()

    all_passed = True

    # CHECK 1: Required keys present
    required_keys = {"doc_name", "section", "position", "text"}
    passages_missing_keys = [
        (i, required_keys - set(p.keys()))
        for i, p in enumerate(passages)
        if required_keys - set(p.keys())
    ]

    if passages_missing_keys:
        print(f"FAIL  Check 1 - Required keys: "
              f"{len(passages_missing_keys)} passages missing keys")
        for idx, missing in passages_missing_keys[:3]:
            print(f"        Passage {idx} is missing: {missing}")
        all_passed = False
    else:
        print(f"PASS  Check 1 - Required keys: all passages have all 4 keys")

    # CHECK 2: No empty text
    empty_text = [
        i for i, p in enumerate(passages)
        if not p.get("text", "").strip()
    ]

    if empty_text:
        print(f"FAIL  Check 2 - Empty text: {len(empty_text)} passages have empty text")
        all_passed = False
    else:
        print(f"PASS  Check 2 - Empty text: no empty passages found")

    # CHECK 3: Section values valid
    valid_sections = {"signature", "description", "parameters", "examples"}
    found_sections = {p["section"] for p in passages if "section" in p}
    unexpected_sections = found_sections - valid_sections

    if unexpected_sections:
        print(f"WARN  Check 3 - Section values: "
              f"unexpected sections found: {unexpected_sections}")
    else:
        print(f"PASS  Check 3 - Section values: "
              f"all sections are valid {valid_sections}")

    # CHECK 4: Passage count reasonable
    if len(passages) < 500:
        print(f"FAIL  Check 4 - Passage count: only {len(passages)} passages")
        all_passed = False
    elif len(passages) > 50000:
        print(f"WARN  Check 4 - Passage count: {len(passages)} passages seems high")
    else:
        print(f"PASS  Check 4 - Passage count: {len(passages)} (in expected range)")

    # CHECK 5: Spot check for DataFrame.merge
    merge_passages = [
        p for p in passages
        if "DataFrame.merge" in p.get("doc_name", "")
    ]

    if not merge_passages:
        print(f"FAIL  Check 5 - Spot check: no passages for DataFrame.merge")
        all_passed = False
    else:
        print(f"PASS  Check 5 - Spot check: "
              f"{len(merge_passages)} passages for DataFrame.merge")
        sample = merge_passages[0]["text"][:100]
        print(f"        Sample: {sample!r}...")

    # CHECK 6: Text length within limits
    too_long = [p for p in passages if len(p.get("text", "")) > 1000]
    if too_long:
        print(f"WARN  Check 6 - Text length: {len(too_long)} passages exceed 1000 chars")
    else:
        print(f"PASS  Check 6 - Text length: all passages within 1000 chars")

    # DISTRIBUTION SUMMARY
    by_section = defaultdict(int)
    for p in passages:
        by_section[p.get("section", "unknown")] += 1

    lengths = [len(p.get("text", "")) for p in passages]

    # Count unique doc pages
    unique_docs = len(set(p.get("doc_name", "") for p in passages))

    print()
    print("-" * 50)
    print(f"  Total unique doc pages: {unique_docs}")
    print(f"  Total passages        : {len(passages)}")
    print(f"  Average per page      : {len(passages) / unique_docs:.1f}")
    print("-" * 50)
    print("  Distribution by section:")
    for section, count in sorted(by_section.items()):
        pct = (count / len(passages)) * 100
        print(f"    {section:<15}  {count:>6}   ({pct:5.1f}%)")

    print()
    print("-" * 50)
    print("  Text length statistics:")
    if lengths:
        print(f"    Average : {sum(lengths) // len(lengths)} chars")
        print(f"    Minimum : {min(lengths)} chars")
        print(f"    Maximum : {max(lengths)} chars")

    print()
    print("=" * 50)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED")
        print("  Day 1 output is valid. Ready for Day 2.")
    else:
        print("  RESULT: SOME CHECKS FAILED")
        print("  Review the FAIL messages above before proceeding.")
    print("=" * 50)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the passages.jsonl output file"
    )
    parser.add_argument(
        "--passages",
        default="data/passages.jsonl",
        help="Path to passages.jsonl (default: data/passages.jsonl)"
    )
    args = parser.parse_args()

    run_sanity_check(args.passages)
