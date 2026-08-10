# Pandas API Documentation QA System

**Intern:** Muhammad Asad Riaz
**Supervisor:** Dr. Tassadaq Hussain
**Co-Supervisor:** Kamran Younis
**Affiliation:** Centre for AI & BigData, Namal University Mianwali
**Duration:** 2 weeks

---

## What This Project Does

Answers questions about the pandas library using **only** the official
documentation. Every answer includes a citation showing which passage
it came from. When a question asks about something not in the docs
(wrong function, fake parameter, deprecated method), the system
refuses to answer instead of guessing.

**Problem it solves:** AI coding assistants regularly hallucinate
function names and parameters that do not exist. This system cannot
hallucinate - it can only repeat what the documentation actually says.

---

## Current Dataset

- **Source:** pandas 3.0.5 API Reference Documentation
- **Pages scraped:** 2,087
- **Passages extracted:** 18,476
- **File size:** 4.9 MB

---

## Project Progress

- [x] Day 1  - Passages extracted from pandas 3.0.5 docs
- [x] Day 2  - 43 answerable questions written across 10 functions
- [x] Day 3  - Complete question set (120 questions - 80 answerable, 40 unanswerable)
- [x] Day 4  - Dev (60) / heldback (60) split done + evaluate.py written and tested
- [x] Day 5  - Baseline pipeline complete (unsupported=21%, recall=77.5%)
- [x] Day 6  - Config study: chunk=400, k=5 wins (unsupp 21.2% → 12.9%)
- [x] Day 7  - Tested bge-base & reranker - NONE beat Day 6 (domain mismatch documented)
- [x] Day 8  - Refusal threshold=0.82 chosen (unsupp 12.9% -> 7.1%)
- [x] Day 9  - Failure analysis done, heldback scored (unsupp=14.8%, refuse-abs=100%)
- [ ] Day 10 - Report + demo recording

---

## Repository Structure

    pandas_qa/
    |-- data/
    |   |-- raw_docs/          cached HTML from pandas website
    |   |-- passages.jsonl     all document passages (Day 1 output)
    |   |-- questions/
    |       |-- dev.jsonl      development questions (Day 4)
    |       |-- heldback.jsonl held-back questions   (Day 4)
    |-- src/
    |   |-- ingest.py          scraper - produces passages.jsonl
    |   |-- sanity_check.py    validates passages.jsonl
    |   |-- embed_index.py     builds FAISS vector index (Day 5)
    |   |-- retrieve.py        retrieves relevant passages (Day 5)
    |   |-- generate.py        generates grounded answers (Day 5)
    |   |-- refusal.py         handles refusal logic (Day 8)
    |   |-- evaluate.py        computes evaluation metrics (Day 4)
    |-- configs/
    |   |-- runs.yaml          experiment configurations
    |-- logs/
    |   |-- run_log.md         one line per experiment run
    |-- notebooks/
    |   |-- analysis.ipynb     graphs and failure analysis
    |-- report/
    |   |-- report.md          final 6-10 page report
    |-- data_card.md           source and licence information
    |-- requirements.txt       required Python packages
    |-- README.md              this file

---

## Quick Start

    # Install dependencies
    pip install -r requirements.txt

    # Day 1: Scrape docs and validate
    python src/ingest.py
    python src/sanity_check.py

    # Day 5+: Run the full pipeline
    python src/embed_index.py
    python src/retrieve.py --question "What does the how parameter do?"

---

## Document Source

- **Source:** pandas API Reference Documentation
- **Version:** 3.0.5 (pinned)
- **URL:** https://pandas.pydata.org/docs/reference/index.html
- **Licence:** BSD 3-Clause
- **Retrieved:** 2026-08-08

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Retrieval recall@k | Fraction of questions where the correct passage is in top-k results |
| Answer correctness | Fraction of answers matching the recorded answer |
| Unsupported answers | Fraction of answers not supported by the cited passage |
| Refusal when absent | Fraction of unanswerable questions correctly declined |
| Refusal when present | Fraction of answerable questions wrongly declined |
| Response time | Median seconds per question |
