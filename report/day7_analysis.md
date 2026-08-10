# Day 7 Analysis: Why Alternative Retrievers Did Not Help

## Experiments Performed

Three alternative retrieval configurations were tested against
the Day 6 baseline (chunk=400, k=5, bge-small-en-v1.5):

| Config                       | Recall | Unsupported | Response Time |
|------------------------------|--------|-------------|---------------|
| BASELINE (Day 6)             | 80.0%  | 12.9%       | 1.29s         |
| A: bge-base-en-v1.5          | 67.5%  | 16.1%       | 0.98s         |
| B: bge-small + reranker      | 67.5%  | 26.7%       | 0.61s         |
| C: bge-base + reranker       | 75.0%  | 21.2%       | 1.43s         |

## Root Cause Analysis

### bge-base: Worse Despite Being 2x Larger

The bge-base model has approximately 2x more parameters and
produces 768-dim embeddings (vs 384 for bge-small). It typically
outperforms bge-small on standard benchmarks (BEIR, MTEB).

However, on pandas API documentation - highly structured
technical text with parameter names, type hints, and code
snippets - bge-small performed 12.5% better on recall at 5.

Hypothesis: bge-small was likely trained with more emphasis on
short, keyword-heavy queries (documentation-style). bge-base's
additional capacity is spent on modeling longer prose contexts
that do not help here.

### Reranker: Actively Harmful

The cross-encoder/ms-marco-MiniLM-L-6-v2 reranker was trained
on the MS MARCO passage ranking dataset - passages about general
web queries. When applied to pandas documentation:

- Baseline (chunk=400, k=5): 12.9% unsupported
- Baseline + reranker      : 26.7% unsupported (13.8% WORSE)

The reranker consistently downranked correct passages containing
API signatures and parameter definitions, preferring passages
with more prose. This is a classic domain mismatch: the reranker's
training distribution differs fundamentally from our target domain.

## Lesson Learned

Alternative components should be evaluated on the target domain
rather than assumed to be improvements. In this case, a smaller,
older embedding model without a reranker outperformed all more
"sophisticated" combinations. This finding suggests that
retrieval-side improvements have plateaued for this domain, and
gains must come from downstream (refusal, prompting).

## Impact on Later Days

Day 8 pivots to refusal threshold tuning - the largest remaining
opportunity to reduce unsupported answers without changing retrieval.
