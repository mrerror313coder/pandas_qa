# Day 9 Failure Analysis + Attempted Fix

## Failure Category Distribution (Day 8 Baseline on Dev)

| Category                     | Count | % of Total |
|------------------------------|-------|-----------|
| A. Correct answer            | 14    | 23%       |
| E. Correctly refused         | 19    | 32%       |
| B. Wrongly refused           | 13    | 22%       |
| C. Hallucinated              |  1    |  2%       |
| D. Wrong answer given        | 12    | 20%       |
| F. Unsupported answer        |  1    |  2%       |
| **Total successes**          | **33**| **55%**   |
| **Total failures**           | **27**| **45%**   |

## Diagnosis of Biggest Failure Group (B)

The 13 wrongly-refused answerable questions all had TOP RETRIEVAL 
SCORES between 0.79 and 0.87. Retrieval was finding the right 
passages - the LLM (Qwen 2.5 3B) was just being overly cautious 
and outputting NOT_FOUND despite having correct context.

Sample:
- q024 (score=0.873): "Which sorting algorithms are stable?"
- q044 (score=0.850): "Default aggregation in pivot_table?"
- q025 (score=0.855): "What does axis=0 mean in DataFrame.apply?"

## Attempted Fix

Add a "force-answer" override: if the top retrieval score exceeds 0.80,
override any LLM refusal and provide an answer based on the top passage.

## Fix Results (Measured on Dev)

| Metric              | Day 8    | Day 9 Fix | Change    |
|---------------------|----------|-----------|-----------|
| Unsupported rate    | 7.1%     | 26.2%     | +19.0% ❌  |
| Refuse when absent  | 95.0%    | 75.0%     | -20.0% ❌  |
| Refuse when present | 32.5%    | 7.5%      | -25.0% ✅  |
| Correctness         | 55.6%    | 48.6%     | -6.9% ❌   |

## Why the Fix Failed

The force-answer rule successfully reduced Category B (from 13 to 3),
but at unacceptable cost to Categories C and F. Several unanswerable
questions had high top scores (up to 0.93 for q119 about deprecated
Series.argmin) because retrieval found *topically similar* passages
even when the docs contained no true answer.

**Root cause:** Bi-encoder similarity scores measure *topical relevance*
not *presence of a specific answer*. A high score means "this passage
is about the same subject" not "this passage contains the answer to
the question." Category C questions (deprecated methods) score high 
because the docs discuss the correct current API, which is topically 
similar to the deprecated version.

## Decision

Fix reverted. Day 8 pipeline (refusal_threshold=0.82, no force-answer)
kept as the final pipeline for heldback evaluation.

## Lesson

Single-signal refusal rules face an inherent trade-off. To reliably
answer more legit questions without hallucinating on wrong-library
questions, the system would need multi-signal refusal (e.g., LLM
uncertainty + retrieval score + answer-passage similarity). This is
beyond the 2-week scope but is noted as future work.
