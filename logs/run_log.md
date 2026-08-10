# Run Log
One line per experiment run. Added chronologically.

| Date | What Changed | What Happened |
|------|-------------|---------------|
| 2026-08-08 | Day 1: ran ingest.py against pandas API reference (two-level crawl) | 18,476 passages from 2,087 pages written to passages.jsonl - sanity check passed all 6 checks |
| 2026-08-09 | Day 2: wrote 43 answerable questions across 10 pandas functions | dev.jsonl created - merge (8), read_csv (5), groupby (4), fillna (4), dropna (4), sort_values (4), apply (4), to_csv (4), Series.map (3), str.contains (3) |
| 2026-08-10 | Day 3: added 37 more answerable + 40 unanswerable questions | dev.jsonl now has 120 questions (80 answerable across 20 functions, 40 unanswerable: 10 wrong_library, 20 fake_parameter, 10 deprecated_method) |
| 2026-08-10 | Day 4: split questions (60 dev / 60 heldback), wrote evaluate.py, tested with hand-crafted predictions | dev.jsonl=60 (40 ans+20 unans), heldback.jsonl=60 (40 ans+20 unans, LOCKED), evaluate.py verified — all 5 metrics match expected values |
| 2026-08-10 | Day 5 v1: initial baseline (bge-small + Qwen2.5-3B, k=5) | recall=77.5%, correct=61.5%, unsupp=39.3%, ref-abs=90%, ref-pres=35% |
| 2026-08-10 | Day 5 v2: fixed parse_answer to strip trailing NOT_FOUND | recall=77.5%, unsupp=51.5% (worse due to more answers given), ref-pres=20% (better) |
| 2026-08-10 | Day 5 v5: BASELINE - use all retrieved passages as citation | recall=77.5%, correct=50%, UNSUPP=21.2% (big win), ref-abs=95%, ref-pres=20%, time=1.13s |
| 2026-08-10 | Day 5 v1: initial baseline (bge-small + Qwen2.5-3B, k=5) | recall=77.5%, correct=61.5%, unsupp=39.3%, ref-abs=90%, ref-pres=35% |
| 2026-08-10 | Day 5 v2: fixed parse_answer to strip trailing NOT_FOUND | recall=77.5%, unsupp=51.5% (worse due to more answers given), ref-pres=20% (better) |
| 2026-08-10 | Day 5 v5: BASELINE - use all retrieved passages as citation | recall=77.5%, correct=50%, UNSUPP=21.2% (big win), ref-abs=95%, ref-pres=20%, time=1.13s |
| 2026-08-10 | Day 6: config sweep (3 chunks × 3 k values = 9 configs) | WINNER chunk=400 k=5: unsupp=12.9% (was 21.2%), recall=80% (was 77.5%), correctness=51.7%, refuse-abs=90%, refuse-pres=27.5%, time=1.29s. Smaller chunks won because pandas parameter descriptions are short and distinct. |
| 2026-08-10 | Day 7 experiments: A) bge-base recall=67.5% unsupp=16.1% (worse); B) bge-small+MS-MARCO reranker unsupp=26.7% (much worse - reranker trained on prose); C) bge-base+reranker unsupp=21.2%. KEY FINDING: baseline unchanged (chunk=400, k=5, bge-small, unsupp=12.9%). Smaller/simpler wins on structured API docs - domain mismatch documented. |
