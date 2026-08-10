# Failure Analysis — Dev Set (Day 8 Final Baseline)

## Summary of All 60 Dev Questions

| Category | Count | Description |
|---|---|---|
| A. Correct answer | 14 | Answered correctly with proper citation |
| E. Correctly refused | 19 | Refused unanswerable (right decision) |
| B. Wrongly refused | 13 | Refused answerable (should have answered) |
| C. Hallucinated | 1 | Answered unanswerable (invented answer) |
| D. Wrong answer | 12 | Gave answer that didn't match expected |
| F. Unsupported answer | 1 | Answer text not in cited passages |

**Total successes:** 33 / 60 (55%)
**Total failures:** 27 / 60 (45%)

## Category B: Wrongly Refused (Root Cause Analysis)

These questions had answers in the docs but the pipeline refused.
Common cause: LLM said NOT_FOUND despite retrieval finding the right passage.

- **q063** (top_score=0.854): What does drop=True do in DataFrame.reset_index?
  - Expected: It does not insert the index into the DataFrame columns. It resets the index to the default integer 

- **q039** (top_score=0.848): What types can the 'func' parameter accept in Series.map?
  - Expected: A function, a collections.abc.Mapping subclass, or a Series.

- **q008** (top_score=0.851): What does a 'cross' merge do in DataFrame.merge?
  - Expected: It creates the cartesian product from both frames and preserves the order of the left keys.

- **q033** (top_score=0.798): What are the available parser engines in read_csv?
  - Expected: 'c', 'python', and 'pyarrow'. The C and pyarrow engines are faster, while the python engine is more 

- **q044** (top_score=0.850): What is the default aggregation function used by DataFrame.pivot_table?
  - Expected: mean

- **q024** (top_score=0.873): Which sorting algorithms are stable in DataFrame.sort_values?
  - Expected: mergesort and stable are the only stable algorithms.

- **q058** (top_score=0.851): What is the default value of the 'origin' parameter in pandas.to_datetime?
  - Expected: 'unix' (which sets the reference date to 1970-01-01)

- **q034** (top_score=0.816): What does the 'sep' parameter do in DataFrame.to_csv?
  - Expected: It specifies the field delimiter for the output file. It must be a string of length 1. Default is a 

- **q026** (top_score=0.840): What does axis=1 mean in DataFrame.apply?
  - Expected: It applies the function to each row.

- **q020** (top_score=0.847): What does 'ignore_index' do in DataFrame.dropna?
  - Expected: If True, the resulting axis will be labeled 0, 1, up to n-1.

- **q031** (top_score=0.813): What does the 'index_col' parameter do in read_csv?
  - Expected: It specifies the column(s) to use as row labels, denoted either by column labels or column indices. 

- **q025** (top_score=0.855): What does axis=0 mean in DataFrame.apply?
  - Expected: It applies the function to each column.

- **q047** (top_score=0.835): What is the default name for the totals row/column in DataFrame.pivot_table?
  - Expected: 'All'

## Category C: Hallucinations

These are unanswerable questions where the pipeline invented an answer.

- **q119** [deprecated_method]: How does Series.argmin work with skipna in pandas 3.0?
  - top_score: 0.932
  - Answer given: Series.argmin works with `skipna` by excluding NA/null values by default. If `skipna` is set to False and there is any NA value in the Series, it will raise a ValueError. Source: pandas.Series.argmin

## Category D: Wrong Answers (Evaluator Strictness)

Manual inspection shows most Category D answers are factually correct
but phrased differently from the recorded answer text. Examples:

- **q023**: What does the 'na_position' parameter do in DataFrame.sort_values?
  - Expected: It controls where NaN values are placed. 'first' puts NaNs at the beginning, and 'last' puts NaNs at
  - Got:      The `na_position` parameter in DataFrame.sort_values() puts NaNs at the beginning if set to 'first';

- **q060**: What is the difference between the 'index' and 'columns' parameters in DataFrame.rename?
  - Expected: 'index' is an alternative to specifying axis=0 with mapper. 'columns' is an alternative to specifyin
  - Got:      The 'index' parameter targets the index of the DataFrame, while the 'columns' parameter targets the 

- **q073**: What does the 'min_periods' parameter do in DataFrame.corr?
  - Expected: It is the minimum number of observations required per pair of columns to have a valid result. Curren
  - Got:      The 'min_periods' parameter specifies the minimum number of observations required per pair of column

- **q006**: What does the 'validate' parameter do in DataFrame.merge?
  - Expected: If specified, it checks if the merge is of a specified type such as one_to_one, one_to_many, many_to
  - Got:      The 'validate' parameter, if specified, checks if the merge keys are unique in both the left and rig

- **q043**: What does the 'flags' parameter do in Series.str.contains?
  - Expected: Flags to pass through to the re module, for example re.IGNORECASE. Default is 0 (no flags).
  - Got:      The 'flags' parameter in Series.str.contains controls flags from the re module, such as re.IGNORECAS

## Category F: Unsupported (Evaluator Matching Issue)

- **q071**: What does DataFrame.corr do?
  - Expected wording: Compute pairwise correlation of columns, excluding NA/null values.
  - Got: DataFrame.corr method computes pairwise correlation of columns, excluding NA/null values. Source: pa

## Key Insight

The biggest failure category is B (Wrongly Refused). All 13 cases had
top retrieval scores between 0.79 and 0.87 - retrieval was successful,
the LLM was overly cautious.

Day 9 attempted to fix this by adding a 'force-answer' override for
high-scoring passages. See day9_failure_analysis.md for the fix
attempt and why it was rejected.
