# Data Card - Pandas API Reference Documentation

## Source Information
- **Name:** pandas API Reference Documentation
- **Version:** 3.0.5 (pinned - will not auto-update)
- **URL:** https://pandas.pydata.org/docs/reference/index.html
- **Date Retrieved:** 2026-08-08
- **Retrieved By:** Muhammad Asad, CAID Internship Project

## Licence
- **Licence Type:** BSD 3-Clause Licence
- **Licence URL:** https://github.com/pandas-dev/pandas/blob/main/LICENSE
- **Permitted Uses:** Storage, processing, redistribution with attribution
- **This Project Use:** Non-commercial academic research only
- **Attribution:** pandas development team

## Collection Statistics
- **Total Pages Scraped:** 2,087
- **Total Passages Extracted:** 18,476
- **Average Passages per Page:** 8.9

## What Was Collected
All function, method, and class pages from the pandas 3.0.5 API reference.
The scraper uses a two-level crawl:
1. Main reference index (17 sub-index pages)
2. Each sub-index page (frame.html, series.html, groupby.html, etc.)
3. Each individual API page (2,087 total)

Four types of passages are extracted from each page:

| Section     | Count   | What it contains                        |
|-------------|---------|-----------------------------------------|
| signature   | 679     | Function name and parameter list        |
| description | 1,930   | What the function does                  |
| parameters  | 12,505  | Each parameter name and its explanation |
| examples    | 3,362   | Code examples showing usage             |

## What Was NOT Collected
- User guide narrative pages (not API reference)
- Changelog and release notes
- Developer/contributor documentation
- Pages that require JavaScript to render content

## Passage Schema
Each line in passages.jsonl contains one JSON object with these fields:

- **doc_name** : the pandas object name (e.g. pandas.DataFrame.merge)
- **section**  : one of signature / description / parameters / examples
- **position** : integer position within the page (0-based)
- **text**     : plain text content of the passage

Example passage:

    {
      "doc_name" : "pandas.DataFrame.merge",
      "section"  : "parameters",
      "position" : 0,
      "text"     : "how : left, right, outer, inner or cross..."
    }

## Known Limitations
- CSS selectors written for PyData Sphinx theme (v3.0.5)
- Code examples stored as plain text only - not executed or validated
- Some short pages may not have all four section types
- Text length capped at 800 characters per passage (longer texts split)
- Minimum passage length: 60 characters

## File Locations
- Raw HTML cache : data/raw_docs/
- Passages file  : data/passages.jsonl (4.9 MB)
- This data card : data_card.md
