#!/usr/bin/env python3
"""
ingest.py
=========
Scrapes the pandas 3.0.5 API reference documentation website and
produces passages.jsonl - the core document store for the QA system.
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

BASE_URL = "https://pandas.pydata.org/docs/reference"
INDEX_URL = f"{BASE_URL}/index.html"
RAW_DIR = "data/raw_docs"
OUT_FILE = "data/passages.jsonl"
REQUEST_DELAY = 1.5
MIN_CHARS = 60
MAX_CHARS = 800

HEADERS = {
    "User-Agent": (
        "pandas-qa-research-bot/1.0 "
        "(CAID Internship, Namal University - academic use only)"
    )
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text(text):
    """Clean up whitespace in raw text extracted from HTML."""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace 3+ consecutive newlines with just two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text


def split_long_text(text, max_chars):
    """Split long text at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while len(text) > max_chars:
        cut_point = text.rfind(". ", 0, max_chars)
        if cut_point == -1:
            cut_point = max_chars
        else:
            cut_point += 1
        chunks.append(text[:cut_point].strip())
        text = text[cut_point:].strip()

    if text:
        chunks.append(text)
    return chunks


def fetch_page(url, use_cache=True):
    """Download an HTML page, using file-based cache."""
    safe_name = url.replace("https://", "").replace("/", "_")
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", safe_name)
    safe_name = safe_name[:150]
    cache_path = os.path.join(RAW_DIR, safe_name + ".html")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            print(f"    SKIP: status {response.status_code} - {url}")
            return None

        html_content = response.text
        os.makedirs(RAW_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        time.sleep(REQUEST_DELAY)
        return html_content

    except requests.RequestException as error:
        print(f"    ERROR: {error} - {url}")
        return None


def get_all_api_links(index_html):
    """
    Extract all API page URLs from the pandas reference documentation.

    The pandas docs use a TWO-LEVEL structure:
        Level 1: /docs/reference/index.html
                 -> lists sub-index pages (frame.html, series.html, etc.)
        Level 2: /docs/reference/frame.html
                 -> lists individual API pages (DataFrame.merge, etc.)

    We must crawl both levels to find all ~700 API pages.
    """
    soup = BeautifulSoup(index_html, "lxml")

    # ── STEP A: Find all sub-index page URLs from the main index ─────────
    # Sub-index pages are things like: frame.html, series.html, groupby.html
    # They live in the same directory as index.html (not in /api/)
    sub_index_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        # Skip API pages (we get those in step B)
        if "api/" in href:
            continue
        # Skip external links, anchors, etc.
        if href.startswith(("http", "#", "mailto:", "..", "javascript:")):
            continue
        # Skip the index page itself
        if "index.html" in href or href == "":
            continue
        # Only keep .html files
        if not href.endswith(".html"):
            continue

        full_url = f"{BASE_URL}/{href}".split("#")[0]
        sub_index_urls.add(full_url)

    print(f"    Level 1: found {len(sub_index_urls)} sub-index pages")

    # ── STEP B: Fetch each sub-index and collect all API links ───────────
    all_api_links = set()

    # Also check the main index itself for any direct API links
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "api/pandas." in href:
            if href.startswith("http"):
                full_url = href
            else:
                full_url = f"{BASE_URL}/{href}"
            full_url = full_url.split("#")[0]
            all_api_links.add(full_url)

    # Now crawl each sub-index page
    for i, sub_url in enumerate(sorted(sub_index_urls), start=1):
        print(f"    Level 2 [{i:2d}/{len(sub_index_urls)}]: {sub_url.split('/')[-1]}")

        sub_html = fetch_page(sub_url, use_cache=True)
        if not sub_html:
            continue

        sub_soup = BeautifulSoup(sub_html, "lxml")

        for a_tag in sub_soup.find_all("a", href=True):
            href = a_tag["href"]

            if "api/pandas." not in href:
                continue

            if href.startswith("http"):
                full_url = href
            else:
                # href might be relative like "api/pandas.DataFrame.merge.html"
                # or "../reference/api/pandas.DataFrame.merge.html"
                if href.startswith("api/"):
                    full_url = f"{BASE_URL}/{href}"
                else:
                    # Handle relative paths
                    from urllib.parse import urljoin
                    full_url = urljoin(sub_url, href)

            full_url = full_url.split("#")[0]
            all_api_links.add(full_url)

    print(f"    Total unique API pages found: {len(all_api_links)}")
    return sorted(all_api_links)


def url_to_doc_name(url):
    """Convert URL to a doc_name like 'pandas.DataFrame.merge'."""
    filename = url.rstrip("/").split("/")[-1]
    doc_name = filename.replace(".html", "")
    return doc_name


def build_passages(doc_name, section, texts, position_start):
    """Convert raw text blocks into passage dictionaries."""
    passages = []
    position = position_start

    for raw_text in texts:
        cleaned = clean_text(raw_text)
        if len(cleaned) < MIN_CHARS:
            continue

        chunks = split_long_text(cleaned, MAX_CHARS)
        for chunk in chunks:
            if len(chunk) >= MIN_CHARS:
                passages.append({
                    "doc_name": doc_name,
                    "section": section,
                    "position": position,
                    "text": chunk
                })
                position += 1

    return passages, position


# =============================================================================
# PAGE PARSER
# =============================================================================

def parse_api_page(html, doc_name):
    """Extract passages from one pandas API documentation page."""
    soup = BeautifulSoup(html, "lxml")

    # Remove Sphinx anchor links
    for anchor in soup.find_all("a", class_="headerlink"):
        anchor.decompose()

    all_passages = []
    position = 0

    # ── SECTION 1: SIGNATURE ────────────────────────────────────────────────
    signature_texts = []
    sig_elements = soup.find_all("dt", class_=re.compile(r"sig"))

    if not sig_elements:
        for dl in soup.find_all("dl", class_=re.compile(r"^py ")):
            dt = dl.find("dt")
            if dt:
                sig_elements.append(dt)

    if sig_elements:
        sig_text = sig_elements[0].get_text(separator=" ")
        signature_texts.append(sig_text)

    sig_passages, position = build_passages(
        doc_name, "signature", signature_texts, position
    )
    all_passages.extend(sig_passages)

    # ── SECTION 2: DESCRIPTION ──────────────────────────────────────────────
    description_texts = []
    py_dl_elements = soup.find_all("dl", class_=re.compile(r"^py "))

    for py_dl in py_dl_elements:
        dd = py_dl.find("dd")
        if not dd:
            continue

        for child in dd.children:
            if hasattr(child, "name"):
                if child.name == "dl":
                    break
                if child.name == "p":
                    text = child.get_text(separator=" ")
                    description_texts.append(text)

        if description_texts:
            break

    if not description_texts:
        main_content = (
            soup.find("div", role="main") or
            soup.find("article") or
            soup.find("div", class_="body")
        )
        if main_content:
            for p_tag in main_content.find_all("p"):
                text = p_tag.get_text(separator=" ")
                description_texts.append(text)
                if len(description_texts) >= 3:
                    break

    desc_passages, position = build_passages(
        doc_name, "description", description_texts[:3], position
    )
    all_passages.extend(desc_passages)

    # ── SECTION 3: PARAMETERS ───────────────────────────────────────────────
    parameter_texts = []
    field_lists = soup.find_all(
        "dl",
        class_=re.compile(r"field-list|simple")
    )

    for field_dl in field_lists:
        dt_tags = field_dl.find_all("dt")
        dd_tags = field_dl.find_all("dd")

        for dt_tag, dd_tag in zip(dt_tags, dd_tags):
            param_name = clean_text(dt_tag.get_text(separator=" "))
            param_desc = clean_text(dd_tag.get_text(separator=" "))

            if param_name and param_desc:
                combined = f"{param_name} : {param_desc}"
                parameter_texts.append(combined)

    param_passages, position = build_passages(
        doc_name, "parameters", parameter_texts, position
    )
    all_passages.extend(param_passages)

    # ── SECTION 4: EXAMPLES ─────────────────────────────────────────────────
    example_texts = []
    highlight_divs = soup.find_all("div", class_="highlight")

    for div in highlight_divs:
        code_text = div.get_text(separator="\n")
        example_texts.append(code_text)

    example_passages, position = build_passages(
        doc_name, "examples", example_texts[:3], position
    )
    all_passages.extend(example_passages)

    return all_passages


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(limit=0):
    """Main entry point - runs the full scraping pipeline."""

    print("=" * 55)
    print("  Pandas API Documentation Scraper")
    print("  CAID Internship - Day 1")
    print("=" * 55)

    os.makedirs("data", exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # STEP 1: Fetch index page
    print(f"\nStep 1: Fetching reference index...")
    print(f"  URL: {INDEX_URL}")

    index_html = fetch_page(INDEX_URL, use_cache=False)
    if not index_html:
        print("  ERROR: Could not fetch the index page.")
        sys.exit(1)

    print("  Index page downloaded successfully.")

    # STEP 2: Collect links
    print(f"\nStep 2: Collecting API page links...")
    api_links = get_all_api_links(index_html)
    print(f"  Found {len(api_links)} API pages in the index.")

    if limit > 0:
        api_links = api_links[:limit]
        print(f"  Limit applied: will process {limit} pages only.")

    # STEP 3: Download and parse pages
    print(f"\nStep 3: Scraping and parsing pages...")

    all_passages = []
    pages_skipped = 0

    for index, url in enumerate(api_links, start=1):
        doc_name = url_to_doc_name(url)

        if index <= 5 or index % 10 == 0:
            print(f"  [{index:4d} / {len(api_links):4d}]  {doc_name}")

        html = fetch_page(url, use_cache=True)
        if not html:
            pages_skipped += 1
            continue

        page_passages = parse_api_page(html, doc_name)
        all_passages.extend(page_passages)

    # STEP 4: Write output
    print(f"\nStep 4: Writing passages to {OUT_FILE}...")

    with open(OUT_FILE, "w", encoding="utf-8") as output_file:
        for passage in all_passages:
            output_file.write(json.dumps(passage, ensure_ascii=False) + "\n")

    print(f"  Written: {len(all_passages)} passages")

    # STEP 5: Print summary
    section_counts = {}
    for passage in all_passages:
        section = passage["section"]
        section_counts[section] = section_counts.get(section, 0) + 1

    text_lengths = [len(p["text"]) for p in all_passages]
    avg_length = sum(text_lengths) // len(text_lengths) if text_lengths else 0

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Pages attempted   : {len(api_links)}")
    print(f"  Pages skipped     : {pages_skipped}")
    print(f"  Pages successful  : {len(api_links) - pages_skipped}")
    print(f"  Total passages    : {len(all_passages)}")
    print(f"\n  Passages by section:")
    for section, count in sorted(section_counts.items()):
        bar = "#" * (count // 15)
        print(f"    {section:<15}  {count:>5}   {bar}")
    if text_lengths:
        print(f"\n  Text length stats:")
        print(f"    Average : {avg_length} characters")
        print(f"    Minimum : {min(text_lengths)} characters")
        print(f"    Maximum : {max(text_lengths)} characters")
    print(f"\n  Output saved to : {OUT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape pandas API docs and produce passages.jsonl"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many pages. 0 means no limit."
    )
    args = parser.parse_args()
    main(limit=args.limit)
