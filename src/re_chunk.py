#!/usr/bin/env python3
"""
re_chunk.py
===========
Takes the existing passages.jsonl and re-splits any passages that
are longer than the new max_chars threshold. Also merges very short
passages within the same doc_name+section.

This lets us test different chunk sizes without re-scraping.

Usage:
    python src/re_chunk.py --max-chars 400 --min-chars 60 \
        --output data/passages_400.jsonl
"""

import argparse
import json
import re
import os


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind(". ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        else:
            cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def main(input_path, output_path, max_chars, min_chars):
    print(f"Re-chunking with max_chars={max_chars}, min_chars={min_chars}")
    
    # Load original passages
    original = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(json.loads(line))
    print(f"  Loaded {len(original):,} original passages")
    
    # Group by (doc_name, section) so we can merge/split within groups
    from collections import defaultdict
    groups = defaultdict(list)
    for p in original:
        key = (p['doc_name'], p['section'])
        groups[key].append(p)
    
    # Sort each group by position
    for key in groups:
        groups[key].sort(key=lambda x: x['position'])
    
    # Rebuild passages
    new_passages = []
    for (doc_name, section), items in groups.items():
        # Combine all text from this group
        combined_text = ' '.join(clean_text(p['text']) for p in items)
        
        # Split into chunks at new size
        chunks = split_long(combined_text, max_chars)
        
        # Filter min size and assign positions
        pos = 0
        for chunk in chunks:
            if len(chunk) >= min_chars:
                new_passages.append({
                    'doc_name': doc_name,
                    'section': section,
                    'position': pos,
                    'text': chunk
                })
                pos += 1
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in new_passages:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Stats
    lengths = [len(p['text']) for p in new_passages]
    print(f"  Wrote {len(new_passages):,} new passages")
    print(f"  Length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/passages.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--min-chars", type=int, default=60)
    args = parser.parse_args()
    
    main(args.input, args.output, args.max_chars, args.min_chars)
