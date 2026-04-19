"""
Chunker Module — Pre-ingestion structural document splitting.

Splits large documents into semantically coherent chunks before they reach the
LLM, preventing the truncation and monolithic-blob ingestion problems that cause
information loss (e.g. a 25.6 KB document being compressed to 7 MemCubes).

Strategy (in order of preference):
  1. Markdown: split on ## / ### headings — each heading + body = one chunk
  2. JSON: valid JSON with top-level dict keys — each key-value pair = one chunk
  3. Plain text: line-boundary splitting via utils.chunk_text()

Size constraints (env-configurable via CHUNK_MAX_CHARS / CHUNK_MIN_CHARS):
  - Chunks smaller than CHUNK_MIN_CHARS are merged with the next chunk.
  - Chunks larger than CHUNK_MAX_CHARS are sub-split.

Return format: list of dicts — {"text": str, "chunk_index": int, "section_title": str}
A single-element list means the document was small enough to ingest as-is.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS
from utils import chunk_text


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _merge_small_chunks(chunks: List[dict], min_chars: int) -> List[dict]:
    """Merge chunks below min_chars into the following chunk to avoid micro-MemCubes.

    Forward pass: tiny chunks merge into their successor.
    Backward pass: a tiny final chunk (no successor) merges into its predecessor.
    """
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        # If this chunk is too small and a next chunk exists, merge forward
        if len(chunk["text"]) < min_chars and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            merged_text = chunk["text"].rstrip() + "\n\n" + next_chunk["text"].lstrip()
            # Prefer the current chunk's section title; fall back to next
            title = chunk["section_title"] or next_chunk["section_title"]
            # Overwrite the next slot so it gets evaluated again (handles chains)
            chunks[i + 1] = {
                "text": merged_text,
                "chunk_index": chunk["chunk_index"],
                "section_title": title,
            }
            i += 1
            continue
        merged.append(chunk)
        i += 1

    # Backward pass: if the final chunk is still too small, merge it into its predecessor
    if len(merged) >= 2 and len(merged[-1]["text"]) < min_chars:
        tail = merged.pop()
        merged[-1]["text"] = merged[-1]["text"].rstrip() + "\n\n" + tail["text"].lstrip()

    # Re-sequence indices after merging
    for idx, ch in enumerate(merged):
        ch["chunk_index"] = idx
    return merged


def _split_large_chunks(chunks: List[dict], max_chars: int) -> List[dict]:
    """Sub-split chunks that exceed max_chars using plain-text line splitting."""
    result = []
    idx = 0
    for chunk in chunks:
        if len(chunk["text"]) <= max_chars:
            chunk["chunk_index"] = idx
            result.append(chunk)
            idx += 1
        else:
            sub_texts = chunk_text(chunk["text"], max_chars=max_chars)
            for sub_pos, sub_text in enumerate(sub_texts):
                title = chunk["section_title"]
                if title and len(sub_texts) > 1:
                    title = f"{title} (part {sub_pos + 1})"
                result.append({
                    "text": sub_text,
                    "chunk_index": idx,
                    "section_title": title,
                })
                idx += 1
    return result


# ─── Chunking Strategies ──────────────────────────────────────────────────────

def _chunk_markdown(text: str) -> Optional[List[dict]]:
    """Split on ## or ### headings at the start of a line.

    Returns None if no qualifying headings are found (so the caller can
    fall through to the next strategy).
    """
    pattern = re.compile(r'^(#{1,3} .+)$', re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in pattern.finditer(text)]

    if not positions:
        return None

    chunks: List[dict] = []

    # Preamble: any content that precedes the first heading
    if positions[0][0] > 0:
        preamble = text[:positions[0][0]].strip()
        if preamble:
            chunks.append({
                "text": preamble,
                "chunk_index": 0,
                "section_title": "preamble",
            })

    for i, (start, heading_line) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append({
                "text": body,
                "chunk_index": len(chunks),
                "section_title": heading_line.lstrip("# ").strip(),
            })

    return chunks if chunks else None


def _chunk_json(text: str) -> Optional[List[dict]]:
    """Split a JSON object by top-level key.

    Returns None if the text is not valid JSON, not a dict, or has only one key
    (no gain from splitting a single-key object).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict) or len(data) < 2:
        return None

    chunks = []
    for idx, (key, value) in enumerate(data.items()):
        sub_text = json.dumps({key: value}, indent=2, ensure_ascii=False)
        chunks.append({
            "text": sub_text,
            "chunk_index": idx,
            "section_title": str(key),
        })

    return chunks if chunks else None


def _chunk_plaintext(text: str) -> List[dict]:
    """Line-boundary plain-text fallback using utils.chunk_text()."""
    sub_texts = chunk_text(text, max_chars=CHUNK_MAX_CHARS)
    return [
        {"text": t, "chunk_index": i, "section_title": ""}
        for i, t in enumerate(sub_texts)
    ]


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_document(text: str, source_path: str) -> List[dict]:
    """Split a document into semantically coherent chunks for per-chunk LLM ingestion.

    Args:
        text: Full document text.
        source_path: File name or relative path — used only for extension detection.

    Returns:
        A list of chunk dicts, each with keys:
          - "text": str          — chunk content
          - "chunk_index": int   — zero-based position in the original document
          - "section_title": str — heading / JSON key / "preamble" / "" for plain

    Always returns at least one element. If the document is short enough to fit
    in a single chunk, a one-element list is returned.
    """
    ext = Path(source_path).suffix.lower()

    chunks: Optional[List[dict]] = None

    # Strategy 1: Markdown heading split (tried for all text types)
    # Many .txt and .log files also use ## headings, so we always try this first.
    chunks = _chunk_markdown(text)

    # Strategy 2: JSON top-level key split (only for .json files)
    if chunks is None and ext == ".json":
        chunks = _chunk_json(text)

    # Strategy 3: Plain text line-boundary split
    if chunks is None:
        chunks = _chunk_plaintext(text)

    # Post-processing: enforce min/max size constraints
    chunks = _merge_small_chunks(chunks, CHUNK_MIN_CHARS)
    chunks = _split_large_chunks(chunks, CHUNK_MAX_CHARS)

    # Defensive: guarantee at least one chunk is always returned
    if not chunks:
        return [{"text": text, "chunk_index": 0, "section_title": ""}]

    return chunks
