"""
Utils Module — Common helper functions for embeddings, retries, and file processing.
"""

import asyncio
import hashlib
import logging
import struct
from pathlib import Path
from typing import List, Optional, Any, Callable, Dict

import numpy as np

from turboquant import get_turboquant

from config import (
    BINARY_EXTENSIONS, RATE_LIMIT, EMBEDDING_MODEL
)

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

log = logging.getLogger("memory-agent.utils")

# Rate limiter for embedding calls
_embed_semaphore = asyncio.Semaphore(4)
_EMBED_DELAY = 1.0 / max(RATE_LIMIT, 4)

def is_binary_file(file_path: Path) -> bool:
    """Quick heuristic: check if a file is binary by reading its first 8KB."""
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        chunk = file_path.read_bytes()[:8192]
        if not chunk:
            return False
        # If more than 10% of bytes are non-text, it's binary
        non_text = sum(1 for b in chunk if b < 8 or (14 <= b < 32 and b != 27))
        return (non_text / len(chunk)) > 0.10
    except Exception:
        return True

def serialize_f32(vector: List[float]) -> bytes:
    """Serialize a list of floats into compact binary format for sqlite-vec."""
    return struct.pack("%sf" % len(vector), *vector)

def serialize_int8(vector: List[float]) -> bytes:
    """
    TurboQuant-enhanced scalar quantization: rotation → int8.
    Uses a persistent random orthogonal rotation to improve quantization fidelity.
    """
    tq = get_turboquant(dim=len(vector))
    return tq.quantize_to_int8(vector)

async def retry_with_backoff(
    coro_fn: Callable, 
    *args: Any, 
    max_retries: int = 5, 
    base_delay: float = 3.0, 
    shutdown_event: Optional[asyncio.Event] = None,
    **kwargs: Any
) -> Any:
    """Retry an async function with exponential backoff on 429 and 503 errors."""
    last_error = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            is_retryable = any(k in error_str for k in ["429", "503", "unavailable", "quota", "resource_exhausted", "high demand"])
            
            if not is_retryable or attempt == max_retries:
                log.error(f"❌ Operation failed after {attempt} attempts: {e}")
                raise e
            
            actual_delay = delay * (2.0 if "503" in error_str or "unavailable" in error_str else 1.0)
            log.warning(f"⚠️ Attempt {attempt+1} failed: {e}. Retrying in {actual_delay:.1f}s...")
            
            if shutdown_event:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=actual_delay)
                    if shutdown_event.is_set():
                        return None
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(actual_delay)
            
            delay *= 2
    raise last_error

_EMBED_CHUNK_SIZE = 4000  # Safe ceiling below gemini-embedding-2-preview's ~2048 token limit

# Process-scoped embedding cache. Keyed on a 20-char SHA-256 prefix of the text.
# Eliminates redundant API calls for repeated queries within a session.
# Eviction: capped at _EMBED_CACHE_MAX entries (oldest silently dropped).
_embed_cache: Dict[str, List[float]] = {}
_EMBED_CACHE_MAX = 512

async def _embed_single(text: str, shutdown_event: Optional[asyncio.Event] = None) -> Optional[List[float]]:
    """Embed a single text segment (must be <= _EMBED_CHUNK_SIZE chars)."""
    async def _do_embed():
        client = genai.Client()
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return result.embeddings[0].values

    async with _embed_semaphore:
        embedding = await retry_with_backoff(_do_embed, shutdown_event=shutdown_event)
        await asyncio.sleep(_EMBED_DELAY)
        return embedding

async def embed_text(text: str, shutdown_event: Optional[asyncio.Event] = None) -> Optional[List[float]]:
    """Generate an embedding for the given text using google-genai (rate-limited).

    For text longer than _EMBED_CHUNK_SIZE characters, splits into chunks and
    returns the element-wise mean of all chunk embeddings so the full content
    is represented without risking INVALID_ARGUMENT errors from the API.

    Results are cached in-process (up to _EMBED_CACHE_MAX entries) to avoid
    redundant API calls for repeated query strings within a session.
    """
    if not HAS_GENAI:
        return None
    if not text or not text.strip():
        return None
    if shutdown_event and shutdown_event.is_set():
        return None

    cache_key = hashlib.sha256(text.encode()).hexdigest()[:20]
    if cache_key in _embed_cache:
        log.debug(f"Embed cache hit (key={cache_key[:8]})")
        return _embed_cache[cache_key]

    try:
        if len(text) <= _EMBED_CHUNK_SIZE:
            result = await _embed_single(text, shutdown_event=shutdown_event)
        else:
            # Long text: mean-pool chunk embeddings
            chunks = chunk_text(text, max_chars=_EMBED_CHUNK_SIZE)
            embeddings = []
            for chunk in chunks:
                if shutdown_event and shutdown_event.is_set():
                    return None
                emb = await _embed_single(chunk, shutdown_event=shutdown_event)
                if emb:
                    embeddings.append(emb)

            if not embeddings:
                return None

            arr = np.array(embeddings, dtype=np.float32)
            result = arr.mean(axis=0).tolist()

        if result is not None and len(_embed_cache) < _EMBED_CACHE_MAX:
            _embed_cache[cache_key] = result

        return result

    except Exception as e:
        log.error(f"Embedding error: {e}")
        return None

async def embed_texts_batch(
    texts: List[str],
    shutdown_event: Optional[asyncio.Event] = None,
) -> List[Optional[List[float]]]:
    """Batch-embed multiple short texts in a single Gemini API call.

    Texts exceeding _EMBED_CHUNK_SIZE are routed through embed_text() individually
    (they need internal sub-chunking). Short texts are batched into one request,
    reducing per-file indexing from N round-trips to 1.
    """
    if not HAS_GENAI or not texts:
        return [None] * len(texts)
    if shutdown_event and shutdown_event.is_set():
        return [None] * len(texts)

    short_texts: List[str] = []
    short_indices: List[int] = []
    results: List[Optional[List[float]]] = [None] * len(texts)

    for i, t in enumerate(texts):
        if not t or not t.strip():
            continue
        cache_key = hashlib.sha256(t.encode()).hexdigest()[:20]
        if cache_key in _embed_cache:
            results[i] = _embed_cache[cache_key]
        elif len(t) <= _EMBED_CHUNK_SIZE:
            short_texts.append(t)
            short_indices.append(i)
        else:
            # Long text: fall back to individual embed_text (handles sub-chunking)
            results[i] = await embed_text(t, shutdown_event=shutdown_event)

    if short_texts:
        try:
            async with _embed_semaphore:
                client = genai.Client()
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=short_texts,
                )
                await asyncio.sleep(_EMBED_DELAY)

            for list_pos, original_idx in enumerate(short_indices):
                emb = response.embeddings[list_pos].values
                results[original_idx] = emb
                cache_key = hashlib.sha256(short_texts[list_pos].encode()).hexdigest()[:20]
                if len(_embed_cache) < _EMBED_CACHE_MAX:
                    _embed_cache[cache_key] = emb
        except Exception as e:
            log.error(f"Batch embedding error: {e}. Falling back to individual calls.")
            for list_pos, original_idx in enumerate(short_indices):
                results[original_idx] = await embed_text(short_texts[list_pos], shutdown_event=shutdown_event)

    return results


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Split text into chunks, trying to break on newlines."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = current_chunk + "\n" + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_code_structural(text: str, file_ext: str, max_chars: int = 6000) -> List[str]:
    """Chunk source code at function/class boundaries instead of fixed character counts.

    Produces semantically coherent chunks where each unit contains a complete
    function or class. Enforces a hard max_chars cap: chunks exceeding it are
    sub-split via chunk_text() to avoid embedding model token limit overflows.

    Falls back to chunk_text(text, 1500) if boundary detection yields nothing
    (e.g., a data file, an unrecognised language, or a file with only top-level
    statements and no function/class definitions).
    """
    import ast as _ast
    import re as _re

    ext = file_ext.lower()
    boundary_lines = _find_code_boundaries(text, ext)

    if not boundary_lines:
        return chunk_text(text, max_chars=1500)

    lines = text.splitlines(keepends=True)
    boundaries = sorted(set([0] + boundary_lines + [len(lines)]))

    chunks: List[str] = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = "".join(lines[start:end])
        if not chunk.strip():
            continue
        if len(chunk) > max_chars:
            # Hard cap: sub-split oversized chunks (e.g. 500-line god-functions)
            chunks.extend(chunk_text(chunk, max_chars=max_chars))
        else:
            chunks.append(chunk)

    return chunks if chunks else chunk_text(text, max_chars=1500)


def _find_code_boundaries(text: str, ext: str) -> List[int]:
    """Return 0-indexed line numbers where a new top-level definition starts."""
    import ast as _ast
    import re as _re

    if ext == ".py":
        try:
            tree = _ast.parse(text)
        except SyntaxError:
            return []
        return sorted({
            node.lineno - 1  # ast is 1-indexed; convert to 0-indexed
            for node in _ast.walk(tree)
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
            and node.col_offset == 0  # top-level only
        })

    _BOUNDARY_PATTERNS: dict = {
        frozenset([".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"]):
            _re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+|^(?:export\s+)?(?:default\s+)?class\s+\w+", _re.MULTILINE),
        frozenset([".go"]):
            _re.compile(r"^func\s+|^type\s+\w+\s+(?:struct|interface)", _re.MULTILINE),
        frozenset([".rs"]):
            _re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+\w+|^(?:pub\s+)?(?:struct|enum)\s+\w+", _re.MULTILINE),
    }

    for exts, pattern in _BOUNDARY_PATTERNS.items():
        if ext in exts:
            return sorted({text[:m.start()].count("\n") for m in pattern.finditer(text)})

    return []
