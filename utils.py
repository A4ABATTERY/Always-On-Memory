"""
Utils Module — Common helper functions for embeddings, retries, and file processing.
"""

import asyncio
import logging
import struct
from pathlib import Path
from typing import List, Optional, Any, Callable

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
    """
    if not HAS_GENAI:
        return None
    if not text or not text.strip():
        return None
    if shutdown_event and shutdown_event.is_set():
        return None

    try:
        if len(text) <= _EMBED_CHUNK_SIZE:
            return await _embed_single(text, shutdown_event=shutdown_event)

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

        # Element-wise mean across all chunk embeddings
        arr = np.array(embeddings, dtype=np.float32)
        mean_emb = arr.mean(axis=0).tolist()
        return mean_emb

    except Exception as e:
        log.error(f"Embedding error: {e}")
        return None

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
