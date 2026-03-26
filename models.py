"""
Models Module — Defines immutable data structures using Pydantic.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class Memory(BaseModel):
    """Represents a single memory unit."""
    model_config = ConfigDict(frozen=True)

    id: Optional[int] = None
    source: str = ""
    raw_text: str
    summary: str
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    importance: float = 0.5
    created_at: str
    consolidated: bool = False
    sector: str = "semantic"
    valid_to: Optional[str] = None
    composite_score: Optional[float] = None
    recall_reason: Optional[List[str]] = None

class Consolidation(BaseModel):
    """Represents a consolidation of multiple memories."""
    model_config = ConfigDict(frozen=True)

    id: Optional[int] = None
    source_ids: List[int]
    summary: str
    insight: str
    created_at: str

class DocumentChunk(BaseModel):
    """Represents a chunk of a document for vector search."""
    model_config = ConfigDict(frozen=True)

    id: Optional[int] = None
    path: str
    content_hash: str
    chunk_text: str
    chunk_index: int = 0
    updated_at: str

class SearchResult(BaseModel):
    """Represents a single search result from vector search."""
    model_config = ConfigDict(frozen=True)

    path: str
    snippet: str
    distance: float
    chunk_index: int
