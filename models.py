"""
Models Module — Defines immutable data structures using Pydantic.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from uuid import uuid4

class MemCube(BaseModel):
    """
    A standardized, portable memory unit inspired by the Always-On-Memory paradigm.
    
    Sectors:
        - semantic: Facts, rules, permanent conventions, system architecture
        - episodic: Events, specific task outcomes, error logs, event history
        - procedural: "How-to" guides, specific workflows, testing patterns
        - reflection: Reasoning paths, dead-ends, lessons learned from failures
    
    Reference: https://github.com/A4ABATTERY/Always-On-Memory
    """
    model_config = ConfigDict(frozen=True)

    id: Optional[int] = None
    cube_id: str = Field(default_factory=lambda: str(uuid4()))
    sector: str = "semantic"
    source: str = ""
    origin_platform: str = "aom-local"
    raw_text: str
    summary: str
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic metadata: valid_from, valid_to, custom tags"
    )
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    created_at: str = ""
    consolidated: bool = False
    valid_to: Optional[str] = None
    composite_score: Optional[float] = None
    recall_reason: Optional[List[str]] = None

    def export_portable(self) -> Dict[str, Any]:
        """Export cube in a portable format for cross-platform migration."""
        return self.model_dump(exclude={"id", "composite_score", "recall_reason"})

class EvalResult(BaseModel):
    """Structured output from the Adversarial Evaluator agent."""
    model_config = ConfigDict(frozen=True)
    score: float = Field(..., ge=0.0, le=1.0)
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Did the synthesis preserve all source facts?")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Were any source facts omitted?")
    redundancy_removed: float = Field(..., ge=0.0, le=1.0, description="Were duplicates properly merged?")
    feedback: str = Field("", description="Specific feedback for the Generator to improve")

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
