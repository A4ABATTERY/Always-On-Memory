"""
Models Module — Defines immutable data structures using Pydantic.
"""

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

class TopicSynthesis(BaseModel):
    """
    An atomic, topically-bound insight synthesized from a semantic sub-cluster
    of source memories. One TopicSynthesis is created per identified theme.
    """
    topic_name: str = Field(
        ...,
        description="Short, descriptive category name (e.g., 'Backend Auth Infrastructure', 'Frontend RBAC', 'E2E Test Suite Status')"
    )
    summary: str = Field(
        ...,
        description="Concise 1-2 sentence high-level summary of this specific topic only."
    )
    insight: str = Field(
        ...,
        description="Deep, full-fidelity synthesized insight for this topic. ALL technical details, entity names, and decisions from the source memories MUST be preserved."
    )
    source_ids: List[int] = Field(
        ...,
        description="The integer memory IDs that were used to construct THIS topic's insight. Each ID should appear in exactly one TopicSynthesis."
    )
    connections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structural links (file_link, memory_link) relevant to this specific topic."
    )


class MultiSynthesisResult(BaseModel):
    """
    The top-level output from the Generator Agent. Encapsulates all topics
    identified in a consolidation batch as a list of independent TopicSynthesis objects.
    No two TopicSynthesis entries should cover substantially the same theme.
    """
    insights: List[TopicSynthesis] = Field(
        ...,
        description="The distinct, topically-isolated insights generated from the source memories."
    )

class AuditResult(BaseModel):
    """Structured output from the Memory-Code Link Integrity Auditor agent."""
    model_config = ConfigDict(frozen=True)
    status: str = Field(..., description="ACTIVE | HISTORICAL | REPAIR")
    reason: str = Field(..., description="A brief technical explanation of the decision")
    suggested_update: Optional[str] = Field(None, description="Corrected version of the insight if status is REPAIR")

class EvalResult(BaseModel):
    """
    Updated grading schema for the Adversarial Evaluator agent.
    'completeness' has been REPLACED by 'source_coverage' (objective, enforceable)
    and 'topic_cohesion' has been ADDED to penalize poor cluster boundaries.
    """
    model_config = ConfigDict(frozen=True)
    score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score.")
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Did the synthesis accurately preserve source facts without hallucination?")
    source_coverage: float = Field(..., ge=0.0, le=1.0, description="Were all source memory IDs utilized in at least one TopicSynthesis?")
    topic_cohesion: float = Field(..., ge=0.0, le=1.0, description="Are the generated topics sufficiently distinct with sharp thematic boundaries?")
    redundancy_removed: float = Field(..., ge=0.0, le=1.0, description="Were overlapping facts within a topic merged efficiently?")
    feedback: str = Field("", description="Actionable feedback for the Generator specifying orphaned memories or overlapping topics.")

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
