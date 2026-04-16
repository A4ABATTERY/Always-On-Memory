# Agent Orchestration

AOM v3 is powered by **PydanticAI**, a framework that enables type-safe, structured interactions with Large Language Models. This document details the roles, responsibilities, and interactions of the specialized agents within the system.

## 🤖 The Agent Fleet

### 1. Ingest Agent
- **Role**: Entry point for all new data.
- **Responsibility**: Analyzes files/content, extracts metadata (entities, topics), and formats them into a `MemCube`.
- **Structured Output**: Returns a validated `MemCube` object.
- **System Prompt Focus**: Extraction accuracy and objective summarization.

### 2. Generator Agent (Consolidation)
- **Role**: The "Dreamer".
- **Responsibility**: Takes a cluster of related memories and synthesizes them into a single, high-density **Insight Cube**. It looks for patterns, contradictions, and overarching themes.
- **Structured Output**: Returns a `SynthesisResult` (summary, refined topics, and a list of source memory IDs).
- **System Prompt Focus**: Synthesis, pattern recognition, and semantic compression.

### 3. Evaluator Agent (Consolidation)
- **Role**: The "Critic".
- **Responsibility**: Performs a rigorous "Adversarial Audit" of the Generator's output. It compares the synthesis against the raw source memories to check for hallucinations or omitted facts.
- **Structured Output**: Returns an `EvalResult` (score 0.0-1.0 and detailed feedback).
- **System Prompt Focus**: Fact-checking, skepticism, and fidelity enforcement.

### 4. Sync Auditor (Self-Healing)
- **Role**: The "Maintenance Worker".
- **Responsibility**: Triggered when the Librarian detects semantic drift in linked code files. It analyzes the new code and decides if the memory link is still valid or needs repair.
- **Structured Output**: Returns an `AuditResult` (status: ACTIVE, REPAIR, or HISTORICAL).
- **System Prompt Focus**: Code analysis and structural integrity.

### 5. Query Agent
- **Role**: The "Librarian".
- **Responsibility**: Decides which memories and code chunks are relevant to a user query. It generates the search parameters and synthesizes the final answer using retrieved context.
- **System Prompt Focus**: Grounding, relevance, and helpfulness.

### 6. Self-Improvement Agent (EvoSkill)
- **Role**: The "Strategist".
- **Responsibility**: Analyzes reflection and episodic memories to discover recurring project patterns or successful strategies.
- **Outcome**: Autonomously writes or updates structured `SKILL.md` files in the `.agents/skills` directory, evolving the system's capabilities over time.
- **System Prompt Focus**: Pattern discovery, capability expansion, and strategic reasoning.

## 🛠️ PydanticAI Integration

### Structured Results (`result_type`)
AOM extensively uses the `result_type` parameter in PydanticAI. This ensures that agents return validated Python objects instead of raw strings or fragile JSON.

```python
# Example: The Evaluator's Structured Result
class EvalResult(BaseModel):
    score: float = Field(description="Fidelity score (0.0 to 1.0)")
    feedback: str = Field(description="Critique or reasons for the score")
    hallucinations: List[str] = Field(default_factory=list)
```

### Self-Correction Loop
If an agent's output fails Pydantic validation, PydanticAI automatically provides the validation error back to the LLM for a retry, ensuring high reliability in the background loops.

### Dependency Injection
AOM uses PydanticAI's `deps` to provide agents with access to the `MemoryStore`, `Librarian`, and system configuration without tight global coupling.
