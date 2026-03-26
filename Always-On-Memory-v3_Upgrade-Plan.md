***

# Always-On-Memory v3 Architecture Upgrade

## Overview
The goal of this upgrade is to transform AOM from a simple vector-backed CRUD application into a **self-optimizing, portable Memory Operating System**. 

**Core Upgrades:**
1.  **Always-On-Memory Architecture:** Transitioning flat memory rows into standardized, portable "MemCubes."
2.  **Adversarial Consolidation:** Implementing a Generator-Evaluator agent harness to strictly audit memory synthesis and prevent context degradation.
3.  **AutoDream Cycles:** Replacing standard decay with an active "sleep" phase that prunes and reorganizes data.
4.  **TurboQuant-Inspired Compression:** Applying scalar quantization to embedding vectors before SQLite storage to drastically reduce database bloat in Librarian Mode.

---

## Phase 1: The Always-On-Memory Paradigm (Data Structure Refactoring)

### Concept & End-Goal
Current AI memory is siloed. We will adopt the **Always-On-Memory** paradigm, treating memory as a core computational resource that can be scheduled, shared, and migrated across different AI tools and platforms. We will implement "MemCubes"—standardized memory units encapsulating text, parameters, and metadata.
*Source: [VentureBeat: Chinese researchers unveil Always-On-Memory](https://venturebeat.com/ai/chinese-researchers-unveil-memos-the-first-memory-operating-system-that-gives-ai-human-like-recall)*

### Implementation Details
* **Refactoring `models.py`:** Deprecate the existing `Memory` class. Introduce a `MemCube` Pydantic model. 
* **Refactoring `database.py`:** Update the SQLite schema to support MemCubes, adding JSON columns for dynamic metadata and cross-platform routing tags.
* **Feature Addition:** Create a new API endpoint `/export_cubes` and `/import_cubes` in `server.py` to allow memory states to be migrated between different physical servers or agent environments.

### Logic Flow & Pseudo-Code (`models.py`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MemCube(BaseModel):
    cube_id: str
    sector: str = Field(..., description="Semantic, Episodic, Procedural, or Reflection")
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stores valid_from, valid_to, and origin_platform")
    importance_score: float
    access_count: int = 0
    last_accessed: datetime

    def export_portable_format(self) -> str:
        # Strips environment-specific IDs for cross-platform migration
        export_data = self.model_dump(exclude={"cube_id", "access_count"})
        return json.dumps(export_data)
```

---

## Phase 2: Adversarial Consolidation (The Anthropic Harness)

### Concept & End-Goal
Currently, a single agent synthesizes and consolidates memories. LLMs are notoriously poor at grading their own work and often hallucinate or lose critical context during synthesis. We will implement a multi-agent harness inspired by Anthropic's long-running application design: a **Generator Agent** to synthesize the memories, and an independent **Evaluator Agent** to rigorously grade the output against the original source facts. 
*Source: [Anthropic: Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)*

### Implementation Details
* **Refactoring `agents_factory.py`:** Split the `ConsolidationAgent` into `MemoryGenerator` and `MemoryEvaluator`.
* **Execution Logic:** The Evaluator will use a strict grading rubric (Fidelity, Redundancy, Completeness). If the Evaluator scores the generated synthesis below a threshold (e.g., 0.8/1.0), it returns feedback to the Generator to retry.

### Logic Flow & Pseudo-Code (`agents_factory.py` / `agent.py`)
```python
async def adversarial_consolidation(raw_memories: List[MemCube]) -> MemCube:
    approved = False
    attempts = 0
    max_attempts = 3
    current_draft = ""
    feedback = "No feedback yet. Please generate the initial synthesis."

    while not approved and attempts < max_attempts:
        # Generator creates the synthesis
        current_draft = await memory_generator.run(
            prompt=f"Synthesize these memories: {raw_memories}. Feedback from last attempt: {feedback}"
        )
        
        # Evaluator grades it blindly against the source
        evaluation = await memory_evaluator.run(
            prompt=f"Source facts: {raw_memories}. Draft synthesis: {current_draft}. Grade strictly on Fidelity and Completeness. Output JSON with 'score' and 'feedback'."
        )
        
        if evaluation.score >= 0.85:
            approved = True
        else:
            feedback = evaluation.feedback
            attempts += 1

    if not approved:
        # Fallback: keep raw memories if synthesis fails quality check
        raise ConsolidationQualityError("Failed to achieve passing grade after max attempts.")
        
    return build_memcube_from_synthesis(current_draft)
```

---

## Phase 3: The "AutoDream" Cycle

### Concept & End-Goal
Current decay logic simply deletes memories that fall below a threshold. We will replace the 24h Deep Re-Consolidation loop with an **AutoDream** sub-agent. This agent activates only during periods of low API activity (mimicking sleep). It actively reorganizes data structures, aggressively prunes redundancies, and groups related `MemCubes` to ensure the system wakes up to a highly streamlined, zero-bloat state.
*Source: [Geeky Gadgets: Claude AutoDream Memory Files](https://www.geeky-gadgets.com/claude-autodream-memory-files/)*

### Implementation Details
* **Refactoring `agent.py`:** Remove the standard `decay_loop`. Implement an `autodream_loop`.
* **Activity Monitoring:** The loop checks the `access_count` and `last_accessed` timestamps. If no new ingestions or queries have occurred in the last X minutes, the dream cycle begins. 

### Logic Flow & Pseudo-Code (`agent.py`)
```python
import asyncio

async def autodream_loop():
    while True:
        await asyncio.sleep(300) # Check every 5 minutes
        
        if system_is_idle(idle_threshold_minutes=30):
            print("Entering AutoDream sequence...")
            
            # Step 1: Pruning
            await execute_redundancy_pruning()
            
            # Step 2: Reorganization (Clustering related MemCubes)
            clusters = await cluster_memcubes_by_embedding()
            for cluster in clusters:
                if len(cluster) > 5:
                    # Trigger the Adversarial Consolidation to compress the cluster
                    await adversarial_consolidation(cluster)
            
            print("AutoDream complete. Memory state optimized.")
```

---

## Phase 4: Vector Compression (TurboQuant Principles)

### Concept & End-Goal
Librarian Mode currently embeds large volumes of source code into `sqlite-vec` using `gemini-embedding-2-preview` (which outputs high-dimensional float32 arrays). As the watched directories grow, memory and indexing time will bottleneck. We will implement a data-oblivious quantization step (inspired by Google's TurboQuant) to compress these vectors by up to 6x before storing them in the database, vastly speeding up retrieval without dataset-specific training.
*Source: [MarkTechPost: Google Introduces TurboQuant](https://www.marktechpost.com/2026/03/25/google-introduces-turboquant-a-new-compression-algorithm-that-reduces-llm-key-value-cache-memory-by-6x-and-delivers-up-to-8x-speedup-all-with-zero-accuracy-loss/?amp)*

### Implementation Details
* **Refactoring `librarian.py` & `utils.py`:** Create a vector preprocessing function. Since we are operating in Python and SQLite, we will implement a 1D scalar quantization (e.g., converting float32 embeddings into int8 representations) before inserting them into `sqlite-vec`. `sqlite-vec` natively supports 8-bit integer vectors, which perfectly aligns with this goal.

### Logic Flow & Pseudo-Code (`utils.py`)
```python
import numpy as np

def turboquant_inspired_compress(embedding: List[float]) -> bytes:
    """
    Applies a simplified scalar quantization to convert float32 to int8.
    Reduces memory footprint by 4x for sqlite-vec storage.
    """
    arr = np.array(embedding, dtype=np.float32)
    
    # Normalize to -1.0 to 1.0 (assuming cosine similarity use-case)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
        
    # Quantize to int8 (-128 to 127)
    quantized = np.round(arr * 127.0).astype(np.int8)
    
    # sqlite-vec expects raw bytes for int8 vectors
    return quantized.tobytes()
```
*Note: The `database.py` schema for `vec0` must be updated to explicitly define the column as `int8[N]` where N is the embedding dimension.*

***
