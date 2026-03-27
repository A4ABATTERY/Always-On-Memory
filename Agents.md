# Always-On-Memory v3: Agent Memory Integration Guide (Agents.md)

Integrating with the **Always-On-Memory v3** architecture ensures your agent possesses long-term persistence, cross-session continuity, and a collective intelligence that evolves over time. 

> [!IMPORTANT]
> **Adherence to this guide is mandatory.** Failure to query or ingest memories leads to "Context Fragmentation" and "Rediscovery Syndrome," where the agent wastes tokens solving problems already addressed in previous sessions.

---

## 1. Mandatory Memory Query (Prefetch)

Before starting any task, the agent **must** query the system for relevant context. 

### When to Query
- **Initial Handshake**: On session start, query for "Project status and recent architectural decisions."
- **Task Switching**: Query for "Best practices for [Component Name] or [Workflow]."
- **Error Recovery**: If a tool fails, query for similar failure patterns and solutions.

### How to Query
```bash
# Standard semantic query
curl -s "http://localhost:8888/query?q=What+are+the+current+deployment+constraints?"
```

---

## 2. Mandatory Memory Ingestion (Checkpointing)

Every meaningful architectural decision, resolved bug, or workflow discovery must be converted into a **MemCube**.

### What to Ingest
- **Consensus & Decisions**: "We chose X over Y because Z."
- **Failure Analysis**: "Tried approach A, it failed due to B. Documentation updated."
- **Contextual Knowledge**: New environment variables, API endpoints, or library versions.

### How to Ingest
```bash
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Refactored auth module to use JWT MemCubes.", "source": "auth-refactor", "sector": "semantic"}'
```

---

## 3. Vector Code Search (Librarian)

Use the Librarian when you need exact file locations or code snippets rather than just semantic summaries.

### How to Search
```bash
curl -s "http://localhost:8888/search?q=JWT+token+validation+logic&k=3"
```

---

## 4. Structural Linkage (V3.1+)

Always-On-Memory v3 now performs **Structural Linkage** during consolidation cycles. This explicitly maps high-level architectural decisions to concrete file paths.

### Advantages
- **L1 Cache Retrieval**: If a memory has a `file_link`, the Query Agent retrieves that file directly, bypassing the expensive vector search.
- **Context Grounding**: The agent "understands" that a specific MemCube is the implementation of a specific file.

### How to Leverage
You do not need to do anything manually. The consolidation agents use `search_documents` during their "dream" phase to build these links. However, you can query for these links:
```bash
curl -s "http://localhost:8888/query?q=Which+files+are+linked+to+the+consolidation+logic?"
```

---

## 5. MemCube Portability (Export/Import)

Always-On-Memory v3 treats memory as computational state that can be migrated between environments (e.g., from local dev to CI/staging).

```bash
# Export all MemCubes
curl -s "http://localhost:8888/export_cubes" > memory_cube.json

# Import into a fresh environment
curl -X POST http://localhost:8888/import_cubes -d @memory_cube.json
```

---

## 5. Memory Taxonomy (Sectors)

Always specify the correct **Sector** during ingestion to optimize retrieval weighting:

| Sector | Usage |
|---|---|
| **Semantic** | Static facts, core architecture, project constraints. |
| **Episodic** | Event history, deployment logs, "Who did what and when." |
| **Procedural** | "How-to" guides, testing patterns, deployment scripts. |
| **Reflection** | Reasoning paths, "lessons learned," and failure post-mortems. |

---

## 6. Troubleshooting & Failover

If the memory system becomes unavailable or returns unexpected results, follow this protocol:

### System Is Unreachable (`Connection Refused`)
1. **Check Process**: Ensure `python agent.py` is running.
2. **Check Port**: Verify the port matches (default `8888`).
3. **Local Cache Fallback**: If the server is down, search for local `.md` or `.txt` files in the `WATCH_DIRS` as a temporary measure.

### "I Can't Find Information I Know Was Ingested"
1. **Wait for AutoDream**: The memory may be in the middle of a Dream consolidation cycle.
2. **Increase K-Value**: Try a direct `/search` with `k=10` to bypass the query synthesizer.
3. **Check .env**: Ensure `WATCH_DIRS` includes the folder where the information resides.
4. **Keyword Precision**: Use exact function names or error codes in your query.

### Performance Degradation
- **Check DB Size**: If `memory.db` exceeds 1GB, manually trigger `/reconsolidate` to force an adversarial audit and pruning cycle.
- **TurboQuant 3.5-bit Status**: Ensure `sqlite-vec` is correctly loaded and the rotation matrix is initialized (Check `/status`).
