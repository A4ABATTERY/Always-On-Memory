# Agent Memory Integration Guide (Agents.md)

Integrating the **Always-On Memory Agent** into your AI agent's workflow ensures long-term persistence, cross-session continuity, and reduced context drift. This guide outlines the mandatory rules for any agent interacting with this memory system.

---

## 1. Mandatory Memory Query (Prefetch)

Before your agent begins any significant task, it must query the memory system for relevant context. This prevents "Rediscovery Syndrome" and ensures past decisions or errors are respected.

### When to Query
- **Starting a New Task**: Search for past approaches or architectural constraints.
- **Encountering an Error**: Check if this error has been seen and resolved before.
- **Resuming Work**: Retrieve the last known state and pending action items.

### How to Query
```bash
curl -s "http://localhost:8888/query?q=YOUR_QUERY"
```

---

## 2. Mandatory Memory Ingestion

To ensure context survives across sessions, your agent must ingest meaningful outcomes before the session concludes.

### What to Ingest
- **Architecture/Design Decisions**: Logic, rationale, and affected components.
- **Resolved Errors**: Root cause analysis and the fix applied.
- **Task Milestones**: Completed steps and current progress.
- **Lessons Learned**: Approaches that failed and why (Reflection).

### How to Ingest
```bash
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  - d '{"text": "FACT_OR_SUMMARY", "source": "context-identifier", "importance": 0.8}'
```

---

## 3. Searching Documents (Librarian)

If your agent needs to find specific code patterns, function definitions, or documentation within the project files, it can use the vector search endpoint directly.

### When to Search
- **Looking for Symbols**: Find where a specific function or class is defined.
- **Understanding Patterns**: Search for "how we handle error X" to find code examples.
- **Researching Configuration**: Find where specific constants or env vars are used.

### How to Search
```bash
curl -s "http://localhost:8888/search?q=YOUR_CODE_QUERY&k=5"
```
Returns a list of matching file paths and code snippets.

---

## 4. Memory Taxonomy & Sectors

Categorize your memories to improve retrieval precision:

| Sector | Content Type |
|---|---|
| **Semantic** | Facts, rules, permanent conventions, system architecture. |
| **Episodic** | Events, specific task outcomes, error logs, event history. |
| **Procedural** | "How-to" guides, specific workflows, testing patterns. |
| **Reflection** | Reasoning paths, dead-ends, "lessons learned" from failures. |

---

## 5. Integration Best Practices

- **Entity-Specific Queries**: Use specific keywords (filenames, function names, error codes) rather than vague descriptions.
- **Synthesized Ingestion**: Do not ingest raw chat history. Ingest concise, structured summaries focused on "What changed and Why."
- **Importance Scoring**: Set importance levels (0.1–1.0). High-importance memories (0.7+) persist through the decay loop longer.
- **Temporal Truth Windows**: If a previously stored fact changes, ingest the update. The consolidation loop will invalidate the old fact.

---

## 6. Quick Troubleshooting

If your agent is losing context:
1. Verify the memory agent is running (`/status`).
2. Check if the agent is performing a query *at the very beginning* of the session.
3. Ensure ingestion happens *immediately* after a decision or resolution is reached.
