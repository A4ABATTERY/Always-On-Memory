"""
Agents Factory Module — Builds PydanticAI agents for various tasks.
"""

from typing import Tuple
from pydantic_ai import Agent

from config import MODEL, SMART_MODEL, RATE_LIMIT, SKILLS_DIR
from memory_store import (
    store_memory, read_unconsolidated_memories, store_consolidation,
    update_memory_validity, reinforce_memory, read_all_memories,
    read_consolidation_history, read_memory_partition
)

# ─── Rate-Limited Model Helper ────────────────────────────────

try:
    from pydantic_ai import ConcurrencyLimitedModel
except ImportError:
    ConcurrencyLimitedModel = None

def make_model(model_str: str):
    """Wrap a model string with ConcurrencyLimitedModel if available."""
    if ConcurrencyLimitedModel and RATE_LIMIT > 0:
        return ConcurrencyLimitedModel(model_str, limiter=RATE_LIMIT)
    return model_str

lite_model = make_model(MODEL)
smart_model = make_model(SMART_MODEL)

# ─── System Prompts ───────────────────────────────────────────

INGEST_SYSTEM_PROMPT = (
    "You are a Memory Ingest Agent. You handle ALL types of input — text, images, "
    "audio, video, and PDFs. For any input you receive:\n"
    "1. Thoroughly describe what the content contains\n"
    "2. Create a concise 1-2 sentence summary\n"
    "3. Extract key entities (people, companies, products, concepts, objects, locations)\n"
    "4. Assign 2-4 topic tags\n"
    "5. Rate importance from 0.0 to 1.0\n"
    "6. Identify the memory sector (semantic, episodic, procedural, or reflection)\n"
    "7. If the fact has an expiration, provide valid_to (ISO datetime string)\n"
    "8. Call store_memory with all extracted information\n\n"
    "For images: describe the scene, objects, text, people, and any visual details.\n"
    "For audio/video: describe the spoken content, sounds, scenes, and key moments.\n"
    "For PDFs: extract and summarize the document content.\n\n"
    "Use the full description as raw_text in store_memory so the context is preserved.\n"
    "Always call store_memory. Be Detailed and accurate.\n"
    "After storing, confirm what was stored in one sentence."
)

CONSOLIDATE_SYSTEM_PROMPT = (
    "You are a Memory Consolidation Agent. You:\n"
    "1. Call read_unconsolidated_memories to see what needs processing\n"
    "2. If fewer than 2 memories, say nothing to consolidate\n"
    "3. Find connections and patterns across the memories.\n"
    "4. Create synthesized summaries and insights. PERFORM MULTIPLE store_consolidation "
    "calls if memories cover disparate themes (e.g. don't mix UI and Database).\n"
    "5. Call store_consolidation with source_ids, summary, insight, and connections\n"
    "6. Contradictions: Call update_memory_validity for old memories if a newer memory contradicts it.\n"
    "7. Reinforcement: Call reinforce_memory for old memories supported by new info.\n\n"
    "Connections: list of dicts with 'from_id', 'to_id', 'relationship' keys.\n"
    "Prioritize thematic clustering over broad summarization."
)

QUERY_SYSTEM_PROMPT = (
    "You are a Memory Query Agent. When asked a question:\n"
    "1. Call read_all_memories to access the memory store\n"
    "2. Call read_consolidation_history for higher-level insights\n"
    "3. Call search_documents to find relevant source code files or documents\n"
    "4. Synthesize an answer based ONLY on stored memories\n"
    "5. Reference memory IDs: [Memory 1], [Memory 2], etc.\n"
    "6. Include a 'Relevant Files' section listing file paths from search_documents results\n"
    "7. If no relevant memories exist, say so honestly\n\n"
    "Be thorough. Always cite sources."
)

DEEP_CONSOLIDATE_SYSTEM_PROMPT = (
    "You are a Deep Memory Consolidation Agent. You are the HIGH-FIDELITY CORRECTIVE layer.\n"
    "Your job is to catch details the 'lite' agent missed or over-summarized.\n"
    "1. Call read_all_memories to see the full context\n"
    "2. Call search_documents and read_document to verify facts against source code/docs\n"
    "3. Directly link insights to relevant file paths in your summary/insight text\n"
    "4. Look for deep architectural patterns, contradictions, and themes\n"
    "5. Reinforce still-relevant memories (reinforce_memory)\n"
    "6. Mark outdated facts as invalid (update_memory_validity)\n"
    "7. PERFORM MULTIPLE store_consolidation calls for different high-level themes.\n\n"
    "Be precise and analytical. Link facts to files."
)

SELF_IMPROVEMENT_SYSTEM_PROMPT = (
    "You are a Self-Improvement Agent. Your goal is to evolve the project's capabilities "
    "by discovering and refining skills based on past performance and failures.\n\n"
    "1. Call read_memory_partition for 'reflection' and 'episodic' sectors.\n"
    "2. Identify recurring failure patterns (EvoSkill thresholds: ≥3 errors, ≥2 hallucinations).\n"
    "3. Identify successful complex workflows that should be codified.\n"
    "4. Use Anthropic's Skill-Creator patterns to write new SKILL.md files:\n"
    "   - Name: Short, descriptive ID.\n"
    "   - Description: When to trigger (MAKE IT PUSHY to avoid undertriggering).\n"
    "   - Instructions: Imperative, clear steps, examples of input/output.\n"
    "5. Use EvoSkill taxonomy: Procedural guide, Scoped constraint, Correction reference, Meta-strategy, Style guide.\n"
    "6. Call write_skill_file to persist the new or updated skill.\n"
    "7. Call search_documents to see if a similar skill already exists before creating a new one.\n\n"
    "Be proactive. If you see a way to make the Orchestrator more reliable, codify it as a skill."
)

# ─── Factory Function ─────────────────────────────────────────

def build_agents() -> Tuple[Agent, Agent, Agent, Agent, Agent]:
    """Build PydanticAI agents for ingest, consolidate, query, and self-improvement."""
    
    # These tools will be provided by MemoryAgent or passed directly
    # Note: search_documents and read_document are defined later in the process
    # To avoid circular imports, we'll assume they will be injected or imported carefully.
    
    # For now, let's keep the tool imports here but note that search_documents
    # might need to be imported from 'librarian' or similar.
    # In agent.py, they were all in one file.
    
    from librarian import search_documents, read_document, write_skill_file # Assume these exist or will be merged

    ingest_agent = Agent(
        lite_model,
        system_prompt=INGEST_SYSTEM_PROMPT,
        tools=[store_memory],
    )

    consolidate_agent = Agent(
        lite_model,
        system_prompt=CONSOLIDATE_SYSTEM_PROMPT,
        tools=[read_unconsolidated_memories, store_consolidation, update_memory_validity, reinforce_memory],
    )

    query_agent = Agent(
        lite_model,
        system_prompt=QUERY_SYSTEM_PROMPT,
        tools=[read_all_memories, read_consolidation_history, search_documents],
    )

    deep_consolidate_agent = Agent(
        smart_model,
        system_prompt=DEEP_CONSOLIDATE_SYSTEM_PROMPT,
        tools=[
            read_all_memories, read_consolidation_history, 
            update_memory_validity, reinforce_memory,
            search_documents, read_document, store_consolidation
        ],
    )

    self_improvement_agent = Agent(
        smart_model,
        system_prompt=SELF_IMPROVEMENT_SYSTEM_PROMPT,
        tools=[
            read_memory_partition, search_documents, 
            read_document, write_skill_file
        ],
    )

    return ingest_agent, consolidate_agent, query_agent, deep_consolidate_agent, self_improvement_agent
