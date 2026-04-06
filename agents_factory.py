"""
Agents Factory Module — Builds PydanticAI agents for various tasks.
"""

from typing import Tuple
from pydantic_ai import Agent

from config import MODEL, SMART_MODEL, RATE_LIMIT
from memory_store import (
    store_memory, read_unconsolidated_memories, 
    read_all_memories, read_consolidation_history, read_memory_partition
)
from models import SynthesisResult, AuditResult, EvalResult

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

# Models are constructed inside build_agents() so that importing this module
# never requires a live API key — model resolution is deferred until runtime.

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

GENERATOR_SYSTEM_PROMPT = (
    "You are a Memory Synthesis Generator. Given a set of raw memories:\n"
    "1. Identify thematic clusters among the memories\n"
    "2. For each cluster, create a single synthesized summary that preserves ALL facts\n"
    "3. Resolve contradictions by keeping the most recent fact (highest created_at) and noting the contradiction\n"
    "4. Maintain entity and topic specificity — do not over-generalize\n"
    "5. STRUCTURAL LINKAGE: Use search_documents to identify the 1-2 most relevant file paths for this cluster.\n"
    "6. If feedback from a previous attempt is provided, address every point specifically\n\n"
    "CRITICAL: Do not drop facts. Every entity, date, and specific detail from the source "
    "memories must appear in your synthesis. Prefer verbatim preservation over paraphrasing.\n"
    "DO NOT call any storage tools yourself."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are a Memory Synthesis Evaluator. You receive source memories and a draft synthesis.\n"
    "Your job is to RIGOROUSLY grade the synthesis. Be skeptical. Be strict.\n\n"
    "Grade on three dimensions (0.0 to 1.0 each):\n"
    "- FIDELITY: Does the synthesis accurately reflect the source facts? Any hallucinated details?\n"
    "- COMPLETENESS: Were ANY facts from the source memories omitted or glossed over?\n"
    "- REDUNDANCY_REMOVED: Were duplicate/overlapping facts properly merged?\n\n"
    "Overall score = min(fidelity, completeness) * 0.7 + redundancy_removed * 0.3\n\n"
    "The 'feedback' field must contain SPECIFIC, ACTIONABLE criticism. "
    "Do NOT say 'good job' — find something to improve."
)

QUERY_SYSTEM_PROMPT = (
    "You are a Memory Query Agent. When asked a question:\n"
    "1. Call read_all_memories to access the memory store\n"
    "2. Call read_consolidation_history for higher-level insights\n"
    "3. Call search_documents ONLY as a fallback if no valid 'file_link' exists in relevant memories\n"
    "4. Prioritize context from 'file_link' paths found in structural connections\n"
    "5. Synthesize an answer based ONLY on stored memories and linked or searched files\n"
    "6. Reference memory IDs: [Memory 1], [Memory 2], etc.\n"
    "7. Include a 'Relevant Files' section listing file paths from structural links or fallback search\n\n"
    "Be thorough. Always cite sources."
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

SYNC_AUDITOR_SYSTEM_PROMPT = (
    "You are a Memory-Code Link Integrity Auditor. Your task is to rigorously evaluate "
    "if a specific Memory Insight is still an accurate and grounded description of a given Code Snippet.\n\n"
    "States:\n"
    "- ACTIVE: The Memory Insight's core assertions, business logic, or architectural rules "
    "are still present and accurately implemented in the Code Snippet. Minor refactors, "
    "whitespace changes, or variable renaming should NOT trigger a status change.\n"
    "- REPAIR: The core logic or intent is still present, but the implementation details "
    "have diverged enough that the Memory Insight's technical details are now slightly "
    "inaccurate. The link should remain active, but the memory needs an update.\n"
    "- HISTORICAL: The core logic described in the Memory Insight has been significantly "
    "changed, replaced (e.g., switched algorithm), or removed. The memory no longer "
    "describes the current state, but is a valid historical trace.\n\n"
    "Decision Logic:\n"
    "1. Does the Code Snippet still contain the functionality described in the Insight?\n"
    "2. If refactored, is the architectural 'spirit' or 'intent' of the memory still honored?\n"
    "3. Has the implementation diverged to a different pattern or behavior?\n\n"
    "Be precise and skeptical. Your decision ensures grounding integrity."
)

# ─── Factory Function ─────────────────────────────────────────

def build_agents() -> Tuple[Agent, Agent, Agent, Agent, Agent, Agent, Agent, Agent]:
    """Build PydanticAI agents."""
    from librarian import search_documents, read_document, write_skill_file

    lite_model = make_model(MODEL)
    smart_model = make_model(SMART_MODEL)

    ingest_agent = Agent(
        lite_model,
        system_prompt=INGEST_SYSTEM_PROMPT,
        tools=[store_memory],
    )

    # Periodic consolidation (lite)
    memory_generator_lite = Agent(
        lite_model,
        output_type=SynthesisResult,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        tools=[read_unconsolidated_memories, search_documents, read_document],
    )
    memory_evaluator_lite = Agent(
        lite_model,
        output_type=EvalResult,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        tools=[],
    )

    # Deep/Dream consolidation (smart)
    memory_generator_smart = Agent(
        smart_model,
        output_type=SynthesisResult,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        tools=[
            read_all_memories, search_documents, read_document
        ],
    )
    memory_evaluator_smart = Agent(
        smart_model,
        output_type=EvalResult,
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        tools=[],
    )

    query_agent = Agent(
        lite_model,
        system_prompt=QUERY_SYSTEM_PROMPT,
        tools=[read_all_memories, read_consolidation_history, search_documents],
    )

    self_improvement_agent = Agent(
        smart_model,
        system_prompt=SELF_IMPROVEMENT_SYSTEM_PROMPT,
        tools=[
            read_memory_partition, search_documents, 
            read_document, write_skill_file
        ],
    )

    sync_agent = Agent(
        lite_model,
        output_type=AuditResult,
        system_prompt=SYNC_AUDITOR_SYSTEM_PROMPT,
        tools=[], # Only analysis
    )

    return (
        ingest_agent, 
        memory_generator_lite, memory_evaluator_lite, 
        memory_generator_smart, memory_evaluator_smart, 
        query_agent, self_improvement_agent,
        sync_agent
    )

