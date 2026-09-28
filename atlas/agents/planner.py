"""
Planner Agent for Atlas.
Decomposes an open-ended research question into 3-6 structured, mutually exclusive, collectively exhaustive sub-questions.
"""

from typing import Optional
from atlas.schemas.state import Plan, SubQuestion, AtlasState
from atlas.llm.client import default_llm_client
from atlas.config import settings


class PlannerAgent:
    """Specialized agent for research question decomposition."""

    SYSTEM_PROMPT = """You are the Lead Planning Agent for Atlas, an autonomous research assistant.
Your mission is to take an open-ended research question and decompose it into 3 to 6 logical, non-overlapping sub-questions that collectively answer the user's primary query with analytical depth.

Rules:
1. Break down the question systematically:
   - Upstream / foundational aspects (e.g. supply, technology, raw materials).
   - Core dynamics / regulatory / geopolitical / market factors.
   - Downstream impacts / mitigation strategies / future outlook.
2. For each sub-question, provide 2-3 focused keyword search queries.
3. Recommend whether to use internal knowledge base, web search, or both.
4. Output MUST conform strictly to the Plan Pydantic schema.
"""

    def __init__(self, llm_client=None):
        self.llm = llm_client or default_llm_client

    def plan(self, state: AtlasState) -> AtlasState:
        """Decompose the research question and update state with the structured Plan."""
        state.current_stage = "planning"
        state.status = "planning"
        
        state.add_log(
            agent_name="Planner",
            action=f"Decomposing research question: '{state.research_question}'",
            thought="Analyzing query scope and identifying analytical sub-dimensions...",
        )

        prompt = f"""Decompose the following research question into a comprehensive research plan:

Research Question: "{state.research_question}"

Generate between {settings.min_subquestions} and {settings.max_subquestions} high-value sub-questions.
"""

        plan_result = self.llm.generate_structured(
            prompt=prompt,
            schema_class=Plan,
            system_instruction=self.SYSTEM_PROMPT,
            temperature=0.2,
        )
        
        # Ensure original question is preserved
        plan_result.original_question = state.research_question
        
        # Ensure subquestion IDs are standardized (SQ1, SQ2, ...)
        for idx, sq in enumerate(plan_result.sub_questions, 1):
            sq.id = f"SQ{idx}"
            sq.status = "pending"

        state.plan = plan_result
        
        state.add_log(
            agent_name="Planner",
            action=f"Generated {len(plan_result.sub_questions)} sub-questions for domain: {plan_result.domain}",
            details={"sub_questions": [f"[{sq.id}] {sq.question}" for sq in plan_result.sub_questions]},
        )
        
        return state
