"""
Researcher Agent for Atlas.
Executes a ReAct loop across Knowledge Base RAG and Live Web Search to gather structured evidence.
"""

from typing import List, Dict, Any, Optional
from atlas.schemas.state import AtlasState, SubQuestion
from atlas.schemas.evidence import EvidenceItem
from atlas.tools.registry import tool_registry
from atlas.llm.client import default_llm_client
from atlas.config import settings


class ResearcherAgent:
    """Specialized agent for evidence retrieval and factual ground truthing."""

    def __init__(self, llm_client=None):
        self.llm = llm_client or default_llm_client

    def research(self, state: AtlasState, target_subquestions: Optional[List[str]] = None) -> AtlasState:
        """Run evidence gathering across planned sub-questions."""
        state.current_stage = "researching"
        state.status = "researching"

        if not state.plan or not state.plan.sub_questions:
            state.add_log("Researcher", "No active plan found. Skipping research phase.", None, {"error": "Missing plan"})
            return state

        # Determine which sub-questions to research
        sub_questions_to_process = state.plan.sub_questions
        if target_subquestions:
            sub_questions_to_process = [sq for sq in state.plan.sub_questions if sq.id in target_subquestions]

        state.add_log(
            agent_name="Researcher",
            action=f"Commencing evidence gathering across {len(sub_questions_to_process)} sub-questions...",
            thought="Executing ReAct loops over internal Knowledge Base and Live Web Search...",
        )

        for sq in sub_questions_to_process:
            sq.status = "researching"
            self._research_single_subquestion(state, sq)
            sq.status = "completed"

        state.add_log(
            agent_name="Researcher",
            action=f"Completed evidence gathering. Total structured evidence items collected: {len(state.evidence_store.items)}",
            details={"evidence_ids": list(state.evidence_store.items.keys())},
        )

        return state

    def _research_single_subquestion(self, state: AtlasState, sq: SubQuestion):
        """Execute ReAct cycle for an individual sub-question."""
        state.add_log(
            agent_name="Researcher",
            action=f"Investigating [{sq.id}]: {sq.question}",
            thought=f"Preferred source channel: {sq.preferred_source}. Initiating tool queries...",
        )

        # 1. Query Knowledge Base RAG if preferred or 'both'
        if sq.preferred_source in ["knowledge_base", "both"] and settings.enable_rag_kb:
            for q in sq.search_queries[:2]:
                rag_res = tool_registry.execute_tool("query_knowledge_base", query=q, top_k=settings.rag_top_k)
                if rag_res.get("status") == "success":
                    for item in rag_res.get("results", []):
                        ev = state.evidence_store.add_evidence(
                            sub_question_id=sq.id,
                            query_used=q,
                            snippet=item["snippet"],
                            source_title=item["title"],
                            source_url=item.get("source_path"),
                            source_type="knowledge_base",
                            relevance_score=item.get("score", 0.9),
                        )
                        state.add_log(
                            agent_name="Researcher",
                            action=f"Retained [{ev.evidence_id}] from Knowledge Base: {item['title']}",
                            details={"evidence_id": ev.evidence_id, "snippet": ev.snippet[:150]},
                        )

        # 2. Query Live Web Search if preferred or 'both'
        if sq.preferred_source in ["web_search", "both"] and settings.enable_web_search:
            for q in sq.search_queries[:2]:
                web_res = tool_registry.execute_tool("web_search", query=q, max_results=settings.web_search_max_results)
                if web_res.get("status") == "success":
                    for item in web_res.get("results", []):
                        ev = state.evidence_store.add_evidence(
                            sub_question_id=sq.id,
                            query_used=q,
                            snippet=item["snippet"],
                            source_title=item["title"],
                            source_url=item.get("source_url"),
                            source_type="web_search",
                            relevance_score=item.get("score", 0.85),
                        )
                        state.add_log(
                            agent_name="Researcher",
                            action=f"Retained [{ev.evidence_id}] from Web Search: {item['title']}",
                            details={"evidence_id": ev.evidence_id, "snippet": ev.snippet[:150]},
                        )

    def targeted_research_for_critic_issues(self, state: AtlasState, target_queries: List[str]) -> AtlasState:
        """Execute targeted repair research for specific factual gaps identified by the Critic."""
        state.add_log(
            agent_name="Researcher",
            action=f"Running targeted repair research for {len(target_queries)} specific Critic feedback gaps...",
            thought="Closing evidence deficiencies to satisfy verification criteria.",
        )

        for query in target_queries:
            web_res = tool_registry.execute_tool("web_search", query=query, max_results=2)
            if web_res.get("status") == "success":
                for item in web_res.get("results", []):
                    ev = state.evidence_store.add_evidence(
                        sub_question_id="REPAIR",
                        query_used=query,
                        snippet=item["snippet"],
                        source_title=item["title"],
                        source_url=item.get("source_url"),
                        source_type="web_search",
                        relevance_score=0.95,
                    )
                    state.add_log(
                        agent_name="Researcher",
                        action=f"Acquired repair evidence [{ev.evidence_id}]: {item['title']}",
                        details={"evidence_id": ev.evidence_id},
                    )

        return state
