"""
Writer Agent for Atlas.
Synthesizes structured evidence into an executive-ready research report strictly citing evidence IDs [E#].
"""

from typing import Optional, List
from atlas.schemas.state import AtlasState
from atlas.schemas.report import DraftReport, ReportSection
from atlas.schemas.critic import CriticVerdict
from atlas.llm.client import default_llm_client


class WriterAgent:
    """Specialized agent for evidence synthesis and cited report drafting."""

    SYSTEM_PROMPT = """You are the Senior Intelligence Writer for Atlas, an autonomous multi-agent research assistant.
Your job is to synthesize raw evidence into a structured, highly coherent, executive-level research report.

CRITICAL CITATION RULES:
1. Every factual statement, statistic, projection, or policy claim MUST cite an Evidence ID in brackets, e.g. [E1], [E2].
2. Do NOT invent citation IDs that are not present in the provided evidence.
3. Every section MUST map to the planned sub-questions and provide rigorous analytical narrative.
4. Output MUST conform strictly to the DraftReport Pydantic schema.
"""

    def __init__(self, llm_client=None):
        self.llm = llm_client or default_llm_client

    def write(self, state: AtlasState) -> AtlasState:
        """Generate or revise a draft report based on available evidence and critic feedback."""
        state.current_stage = "drafting"
        state.status = "drafting"
        
        iteration = state.iteration_count + 1
        
        state.add_log(
            agent_name="Writer",
            action=f"Drafting comprehensive report (Revision iteration {iteration})...",
            thought="Synthesizing discrete evidence items into a unified, cited narrative...",
        )

        evidence_context = state.evidence_store.to_context_prompt()
        
        # Build prompt with Plan, Evidence, and any prior Critic feedback
        subq_list = "\n".join([f"- [{sq.id}] {sq.question}" for sq in state.plan.sub_questions]) if state.plan else "N/A"
        
        critic_guidance = ""
        if state.critic_verdicts:
            last_verdict: CriticVerdict = state.critic_verdicts[-1]
            critic_guidance = f"""
PREVIOUS CRITIC FEEDBACK TO FIX:
- Summary: {last_verdict.feedback_summary}
- Issues to Address:
""" + "\n".join([f"  * [{issue.category}] {issue.problem_statement} -> {issue.suggested_action}" for issue in last_verdict.issues])

        prompt = f"""Write a complete, structured research report answering:
"{state.research_question}"

PLANNED SUB-QUESTIONS TO ADDRESS:
{subq_list}

{critic_guidance}

AVAILABLE VERIFIED EVIDENCE ITEMS (YOU MUST CITE THESE USING [E1], [E2], ETC.):
{evidence_context}

Draft a complete, authoritative report with:
1. A clear Title.
2. An Executive Summary (with inline [E#] citations).
3. Thematic Sections corresponding to each sub-question with rich content and inline [E#] citations.
4. Strategic Key Takeaways bullet list.
"""

        draft: DraftReport = self.llm.generate_structured(
            prompt=prompt,
            schema_class=DraftReport,
            system_instruction=self.SYSTEM_PROMPT,
            temperature=0.2,
        )

        draft.iteration = iteration
        
        # Compile markdown report with auto-generated references section
        bibliography_md = state.evidence_store.to_bibliography_markdown()
        draft.compile_markdown(bibliography_md)

        state.current_draft = draft
        state.draft_history.append(draft)

        state.add_log(
            agent_name="Writer",
            action=f"Completed draft '{draft.title}' with {len(draft.sections)} sections.",
            details={
                "iteration": iteration,
                "sections": [s.heading for s in draft.sections],
                "total_length_chars": len(draft.raw_markdown),
            },
        )

        return state
