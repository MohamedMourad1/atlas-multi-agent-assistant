"""
Critic Agent for Atlas.
Performs programmatic and semantic verification of draft reports against the evidence store.
"""

import re
from typing import List, Dict, Any
from atlas.schemas.state import AtlasState
from atlas.schemas.critic import CriticVerdict, CritiqueIssue, CriticAction
from atlas.llm.client import default_llm_client
from atlas.config import settings


class CriticAgent:
    """Specialized verification and fact-checking agent."""

    SYSTEM_PROMPT = """You are the Lead Verification & Quality Critic for Atlas, an autonomous multi-agent research assistant.
Your job is to rigorously review the draft report against the raw evidence store side by side.

VERIFICATION EVALUATION CRITERIA:
1. Faithfulness: Is every statement grounded in the cited evidence? Are there unsupported or hallucinated claims?
2. Citation Precision: Do all [E#] citations accurately reflect the content of the cited evidence?
3. Coverage: Does the draft adequately address all planned sub-questions?
4. Coherence: Does the report read as one polished, executive document?

ROUTING DECISIONS:
- 'approve': Overall quality and grounding are high (Score >= 0.85), no critical hallucinations.
- 'research_more': Specific sub-questions lack facts, or claims require additional external evidence.
- 'rewrite': Evidence exists in the store, but the draft misattributed citations, missed sections, or had poor structure.

Output MUST conform strictly to the CriticVerdict Pydantic schema.
"""

    def __init__(self, llm_client=None):
        self.llm = llm_client or default_llm_client

    def critique(self, state: AtlasState) -> AtlasState:
        """Evaluate the active draft against the evidence store."""
        state.current_stage = "critiquing"
        state.status = "critiquing"
        
        if not state.current_draft:
            state.add_log("Critic", "No active draft to critique.", None, {"error": "Missing draft"})
            return state

        iteration = state.iteration_count + 1

        state.add_log(
            agent_name="Critic",
            action=f"Commencing rigorous fact-check of Draft Iteration {iteration}...",
            thought="Comparing draft claims against raw evidence store and verifying sub-question coverage...",
        )

        # 1. Programmatic Pre-Check: Scan for invalid citation IDs
        programmatic_issues = self._programmatic_citation_check(state)

        # 2. LLM Semantic & Grounding Verification
        evidence_context = state.evidence_store.to_context_prompt()
        draft_markdown = state.current_draft.raw_markdown
        subq_list = "\n".join([f"- [{sq.id}] {sq.question}" for sq in state.plan.sub_questions]) if state.plan else "N/A"

        prompt = f"""EVALUATE THIS DRAFT REPORT AGAINST THE VERIFIED EVIDENCE:

RESEARCH QUESTION:
"{state.research_question}"

PLANNED SUB-QUESTIONS:
{subq_list}

VERIFIED EVIDENCE SET:
{evidence_context}

DRAFT REPORT TO EVALUATE:
{draft_markdown}

Perform claim-by-claim verification. Output your evaluation as a strict CriticVerdict JSON object.
"""

        verdict: CriticVerdict = self.llm.generate_structured(
            prompt=prompt,
            schema_class=CriticVerdict,
            system_instruction=self.SYSTEM_PROMPT,
            temperature=0.1,
        )

        verdict.iteration = iteration
        
        # Merge programmatic issues if any were detected
        for p_issue in programmatic_issues:
            verdict.issues.append(p_issue)

        # Apply approval threshold logic
        if (
            verdict.faithfulness_score >= settings.critic_approval_threshold
            and verdict.citation_precision_score >= settings.critic_approval_threshold
            and verdict.coverage_score >= 0.75
            and not any(issue.severity == "critical" for issue in verdict.issues)
        ):
            verdict.approved = True
            verdict.action = "approve"
        else:
            verdict.approved = False
            if verdict.coverage_score < 0.70 or any("missing" in i.problem_statement.lower() for i in verdict.issues):
                verdict.action = "research_more"
            else:
                verdict.action = "rewrite"

        state.critic_verdicts.append(verdict)

        state.add_log(
            agent_name="Critic",
            action=f"Verdict rendered: {'APPROVED' if verdict.approved else 'REJECTED (Action: ' + verdict.action + ')'}",
            thought=f"Scores -> Faithfulness: {verdict.faithfulness_score:.2f}, Citations: {verdict.citation_precision_score:.2f}, Coverage: {verdict.coverage_score:.2f}",
            details={
                "approved": verdict.approved,
                "action": verdict.action,
                "issues_count": len(verdict.issues),
                "summary": verdict.feedback_summary,
            },
        )

        return state

    def _programmatic_citation_check(self, state: AtlasState) -> List[CritiqueIssue]:
        """Perform regex scan on the draft markdown to find non-existent citation IDs."""
        issues = []
        valid_ids = set(state.evidence_store.items.keys())
        cited_ids_found = set(re.findall(r"\[(E\d+)\]", state.current_draft.raw_markdown))
        
        invalid_ids = cited_ids_found - valid_ids
        for inv_id in invalid_ids:
            issues.append(
                CritiqueIssue(
                    issue_id=f"PROG-{inv_id}",
                    category="unsupported_claim",
                    severity="critical",
                    problem_statement=f"Draft cites non-existent evidence ID [{inv_id}] not found in the evidence store.",
                    suggested_action=f"Remove or replace [{inv_id}] with a valid verified evidence ID from the store.",
                    cited_evidence_id=inv_id,
                )
            )

        if not cited_ids_found and len(valid_ids) > 0:
            issues.append(
                CritiqueIssue(
                    issue_id="PROG-NO-CITATIONS",
                    category="missing_citation",
                    severity="critical",
                    problem_statement="Draft contains no inline [E#] citation brackets despite evidence being available.",
                    suggested_action="Ensure every paragraph explicitly cites evidence IDs in [E#] format.",
                )
            )

        return issues
