"""
Critic schemas for Atlas verification and fact-checking.
Enables programmatic, claim-by-claim verification of drafts against evidence.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

CriticAction = Literal["approve", "research_more", "rewrite"]
IssueSeverity = Literal["low", "medium", "critical"]
IssueCategory = Literal[
    "unsupported_claim",     # Draft makes a claim with an invalid or misattributed citation ID
    "missing_coverage",      # A key sub-question from the plan was omitted or inadequately answered
    "missing_citation",      # Factual statistics/claims given without citing an evidence ID [E#]
    "contradiction",         # Claim directly contradicts the cited evidence
    "structure_clarity",     # Report has formatting flaws, poor transitions, or broken markdown
    "hallucination"          # Fabricated fact with no basis in any evidence
]


class CritiqueIssue(BaseModel):
    """Specific flaw detected in the draft by the Critic agent."""
    
    issue_id: str = Field(description="Unique ID for this issue, e.g. ISS-1")
    category: IssueCategory = Field(description="Classification of the critique issue")
    severity: IssueSeverity = Field(default="medium", description="Importance: critical, medium, or low")
    target_section_or_paragraph: Optional[str] = Field(default=None, description="Section heading or paragraph location")
    target_sub_question_id: Optional[str] = Field(default=None, description="Related sub-question ID if missing coverage")
    problem_statement: str = Field(description="Clear explanation of why this claim/section is problematic")
    cited_evidence_id: Optional[str] = Field(default=None, description="The cited [E#] if claim is unsupported or contradictory")
    suggested_action: str = Field(description="Actionable instruction for the Researcher or Writer to fix this issue")


class CriticVerdict(BaseModel):
    """The structured verdict produced by the Critic agent evaluating a draft report."""
    
    approved: bool = Field(description="True if report meets all quality & grounding thresholds, False otherwise")
    overall_score: float = Field(default=0.0, description="Composite quality score from 0.0 to 1.0")
    faithfulness_score: float = Field(default=0.0, description="Percentage of claims factually grounded in cited evidence (0.0 - 1.0)")
    citation_precision_score: float = Field(default=0.0, description="Percentage of citation IDs that accurately point to valid supporting evidence")
    coverage_score: float = Field(default=0.0, description="Percentage of planned sub-questions adequately answered (0.0 - 1.0)")
    action: CriticAction = Field(description="'approve' to finalize, 'research_more' for missing facts, 'rewrite' for structure/grounding fixes")
    issues: List[CritiqueIssue] = Field(default_factory=list, description="List of specific issues found")
    feedback_summary: str = Field(description="Executive summary of the critique for routing and logs")
    iteration: int = Field(default=1, description="Which revision cycle this verdict belongs to")

    def to_readable_summary(self) -> str:
        """Format verdict for terminal logs and UI displays."""
        status = "✅ APPROVED" if self.approved else f"⚠️ REJECTED -> ROUTE TO: {self.action.upper()}"
        lines = [
            f"=== Critic Verdict (Iteration {self.iteration}) ===",
            f"Status: {status}",
            f"Scores -> Overall: {self.overall_score:.2f} | Faithfulness: {self.faithfulness_score:.2f} | Citations: {self.citation_precision_score:.2f} | Coverage: {self.coverage_score:.2f}",
            f"Summary: {self.feedback_summary}",
        ]
        if self.issues:
            lines.append(f"Issues Detected ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  - [{issue.severity.upper()}] {issue.category}: {issue.problem_statement} -> {issue.suggested_action}")
        return "\n".join(lines)
