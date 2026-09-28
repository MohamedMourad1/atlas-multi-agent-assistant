"""
State schemas for the Atlas multi-agent state machine.
Defines the strictly typed global state passed across Planner, Researcher, Writer, and Critic.
"""

from typing import List, Dict, Optional, Literal, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from atlas.schemas.evidence import EvidenceStore
from atlas.schemas.report import DraftReport
from atlas.schemas.critic import CriticVerdict, CritiqueIssue


class SubQuestion(BaseModel):
    """An atomic sub-question generated during the planning phase."""
    
    id: str = Field(description="Unique ID, e.g. SQ1, SQ2, SQ3")
    question: str = Field(description="The specific research sub-question")
    rationale: str = Field(description="Why this sub-question is critical to solving the main question")
    search_queries: List[str] = Field(default_factory=list, description="Targeted keyword search queries")
    preferred_source: Literal["knowledge_base", "web_search", "both"] = Field(
        default="both", description="Recommended retrieval mode for this sub-question"
    )
    status: Literal["pending", "researching", "completed", "insufficient_evidence"] = Field(
        default="pending", description="Current research status"
    )


class Plan(BaseModel):
    """The structured research decomposition produced by the Planner Agent."""
    
    original_question: str = Field(description="The user's original research prompt")
    domain: str = Field(description="Identified domain / industry, e.g. 'EV Technology & Supply Chain'")
    sub_questions: List[SubQuestion] = Field(default_factory=list, description="3 to 6 structured sub-questions")
    estimated_scope: str = Field(default="In-depth analyst briefing")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))


class AgentStepLog(BaseModel):
    """An observable trace log entry from any agent execution."""
    
    step_number: int = Field(description="Sequential log step index")
    agent_name: str = Field(description="Agent identifier: Planner, Researcher, Writer, Critic, System")
    action: str = Field(description="Short action summary, e.g. 'Executed web search for battery mineral prices'")
    thought: Optional[str] = Field(default=None, description="Internal reasoning or observation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata, tool outputs, or issue lists")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))


class AtlasState(BaseModel):
    """The complete, observable state of an Atlas research run."""
    
    # Input
    research_question: str = Field(description="The primary user research question")
    
    # Process Artifacts
    plan: Optional[Plan] = Field(default=None, description="Decomposition plan")
    evidence_store: EvidenceStore = Field(default_factory=EvidenceStore, description="Indexed evidence repository")
    current_draft: Optional[DraftReport] = Field(default=None, description="Active working draft")
    draft_history: List[DraftReport] = Field(default_factory=list, description="All draft revisions")
    critic_verdicts: List[CriticVerdict] = Field(default_factory=list, description="History of critic evaluations")
    
    # Control Flow & State Machine Flags
    iteration_count: int = Field(default=0, description="Current repair loop iteration")
    max_retries: int = Field(default=3, description="Hard cap on repair iterations to prevent infinite loops")
    current_stage: str = Field(default="idle", description="Current stage: planning, researching, drafting, critiquing, completed, failed")
    status: Literal["planning", "researching", "drafting", "critiquing", "repairing", "completed", "insufficient_evidence", "failed"] = Field(
        default="planning"
    )
    
    # Observability & Trace
    step_logs: List[AgentStepLog] = Field(default_factory=list, description="Timeline of agent actions for live UI tracing")
    final_report_markdown: Optional[str] = Field(default=None, description="Final approved markdown report with citations")
    error: Optional[str] = Field(default=None, description="Error message if run failed")
    total_tokens_used: int = Field(default=0, description="Estimated total token consumption")
    execution_time_seconds: float = Field(default=0.0, description="Total wall-clock duration in seconds")

    def add_log(self, agent_name: str, action: str, thought: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Append an observable trace step."""
        log = AgentStepLog(
            step_number=len(self.step_logs) + 1,
            agent_name=agent_name,
            action=action,
            thought=thought,
            details=details or {},
        )
        self.step_logs.append(log)
        return log
