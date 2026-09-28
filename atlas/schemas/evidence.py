"""
Evidence schemas for Atlas Autonomous Research Assistant.
Ensures every retrieved fact is strictly tracked with an Evidence ID and full metadata.
"""

from typing import List, Dict, Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

SourceType = Literal["knowledge_base", "web_search", "academic", "url_scraper", "synthetic_test"]


class EvidenceItem(BaseModel):
    """Represents a discrete, atomic piece of factual evidence retrieved by Researcher."""
    
    evidence_id: str = Field(description="Unique ID for this evidence item, e.g. E1, E2, E3")
    sub_question_id: str = Field(description="The sub-question this evidence attempts to answer, e.g. SQ1")
    query_used: str = Field(description="The exact search query or vector retrieval query used")
    snippet: str = Field(description="The extracted factual text or passage")
    source_title: str = Field(description="Title of the source document or webpage")
    source_url: Optional[str] = Field(default=None, description="Direct URL if web source, or document path if KB")
    source_type: SourceType = Field(default="knowledge_base", description="Channel from which evidence was retrieved")
    published_date: Optional[str] = Field(default=None, description="Publication date if available")
    relevance_score: float = Field(default=1.0, description="Calculated or estimated relevance score (0.0 - 1.0)")
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))

    def to_citation_reference(self) -> str:
        """Format as a markdown bibliography entry."""
        url_part = f" ([Source Link]({self.source_url}))" if self.source_url else ""
        date_part = f", published {self.published_date}" if self.published_date else ""
        return f"**[{self.evidence_id}]** *{self.source_title}* ({self.source_type.replace('_', ' ').title()}{date_part}){url_part}: \"{self.snippet[:250]}...\""


class EvidenceStore(BaseModel):
    """Collection of structured evidence items indexed by ID."""
    
    items: Dict[str, EvidenceItem] = Field(default_factory=dict, description="Map of evidence_id -> EvidenceItem")
    next_counter: int = Field(default=1, description="Counter for auto-incrementing evidence IDs")

    def add_evidence(
        self,
        sub_question_id: str,
        query_used: str,
        snippet: str,
        source_title: str,
        source_url: Optional[str] = None,
        source_type: SourceType = "knowledge_base",
        published_date: Optional[str] = None,
        relevance_score: float = 1.0,
    ) -> EvidenceItem:
        """Add a new evidence item, assigning a unique E# identifier."""
        # Check for near-duplicate snippets to prevent evidence bloat
        for item in self.items.values():
            if item.snippet.strip() == snippet.strip():
                return item

        evidence_id = f"E{self.next_counter}"
        self.next_counter += 1
        
        item = EvidenceItem(
            evidence_id=evidence_id,
            sub_question_id=sub_question_id,
            query_used=query_used,
            snippet=snippet.strip(),
            source_title=source_title.strip(),
            source_url=source_url,
            source_type=source_type,
            published_date=published_date,
            relevance_score=relevance_score,
        )
        self.items[evidence_id] = item
        return item

    def get_by_subquestion(self, sub_question_id: str) -> List[EvidenceItem]:
        """Filter evidence items for a given sub-question."""
        return [item for item in self.items.values() if item.sub_question_id == sub_question_id]

    def to_context_prompt(self) -> str:
        """Generate formatted evidence prompt for the Writer and Critic agents."""
        if not self.items:
            return "No evidence collected yet."
        
        lines = []
        for eid, item in self.items.items():
            lines.append(
                f"[{eid}] (Source: {item.source_title} | Type: {item.source_type} | URL: {item.source_url or 'Local KB'})\n"
                f"Evidence: {item.snippet}\n"
            )
        return "\n".join(lines)

    def to_bibliography_markdown(self) -> str:
        """Generate final report references section."""
        if not self.items:
            return "No references available."
        
        lines = ["### References & Evidence Index\n"]
        for eid in sorted(self.items.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
            item = self.items[eid]
            lines.append(f"- {item.to_citation_reference()}")
        return "\n".join(lines)
