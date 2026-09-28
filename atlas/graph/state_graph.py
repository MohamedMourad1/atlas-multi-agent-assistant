"""
State Graph & Multi-Agent Orchestrator for Atlas.
Executes the directed loop (Planner -> Researcher -> Writer -> Critic -> Repair Loop).
"""

import time
import traceback
from typing import Optional, Callable, Generator
from atlas.schemas.state import AtlasState
from atlas.agents.planner import PlannerAgent
from atlas.agents.researcher import ResearcherAgent
from atlas.agents.writer import WriterAgent
from atlas.agents.critic import CriticAgent
from atlas.graph.router import route_after_critic


class AtlasOrchestrator:
    """Directed state graph orchestrator managing agent transitions, loops, and telemetry."""

    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
        critic: Optional[CriticAgent] = None,
    ):
        self.planner = planner or PlannerAgent()
        self.researcher = researcher or ResearcherAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()

    def run(self, question: str, max_retries: int = 3, on_step: Optional[Callable[[AtlasState], None]] = None) -> AtlasState:
        """Synchronously execute the full Atlas research loop from start to completion."""
        state = AtlasState(research_question=question, max_retries=max_retries)
        start_time = time.time()

        try:
            # 1. Planning Stage
            state = self.planner.plan(state)
            if on_step:
                on_step(state)

            # 2. Initial Evidence Gathering
            state = self.researcher.research(state)
            if on_step:
                on_step(state)

            # 3. Directed Loop (Draft -> Critique -> Conditional Repair)
            while True:
                # Drafting Stage
                state = self.writer.write(state)
                if on_step:
                    on_step(state)

                # Verification Stage
                state = self.critic.critique(state)
                if on_step:
                    on_step(state)

                # Route verdict
                next_step = route_after_critic(state)
                
                if next_step == "finalize":
                    break
                elif next_step == "research_more":
                    state.iteration_count += 1
                    state.status = "repairing"
                    
                    # Extract search queries from critic issues
                    last_verdict = state.critic_verdicts[-1]
                    repair_queries = [
                        issue.problem_statement for issue in last_verdict.issues if issue.category in ["missing_coverage", "unsupported_claim"]
                    ]
                    if not repair_queries:
                        repair_queries = [state.research_question + " latest details"]
                    
                    state = self.researcher.targeted_research_for_critic_issues(state, repair_queries[:2])
                    if on_step:
                        on_step(state)
                elif next_step == "rewrite":
                    state.iteration_count += 1
                    state.status = "repairing"
                    state.add_log(
                        agent_name="System",
                        action=f"Routing back to Writer for Revision (Iteration {state.iteration_count + 1})...",
                        thought="Applying structural, citation, and clarity feedback from Critic.",
                    )
                    if on_step:
                        on_step(state)

            # Finalize Report
            if state.current_draft:
                state.final_report_markdown = state.current_draft.raw_markdown
                state.status = "completed"
                state.current_stage = "completed"
                state.add_log(
                    agent_name="System",
                    action="Report research pipeline completed successfully.",
                    thought="Final cited markdown report is compiled and ready.",
                )
            else:
                state.status = "failed"
                state.error = "No draft was produced during the research run."

        except Exception as e:
            state.status = "failed"
            state.error = f"Pipeline execution error: {str(e)}\n{traceback.format_exc()}"
            state.add_log("System", f"Fatal Error: {str(e)}", thought="Pipeline encountered an unhandled exception.")

        state.execution_time_seconds = round(time.time() - start_time, 2)
        return state

    def stream_run(self, question: str, max_retries: int = 3) -> Generator[AtlasState, None, None]:
        """Stream state updates generator for real-time SSE API or Streamlit dashboards."""
        state = AtlasState(research_question=question, max_retries=max_retries)
        start_time = time.time()

        try:
            # 1. Planning
            state = self.planner.plan(state)
            yield state

            # 2. Researching
            state = self.researcher.research(state)
            yield state

            # 3. Directed Loop
            while True:
                state = self.writer.write(state)
                yield state

                state = self.critic.critique(state)
                yield state

                next_step = route_after_critic(state)
                if next_step == "finalize":
                    break
                elif next_step == "research_more":
                    state.iteration_count += 1
                    state.status = "repairing"
                    last_verdict = state.critic_verdicts[-1]
                    repair_queries = [
                        issue.problem_statement for issue in last_verdict.issues if issue.category in ["missing_coverage", "unsupported_claim"]
                    ]
                    if not repair_queries:
                        repair_queries = [state.research_question + " data"]
                    state = self.researcher.targeted_research_for_critic_issues(state, repair_queries[:2])
                    yield state
                elif next_step == "rewrite":
                    state.iteration_count += 1
                    state.status = "repairing"
                    state.add_log("System", f"Routing to Writer for Revision {state.iteration_count + 1}...")
                    yield state

            if state.current_draft:
                state.final_report_markdown = state.current_draft.raw_markdown
                state.status = "completed"
                state.current_stage = "completed"
                state.add_log("System", "Research completed successfully.")
            yield state

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            yield state

        state.execution_time_seconds = round(time.time() - start_time, 2)
        yield state


# Global orchestrator singleton
atlas_orchestrator = AtlasOrchestrator()
