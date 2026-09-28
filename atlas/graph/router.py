"""
Conditional Router for the Atlas State Machine.
Determines next state transition based on Critic verdicts and retry counters.
"""

from typing import Literal
from atlas.schemas.state import AtlasState


def route_after_critic(state: AtlasState) -> Literal["finalize", "research_more", "rewrite"]:
    """Evaluate Critic verdict and iteration limits to determine next node."""
    if not state.critic_verdicts:
        return "finalize"

    last_verdict = state.critic_verdicts[-1]

    # Check approval
    if last_verdict.approved:
        return "finalize"

    # Check retry cap to prevent infinite loops
    if state.iteration_count >= state.max_retries:
        state.add_log(
            agent_name="System",
            action=f"Reached maximum repair iterations ({state.max_retries}). Finalizing best available draft.",
            thought="Terminating repair loop due to retry budget cap.",
        )
        return "finalize"

    # Route based on Critic action recommendation
    if last_verdict.action == "research_more":
        return "research_more"
    else:
        return "rewrite"
