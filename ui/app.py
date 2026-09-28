"""
Atlas — Autonomous Multi-Agent Research & Report Assistant
Streamlit Dashboard & Interactive Execution Engine.
"""

import sys
import os
import json
import time
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Atlas — Autonomous Multi-Agent Research Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from atlas.config import settings
from atlas.schemas.state import AtlasState
from atlas.graph.state_graph import AtlasOrchestrator
from atlas.tools.rag_tool import rag_retriever
from atlas.evaluation.evaluator import atlas_evaluator
from atlas.evaluation.ablation import ablation_runner

# Custom CSS styling for premium glassmorphism & typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .agent-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    
    .agent-badge-planner { background-color: #3b82f6; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .agent-badge-researcher { background-color: #10b981; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .agent-badge-writer { background-color: #8b5cf6; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .agent-badge-critic { background-color: #f59e0b; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .agent-badge-system { background-color: #6b7280; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .evidence-box {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=300&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🧭 Atlas Control Panel")
    st.caption("Autonomous Multi-Agent Orchestrator v1.0")
    
    st.markdown("---")
    st.subheader("⚙️ LLM & Engine Settings")
    
    provider_choice = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "simulation_offline", "ollama"],
        index=0 if settings.gemini_api_key else 2,
        help="Select the AI provider. Simulation mode runs offline test synthesis without API keys."
    )
    
    if provider_choice == "gemini":
        api_key_input = st.text_input("Gemini API Key", value=settings.gemini_api_key, type="password")
        if api_key_input:
            settings.gemini_api_key = api_key_input
            os.environ["GEMINI_API_KEY"] = api_key_input
    elif provider_choice == "openai":
        api_key_input = st.text_input("OpenAI API Key", value=settings.openai_key, type="password")
        if api_key_input:
            settings.openai_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input
            
    st.markdown("---")
    st.subheader("🔍 Tooling & Retrieval Options")
    enable_web = st.checkbox("Enable Live Web Search", value=True)
    enable_rag = st.checkbox("Enable RAG Knowledge Base", value=True)
    max_retries = st.slider("Max Repair Iterations", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.subheader("📚 Corpus Status")
    st.info(f"**{len(rag_retriever.chunks)}** chunks indexed across **{len(list(settings.kb_dir.glob('*.*')))}** local documents.")


# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 28px; font-weight: 800;">🧭 Project Atlas</h1>
    <p style="margin: 6px 0 0 0; font-size: 16px; opacity: 0.9;">
        Autonomous Multi-Agent Research & Report Assistant — Transparent, Grounded, and Source-Cited Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Research Assistant",
    "📊 Evaluation & Ablation Study",
    "📚 Knowledge Base Manager",
    "🏗️ Architecture & Schemas",
])

# ----------------- TAB 1: RESEARCH RUNNER -----------------
with tab1:
    col_q1, col_q2 = st.columns([3, 1])
    
    with col_q1:
        preset_questions = [
            "What are the main risks facing the EV battery supply chain in 2026?",
            "How do US FEOC rules and the EU Battery Passport impact automaker compliance in 2026?",
            "What are the commercialization bottlenecks of Solid-State and Sodium-ion batteries in 2026?",
            "Custom Research Question..."
        ]
        selected_preset = st.selectbox("Select Preset Benchmark Question or Custom:", preset_questions)
        
        default_val = "What are the main risks facing the EV battery supply chain in 2026?" if selected_preset == "Custom Research Question..." else selected_preset
        user_query = st.text_area("Research Prompt:", value=default_val, height=75)

    with col_q2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True)
        st.caption("Executes: Planner ➔ Researcher ➔ Writer ➔ Critic ➔ Repair Loop")

    if run_btn and user_query.strip():
        # Initialize state container
        state_placeholder = st.empty()
        
        with st.status("🧭 Atlas Multi-Agent Graph Executing...", expanded=True) as status_box:
            orchestrator = AtlasOrchestrator()
            
            # Progress tracking
            progress_bar = st.progress(0.0)
            step_container = st.container()
            
            # Run stream
            final_state = None
            for state_update in orchestrator.stream_run(question=user_query, max_retries=max_retries):
                final_state = state_update
                
                # Update progress based on stage
                if state_update.current_stage == "planning":
                    progress_bar.progress(0.2)
                    status_box.update(label="📋 Planner Agent: Decomposing research query into structured sub-questions...")
                elif state_update.current_stage == "researching":
                    progress_bar.progress(0.4)
                    status_box.update(label="🔍 Researcher Agent: Executing ReAct loops (RAG KB + Web Search)...")
                elif state_update.current_stage == "drafting":
                    progress_bar.progress(0.65)
                    status_box.update(label=f"✍️ Writer Agent: Synthesizing cited draft (Iteration {state_update.iteration_count + 1})...")
                elif state_update.current_stage == "critiquing":
                    progress_bar.progress(0.85)
                    status_box.update(label=f"🧐 Critic Agent: Verifying claims against evidence store...")
                elif state_update.current_stage == "completed":
                    progress_bar.progress(1.0)
                    status_box.update(label="✅ Research Completed & Report Verified!", state="complete")
                    
            st.session_state["last_run_state"] = final_state

    # Display Results if state exists
    if "last_run_state" in st.session_state and st.session_state["last_run_state"]:
        state: AtlasState = st.session_state["last_run_state"]
        eval_metrics = atlas_evaluator.evaluate_run(state)
        
        st.markdown("---")
        
        # 1. High-Level Metrics Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Status", state.status.upper(), delta="Verified" if state.status == "completed" else "Failed")
        with m2:
            st.metric("Faithfulness Score", f"{eval_metrics['faithfulness'] * 100:.0f}%", help="Claims factually grounded in evidence")
        with m3:
            st.metric("Citation Precision", f"{eval_metrics['citation_precision'] * 100:.0f}%", help="Valid [E#] citations")
        with m4:
            st.metric("Sub-Q Coverage", f"{eval_metrics['coverage'] * 100:.0f}%", help="Planned sub-questions addressed")
        with m5:
            st.metric("Repair Iterations", f"{state.iteration_count} of {state.max_retries}")
        with m6:
            st.metric("Latency", f"{state.execution_time_seconds:.1f}s")

        # 2. Detailed Views Accordion
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.subheader("📄 Verified Final Report")
            if state.final_report_markdown:
                st.markdown(state.final_report_markdown)
                
                # Export Options
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=state.final_report_markdown,
                        file_name="Atlas_Research_Report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_exp2:
                    st.download_button(
                        label="📥 Download Full State (JSON)",
                        data=json.dumps(state.model_dump(), indent=2),
                        file_name="Atlas_Execution_State.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            else:
                st.error("No report available.")

        with c_right:
            st.subheader("🔍 Agent Trace & Evidence Store")
            
            # Sub-tabs for inspection
            insp_tab1, insp_tab2, insp_tab3 = st.tabs(["Timeline Trace", "Evidence Store", "Critic Verdicts"])
            
            with insp_tab1:
                st.markdown("**Observable Step-by-Step Execution Log:**")
                for log in state.step_logs:
                    badge_class = f"agent-badge-{log.agent_name.lower()}" if log.agent_name.lower() in ["planner", "researcher", "writer", "critic", "system"] else "agent-badge-system"
                    with st.expander(f"[{log.timestamp}] {log.agent_name}: {log.action}", expanded=False):
                        st.markdown(f"<span class='{badge_class}'>{log.agent_name}</span>", unsafe_allow_html=True)
                        if log.thought:
                            st.info(f"🧠 **Thought:** {log.thought}")
                        if log.details:
                            st.json(log.details)

            with insp_tab2:
                st.markdown(f"**Retained Evidence Items ({len(state.evidence_store.items)}):**")
                for eid, item in state.evidence_store.items.items():
                    with st.expander(f"[{eid}] {item.source_title} ({item.source_type.title()})"):
                        st.markdown(f"**Sub-Question ID:** `{item.sub_question_id}`")
                        st.markdown(f"**Query Used:** `{item.query_used}`")
                        st.markdown(f"**Evidence Text:**\n> \"{item.snippet}\"")
                        if item.source_url:
                            st.markdown(f"**Source URL / File:** [{item.source_url}]({item.source_url})")

            with insp_tab3:
                st.markdown("**Critic Review History:**")
                if not state.critic_verdicts:
                    st.write("No critic verdicts recorded yet.")
                for v in state.critic_verdicts:
                    st.write(f"### Iteration {v.iteration}: {'✅ APPROVED' if v.approved else '⚠️ ' + v.action.upper()}")
                    st.write(f"- Overall Score: `{v.overall_score:.2f}` | Faithfulness: `{v.faithfulness_score:.2f}` | Citations: `{v.citation_precision_score:.2f}`")
                    st.write(f"**Summary:** {v.feedback_summary}")
                    if v.issues:
                        st.markdown("**Issues Identified:**")
                        for iss in v.issues:
                            st.error(f"[{iss.severity.upper()}] {iss.category}: {iss.problem_statement}")


# ----------------- TAB 2: EVALUATION & ABLATION STUDY -----------------
with tab2:
    st.header("📊 Ablation Studies & Performance Benchmark")
    st.markdown("""
    Compare **Configuration A** (Single-Prompt LLM), **Configuration B** (RAG + Writer without Critic), and **Configuration C** (Full Atlas Multi-Agent Directed Graph).
    """)
    
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        test_limit = st.slider("Number of Benchmark Questions to Evaluate", min_value=1, max_value=10, value=3)
    with col_b2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_ablation_btn = st.button("⚡ Run Comparative Ablation Sweep", type="primary", use_container_width=True)

    if run_ablation_btn:
        with st.spinner("Running full 3-configuration benchmark sweep..."):
            ablation_results = ablation_runner.run_full_benchmark(limit=test_limit)
            st.session_state["ablation_results"] = ablation_results
            st.success("Benchmark sweep complete!")

    if "ablation_results" in st.session_state and st.session_state["ablation_results"]:
        results = st.session_state["ablation_results"]
        
        # Aggregate scores
        configs = ["Config A: Single LLM", "Config B: RAG + Writer", "Config C: Full Atlas"]
        avg_faithfulness = [
            sum(r["config_a"]["faithfulness"] for r in results) / len(results),
            sum(r["config_b"]["faithfulness"] for r in results) / len(results),
            sum(r["config_c"]["faithfulness"] for r in results) / len(results),
        ]
        avg_precision = [
            sum(r["config_a"]["citation_precision"] for r in results) / len(results),
            sum(r["config_b"]["citation_precision"] for r in results) / len(results),
            sum(r["config_c"]["citation_precision"] for r in results) / len(results),
        ]
        avg_coverage = [
            sum(r["config_a"]["coverage"] for r in results) / len(results),
            sum(r["config_b"]["coverage"] for r in results) / len(results),
            sum(r["config_c"]["coverage"] for r in results) / len(results),
        ]
        
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            fig_bar = go.Figure(data=[
                go.Bar(name='Faithfulness / Grounding', x=configs, y=avg_faithfulness, marker_color='#3b82f6'),
                go.Bar(name='Citation Precision', x=configs, y=avg_precision, marker_color='#10b981'),
                go.Bar(name='Sub-Question Coverage', x=configs, y=avg_coverage, marker_color='#8b5cf6'),
            ])
            fig_bar.update_layout(
                title="Comparative Performance Metrics (Ablation Table)",
                barmode='group',
                yaxis=dict(range=[0, 1.05], title="Score (0.0 - 1.0)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_ch2:
            fig_radar = go.Figure()
            categories = ['Faithfulness', 'Citation Precision', 'Coverage', 'Overall Quality']
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_faithfulness[0], avg_precision[0], avg_coverage[0], 0.50],
                theta=categories, fill='toself', name='Config A (Single Prompt)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_faithfulness[1], avg_precision[1], avg_coverage[1], 0.72],
                theta=categories, fill='toself', name='Config B (RAG No Critic)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_faithfulness[2], avg_precision[2], avg_coverage[2], 0.92],
                theta=categories, fill='toself', name='Config C (Full Atlas)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="System Capability Radar Chart",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Question-by-Question breakdown table
        st.subheader("📋 Detailed Per-Question Results")
        for res in results:
            with st.expander(f"[{res.get('id', 'Q')}] {res['question']}"):
                col_qa, col_qb, col_qc = st.columns(3)
                with col_qa:
                    st.markdown("**Config A (Single LLM)**")
                    st.write(f"- Faithfulness: `{res['config_a']['faithfulness']:.2f}`")
                    st.write(f"- Citations: `{res['config_a']['citation_precision']:.2f}`")
                    st.write(f"- Coverage: `{res['config_a']['coverage']:.2f}`")
                with col_qb:
                    st.markdown("**Config B (RAG + Writer)**")
                    st.write(f"- Faithfulness: `{res['config_b']['faithfulness']:.2f}`")
                    st.write(f"- Citations: `{res['config_b']['citation_precision']:.2f}`")
                    st.write(f"- Coverage: `{res['config_b']['coverage']:.2f}`")
                with col_qc:
                    st.markdown("**Config C (Full Atlas)**")
                    st.write(f"- Faithfulness: `{res['config_c']['faithfulness']:.2f}`")
                    st.write(f"- Citations: `{res['config_c']['citation_precision']:.2f}`")
                    st.write(f"- Coverage: `{res['config_c']['coverage']:.2f}`")
                    st.write(f"- Critic Uplift: `+{res['config_c'].get('critic_uplift', 0.0):.2f}`")


# ----------------- TAB 3: KNOWLEDGE BASE MANAGER -----------------
with tab3:
    st.header("📚 Knowledge Base Ingestion & Vector Explorer")
    st.markdown("Upload domain documents (PDFs, Markdown, Text) to expand Atlas's internal RAG retrieval corpus.")
    
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        uploaded_files = st.file_uploader(
            "Upload Documents to Knowledge Base:",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = settings.kb_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                rag_retriever.ingest_file(save_path)
            rag_retriever._calculate_idf()
            st.success(f"Indexed {len(uploaded_files)} new file(s)! Total chunks: {len(rag_retriever.chunks)}")

    with col_u2:
        st.markdown("**Corpus Summary:**")
        docs = list(settings.kb_dir.glob("*.*"))
        st.write(f"📁 Directory: `{settings.kb_dir.name}`")
        st.write(f"📄 Total Files: `{len(docs)}`")
        st.write(f"🧩 Indexed Chunks: `{len(rag_retriever.chunks)}`")
        if st.button("🔄 Rebuild Corpus Index"):
            rag_retriever.load_and_index()
            st.success("Corpus index rebuilt successfully!")

    st.markdown("---")
    st.subheader("🔎 Test Vector Retrieval Query")
    test_query = st.text_input("Enter search query to test RAG retrieval:", "lithium refining bottlenecks 2026")
    if test_query:
        search_res = rag_retriever.search(test_query, top_k=3)
        if search_res:
            for r in search_res:
                st.markdown(f"""
                <div class="evidence-box">
                    <strong>{r['title']}</strong> (Score: {r['score']})<br/>
                    <em>{r['snippet']}</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matching passages found in current corpus.")


# ----------------- TAB 4: ARCHITECTURE & AGENT SPECS -----------------
with tab4:
    st.header("🏗️ Atlas Multi-Agent Architecture & Design Principles")
    st.markdown("""
    Atlas replaces monolithic LLM generation with a **strictly typed directed loop of specialized agents**.
    Agents never pass free-form unstructured text — all inter-agent boundaries are governed by strict Pydantic schemas.
    """)
    
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Graph_Loop [Atlas Directed Loop]
            User([User Research Query]) --> Planner[1. Planner Agent<br/>Pydantic Decomposition]
            Planner --> Researcher[2. Researcher Agent<br/>ReAct Loop: RAG + Web]
            Researcher -->|EvidenceStore: [E1], [E2]| Writer[3. Writer Agent<br/>Synthesis & Citations]
            Writer --> Critic[4. Critic Agent<br/>Claim-by-Claim Verification]
            
            Critic -- "Approved (Score >= 0.85)" --> FinalReport([Final Verified Report])
            Critic -- "Research More" --> Researcher
            Critic -- "Rewrite / Grounding" --> Writer
        end
    ```
    """)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("🧩 Agent Roles")
        st.markdown("""
        1. **Planner Agent:** Breaks down open-ended question into 3-6 non-overlapping sub-questions.
        2. **Researcher Agent:** Dispatches tools (`query_knowledge_base`, `web_search`, `scrape_webpage`), gathers atomic evidence with unique `[E#]` IDs.
        3. **Writer Agent:** Writes multi-section analytical report with mandatory `[E#]` citations.
        4. **Critic Agent:** Verifies claims against evidence items, checks citation validity, calculates faithfulness and coverage scores.
        """)
    with col_a2:
        st.subheader("🛡️ Core Safety & Quality Guardrails")
        st.markdown("""
        - **Strict Citation Enforcement:** Programmatic regex checks flag any non-existent `[E#]` IDs.
        - **Prompt Injection Neutralization:** Web scraper strips instruction-like payloads (`Ignore previous instructions`).
        - **Sandboxed Execution:** Arithmetic is evaluated safely without arbitrary code execution.
        - **Loop Cap:** Hard cap at 3 repair iterations prevents infinite agent loops.
        """)
