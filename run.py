"""
Atlas CLI and Launcher Script.
Run research queries via CLI, start Streamlit UI, launch FastAPI server, or execute benchmarks.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Ensure Windows stdout handles UTF-8 strings gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def run_cli_research(query: str, max_retries: int = 3):
    """Execute research on a query directly from terminal and display report."""
    from atlas.graph.state_graph import atlas_orchestrator
    from atlas.evaluation.evaluator import atlas_evaluator

    print(f"\n🧭 [Atlas] Launching Autonomous Multi-Agent Research...")
    print(f"📌 Research Question: \"{query}\"\n")

    def print_step(state):
        if state.step_logs:
            latest = state.step_logs[-1]
            print(f"[{latest.timestamp}] [{latest.agent_name.upper()}] {latest.action}")
            if latest.thought:
                print(f"   💡 Thought: {latest.thought}")

    state = atlas_orchestrator.run(question=query, max_retries=max_retries, on_step=print_step)
    metrics = atlas_evaluator.evaluate_run(state)

    print("\n" + "=" * 60)
    print("📊 EVALUATION METRICS:")
    print(f"- Status: {state.status.upper()}")
    print(f"- Faithfulness / Grounding: {metrics['faithfulness'] * 100:.1f}%")
    print(f"- Citation Precision: {metrics['citation_precision'] * 100:.1f}%")
    print(f"- Sub-Question Coverage: {metrics['coverage'] * 100:.1f}%")
    print(f"- Repair Iterations: {state.iteration_count}")
    print(f"- Latency: {state.execution_time_seconds:.2f}s")
    print("=" * 60)

    print("\n📄 FINAL CITATION-GROUNDED REPORT:")
    print("-" * 60)
    print(state.final_report_markdown)
    print("-" * 60)


def start_ui():
    """Launch the Streamlit interactive dashboard."""
    ui_path = BASE_DIR / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
    print(f"🚀 Starting Atlas Streamlit UI: {' '.join(cmd)}")
    subprocess.run(cmd)


def start_api(port: int = 8000):
    """Launch the FastAPI server."""
    import uvicorn
    print(f"🚀 Starting Atlas FastAPI Server on http://localhost:{port}")
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=True)


def run_benchmark(limit: int = 3):
    """Run ablation comparison sweep."""
    from atlas.evaluation.ablation import ablation_runner
    print(f"⚡ Running Comparative Ablation Sweep across {limit} benchmark questions...")
    results = ablation_runner.run_full_benchmark(limit=limit)
    print("\n" + "=" * 60)
    print(f"ABLATION RESULTS SUMMARY ({len(results)} questions evaluated):")
    for r in results:
        print(f"\nQuestion: {r['question']}")
        print(f"  - Config A (Single LLM): Faithfulness={r['config_a']['faithfulness']:.2f}, Citations={r['config_a']['citation_precision']:.2f}")
        print(f"  - Config B (RAG + Writer): Faithfulness={r['config_b']['faithfulness']:.2f}, Citations={r['config_b']['citation_precision']:.2f}")
        print(f"  - Config C (Full Atlas): Faithfulness={r['config_c']['faithfulness']:.2f}, Citations={r['config_c']['citation_precision']:.2f}, Uplift=+{r['config_c']['critic_uplift']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas Multi-Agent Research Assistant")
    parser.add_argument("mode", choices=["ui", "api", "research", "benchmark"], default="ui", nargs="?", help="Operating mode")
    parser.add_argument("-q", "--query", type=str, default="What are the main risks facing the EV battery supply chain in 2026?", help="Research question for CLI mode")
    parser.add_argument("--retries", type=int, default=3, help="Max repair iterations")
    parser.add_argument("--limit", type=int, default=3, help="Limit benchmark questions count")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    args = parser.parse_args()

    if args.mode == "ui":
        start_ui()
    elif args.mode == "api":
        start_api(port=args.port)
    elif args.mode == "research":
        run_cli_research(query=args.query, max_retries=args.retries)
    elif args.mode == "benchmark":
        run_benchmark(limit=args.limit)
