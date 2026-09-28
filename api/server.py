"""
FastAPI Server for Atlas Autonomous Research Assistant.
Provides REST and Server-Sent Events (SSE) endpoints for agents, knowledge base, and benchmarking.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from atlas.config import settings
from atlas.graph.state_graph import atlas_orchestrator
from atlas.tools.rag_tool import rag_retriever
from atlas.evaluation.evaluator import atlas_evaluator
from atlas.evaluation.ablation import ablation_runner

app = FastAPI(
    title="Atlas Multi-Agent Research API",
    version="1.0.0",
    description="Autonomous Multi-Agent Research & Report Assistant Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str
    max_retries: int = 3
    enable_web_search: bool = True
    enable_rag_kb: bool = True


class BenchmarkRequest(BaseModel):
    limit: int = 3


@app.get("/health")
def health_check():
    """Service health and component status check."""
    return {
        "status": "healthy",
        "service": "Atlas Multi-Agent Engine",
        "version": settings.version,
        "indexed_chunks": len(rag_retriever.chunks),
        "llm_provider": settings.default_provider,
    }


@app.post("/api/research/run")
def run_research(req: ResearchRequest):
    """Run full autonomous research workflow synchronously."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Research question cannot be empty.")

    settings.enable_web_search = req.enable_web_search
    settings.enable_rag_kb = req.enable_rag_kb

    state = atlas_orchestrator.run(question=req.question, max_retries=req.max_retries)
    metrics = atlas_evaluator.evaluate_run(state)

    return {
        "state": state.model_dump(),
        "metrics": metrics,
        "report_markdown": state.final_report_markdown,
    }


@app.post("/api/research/stream")
def stream_research(req: ResearchRequest):
    """Stream live state updates as Server-Sent Events (SSE)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Research question cannot be empty.")

    settings.enable_web_search = req.enable_web_search
    settings.enable_rag_kb = req.enable_rag_kb

    def event_generator():
        for state in atlas_orchestrator.stream_run(question=req.question, max_retries=req.max_retries):
            data = json.dumps({
                "stage": state.current_stage,
                "status": state.status,
                "iteration": state.iteration_count,
                "logs_count": len(state.step_logs),
                "latest_log": state.step_logs[-1].model_dump() if state.step_logs else None,
                "evidence_count": len(state.evidence_store.items),
                "has_draft": state.current_draft is not None,
                "verdict": state.critic_verdicts[-1].model_dump() if state.critic_verdicts else None,
                "final_report": state.final_report_markdown,
            })
            yield f"data: {data}\n\n"
            time.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/kb/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF, Markdown, or text file into the knowledge base."""
    allowed_exts = [".pdf", ".md", ".txt"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_exts}")

    save_path = settings.kb_dir / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Ingest into retriever
    rag_retriever.ingest_file(save_path)
    rag_retriever._calculate_idf()

    return {
        "status": "success",
        "filename": file.filename,
        "total_indexed_chunks": len(rag_retriever.chunks),
        "message": f"Successfully indexed {file.filename} into Atlas knowledge base.",
    }


@app.get("/api/kb/stats")
def get_kb_stats():
    """Retrieve knowledge base corpus statistics."""
    files = list(settings.kb_dir.glob("*.*"))
    return {
        "corpus_directory": str(settings.kb_dir),
        "total_documents": len(files),
        "document_names": [f.name for f in files],
        "total_chunks_indexed": len(rag_retriever.chunks),
    }


@app.post("/api/benchmark/run")
def run_benchmark(req: BenchmarkRequest):
    """Run ablation evaluation suite."""
    results = ablation_runner.run_full_benchmark(limit=req.limit)
    return {"benchmark_results": results}
