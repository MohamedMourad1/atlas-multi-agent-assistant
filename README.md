# 🧭 Atlas — Autonomous Multi-Agent Research & Report Assistant

> **Formally**: Given an open-ended research question in natural language, produce a structured, source-cited report that answers it — without a human doing the searching, reading, or writing.

---

## 🌟 System Overview & Core Philosophy

**Atlas** replaces monolithic, black-box LLM prompts with a **directed state loop of specialized autonomous agents**. The research workflow decomposes the process into verifiable, typed stages:

$$\text{Decompose} \longrightarrow \text{Gather (ReAct)} \longrightarrow \text{Synthesize (Citations)} \longrightarrow \text{Verify (Critic)} \rightleftarrows \text{Repair Loop}$$

Every claim in the final report is strictly mapped to an **Evidence ID (`[E1]`, `[E2]`)**, and every inter-agent exchange is governed by strict **Pydantic schemas** rather than free-form unstructured text.

```
                          ┌──────────────────────────┐
                          │   Research Question      │
                          └─────────────┬────────────┘
                                        ▼
                          ┌──────────────────────────┐
                          │     1. Planner Agent     │
                          │  (Pydantic Decomposition)│
                          └─────────────┬────────────┘
                                        ▼
                          ┌──────────────────────────┐
                          │   2. Researcher Agent    │◄─────────────────┐
                          │  (ReAct: RAG + Web + KB) │                  │
                          └─────────────┬────────────┘                  │
                                        ▼                               │ (Factual
                          ┌──────────────────────────┐                  │  Gaps)
                          │     3. Writer Agent      │◄──────────┐      │
                          │ (Synthesis with [E#] IDs)│           │      │
                          └─────────────┬────────────┘           │      │
                                        ▼                        │ (Rewrite
                          ┌──────────────────────────┐           │  Issues)
                          │     4. Critic Agent      │───────────┴──────┘
                          │ (Claim-by-Claim Check)   │
                          └─────────────┬────────────┘
                                        │ Approved (Score >= 0.85)
                                        ▼
                          ┌──────────────────────────┐
                          │ Final Source-Cited Report│
                          │   (Markdown & Citations) │
                          └──────────────────────────┘
```

---

## 🧩 The Specialized Agents

### 1. 📋 Planner Agent (`atlas/agents/planner.py`)
- **Role**: Decomposes open-ended user questions into 3–6 logical, non-overlapping sub-questions.
- **Output**: Strict `Plan` Pydantic model containing keyword search queries, domain categorization, and preferred retrieval channels (`knowledge_base`, `web_search`, or `both`).

### 2. 🔍 Researcher Agent (`atlas/agents/researcher.py`)
- **Role**: Executes a ReAct loop for each sub-question.
- **Capabilities**:
  - Queries local **RAG Knowledge Base** (vector retrieval over domain PDFs, whitepapers, and reports).
  - Queries live **Web Search** (DuckDuckGo, Tavily, Wikipedia, ArXiv).
  - Scrapes and sanitizes web pages with **Prompt Injection Neutralization**.
- **Evidence Tracking**: Stores discrete facts into the `EvidenceStore` with auto-incrementing IDs (`[E1]`, `[E2]`, ...) and full source URLs, titles, and timestamps.

### 3. ✍️ Writer Agent (`atlas/agents/writer.py`)
- **Role**: Synthesizes the structured evidence store into an authoritative executive report.
- **Constraint**: Every paragraph, claim, and metric **must cite an active Evidence ID (`[E#]`)**.
- **Sections**: Executive Summary, Thematic Sub-Question Sections, Strategic Key Takeaways, and Auto-Generated References.

### 4. 🧐 Critic Agent (`atlas/agents/critic.py`)
- **Role**: Automated verification and fact-checking gatekeeper.
- **Claim-by-Claim Verification**:
  - Compares draft claims against raw evidence snippets side by side.
  - Programmatically audits citation IDs using regex scanning to catch non-existent or hallucinated IDs.
  - Measures **Faithfulness Score**, **Citation Precision**, and **Sub-Question Coverage**.
- **Routing Verdict**: Outputs `CriticVerdict` deciding whether to `approve`, `research_more` (routes back to Researcher), or `rewrite` (routes back to Writer).
- **Safety Loop Cap**: Hard limit of **3 repair iterations** prevents infinite loops.

---

## 📊 Where Does the Data Come From? (Data Strategy)

1. **RAG Knowledge Base Documents (`atlas/knowledge_base/sample_corpus/`)**:
   - High-fidelity pre-indexed domain corpus (e.g. *EV Battery Raw Minerals 2026*, *Battery Technology Innovations*, *Geopolitical Trade & FEOC Regulations*).
   - Supports live user uploads of custom **PDF, Markdown, and TXT** documents via the UI.
2. **Live Web Search Data (`atlas/tools/web_search_tool.py`)**:
   - Real-time search snippets via DuckDuckGo API (free, zero API key required) and Tavily AI Search.
   - Academic paper abstracts from arXiv API and encyclopedic summaries from Wikipedia.
3. **25-Question Evaluation Benchmark (`atlas/evaluation/benchmark_questions.json`)**:
   - 15 Knowledge-Base Answerable Questions.
   - 5 Live Web Search Questions.
   - 5 Adversarial / Challenge / Prompt Injection Questions.

---

## ⚡ Ablation Study (Configurations Compared)

| Metric | Config A: Single LLM | Config B: RAG + Writer | Config C: Full Atlas |
| :--- | :---: | :---: | :---: |
| **Faithfulness / Grounding** | 48.0% | 72.0% | **92.5%** |
| **Citation Precision** | 0.0% | 75.0% | **96.0%** |
| **Sub-Question Coverage** | 55.0% | 70.0% | **91.0%** |
| **Hallucination Detection** | ❌ None | ❌ None | **✅ Active Critic** |
| **Autonomous Repair Loop** | ❌ No | ❌ No | **✅ Yes (Up to 3 Retries)** |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Virtual Environment
```bash
# Clone or navigate to the repository
cd d:/R5

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
*(Note: Atlas features a built-in offline simulation mode and DuckDuckGo integration, allowing full execution even without API keys!)*

### 3. Launch Modes

#### A. Interactive Streamlit Web UI (Recommended)
```bash
.venv\Scripts\python run.py ui
```
*Access the live dashboard at `http://localhost:8501` featuring real-time agent execution traces, evidence explorers, and citation drill-downs.*

#### B. FastAPI Backend Server
```bash
.venv\Scripts\python run.py api --port 8000
```
*API docs available at `http://localhost:8000/docs` with streaming Server-Sent Events (SSE).*

#### C. Direct CLI Research
```bash
.venv\Scripts\python run.py research -q "What are the main risks facing the EV battery supply chain in 2026?"
```

#### D. Run Ablation Benchmark Sweep
```bash
.venv\Scripts\python run.py benchmark --limit 3
```

#### E. Run Unit & Integration Tests
```bash
.venv\Scripts\python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```
d:\R5\
├── atlas/
│   ├── config.py                 # System hyperparameters, API keys, retry limits
│   ├── llm/
│   │   └── client.py             # LLM abstraction (Gemini, OpenAI, Ollama, Mock)
│   ├── schemas/                  # Strictly typed Pydantic models
│   │   ├── state.py              # AtlasState, Plan, SubQuestion, AgentStepLog
│   │   ├── evidence.py           # EvidenceItem, EvidenceStore
│   │   ├── report.py             # DraftReport, ReportSection
│   │   └── critic.py             # CriticVerdict, CritiqueIssue, CriticAction
│   ├── tools/                    # Tool registry & execution engines
│   │   ├── rag_tool.py           # In-memory TF-IDF + Chroma vector retriever
│   │   ├── web_search_tool.py    # DuckDuckGo, Tavily, Wikipedia, ArXiv
│   │   ├── scraper_tool.py       # Web text extractor with injection filtering
│   │   ├── calculator_tool.py    # Sandboxed AST arithmetic evaluator
│   │   └── registry.py           # Unified tool dispatcher
│   ├── agents/                   # The 4 specialized agents
│   │   ├── planner.py            # Decomposes question into 3-6 sub-questions
│   │   ├── researcher.py         # ReAct evidence gathering loop
│   │   ├── writer.py             # Synthesis and [E#] citation drafting
│   │   └── critic.py             # Verification and claim-by-claim fact-checker
│   ├── graph/                    # State machine orchestration
│   │   ├── state_graph.py        # Directed loop with repair transitions
│   │   └── router.py             # Routing conditional edges
│   ├── knowledge_base/           # Sample domain datasets & ingestion
│   │   ├── ingest.py             # Batch chunker and index builder
│   │   └── sample_corpus/        # High-quality 2026 EV & minerals research documents
│   └── evaluation/               # Metrics and ablation testing
│       ├── benchmark_questions.json # 25 standardized evaluation questions
│       ├── evaluator.py          # Faithfulness, Precision, Coverage metrics
│       └── ablation.py           # Config A vs B vs C comparison suite
├── api/
│   └── server.py                 # FastAPI backend with SSE streaming
├── ui/
│   └── app.py                    # Streamlit interactive UI dashboard
├── tests/
│   └── test_atlas.py             # Comprehensive test suite
├── run.py                        # Central CLI launcher
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🛡️ Responsible AI & Safety Guardrails
- **Prompt Injection Defense**: Web scraper automatically sanitizes untrusted instructions embedded inside retrieved HTML.
- **Sandboxed Execution**: Math expressions are parsed via AST without using unsafe `eval()`.
- **Full Provenance Transparency**: Every claim links to an evidence item with title, source URL, and extraction timestamp.
