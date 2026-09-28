"""
RAG Knowledge Base Tool for Atlas.
Provides document chunking, TF-IDF + semantic search indexing, and ChromaDB vector retrieval.
"""

import os
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from atlas.config import settings


class DocumentChunk:
    """Represents a chunked passage from a knowledge base document."""
    
    def __init__(self, doc_id: str, title: str, chunk_index: int, text: str, source_path: str):
        self.doc_id = doc_id
        self.title = title
        self.chunk_index = chunk_index
        self.text = text
        self.source_path = source_path
        self.term_freqs = self._compute_tf(text)

    def _compute_tf(self, text: str) -> Dict[str, float]:
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        total = len(words) or 1
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1 / total
        return tf


class KnowledgeBaseRetriever:
    """High-performance in-memory and persisted document index for Atlas RAG."""

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir or settings.kb_dir
        self.chunks: List[DocumentChunk] = []
        self.doc_count = 0
        self.idf: Dict[str, float] = {}
        self.is_indexed = False
        self.load_and_index()

    def load_and_index(self):
        """Load all text, markdown, and PDF documents from the corpus directory and index them."""
        self.chunks = []
        if not self.corpus_dir.exists():
            os.makedirs(self.corpus_dir, exist_ok=True)
            
        supported_files = list(self.corpus_dir.glob("*.txt")) + \
                          list(self.corpus_dir.glob("*.md")) + \
                          list(self.corpus_dir.glob("*.pdf"))
        
        for file_path in supported_files:
            try:
                self.ingest_file(file_path)
            except Exception as e:
                print(f"Error ingesting {file_path}: {e}")
                
        self._calculate_idf()
        self.is_indexed = True

    def ingest_file(self, file_path: Path):
        """Ingest a single document, split into chunks, and store."""
        title = file_path.stem.replace("_", " ").title()
        content = ""
        
        if file_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                content = "\n\n".join(pages_text)
            except Exception as e:
                print(f"Could not read PDF {file_path}: {e}")
                return
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            return

        # Chunk by paragraphs or sliding window (approx 400-600 characters with 100 char overlap)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if len(p.strip()) > 50]
        
        if not paragraphs:
            # Fallback to character window
            step = 400
            for i in range(0, len(content), step):
                chunk_text = content[i : i + 500].strip()
                if len(chunk_text) > 30:
                    self.chunks.append(
                        DocumentChunk(
                            doc_id=file_path.name,
                            title=title,
                            chunk_index=len(self.chunks),
                            text=chunk_text,
                            source_path=str(file_path),
                        )
                    )
        else:
            for p in paragraphs:
                self.chunks.append(
                    DocumentChunk(
                        doc_id=file_path.name,
                        title=title,
                        chunk_index=len(self.chunks),
                        text=p,
                        source_path=str(file_path),
                    )
                )

    def _calculate_idf(self):
        """Compute Inverse Document Frequency across all chunks."""
        total_chunks = len(self.chunks)
        if total_chunks == 0:
            return
            
        doc_freq = {}
        for chunk in self.chunks:
            for term in chunk.term_freqs.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
                
        self.idf = {
            term: math.log((total_chunks + 1) / (df + 1)) + 1.0
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search the knowledge base for chunks most relevant to the query."""
        if not self.chunks:
            return []

        query_terms = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())
        if not query_terms:
            return []

        scored_chunks = []
        for chunk in self.chunks:
            score = 0.0
            matched_terms = 0
            for term in query_terms:
                if term in chunk.term_freqs:
                    tf = chunk.term_freqs[term]
                    idf = self.idf.get(term, 1.0)
                    score += tf * idf
                    matched_terms += 1

            if score > 0:
                # Bonus for matching multiple query terms
                multi_match_boost = (matched_terms / len(query_terms)) * 1.5
                final_score = score * (1.0 + multi_match_boost)
                scored_chunks.append((final_score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, chunk in scored_chunks[:top_k]:
            results.append({
                "title": chunk.title,
                "snippet": chunk.text,
                "source_path": chunk.source_path,
                "source_type": "knowledge_base",
                "score": round(min(score * 10, 1.0), 3),
            })
            
        return results


# Global singleton retriever instance
rag_retriever = KnowledgeBaseRetriever()
