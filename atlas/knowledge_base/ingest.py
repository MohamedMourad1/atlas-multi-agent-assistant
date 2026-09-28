"""
Ingestion script for Atlas Knowledge Base.
Scans corpus directories, parses PDFs/Markdown/Text, chunks content, and rebuilds the vector index.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import settings
from atlas.tools.rag_tool import rag_retriever


def run_ingestion(custom_dir: str = None):
    """Run batch document ingestion."""
    target_dir = Path(custom_dir) if custom_dir else settings.kb_dir
    print(f"=== Atlas Knowledge Base Ingestion ===")
    print(f"Scanning target directory: {target_dir}")
    
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist. Creating it.")
        os.makedirs(target_dir, exist_ok=True)
        return

    rag_retriever.corpus_dir = target_dir
    rag_retriever.load_and_index()
    
    print(f"Successfully indexed {len(rag_retriever.chunks)} passages across documents.")
    print("Knowledge base ready for agent queries.\n")


if __name__ == "__main__":
    run_ingestion()
