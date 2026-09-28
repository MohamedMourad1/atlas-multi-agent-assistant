"""
Tool Registry for Atlas Researcher Agent.
Provides unified dispatching and introspection for all tools.
"""

from typing import Dict, Any, List, Optional
from atlas.tools.rag_tool import rag_retriever
from atlas.tools.web_search_tool import web_search_engine
from atlas.tools.scraper_tool import safe_scraper
from atlas.tools.calculator_tool import safe_calculator


class ToolRegistry:
    """Central registry and executor for agent tools."""

    def __init__(self):
        self.tools = {
            "query_knowledge_base": self._tool_rag_search,
            "web_search": self._tool_web_search,
            "scrape_webpage": self._tool_scrape_url,
            "calculator": self._tool_calculator,
        }

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with arguments."""
        if tool_name not in self.tools:
            return {"status": "error", "error": f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"}
        try:
            return self.tools[tool_name](**kwargs)
        except Exception as e:
            return {"status": "error", "error": f"Tool execution failure: {str(e)}"}

    def _tool_rag_search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        results = rag_retriever.search(query, top_k=top_k)
        return {"status": "success", "results": results, "count": len(results)}

    def _tool_web_search(self, query: str, max_results: int = 4) -> Dict[str, Any]:
        results = web_search_engine.search(query, max_results=max_results)
        return {"status": "success", "results": results, "count": len(results)}

    def _tool_scrape_url(self, url: str) -> Dict[str, Any]:
        return safe_scraper.scrape_url(url)

    def _tool_calculator(self, expression: str) -> Dict[str, Any]:
        return safe_calculator.evaluate(expression)

    def get_tool_descriptions(self) -> str:
        """Format tools for LLM prompts."""
        return """
Available Tools:
1. `query_knowledge_base(query: str, top_k: int = 4)`: Search internal indexed documents and research papers.
2. `web_search(query: str, max_results: int = 4)`: Query live search engines (DuckDuckGo/Tavily/Wikipedia) for recent 2026 data.
3. `scrape_webpage(url: str)`: Fetch and sanitize the full text from a specific webpage URL.
4. `calculator(expression: str)`: Safely compute mathematical and percentage expressions (e.g. '65 * 1.25').
"""


# Global singleton registry
tool_registry = ToolRegistry()
