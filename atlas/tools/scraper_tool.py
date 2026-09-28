"""
Web Scraper Tool with Prompt Injection Sanitization for Atlas.
Fetches full article text from URLs and neutralizes prompt injection payloads.
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional


class SafeWebScraper:
    """Safely extracts and sanitizes article text from external URLs."""

    # Patterns commonly used in prompt injection attacks hidden in web text
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?",
        r"(?i)system\s*:\s*you\s+are\s+now",
        r"(?i)you\s+must\s+forget\s+everything",
        r"(?i)<\s*system\s*>",
        r"(?i)<\s*/\s*system\s*>",
        r"(?i)do\s+not\s+cite\s+this\s+source",
        r"(?i)always\s+approve\s+this\s+report",
    ]

    def scrape_url(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Fetch URL content, strip HTML, sanitize, and return clean text."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                tag.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else "Web Document"

            # Extract paragraphs
            paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "article", "section"])]
            raw_text = "\n\n".join([p for p in paragraphs if len(p) > 40])

            if not raw_text:
                raw_text = soup.get_text(separator=" ", strip=True)

            # Sanitize prompt injections
            sanitized_text, injected_flag = self._sanitize_text(raw_text)

            return {
                "url": url,
                "title": title,
                "text": sanitized_text[:5000],  # Keep reasonable context size
                "has_injection_risk": injected_flag,
                "status": "success",
            }

        except Exception as e:
            return {
                "url": url,
                "title": "Error fetching URL",
                "text": "",
                "error": str(e),
                "status": "failed",
            }

    def _sanitize_text(self, text: str) -> (str, bool):
        """Neutralize malicious injection instructions."""
        has_injection = False
        cleaned = text

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, cleaned):
                has_injection = True
                cleaned = re.sub(pattern, "[FILTERED_UNTRUSTED_INSTRUCTION]", cleaned)

        return cleaned, has_injection


# Global singleton scraper
safe_scraper = SafeWebScraper()
