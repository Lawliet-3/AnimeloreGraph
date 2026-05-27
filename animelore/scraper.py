"""
Sitemap discovery and Fandom scraping utilities.

Implements the three-phase ingestion pipeline:
1) Sitemap discovery + filtering (NS_0 only, URL sanitation).
2) Content cleaning + HTML-to-Markdown conversion.
3) Downstream ingestion handled by the pipeline.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from .models import Universe

logger = logging.getLogger(__name__)

SITEMAP_INDEX_URLS = {
    Universe.one_piece: "https://onepiece.fandom.com/sitemap-newsitemapxml-index.xml",
    Universe.jujutsu_kaisen: "https://jujutsu-kaisen.fandom.com/sitemap-newsitemapxml-index.xml",
    Universe.one_punch_man: "https://onepunchman.fandom.com/sitemap-newsitemapxml-index.xml",
}

_INVALID_PREFIXES = (
    "File:",
    "User:",
    "Special:",
    "Category:",
    "Template:",
    "Help:",
    "Talk:",
)

_DROP_SELECTORS = [
    "aside",
    "header",
    "footer",
    "nav",
    "script",
    "style",
    "noscript",
    "figure",
    "table",
    "sup",
    ".mw-editsection",
    ".toc",
    ".portable-infobox",
    ".reference",
    ".reflist",
    ".navbox",
    ".metadata",
    ".categorylinks",
    ".comments",
    ".comment",
    ".wikia-ad",
    ".mcf-wrapper",
]


def _xml_namespace(root: ElementTree.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0].strip("{")
    return ""


def _find_all(root: ElementTree.Element, tag: str) -> List[ElementTree.Element]:
    ns = _xml_namespace(root)
    if ns:
        return list(root.findall(f".//{{{ns}}}{tag}"))
    return list(root.findall(f".//{tag}"))


def parse_sitemap_index(xml_text: str) -> List[str]:
    """Return NS_0 sitemap URLs from a sitemap index document."""
    root = ElementTree.fromstring(xml_text)
    locs = [loc.text.strip() for loc in _find_all(root, "loc") if loc.text]
    return [loc for loc in locs if "NS_0" in loc]


def parse_sitemap_urls(xml_text: str) -> List[str]:
    """Return all URLs listed in a sitemap document."""
    root = ElementTree.fromstring(xml_text)
    return [loc.text.strip() for loc in _find_all(root, "loc") if loc.text]


def is_article_url(url: str) -> bool:
    """Return True if the URL looks like a valid article page."""
    parsed = urlparse(url)
    if "/wiki/" not in parsed.path:
        return False
    slug = parsed.path.split("/wiki/")[-1]
    if any(slug.startswith(prefix) for prefix in _INVALID_PREFIXES):
        return False
    if parsed.query:
        if re.search(r"(?:^|[&])(action|oldid)=", parsed.query):
            return False
    return True


def filter_article_urls(urls: Iterable[str]) -> List[str]:
    """Filter and deduplicate article URLs."""
    seen = set()
    filtered: List[str] = []
    for url in urls:
        if not is_article_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        filtered.append(url)
    return filtered


def extract_markdown(html_text: str) -> str:
    """Extract the main article body and return Markdown."""
    soup = BeautifulSoup(html_text, "html.parser")
    container = soup.select_one("div.mw-parser-output")
    if container is None:
        return ""
    for selector in _DROP_SELECTORS:
        for element in container.select(selector):
            element.decompose()
    markdown = md(str(container), heading_style="ATX")
    return _clean_markdown(markdown)


def _clean_markdown(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.splitlines()]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


@dataclass
class FandomSitemapScraper:
    """Scraper that respects sitemap discovery and rate limiting."""

    request_delay: float = 2.0
    timeout: int = 30
    user_agent: str = "AnimeloreGraphBot/1.0"
    session: requests.Session = field(default_factory=requests.Session)
    _last_request: float = field(default=0.0, init=False)

    def _throttle(self) -> None:
        if self._last_request:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> str:
        self._throttle()
        response = self.session.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def discover_article_urls(self, index_url: str) -> List[str]:
        """Discover and filter article URLs from a sitemap index."""
        index_xml = self._get(index_url)
        sitemap_urls = parse_sitemap_index(index_xml)
        urls: List[str] = []
        for sitemap_url in sitemap_urls:
            try:
                sitemap_xml = self._get(sitemap_url)
            except requests.RequestException as exc:
                logger.warning("Failed to fetch sitemap %s: %s", sitemap_url, exc)
                continue
            urls.extend(parse_sitemap_urls(sitemap_xml))
        return filter_article_urls(urls)

    def fetch_article_markdown(self, url: str) -> str:
        """Fetch a wiki article and return cleaned Markdown."""
        try:
            html_text = self._get(url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch article %s: %s", url, exc)
            return ""
        return extract_markdown(html_text)
