"""
Tests for animelore/scraper.py
"""
from animelore.scraper import (
    extract_markdown,
    filter_article_urls,
    is_article_url,
    parse_sitemap_index,
    parse_sitemap_urls,
)


def test_parse_sitemap_index_filters_ns0():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://example.com/sitemap-NS_0.xml</loc></sitemap>
        <sitemap><loc>https://example.com/sitemap-NS_1.xml</loc></sitemap>
    </sitemapindex>
    """
    assert parse_sitemap_index(xml) == ["https://example.com/sitemap-NS_0.xml"]


def test_parse_sitemap_urls():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/wiki/Page_1</loc></url>
        <url><loc>https://example.com/wiki/Page_2</loc></url>
    </urlset>
    """
    assert parse_sitemap_urls(xml) == [
        "https://example.com/wiki/Page_1",
        "https://example.com/wiki/Page_2",
    ]


def test_article_url_filtering():
    urls = [
        "https://example.com/wiki/Page_1",
        "https://example.com/wiki/File:Image.png",
        "https://example.com/wiki/User:Someone",
        "https://example.com/wiki/Page_2?action=edit",
        "https://example.com/wiki/Page_3?oldid=123",
    ]
    assert filter_article_urls(urls) == ["https://example.com/wiki/Page_1"]
    assert is_article_url("https://example.com/wiki/Page_1")
    assert not is_article_url("https://example.com/wiki/File:Image.png")


def test_extract_markdown_strips_boilerplate():
    html = """
    <html>
      <body>
        <div class="mw-parser-output">
          <header>Header</header>
          <p>Main content.</p>
          <aside>Ad</aside>
        </div>
      </body>
    </html>
    """
    markdown = extract_markdown(html)
    assert "Main content." in markdown
    assert "Header" not in markdown
    assert "Ad" not in markdown
