"""Unit tests for scrape.py parsing and cleaning logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scrape import clean_html, parse_feed_page

# ---------------------------------------------------------------------------
# clean_html
# ---------------------------------------------------------------------------

class TestCleanHtml:

    def test_strips_html_tags(self):
        assert clean_html("<p>BMW announces</p>") == "BMW announces"

    def test_removes_table_content(self):
        raw = (
            "<p>Introduction.</p>"
            "<table><tr><td>420 kW</td><td>4.6s</td></tr></table>"
            "<p>Conclusion.</p>"
        )
        result = clean_html(raw)
        assert "420 kW" not in result
        assert "Introduction." in result
        assert "Conclusion." in result

    def test_removes_images(self):
        raw = '<p>Text before.</p><img src="photo.jpg" alt="BMW iX"/><p>Text after.</p>'
        result = clean_html(raw)
        assert "img" not in result
        assert "photo.jpg" not in result
        assert "Text before." in result
        assert "Text after." in result

    def test_removes_script_and_style(self):
        raw = "<style>.cls{color:red}</style><p>Visible.</p><script>alert(1)</script>"
        assert clean_html(raw) == "Visible."

    def test_preserves_bare_urls(self):
        raw = "<p>More info at https://www.bmwgroup.com/innovation and here.</p>"
        result = clean_html(raw)
        assert "https://www.bmwgroup.com/innovation" in result

    def test_preserves_anchor_text(self):
        raw = '<p>Read the <a href="https://bmw.com/report">BMW Group Report</a> for details.</p>'
        result = clean_html(raw)
        assert "BMW Group Report" in result
        assert "https://bmw.com" not in result

    def test_decodes_html_entities(self):
        raw = "&lt;p&gt;BMW &amp; MINI&lt;/p&gt;"
        result = clean_html(raw)
        assert "BMW & MINI" in result
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_entities_fully_resolved_after_parsing(self):
        """html.unescape resolves &amp;lt; → &lt;, then BeautifulSoup decodes &lt; → <."""
        raw = "&amp;lt;strong&amp;gt;bold&amp;lt;/strong&amp;gt;"
        result = clean_html(raw)
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "bold" in result

    def test_normalizes_non_breaking_spaces(self):
        raw = "<p>BMW\xa0Group\xa0Press\xa0Release</p>"
        result = clean_html(raw)
        assert "\xa0" not in result
        assert "BMW Group Press Release" == result

    def test_collapses_multiple_spaces(self):
        raw = "<p>BMW    iX   xDrive50</p>"
        assert clean_html(raw) == "BMW iX xDrive50"

    def test_collapses_excessive_newlines(self):
        raw = "<p>First paragraph.</p><br><br><br><br><br><p>Second paragraph.</p>"
        result = clean_html(raw)
        assert "\n\n\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result


# ---------------------------------------------------------------------------
# parse_feed_page
# ---------------------------------------------------------------------------

def _build_rss(items: list[dict], has_next: bool = False) -> str:
    """Build a minimal RSS XML string for testing."""
    item_xml = ""
    for item in items:
        item_xml += (
            f'<item>'
            f'<guid>https://www.press.bmwgroup.com/global/article/detail/{item["id"]}</guid>'
            f'<title>{item.get("title", "Test Article Title Here")}</title>'
            f'<link>https://www.press.bmwgroup.com/global/article/detail/{item["id"]}</link>'
            f'<pubDate>{item.get("date", "Mon, 15 Jan 2024 10:00:00 GMT")}</pubDate>'
            f'<description>{item.get("text", "A" * 500)}</description>'
            f'</item>'
        )

    next_link = ""
    if has_next:
        next_link = '<atom:link rel="next" href="http://example.com/?page=2"/>'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">'
        f'<channel><title>BMW PressClub</title>{next_link}{item_xml}</channel>'
        '</rss>'
    )


class TestParseFeedPage:

    def test_parses_valid_article(self):
        xml = _build_rss([{"id": "T0441169EN"}])
        articles, has_next, raw_count = parse_feed_page(xml)
        assert len(articles) == 1
        assert articles[0]["article_id"] == "T0441169EN"
        assert raw_count == 1
        assert not has_next

    def test_detects_next_page(self):
        xml = _build_rss([{"id": "T0441169EN"}], has_next=True)
        _, has_next, _ = parse_feed_page(xml)
        assert has_next

    def test_filters_old_articles(self):
        xml = _build_rss([{
            "id": "T0441169EN",
            "date": "Tue, 15 Mar 2022 10:00:00 GMT",
        }])
        articles, _, raw_count = parse_feed_page(xml)
        assert len(articles) == 0
        assert raw_count == 1  # item exists but was filtered

    def test_filters_short_articles(self):
        xml = _build_rss([{
            "id": "T0441169EN",
            "text": "Attached please find the specifications.",
        }])
        articles, _, raw_count = parse_feed_page(xml)
        assert len(articles) == 0
        assert raw_count == 1

    def test_skips_invalid_guid(self):
        xml = _build_rss([{
            "id": "INVALID_ID",
            "title": "Valid title here",
        }])
        articles, _, raw_count = parse_feed_page(xml)
        assert len(articles) == 0

    def test_handles_malformed_xml(self):
        articles, has_next, raw_count = parse_feed_page("<not valid xml")
        assert articles == []
        assert not has_next
        assert raw_count == 0

    def test_keeps_recent_long_article(self):
        xml = _build_rss([{
            "id": "T0441169EN",
            "date": "Wed, 10 Jan 2024 10:00:00 GMT",
            "title": "BMW announces new electric vehicle platform",
            "text": "B" * 500,
        }])
        articles, _, _ = parse_feed_page(xml)
        assert len(articles) == 1
        assert articles[0]["date"] == "2024-01-10"
