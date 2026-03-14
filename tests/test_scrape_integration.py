"""Integration tests: verify scraped corpus quality on real articles.

Tests 12 articles across the full length spectrum (462–37,727 chars)
against structural invariants that must hold for any correctly scraped
BMW PressClub article.
"""

import json
import re
from pathlib import Path

import pytest

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "articles.jsonl"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "article_categories.json"

# 12 articles covering shortest → P10 → P25 → median → P75 → P90 → longest
SAMPLE_ARTICLES = {
    # Shortest — boundary articles just above 450-char filter
    "T0407238EN": {"min_len": 400, "max_len": 600, "date": "2023-01-05"},
    "T0449574EN": {"min_len": 400, "max_len": 750, "date": "2025-04-19"},
    "T0421772EN": {"min_len": 400, "max_len": 600, "date": "2023-06-21"},
    # P10–P25
    "T0446913EN": {"min_len": 1800, "max_len": 3000, "date": "2024-12-13"},
    "T0441049EN": {"min_len": 2800, "max_len": 4500, "date": "2024-04-11"},
    # Median
    "T0441150EN": {"min_len": 4000, "max_len": 6000, "date": "2024-04-18"},
    "T0447420EN": {"min_len": 4000, "max_len": 6000, "date": "2025-01-12"},
    # P75–P90
    "T0450061EN": {"min_len": 5500, "max_len": 8500, "date": "2025-05-10"},
    "T0455249EN": {"min_len": 8500, "max_len": 13000, "date": "2026-01-25"},
    # Longest — full model launches and annual conference statements
    "T0410918EN": {"min_len": 25000, "max_len": 35000, "date": "2023-03-15"},
    "T0439260EN": {"min_len": 30000, "max_len": 42000, "date": "2024-01-31"},
    "T0433118EN": {"min_len": 32000, "max_len": 45000, "date": "2023-09-01"},
}

HTML_TAG_RE = re.compile(r"<[^>]+>")


@pytest.fixture(scope="module")
def corpus() -> dict[str, dict]:
    if not CORPUS_PATH.exists():
        pytest.skip("corpus not found — run scraper first")
    articles = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            articles[a["article_id"]] = a
    return articles


@pytest.fixture(scope="module")
def category_index() -> dict[str, list[str]]:
    if not INDEX_PATH.exists():
        pytest.skip("category index not found — run scraper first")
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Corpus-level checks
# ---------------------------------------------------------------------------

class TestCorpusIntegrity:

    def test_all_sample_articles_present(self, corpus):
        missing = [aid for aid in SAMPLE_ARTICLES if aid not in corpus]
        assert not missing, f"missing articles: {missing}"

    def test_no_duplicate_ids(self):
        if not CORPUS_PATH.exists():
            pytest.skip("corpus not found")
        ids = []
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                ids.append(json.loads(line)["article_id"])
        assert len(ids) == len(set(ids)), "duplicate article IDs in corpus"

    def test_corpus_and_index_match(self, corpus, category_index):
        corpus_ids = set(corpus.keys())
        index_ids = set(category_index.keys())
        assert corpus_ids == index_ids, (
            f"corpus has {len(corpus_ids - index_ids)} articles not in index, "
            f"index has {len(index_ids - corpus_ids)} articles not in corpus"
        )

    def test_all_dates_after_cutoff(self, corpus):
        old = [a["article_id"] for a in corpus.values() if a["date"] < "2023-01-01"]
        assert not old, f"articles before 2023: {old[:5]}"


# ---------------------------------------------------------------------------
# Per-article checks across the length spectrum
# ---------------------------------------------------------------------------

class TestArticleQuality:

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_required_fields(self, corpus, article_id):
        a = corpus[article_id]
        for field in ("article_id", "url", "title", "date", "text"):
            assert field in a, f"missing field: {field}"
            assert a[field], f"empty field: {field}"

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_no_html_tags(self, corpus, article_id):
        text = corpus[article_id]["text"]
        match = HTML_TAG_RE.search(text)
        assert not match, f"HTML tag found: {match.group()!r}"

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_no_html_entities(self, corpus, article_id):
        text = corpus[article_id]["text"]
        assert "&amp;" not in text, "unescaped &amp;"
        assert "&lt;" not in text, "unescaped &lt;"
        assert "&gt;" not in text, "unescaped &gt;"
        assert "&nbsp;" not in text, "unescaped &nbsp;"

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_no_excessive_whitespace(self, corpus, article_id):
        text = corpus[article_id]["text"]
        assert "\n\n\n" not in text, "triple newline found"
        assert "  " not in text, "double space found"

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_text_length_in_range(self, corpus, article_id):
        text = corpus[article_id]["text"]
        expected = SAMPLE_ARTICLES[article_id]
        length = len(text)
        assert expected["min_len"] <= length <= expected["max_len"], (
            f"{article_id}: {length} chars, expected {expected['min_len']}–{expected['max_len']}"
        )

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_date_matches(self, corpus, article_id):
        assert corpus[article_id]["date"] == SAMPLE_ARTICLES[article_id]["date"]

    @pytest.mark.parametrize("article_id", SAMPLE_ARTICLES.keys())
    def test_url_format(self, corpus, article_id):
        url = corpus[article_id]["url"]
        assert url.startswith("https://www.press.bmwgroup.com/"), f"unexpected URL: {url}"
        assert article_id in url


# ---------------------------------------------------------------------------
# Category index checks
# ---------------------------------------------------------------------------

class TestCategoryIndex:

    def test_all_categories_valid(self, category_index):
        valid = {
            "corporate", "brands", "bmw", "mini", "technology",
            "people", "heritage", "motorshows", "motorrad",
            "bmw-i", "bmw-m", "sports",
        }
        for aid, cats in category_index.items():
            for cat in cats:
                assert cat in valid, f"{aid} has unknown category: {cat}"

    def test_every_article_has_category(self, category_index):
        empty = [aid for aid, cats in category_index.items() if not cats]
        assert not empty, f"articles without categories: {empty[:5]}"

    def test_multi_category_articles_exist(self, category_index):
        multi = sum(1 for cats in category_index.values() if len(cats) > 1)
        assert multi > 0, "no multi-category articles found"
