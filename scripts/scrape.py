"""Scrape recent BMW PressClub press releases into a deduplicated corpus.

Fetches articles from 2023 onwards across all 12 BMW PressClub RSS
category feeds. Extracts body text only — no tables, images, or scripts.

Categories are tags, not partitions — the same article appears in
multiple feeds. The corpus stores each article exactly once. The
category index preserves the full tagging structure separately.

Output:
    data/raw/articles.jsonl          — deduplicated corpus, one article per line
    data/raw/article_categories.json — article_id → list of categories
    logs/scrape-{timestamp}.log

Usage:
    python scripts/scrape.py
"""

import html
import json
import logging
import random
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.press.bmwgroup.com/global/rss2/topic"
ARTICLE_ID_RE = re.compile(r"T\d{7}[A-Z]{2}")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DELAY = 1.5
MAX_RETRIES = 3
MIN_TEXT_LENGTH = 450
EARLIEST_DATE = "2023-01-01"

TOPICS = {
    "803": "corporate",
    "4099": "brands",
    "4100": "bmw",
    "5127": "mini",
    "5236": "technology",
    "5249": "people",
    "5254": "heritage",
    "4098": "motorshows",
    "6629": "motorrad",
    "6728": "bmw-i",
    "7309": "bmw-m",
    "10837": "sports",
}

OUTPUT_DIR = Path("data/raw")
LOG_DIR = Path("logs")
CORPUS_PATH = OUTPUT_DIR / "articles.jsonl"
INDEX_PATH = OUTPUT_DIR / "article_categories.json"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BMW-PressClub-Scraper/1.0 (research)"})

log = logging.getLogger("scraper")


def setup_logging() -> Path:
    """Configure logging to both stderr and a timestamped log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"scrape-{run_ts}.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    log.addHandler(stderr_handler)

    return log_path


def clean_html(raw: str) -> str:
    """Decode HTML entities, strip all tags, normalize whitespace."""
    decoded = html.unescape(raw)
    soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup.find_all(["img", "script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_feed_page(xml_text: str) -> tuple[list[dict], bool, int]:
    """Parse one RSS page into article dicts.

    Returns (articles, has_next_page, raw_item_count).
    raw_item_count is the number of <item> elements before filtering.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("XML parse error: %s", e)
        return [], False, 0

    channel = root.find("channel")
    if channel is None:
        return [], False, 0

    has_next = any(
        link.get("rel") == "next"
        for link in channel.findall("atom:link", ATOM_NS)
    )

    items = channel.findall("item")
    articles = []
    for item in items:
        guid = item.findtext("guid", "")
        match = ARTICLE_ID_RE.search(guid)
        if not match:
            continue
        article_id = match.group(0)

        pub_date = item.findtext("pubDate", "")
        try:
            dt = parsedate_to_datetime(pub_date)
        except (ValueError, TypeError):
            continue
        date = dt.strftime("%Y-%m-%d")

        if date < EARLIEST_DATE:
            continue

        title = (item.findtext("title", "") or "").strip()
        link = (item.findtext("link", "") or "").split("?")[0]
        description = item.findtext("description", "") or ""
        text = clean_html(description)

        if len(text) < MIN_TEXT_LENGTH:
            preview = text[:80].replace("\n", " ")
            log.debug("skip %s: %d chars | %s", article_id, len(text), preview)
            continue

        articles.append({
            "article_id": article_id,
            "url": link,
            "title": title,
            "date": date,
            "text": text,
        })

    return articles, has_next, len(items)


def scrape_topic(
    topic_id: str, category: str, corpus: dict[str, dict], index: dict[str, list[str]]
) -> int:
    """Scrape all pages of one RSS topic feed.

    Adds new articles to corpus dict. Appends category to index for
    every encounter, including articles already in the corpus.
    """
    written = 0
    page = 1
    retries = 0
    empty_streak = 0

    while True:
        url = f"{BASE_URL}/{topic_id}?page={page}"

        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            retries += 1
            log.warning("[%s] page %d error (%d/%d): %s",
                        category, page, retries, MAX_RETRIES, e)
            if retries >= MAX_RETRIES:
                log.error("[%s] giving up on page %d after %d retries",
                          category, page, MAX_RETRIES)
                break
            time.sleep(DELAY * (2 ** retries))
            continue
        retries = 0

        body = resp.text
        stripped = body.lstrip()
        if not stripped.startswith("<?xml") and not stripped.startswith("<rss"):
            log.debug("[%s] page %d: end of feed (non-XML response)", category, page)
            break

        articles, has_next, raw_count = parse_feed_page(body)
        if raw_count == 0:
            log.debug("[%s] page %d: no items in feed", category, page)
            break

        page_new = 0
        for article in articles:
            aid = article["article_id"]

            # Update category index for every encounter
            if aid not in index:
                index[aid] = []
            if category not in index[aid]:
                index[aid].append(category)

            # Add to corpus only on first encounter
            if aid not in corpus:
                corpus[aid] = article
                written += 1
                page_new += 1

        log.info("[%s] page %d: %d new (%d total)", category, page, page_new, written)

        if page_new > 0:
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 3:
                log.info("[%s] stopping early: %d consecutive pages with no new articles",
                         category, empty_streak)
                break

        if not has_next:
            break

        page += 1
        time.sleep(DELAY + random.uniform(0, 0.5))

    log.info("[%s] done: %d new articles", category, written)
    return written


def write_outputs(corpus: dict[str, dict], index: dict[str, list[str]]):
    """Write corpus and category index to disk."""
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for article in sorted(corpus.values(), key=lambda a: a["article_id"]):
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    log.info("wrote %d articles to %s", len(corpus), CORPUS_PATH)
    log.info("wrote %d entries to %s", len(index), INDEX_PATH)


def main():
    log_path = setup_logging()
    log.info("BMW PressClub scraper started")
    log.info("log file: %s", log_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus: dict[str, dict] = {}
    index: dict[str, list[str]] = {}

    total_new = 0
    first_seen = {}
    for topic_id, category in TOPICS.items():
        count = scrape_topic(topic_id, category, corpus, index)
        first_seen[category] = count
        total_new += count

    write_outputs(corpus, index)

    # Run summary
    log.info("--- run summary (first-seen per feed) ---")
    for cat, count in first_seen.items():
        log.info("  %-14s %d first-seen", cat, count)
    log.info("  total new: %d | corpus size: %d", total_new, len(corpus))

    lengths = [len(a["text"]) for a in corpus.values()]
    if lengths:
        dates = sorted(a["date"] for a in corpus.values())
        total_chars = sum(lengths)
        multi_tag = sum(1 for cats in index.values() if len(cats) > 1)
        log.info("--- data quality ---")
        log.info("  text length : min %d | median %d | max %d chars",
                 min(lengths), int(statistics.median(lengths)), max(lengths))
        log.info("  date range  : %s to %s", dates[0], dates[-1])
        log.info("  corpus size : %s chars | ~%dk tokens",
                 f"{total_chars:,}", total_chars // 4000)
        log.info("  articles in 2+ categories: %d (%.0f%%)",
                 multi_tag, multi_tag / len(index) * 100)


if __name__ == "__main__":
    main()
