"""Preprocess BMW press releases: clean text, split into train/eval sets.

Reads raw articles from data/raw/articles.jsonl, applies text cleaning
(contact block removal, URL/email/phone stripping, whitespace normalization),
prepends titles, and writes train/eval splits.

Output:
    data/processed/train.jsonl  — 90% of articles
    data/processed/eval.jsonl   — 10% of articles
    data/processed/stats.json   — corpus statistics before/after cleaning

Usage:
    python scripts/preprocess.py
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TRAIN_RATIO = 0.9

RAW_PATH = Path("data/raw/articles.jsonl")
OUTPUT_DIR = Path("data/processed")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
EVAL_PATH = OUTPUT_DIR / "eval.jsonl"
STATS_PATH = OUTPUT_DIR / "stats.json"

CONTACT_MARKERS = [
    "in case of queries",
    "if you have any questions",
    "for further information please contact",
    "in the event of enquiries",
]

# Catches http(s):// and www. prefixed URLs. Bare domains without prefix
# (e.g. "ran.de", "lifestyle.bmw.com", "Amazon.com") are intentionally NOT
# matched — ~54 occurrences across 1429 articles, most are brand names
# ("Electrifying.com Awards") or inline media references ("shown on ran.de")
# where a broader regex would risk removing legitimate text.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+\d[\d\s\-()]{7,}")


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def cut_contact_block(text: str) -> str:
    """Remove everything from the last occurrence of a contact marker onward."""
    lower = text.lower()
    cut_pos = -1
    for marker in CONTACT_MARKERS:
        pos = lower.rfind(marker)
        if pos != -1:
            if cut_pos == -1 or pos < cut_pos:
                cut_pos = pos
    if cut_pos != -1:
        return text[:cut_pos]
    return text


def strip_urls(text: str) -> str:
    """Remove bare URLs and www-prefixed URLs."""
    return URL_RE.sub("", text)


def strip_emails(text: str) -> str:
    """Remove email addresses."""
    return EMAIL_RE.sub("", text)


def strip_phones(text: str) -> str:
    """Remove phone numbers (international format)."""
    return PHONE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Replace newlines with spaces and collapse all whitespace.

    Newlines in scraped HTML are rendering artifacts, not semantic
    structure — we normalize everything to continuous text.
    """
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def prepend_title(title: str, text: str) -> str:
    """Combine title and body text."""
    return f"{title}\n{text}"


def clean_article(title: str, text: str) -> str:
    """Apply full cleaning pipeline to one article. Returns cleaned text with title."""
    text = cut_contact_block(text)
    text = strip_urls(text)
    text = strip_emails(text)
    text = strip_phones(text)
    text = normalize_whitespace(text)
    return prepend_title(title, text)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_articles(path: Path) -> list[dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def process_corpus(articles: list[dict]) -> tuple[list[dict], dict]:
    """Clean all articles, return processed articles and stats."""
    raw_count = len(articles)
    raw_lengths = [len(a["text"]) for a in articles]

    contact_cut_count = 0
    url_count = 0
    email_count = 0
    phone_count = 0
    processed = []

    for a in articles:
        text = a["text"]
        title = a["title"]

        # Track which steps affect this article
        after_contact = cut_contact_block(text)
        if after_contact != text:
            contact_cut_count += 1

        after_urls = strip_urls(after_contact)
        if after_urls != after_contact:
            url_count += 1

        after_emails = strip_emails(after_urls)
        if after_emails != after_urls:
            email_count += 1

        after_phones = strip_phones(after_emails)
        if after_phones != after_emails:
            phone_count += 1

        cleaned_body = normalize_whitespace(after_phones)
        full_text = prepend_title(title, cleaned_body)

        processed.append({
            "article_id": a["article_id"],
            "text": full_text,
        })

    after_lengths = [len(a["text"]) for a in processed]

    stats = {
        "raw_articles": raw_count,
        "final_articles": len(processed),
        "cleaning_applied": {
            "contact_blocks_cut": contact_cut_count,
            "urls_stripped": url_count,
            "emails_stripped": email_count,
            "phones_stripped": phone_count,
        },
        "text_stats": {
            "before": {
                "min": min(raw_lengths),
                "median": int(statistics.median(raw_lengths)),
                "max": max(raw_lengths),
                "total_chars": sum(raw_lengths),
            },
            "after": {
                "min": min(after_lengths),
                "median": int(statistics.median(after_lengths)),
                "max": max(after_lengths),
                "total_chars": sum(after_lengths),
            },
        },
    }

    return processed, stats


def split_train_eval(
    articles: list[dict], seed: int = RANDOM_SEED, ratio: float = TRAIN_RATIO
) -> tuple[list[dict], list[dict]]:
    """Randomly split articles into train and eval sets."""
    shuffled = list(articles)
    random.seed(seed)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def write_jsonl(path: Path, articles: list[dict]):
    """Write articles to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")


def main():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found. Run scrape.py first.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    articles = load_articles(RAW_PATH)
    print(f"Loaded {len(articles)} articles from {RAW_PATH}")

    # Clean
    processed, stats = process_corpus(articles)
    print(f"After cleaning: {len(processed)} articles")

    # Split
    train, eval_ = split_train_eval(processed)
    stats["train_articles"] = len(train)
    stats["eval_articles"] = len(eval_)
    stats["split"] = {
        "method": "random",
        "seed": RANDOM_SEED,
        "ratio": "90/10",
    }
    print(f"Split: {len(train)} train / {len(eval_)} eval")

    # Write
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(EVAL_PATH, eval_)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Written to {OUTPUT_DIR}/")
    print(f"  train.jsonl: {len(train)} articles")
    print(f"  eval.jsonl:  {len(eval_)} articles")
    print("  stats.json:  corpus statistics")


if __name__ == "__main__":
    main()
