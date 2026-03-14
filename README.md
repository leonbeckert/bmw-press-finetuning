# BMW Press Release Fine-Tuning

Fine-tune a small LLM on BMW PressClub press releases.

Status: **data collection complete** — preprocessing next.

## Data Pipeline

### Source

BMW PressClub Global RSS feeds (`www.press.bmwgroup.com/global/rss2/topic/`).
All 12 category feeds on the Global domain, paginated, ~20 articles per
page. A 13th feed (Rolls-Royce Motor Cars) is hosted on a separate domain
(`press.rolls-roycemotorcars.com`) and excluded. Only articles from 2023
onwards — older press releases use different formatting conventions and
vocabulary (pre-electrification era), which would add noise to domain
adaptation on current BMW communication style.

PressClub exposes two RSS endpoints — `/rss/` with teasers and `/rss2/` with
full article text in the `<description>` field. Verified that `/rss2/` content
matches the article detail page character-for-character.

### What gets extracted

`article_id`, `url`, `title`, `date`, `text`.

Excluded deliberately:
- **Tables** — spec sheets and timelines. Same key figures appear in the
  article body as prose ("delivers 420 kW and accelerates in 4.6 seconds").
  Keeping the tabular version would teach the model to reproduce layout
  markup instead of BMW's writing patterns.
- **Images** — press photos. Would need multimodal pipeline.
- **Links** — `<a>` tags rendered as plain text (anchor text kept, href dropped).

### Deduplication

PressClub categories are tags, not partitions — same article appears in
multiple feeds. Corpus stores each article exactly once in `articles.jsonl`.
Category tagging structure preserved separately in `article_categories.json`.

### Filtering

- **450-char minimum.** Below that: stub references to PDF attachments and
  quarterly reports ("Attached please find..."). Inspected the boundary by
  sampling articles in the 400-500 char range: found a real article at 391
  chars (MINI special edition blurb — confirmed on detail page, but
  generic marketing copy with no domain depth). First article with actual
  technical content at 462 chars (CES Head-up Display piece). 450 is the
  cleanest cut I found in the data.
- **No max-length cap.** Longest article is ~38k chars. Capping happens
  at training time, not here.

### Scraping behavior

- Sequential, 1.5s + random jitter between requests
- One-time job against a single host — no need to optimize for speed,
  polite scraping avoids rate-limiting
- Exponential backoff on errors (3 retries)
- Early stop after 3 consecutive pages with no new articles
- Full run: ~9 minutes, ~280 requests

## Data Verification

### Manual checks

- Compared 12 articles against `press.bmwgroup.com` website — prose content
  matches after cleaning (whitespace normalization, table removal)
- Sampled filtered articles near the 450-char boundary — confirmed
  stub pattern (PDF/media attachment pointers, no article text)
- Spot-checked cross-category overlap: BMW articles appear in Brands feed
  as expected, category index captures both

### Corpus statistics

| Metric | Value |
|---|---|
| Unique articles (2023+) | 1,429 |
| Categories | 12 |
| Articles in 2+ categories | 399 (28%) |
| Date range | 2023-01-05 to 2026-03-12 |
| Text length (min) | 462 chars |
| Text length (median) | 4,910 chars |
| Text length (max) | 37,727 chars |
| Total corpus size | 8.5M chars (~2.1M tokens) |

## Usage

```bash
pip install -e .
python scripts/scrape.py
```

### Testing

```bash
pytest tests/test_scrape.py              # unit tests — parsing, HTML cleaning
pytest tests/test_scrape_integration.py  # integration tests — corpus quality
ruff check .                             # lint
```

**Unit tests** (`test_scrape.py`): verify `clean_html` and `parse_feed_page` in
isolation — table stripping, entity decoding, date filtering,
short-article filtering, malformed XML handling.

**Integration tests** (`test_scrape_integration.py`): run against the actual
scraped corpus. 12 articles sampled across the full length spectrum
(462–37,727 chars) checked for: no residual HTML tags, no unescaped entities,
no excessive whitespace, text length within expected range, correct dates,
valid URL format. Also verifies corpus-level invariants: no duplicate IDs, all
dates after 2023 cutoff, corpus and category index in sync.

### Output

**`data/raw/articles.jsonl`** — deduplicated corpus, one article per line:
```json
{
  "article_id": "T0407238EN",
  "url": "https://www.press.bmwgroup.com/global/article/detail/T0407238EN",
  "title": "BMW announces a completely new Head-up Display for the Neue Klasse.",
  "date": "2023-01-05",
  "text": "Full article text..."
}
```

**`data/raw/article_categories.json`** — maps article IDs to category lists:
```json
{
  "T0407238EN": ["corporate", "technology", "motorshows"],
  "T0407845EN": ["corporate", "technology", "people"]
}
```

## What's next

- [ ] Preprocessing: train/eval split, tokenization
- [ ] Model selection + training config
- [ ] Training + evaluation metrics
- [ ] Results + model comparison
