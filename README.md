# BMW Press Release Fine-Tuning

Fine-tune a small LLM on BMW PressClub press releases. Config-driven
pipeline with MLflow experiment tracking, automated evaluation metrics,
and side-by-side sample generations.

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

## Preprocessing

Cleaning pipeline applied to all 1,429 articles before splitting:

| Step | What | Why |
|---|---|---|
| Contact block removal | Cut from last contact marker onward | Template boilerplate identical across hundreds of articles — no domain signal |
| URL/email/phone stripping | Conservative regex (only `https?://`, `www.`, `+\d` international format) | No semantic signal; conservative to avoid removing brand names like "Electrifying.com Awards" |
| Newline normalization | All `\n` → space, collapse multiple spaces | Newlines in web-scraped HTML are rendering artifacts, not semantic structure |
| Title prepending | `Title\nBody` format | Titles carry concentrated BMW vocabulary; single newline preserves semantic separation between title and body |

No minimum-length filter post-cleaning — all 1,429 articles preserved.

### Train/eval split

90/10 random split with seed 42. Small corpus (1,429 articles) — maximize
training data. 143 eval articles produce enough tokens for stable perplexity
measurement. A chronological split (train on older, evaluate on newer) would
better simulate real-world deployment and prevent temporal leakage, but risks
systematic topic imbalance in the eval set if recent months are dominated by
a single product launch.

### Preprocessing statistics

| Metric | Value |
|---|---|
| Articles cleaned | 1,429 |
| Contact blocks removed | 351 (25%) |
| Articles with URLs stripped | 267 |
| Articles with emails stripped | 47 |
| Articles with phones stripped | 47 |
| Train set | 1,286 articles |
| Eval set | 143 articles |

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

## Training

### Model selection

Qwen3-0.6B-Base — 596M parameters, 28 transformer layers, 896 hidden
dimensions. Released May 2025, current-generation architecture. Base model
(not instruct) because the task is domain adaptation via continued
pretraining, not instruction-following.

Full fine-tuning, not LoRA. At 0.6B parameters with bf16 precision, the
full model + optimizer states fit comfortably on a single RTX 4090 (24 GB).
LoRA would add adapter complexity without a VRAM-driven reason.

### Training method

SFTTrainer from `trl` with `packing=True`. Packing concatenates multiple
short articles into single 1024-token sequences, eliminating padding waste.
With a median article length of ~1,200 tokens and many articles well below
that, packing significantly improves GPU utilization compared to padding
each article individually.

All hyperparameters live in `configs/base.yaml`, with the option to pass a
custom config via CLI argument (`python scripts/train.py configs/custom.yaml`).
Config-driven pipeline: change a YAML value, get a different experiment.

### Training configuration

| Parameter | Value | Rationale |
|---|---|---|
| `per_device_train_batch_size` | 2 | Started at 4, hit OOM on backward pass — activation memory at seq_length=1024 with packing exceeded estimates |
| `gradient_accumulation_steps` | 8 | Maintains effective batch size of 16 (2 × 8) after reducing micro-batch size |
| `gradient_checkpointing` | false | Initially enabled after the OOM at batch_size=4. After reducing to batch_size=2, the model fits in 12.6 GB — well within the 24 GB budget. Disabling checkpointing recovered 13% training speed (6:06 vs 7:00 min) at the cost of ~4 GB more VRAM |
| `learning_rate` | 2e-4 | Standard for small-model full fine-tuning with AdamW |
| `lr_scheduler_type` | cosine | Gradual decay avoids abrupt learning rate drops |
| `warmup_ratio` | 0.05 | 5% of steps — stabilizes Adam moment estimates at the start |
| `max_seq_length` | 1024 | Covers the majority of articles; longer ones get truncated at training time |
| `num_train_epochs` | 3 | Small corpus — multiple passes needed for convergence |
| `eval_steps` | 25 | Refined from 50 after observing overfitting between evaluation points — finer granularity captured a better checkpoint |
| `bf16` | true | Native RTX 4090 precision, halves memory vs fp32 |

### Experiment tracking

MLflow tracks all training runs. The HuggingFace `MLflowCallback` handles
the run lifecycle and logs training metrics (loss, learning rate, gradient
norm, eval_loss) automatically. Post-training, the script reopens the same
MLflow run to log custom metrics (perplexity, inference throughput, GPU
memory), sample generations as a browsable table, the model artifact, and
the full config YAML.

`mlflow.enable_system_metrics_logging()` provides continuous GPU utilization,
memory, and power consumption charts throughout training — no custom
instrumentation needed.

### Overfitting and checkpoint selection

Eval loss drops steadily through epoch 1, continues improving into epoch 2,
then begins rising — classic overfitting on a small corpus. The config uses
`load_best_model_at_end: true` with `metric_for_best_model: eval_loss`,
so the trainer automatically loads the checkpoint with lowest eval loss
rather than the final (overfitting) weights.

Refining `eval_steps` from 50 to 25 captured a better checkpoint at the
minimum of the loss curve, improving perplexity from 10.46 to 9.96.

### VRAM utilization

Training peaked at 12.6 GB allocated out of 24 GB available. The progression:
batch_size=4 caused OOM on the backward pass, so micro-batch was reduced to 2.
With batch_size=2, gradient checkpointing brought memory down to 8.3 GB but
added ~30% compute overhead. Since 12.6 GB still leaves 11 GB headroom on a
24 GB card, gradient checkpointing was disabled — recovering 13% training
speed (6:06 vs 7:00 min) while staying well within the VRAM budget.

## Results

### Metrics

| Metric | Value |
|---|---|
| Base model eval loss | 3.04 |
| Base model perplexity | 20.85 |
| Fine-tuned eval loss | 2.30 |
| Fine-tuned perplexity | 9.96 |
| Perplexity reduction | 2.1× |
| Training time | 6:06 min (RTX 4090) |
| Inference throughput | 84.0 tokens/sec |
| Train peak GPU memory (allocated) | 12,566 MB |
| Train peak GPU memory (reserved) | 14,454 MB |
| Inference peak GPU memory (allocated) | 3,452 MB |

### Sample generations (base model → fine-tuned)

Six prompts tested before and after fine-tuning. The base model produces
generic, often nonsensical text (math problems, stock market calculations,
repetitive filler). The fine-tuned model generates BMW press release style
prose with correct product names, plant locations, technical specifications,
and corporate communication patterns.

Example — prompt: *"BMW Group reported in 2025 that"*

**Base model:** Generates a stock valuation exercise ("the company's share
price has grown by 22%... What is the current market value?") — completely
off-domain.

**Fine-tuned:** "BMW Group reported in 2025 that the company's global sales
in the Automotive Segment were down 1.4% on the previous year, despite the
continued strength of its brands and a strong product mix." — adopts BMW's
reporting style, quotes a board member by name and title, uses the
company's characteristic structure of headline → location → body → quote.

The fine-tuned model convincingly adopts BMW's writing style but hallucinates
factual details — invented sales figures, conflated model specifications,
fictional board member quotes. This is expected: the model learned the
*form* of BMW press releases (structure, vocabulary, tone) but has no
mechanism to verify facts. Domain adaptation shifts style, not knowledge.

Full side-by-side comparison in `results/sample_generations.json`.

## What I would investigate next

- **Flash Attention 2** — eliminates cross-attention contamination between
  packed articles (tokens currently attend across article boundaries without
  it) and improves training throughput
- **Model serving** — evaluate MLflow's built-in serving (`mlflow models
  serve`) against a dedicated vLLM endpoint for throughput-critical workloads
- **Chronological train/eval split** — better simulates real-world
  deployment (train on past, evaluate on future), at the cost of potential
  topic imbalance in the eval set

## Usage

**Requirements:** Python ≥ 3.10, CUDA-capable GPU with ≥ 16 GB VRAM
(trained on RTX 4090, CUDA 12.4). Scraping and preprocessing run on CPU.

```bash
pip install -e .
python scripts/scrape.py       # collect articles → data/raw/
python scripts/preprocess.py   # clean + split → data/processed/

pip install -e ".[train]"      # install training dependencies (transformers, trl, mlflow, etc.)
python scripts/train.py configs/base.yaml   # train on GPU → results/

mlflow ui                      # browse experiment runs → http://127.0.0.1:5000
```

### Testing

```bash
pytest tests/test_scrape.py              # unit tests — parsing, HTML cleaning
pytest tests/test_preprocess.py          # unit tests — cleaning, splitting
pytest tests/test_train.py               # unit tests — config, dataset format
pytest tests/test_scrape_integration.py  # integration tests — corpus quality
ruff check .                             # lint
```

**Scraper unit tests** (`test_scrape.py`): verify `clean_html` and `parse_feed_page` in
isolation — table stripping, entity decoding, date filtering,
short-article filtering, malformed XML handling.

**Preprocessing unit tests** (`test_preprocess.py`): 38 tests covering each
cleaning function in isolation — contact block cutting (including false
positives like "Corporate Communications" in job titles), URL/email/phone
stripping (preserving model numbers and fuel consumption data), whitespace
normalization, title prepending, and train/eval split (determinism,
no overlap, no data loss).

**Training unit tests** (`test_train.py`): config loading and validation
(required keys, types, data paths), dataset format checks (text field
presence, train/eval no overlap), flatten_dict utility.

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

**`data/processed/train.jsonl`** / **`eval.jsonl`** — cleaned articles, one per line:
```json
{"article_id": "T0407238EN", "text": "BMW announces a completely new Head-up Display...\nFull cleaned article text..."}
```

**`data/processed/stats.json`** — corpus statistics before/after cleaning.

**`results/metrics.json`** — evaluation metrics from the training run.

**`results/sample_generations.json`** — base model vs. fine-tuned generations
for six BMW-domain prompts.
