# Phased implementation plan

Each phase was implemented and its tests were green before starting the next.

1. **Scaffold** — repo layout, `.env.example`, `.gitignore`, `requirements.txt`
   (Python 3.12 venv; added `google-play-scraper`, `langdetect`, `google-genai`,
   `plotly`, which the original scaffold was missing).
2. **Scrapers** — real Google Play (`google-play-scraper`), App Store (Apple
   customer-reviews RSS), and Spotify Community (Khoros LiQL API) scrapers,
   each verified against the live API before writing code, with unit tests
   mocking the network layer for pagination, malformed fields, and failures.
3. **Merge / clean / recency** — `analysis/merge.py` (unified schema, dedup,
   language detection, source/date-range/% non-English summary) wired to
   `analysis/cleaning.py`'s recency filter (24-month window for Play/App
   Store; vote-weighted exception for Community items).
4. **LLM wrapper** — `analysis/llm_client.py`: one Gemini (`gemini-2.5-flash`)
   wrapper with retry/backoff and on-disk, row-id-keyed response caching, so
   the provider is swappable and re-runs never re-pay for already-tagged rows.
5. **Tagging** — `analysis/tagging.py`: batched relevance filtering and
   structured tagging (sentiment/theme/job-to-be-done/segment/severity) on top
   of the LLM wrapper, with enum validation so malformed model output can't
   corrupt the schema.
6. **Aggregation** — `analysis/aggregate.py`: theme/segment frequency tables
   (community items weighted by votes), ranked unmet needs, representative
   quotes, and Gemini-generated answers to the six brief questions, grounded
   only in the aggregated tables.
7. **Dashboard** — `app.py` (Streamlit): Overview, the six questions, Themes,
   Segments, and Unmet needs sections, reading only saved data under `data/`.
8. **End-to-end verification** — full pipeline run against live sources,
   README rewrite with real per-source counts and stated limitations.

See `README.md` for the testing table (what was covered at each phase) and the
current real per-source scrape counts.
