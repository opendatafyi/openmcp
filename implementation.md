# Implementation Plan: EmbeddingGemma 2 + Catalogue Expansion

> **Status:** Revised after review (v2)
> **Repository:** `opendatafyi/openmcp`
> **Scope:** Build and validate everything locally (MCP server, semantic index,
> retrieval quality, new sources).
> **Later:** Deploy to Fly.io once local work is done. Deployment is deferred, but
> local choices must not block it (see §2a).

---

## 0. Review summary (what changed from v1)

Each claim below was checked against the code, `catalog.duckdb`, and the
`google/embeddinggemma-2` model card.

| v1 claim | Finding | Fix |
|---|---|---|
| Truncate to **384 dims** via MRL, keep `FLOAT[384]`, "zero migration" | Model card: supported MRL dims are **768, 512, 256, 128**. 384 is not a trained point. Every vector must be regenerated anyway, since bge and Gemma vectors are incompatible. | Pick 768 or 256, and parameterise the dimension in `store.py` |
| Document prefix `"title: "`, plus a custom query instruction | Model card format: query `task: search result \| query: {q}`, document `title: {title} \| text: {doc}` | Use the documented prompts |
| 512-token truncation drops a lot of metadata | Measured title+notes+topic length: p90 is ~1,200 chars (~300 tokens). Only **~2.8%** of federal/Ontario records and **~0.1%** of Alberta records exceed ~500 tokens. | The 8K window only pays off **after** enrichment (Phase 3). Gains will come from multilinguality and model quality. |
| "Only parent titles are embedded" | `doc_text` already has title, notes, keywords, publisher, and topic | Corrected wording |
| Add StatCan dimensions to the index | Already done in `process_statcan_cube` (dimension names go into notes and keywords) | Dropped. Member labels would need ~8,215 `getCubeMetadata` calls, so they're optional. |
| FTS index on `keywords` column | `datasets` has **no `keywords` column** (schema: id, title, org, notes, topic, resources_json, …) | Index title/notes/topic/org, or add a keywords column |
| Remove GEOJSON from `EXCLUDED_FORMATS` | GEOJSON isn't in that set. It's missing from the allowlist. | Corrected |
| Hybrid search in under 40 ms | Live CKAN calls stay in the path, because README says they cover records the index doesn't have. Latency depends on the slowest portal (`HTTP_TIMEOUT = 20`). | Dropped the target. Added a per-portal search timeout instead. |
| New portals add "~5,500 curated tabular datasets" | 3,364 / 1,611 / 555 are **raw package counts**. Many BC packages are WMS/geospatial and will fail `resource_policy`. | Measure the queryable counts before quoting numbers |
| PR #1 "merged" | GitHub showed PR #1 closed without merge (2026-09-04). | **Resolved:** owner approved. It's merged into local `main` (fast-forward, 44 tests pass). Not pushed yet. |
| No evaluation step | Without a benchmark, we can't show Gemma 2 beats bge. Telemetry doesn't log query text, so real queries aren't available. | Added **Phase 0: eval harness** |

---

## 1. Goals

1. Replace `BAAI/bge-small-en-v1.5` (33M, English-only, 512 tokens) with
   `google/embeddinggemma-2` (270M text backbone, 100+ languages, 8K tokens,
   Apache 2.0) and **prove** it improves retrieval.
2. Use the larger context to enrich what gets embedded per dataset.
3. Add local keyword (BM25) retrieval to the existing RRF fusion.
4. Expand sources (BC, Toronto, Québec; BoC discoverability).

---

## 2a. Deployment-readiness constraints (apply during local work)

No Fly work now. These rules keep a later deploy simple:

- **Transport-agnostic server:** don't put stdio-specific assumptions in tool
  code. A later `MCP_TRANSPORT=sse|streamable-http` switch should be the only change.
- **Load the model once per process,** not per call. That's cheap for a
  long-lived hosted process and keeps it tolerable for local stdio sessions.
- **Read-only catalog at query time:** all writes (upserts, FTS index) happen in
  `build_index.py`, so the built `catalog.duckdb` can be baked into an image as is.
- **No runtime network installs:** DuckDB extensions (`fts`, `spatial`) and model
  weights must be installable ahead of time (e.g., a `setup` step), not only on first query.
- **Record resource numbers during Phase 1:** model RSS, cold-start time, and
  per-query CPU latency. These numbers will size the Fly machine later.

## 2. Phases (ordered; each phase is gated on the eval)

### Phase 0 — Evaluation harness (do first)

- `semantic/eval/queries.jsonl` with 60–100 hand-written cases:
  `{query, expected_ids[], lang, category}`.
  - Mix: layman phrasing vs. official jargon, acronyms/codes (NAICS,
    CANSIM IDs), French queries, StatCan vs. CKAN, single-portal intent.
- `semantic/eval/run_eval.py`: reports Recall@5, Recall@10, and MRR for
  (a) vector-only and (b) full hybrid `semantic_search_datasets` (with live
  CKAN mocked or disabled so runs are reproducible).
- Record the **bge baseline** before changing anything.
- Each later phase merges only if it doesn't regress overall and improves
  its target category.

### Phase 1 — Model swap behind a catalog contract

**`semantic/embed.py`**
- `MODEL_NAME = "google/embeddinggemma-2"`, loaded via `sentence-transformers`.
- Queries: `prompt_name="SearchQuery"` (`task: search result | query: …`).
- Documents: format manually as `title: {title} | text: {body}`
  (`prompt_name="Document"` applies `title: none`, which wastes the title).
- `truncate_dim=EMBED_DIM`, `normalize_embeddings=True`.
- Device: MPS → CUDA → CPU.

**`semantic/store.py`**
- Replace the hard-coded `FLOAT[384]` (in `init_db`, `save_datasets`, `top_k`)
  with `FLOAT[{EMBED_DIM}]`.
- New `catalog_meta` table: `embed_model`, `embed_dim`, `prompt_version`,
  `built_at`, `sources`.

**`mcp_server.py`**
- At startup, compare `catalog_meta` with `embed.py`. If they don't match, return
  a `CatalogModelMismatch` error with recovery text. Users download
  `catalog.duckdb` from releases, so an old catalog with new code would
  otherwise return **silently wrong** results.
- Update the docstring that mentions bge-small.

**Dimension decision:** locked to **768** dimensions per user instruction. Parameterise `store.py` with `EMBED_DIM = 768`.

**Dependency / UX costs to measure and document:**
- Query-time `torch` becomes a hard dependency (today the query path is
  fastembed/ONNX only). That means a much larger install.
- Cold start: stdio servers load the model per client session. Measure load
  time and RSS on CPU and MPS.
- Per-query CPU latency vs. bge.
- Check whether an ONNX/quantised export exists so the no-torch install path survives.
- Full build time: measure on `--limit 2000` and extrapolate. Don't quote
  estimates until then.

Exit criteria: beats the bge baseline on Phase 0 metrics, and cold start and
query latency are acceptable.

### Phase 2 — Bilingual metadata

- `process_dataset_dict`: include FR title/notes alongside EN when present
  (`title_translated.fr`, `notes_translated.fr`), instead of `_get_english()` only.
- Respect `SourceConfig.language_priority` for display fields. Today it's
  ignored and English is always preferred.
- Eval target: French-query category.

### Phase 3 — Enrichment (this is where 8K context matters)

- Resource lines in `doc_text`: name, format, and **description**. Description is
  not extracted today; `process_dataset_dict` keeps only id/name/format/url/
  datastore_active.
- Cap enriched docs (e.g., 1,024–2,048 tokens). Build cost grows with sequence
  length, and most records are short anyway.
- Optional, measured separately: StatCan dimension member labels via
  `getCubeMetadata` (8,215 calls, so cache them).

### Phase 3b — Content enrichment (schemas, PDFs, map previews)

Census of the federal dump (`od-do-canada.jsonl.gz`, 47,731 datasets;
Alberta/Ontario/BC not counted yet):

| Asset | Federal datasets | Modality | Value |
|---|---:|---|---|
| CSV/XLSX/datastore column headers | ~15.7k CSV + 2.2k XLSX | text/code | **Highest:** enables "which dataset has a column for X" |
| PDF reports and guides | 8,262 (32k PDFs) | text, plus vision for scanned/figure pages | High |
| Map previews rendered from WMS / ESRI REST | 1,225 + 2,646 | vision | Medium: geo datasets with thin text |
| Existing images (JPG/PNG/GIF) | ~850 | vision | Low |
| JP2/GeoTIFF rasters | ~1.8k | vision (after downsampling) | Low: large imagery files |
| Audio/video | 2 (AVI) | audio | **Skip** |

Design:
- New `semantic/enrich.py`, separate from `build_index.py`. Fetches are
  resumable and cached on disk (`semantic/enrich_cache/`), rate-limited, with a
  per-asset size cap.
  - CSV header: HTTP Range for the first ~64 KB.
  - XLSX: the header row of each sheet.
  - Datastore: `datastore_search?limit=0`.
  - PDF: text from the first N pages; page images only when there's no text layer.
  - Maps: WMS `GetMap` / ESRI `export?f=image` at ~512 px.
- New table `dataset_assets(asset_id, dataset_id, kind, source_url, text,
  embedding FLOAT[D])`, with multiple vectors per dataset.
- Search: score the query against `dataset_assets`, collapse to the best asset
  per dataset, and fuse that as another RRF list. Return a "matched via" hint
  (e.g., `column AIR50P`, `PDF p.3`, `map preview`).
- Query-time cost is unchanged. Queries are text, so only the text backbone
  loads at runtime. The vision encoder is needed **only at build time**, which
  matters for Fly sizing later.
- **Decided:** shapefile and geo-only datasets stay out of the index for now.
- **Embeddings find images; they don't explain them.** A vector can't be
  turned back into a description. **Decided: no vision LLM in the pipeline.**
  The client's LLM does the understanding:
  1. **On-demand image tool:** `get_asset_image(asset_id)` returns the actual
     image as MCP image content (FastMCP `Image`), downscaled to ≤1024 px. A
     multimodal client LLM (Claude, Gemini) looks at the pixels and writes the
     answer. Also extend `read_pdf` with an option to return a page as an image
     (for charts and scanned pages).
  2. **Context text with no LLM:** store the text that already sits next to
     each image in `dataset_assets.text`, so search results and BM25 have
     something to work with:
     - WMS `GetCapabilities` layer title/abstract; ESRI REST layer name,
       description, and field list.
     - For PDF pages: that page's text layer and any "Figure N: …" captions.
     - For image files: resource name and description.
  3. **Optional embedding-only tags:** score each image embedding against a
     fixed list of text labels ("choropleth map", "line chart", "bar chart",
     "data table", "satellite image", "scanned document") and store the top
     label. This uses the shared text–image space; no LLM involved. Spot-check
     accuracy before relying on it.
  - Caveats: text-only MCP clients won't see the image, only the context text.
    Numbers read off charts by the client LLM can be wrong, so tool text should
    tell the LLM to confirm figures against the underlying data when it exists.
  - Today: **none of this exists.** No image assets are indexed, no tool
    returns images, and `read_pdf` returns text only.
- Order: schemas → PDF text → map previews. Each step is gated on the eval;
  add a "column-level" query category.

### Phase 4 — Local BM25 (DuckDB FTS) in the fusion

- At build time, after all upserts:
  `PRAGMA create_fts_index('datasets', 'id', 'title', 'notes', 'topic', 'org', overwrite=1)`.
- The FTS index doesn't update incrementally, so `--refresh` must rebuild it.
- Add `bm25_search()` to `store.py` and fuse it as another RRF list in
  `semantic_search_datasets`. **Keep** live CKAN lists, because they cover records
  outside the index.
- Add a short per-portal timeout for live search (separate from the 20 s
  data timeout) so one slow portal can't stall discovery.
- Runtime needs the `fts` extension. Decide on offline handling (see §5).
- Eval target: acronym/code category, StatCan keyword recall.

### Phase 5 — Sources

**BC (`bc`) and Toronto (`toronto`):** CKAN, so register them in `source_registry.py`.
- Measure **queryable** counts through `resource_policy` first.
- Verify page templates. Toronto's public pages may use the dataset slug, not the UUID.
- Every CKAN source joins the live fan-out (3 → 5–6 calls per search), which
  makes the Phase 4 timeout more important.
- Update the hard-coded "canada, alberta, or ontario" text in tool
  docstrings and recovery messages (`search_datasets`, `query_datastore`).

**Québec (`quebec`):** French-primary, so ingest **only after Phase 1/2**.
With bge it would be largely unsearchable.

**Bank of Canada:** PR #1's `query_boc_valet` is live-query only, so
`semantic_search_datasets` can never route "interest rates" questions to it.
Proposal: index Valet **groups** (and optionally series) as `boc:` catalog
records that point to `query_boc_valet`. Even all ~16k series is small next to
75k datasets. Measure the group count first.

### Phase 6 — Geospatial (separate track, after search work)

Not part of the embedding upgrade.
- Add `GEOJSON` and `.geojson` to the allowlist. GPKG is a SQLite file and needs a full
  download, not range streaming.
- `ST_Read` over HTTP depends on GDAL/vsicurl support. Verify it.
- **Geometry columns must be dropped or summarised** in tool output, or
  responses blow past MCP size limits.
- Needs the `spatial` extension. Same offline question as FTS.

---

## 3. Files touched

| File | Phase |
|---|---|
| `semantic/eval/*` (new) | 0 |
| `semantic/embed.py` | 1 |
| `semantic/store.py` | 1, 4 |
| `semantic/build_index.py` | 1–5 |
| `mcp_server.py` | 1, 4, 5 |
| `source_registry.py` | 5 |
| `resource_policy.py` | 6 |
| `requirements.txt`, `README.md`, `CHANGELOG.md` | 1, 5 |
| `test_sources.py` (+ new tests) | all |

---

## 4. Risks

- **Breaking release:** the new catalog and new code must ship together.
  The `catalog_meta` guard covers this.
- **Install weight:** torch at query time vs. the current "lightweight, no
  keys" positioning.
- **Unproven gain:** for short English metadata, bge-small is already decent.
  Phase 0 decides whether the swap is worth it.

---

## 5. Open questions for the owner

1. ~~**PR #1:** keep or reset?~~ Kept and merged into local `main`. Push pending.
2. ~~**Dimension:** ship 768 or 256?~~ Resolved: locked to **768** dimensions.
3. **Torch dependency:** hosting on Fly later makes this mostly a server-side
   cost. Is that acceptable for local/self-hosted users too, or do we keep a
   bge/ONNX fallback for them?
4. **DuckDB extensions (`fts`, `spatial`):** install on first run (needs
   network), or document a one-time `INSTALL` step?
5. **BoC in the index:** groups only, or groups plus series?


