# opendata.fyi — Explore public data with your AI assistant

**Current release: v1.1.0 · August 3, 2026** · [Changelog](CHANGELOG.md)

An open-source MCP server that lets AI assistants and MCP clients (Claude,
Cursor, Codex, ChatGPT, Gemini CLI, Zed, and others) discover and query public
datasets from [open.canada.ca](https://open.canada.ca),
[open.alberta.ca](https://open.alberta.ca),
[data.ontario.ca](https://data.ontario.ca), the complete
[Statistics Canada Web Data Service](https://www.statcan.gc.ca/en/developers/wds)
table inventory, and the [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs).

Ask *"how do interest rates affect housing prices?"* and your AI assistant can
use opendata.fyi to find relevant datasets, query their resources, and answer
with source links—without manual downloads or hunting through the portal.

**Zero API keys required.** Semantic search runs on a local embedding model
(bge-small-en-v1.5 via [fastembed](https://github.com/qdrant/fastembed)); data
querying is powered by the public CKAN and StatCan WDS APIs plus DuckDB.

## Source coverage

The v1.1.0 catalogue contains **74,728 queryable datasets and statistical
tables** in a single searchable index. Counts reflect the current release and
will change as source catalogues are refreshed.

| Source | Indexed records | Discovery and access |
|---|---:|---|
| [Government of Canada Open Data](https://open.canada.ca/en/open-data) | 34,326 | Semantic index + live CKAN keyword search and datastore queries |
| [Alberta Open Data](https://open.alberta.ca) | 31,341 | Semantic index + live CKAN keyword search and datastore/file queries |
| [Ontario Open Data](https://data.ontario.ca) | 846 | Semantic index + live CKAN keyword search and datastore/file queries |
| [Statistics Canada data tables](https://www.statcan.gc.ca/en/developers/wds) | 8,215 | Complete WDS table inventory, full-table CSV ZIPs, metadata, vectors, coordinates, and time-series data |

All sources share collision-safe IDs such as `canada:…`, `alberta:…`,
`ontario:…`, and `statcan:17100009`.

The indexed counts intentionally include records with a supported MCP query
path. Live CKAN keyword discovery still searches the broader portal catalogues.
The Statistics Canada count is the complete inventory exposed by WDS, which is
smaller than the broader table count shown by the Statistics Canada website.

The [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs)
(interest rates, exchange rates, and other financial series) is accessed live
via `query_boc_valet` and is not part of the semantic index or the counts
above — its series and group catalogues are large enough that discovery is
left to Valet's own `series_list`/`group_list` endpoints rather than a local
embedding index.

## How it works

```
"which neighbourhoods in Toronto have the worst air quality?"
        │
        ▼
semantic_search_datasets ──── hybrid search: shared local vector index
        │                     (all sources) fused with concurrent live CKAN
        │                     keyword searches (Canada, Alberta, Ontario)
        │                     via Reciprocal Rank Fusion
        ▼
get_dataset ────────────────── resources + which are API-queryable
        │
        ├─ datastore-backed? ──▶ query_datastore    (server-side, no download)
        ├─ CSV / Parquet?    ──▶ query_remote_file  (DuckDB streams over HTTP)
        ├─ ZIP of CSV?       ──▶ query_remote_file  (auto-extracted — StatCan bulk files)
        ├─ StatCan series?   ──▶ query_statcan_wds  (official WDS metadata + data points)
        ├─ Excel?            ──▶ list_excel_sheets → query_excel_sheet
        └─ PDF report?       ──▶ read_pdf           (page-ranged text extraction)

query_boc_valet ─────────────── Bank of Canada interest rate, FX, and other
                                 series — series/group metadata + observations,
                                 outside the semantic-search flow above
```

Every result preserves its authoritative catalog or publication page.

For example, `semantic_search_datasets("Alberta municipal finances")` searches
the shared index and Alberta's live catalogue. A StatCan result such as
`statcan:17100009` can be opened with `get_dataset`, queried as a full CSV ZIP,
or passed to `query_statcan_wds` for metadata and individual time series.

The core discovery and query tools also return MCP `structuredContent` with a
consistent schema (`datasets`, `columns`, `rows`, `sources`, `query`, `warnings`,
and `error`). A Markdown version is returned alongside it for compatibility with
clients that only render text tool results.

## Quick start

```bash
git clone https://github.com/opendatafyi/openmcp.git
cd openmcp
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Get the search index

The semantic index (`catalog.duckdb`) is too large for git. Download it
from the [latest release](https://github.com/opendatafyi/openmcp/releases) and put it in the project root.

### Updating an existing local install

```bash
git pull --ff-only
venv/bin/pip install -r requirements.txt
```

Then replace `catalog.duckdb` when the release notes include a rebuilt search
index, and restart the MCP server/client so tool schemas are refreshed. A hosted
connector is updated by its operator; connector users do not run `git pull`.

### Client Setup (Universal MCP Standard)

opendata.fyi uses the standard **MCP stdio protocol**, making it compatible out of the box with any MCP client application (**Claude Desktop**, **Claude Code**, **Gemini CLI**, **ChatGPT Desktop**, **Cursor**, **Cline / Roo Code**, **Windsurf**, **Zed**, **Continue.dev**, **Goose**, etc.).

Add this standard JSON block to your client's MCP configuration file (e.g. `claude_desktop_config.json`, `.mcp.json`, `cline_mcp_settings.json`, `mcp_config.json`):

```json
{
  "mcpServers": {
    "openmcp": {
      "command": "/absolute/path/to/openmcp/venv/bin/python",
      "args": ["/absolute/path/to/openmcp/mcp_server.py"]
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `semantic_search_datasets(query)` | Hybrid semantic + keyword dataset discovery (RRF) |
| `search_datasets(query, source_id)` | Plain keyword search of one CKAN portal (`canada`, `alberta`, or `ontario`) |
| `get_dataset(id)` | A dataset's resources + which are API-queryable |
| `query_statcan_wds(method, ...)` | Read-only access to the official StatCan WDS methods |
| `query_boc_valet(resource, ...)` | Read-only access to the Bank of Canada Valet API (series, groups, observations) |
| `get_resource_fields(resource_id, source_id)` | Columns/types of a datastore resource, no download |
| `query_datastore(resource_id, ..., source_id)` | **Server-side** filter/search — the fast path |
| `get_file_schema(url)` | Schema of a remote file (DuckDB `DESCRIBE`, minimal download) |
| `preview_remote_file(url)` | First rows of a remote CSV/Parquet/JSON/Excel/ZIP |
| `query_remote_file(url, sql)` | Read-only DuckDB SQL on a remote file (ZIP auto-extracted) |
| `list_excel_sheets(url)` | Sheets, shapes, and columns of a workbook (header auto-detected) |
| `query_excel_sheet(url, sheet, sql)` | SQL against one sheet of a workbook |
| `read_pdf(url, pages)` | Page-ranged text extraction from PDF resources |

Plus three MCP prompts (`query_canada_data`, `explore_dataset`,
`compare_datasets`) that encode the full workflow for one-click use.

### Alberta Open Data

Alberta participates in the same hybrid discovery and retrieval flow as the
federal and Ontario CKAN catalogues:

```text
semantic_search_datasets("population projections for Alberta municipalities")
  → get_dataset("alberta:<dataset-id>")
  → query_datastore(..., source_id="alberta")
     or query_remote_file(...)
```

The index includes machine-readable resources and relevant PDF publications;
unsupported geospatial-only and unknown formats are excluded because the server
cannot query them reliably.

### Statistics Canada WDS

StatCan's complete active and archived table inventory is part of semantic
discovery. The `query_statcan_wds` tool is a read-only, allowlisted gateway to
all 15 documented WDS methods plus `getCodeSets`. It validates PIDs, vectors,
coordinates, dates, languages, and response sizes before making a request.

```text
query_statcan_wds(
  method="getCubeMetadata",
  product_id="17100009"
)

query_statcan_wds(
  method="getDataFromVectorsAndLatestNPeriods",
  vector_ids="V1,V2",
  latest_n=4
)

query_statcan_wds(
  method="getFullTableDownloadCSV",
  product_id="17100009",
  language="en"
)
```

Use WDS for discrete metadata and data-point requests. For whole-table analysis,
`get_dataset("statcan:17100009")` returns the official CSV ZIP resource for
`query_remote_file`.

### Bank of Canada Valet API

`query_boc_valet` is a read-only, allowlisted gateway to the
[Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) — interest
rates, exchange rates, monetary aggregates, and other financial and economic
series. No API key is required. It is a live-query tool: series and group
names are Valet identifiers, not opendata.fyi dataset IDs, and this source is
not part of semantic search or `get_dataset`.

```text
query_boc_valet(resource="series", series_names="FXUSDCAD")

query_boc_valet(
  resource="observations",
  series_names="FXUSDCAD,FXEURCAD",
  recent=10
)

query_boc_valet(
  resource="group_observations",
  group_name="FX_RATES_DAILY",
  start_date="2026-01-01",
  end_date="2026-01-31",
  order_dir="desc"
)

query_boc_valet(resource="series_list")   # discover series names
query_boc_valet(resource="group_list")    # discover group names
```

`recent`, `recent_weeks`, `recent_months`, and `recent_years` are mutually
exclusive with each other and with `start_date`/`end_date`.

## Design notes

- **Server-side first.** Resources with `datastore_active: true` are filtered by
  the portal's own database (`datastore_search`) — only matching rows travel.
  Files are the fallback, streamed by DuckDB over HTTP range requests where
  possible.
- **StatCan WDS access.** `query_statcan_wds` exposes the official allowlisted
  metadata, vector, coordinate, change-list, range, and full-table methods with
  validated inputs and bounded MCP responses.
- **Bank of Canada Valet access.** `query_boc_valet` exposes the allowlisted
  series, group, and observation resources with validated inputs and bounded
  MCP responses, mirroring the StatCan WDS gateway's design.
- **The vector "database" is one DuckDB file.** Records from every enabled
  source share 384-dimensional vectors; no external vector service is needed.
- **Real-world Excel/CSV handling**: multi-sheet workbooks, title rows before
  headers (auto-detected), bilingual descriptions, zipped StatCan bulk tables,
  multiple encodings.
- **Read-only by construction**: SQL is screened against write/DDL patterns;
  CKAN access is GET-only.
- **Source-qualified IDs** (`canada:…`, `alberta:…`, `ontario:…`,
  `statcan:…`) prevent collisions. Existing bare federal IDs and known dataset
  page URLs remain valid inputs to `get_dataset`.
- **Refresh** the index without a full rebuild:
  `venv/bin/python semantic/build_index.py --refresh 1000` re-indexes up to the
  1000 most recently modified records per enabled catalog source. To harvest
  and index the complete Statistics Canada table
  inventory, run:
  `venv/bin/python semantic/build_index.py --sources statcan`.
- **Rebuilding from scratch** is optional — most users should just download the
  release asset. If you want to: `venv/bin/python semantic/build_index.py`
  (download and embedding time depend on the selected sources). The indexer
  uses PyTorch to auto-detect MPS on Apple Silicon or CUDA on supported NVIDIA
  systems, and falls back to FastEmbed on CPU when neither is available.

## Limitations

- Discovery indexes datasets with a CKAN datastore or CSV, Excel, Parquet,
  JSON, PDF, TXT, or ZIP resources. SHP, WMS/WFS, HTML, other geospatial-only
  services, and unknown formats are excluded; ZIPs explicitly identified as
  shapefiles/geodatabases are excluded as well.
- Statistics Canada indexing includes active and archived tables. Their status,
  date coverage, subjects, surveys, frequency, corrections, bilingual title,
  and dimensions are retained in each record's metadata.
- CKAN datastore querying uses portable `q`/`filters`/`sort` parameters rather
  than assuming that a portal enables `datastore_search_sql`.

## Optional Anonymous Telemetry

Telemetry is off by default. Users can explicitly enable lightweight,
non-blocking usage telemetry to help maintainers understand tool reliability,
performance, and which public datasets are surfaced, inspected, and queried.

### Privacy & Anonymity
- **Collected:** a temporary session ID, tool name, success/failure status,
  normalized error code, latency, server version, and an ordered list of unique
  public dataset IDs when datasets appear in search results, are opened, or are
  queried. Each tool call produces one telemetry row.
- **Not collected:** raw questions or search queries, SQL, filters, complete
  URLs, full error messages, file paths, or resource contents.
- **Anonymous discovery funnel:** the session ID groups dataset-result,
  inspection, and query events from one server run without identifying a user.
- **Non-blocking:** events dispatch asynchronously in a background thread.

### Enabling Telemetry
Telemetry runs only after you explicitly opt in using one of these methods:

#### Method 1: Via MCP Client Configuration (Recommended)
Add `"env": { "OPENDATA_FYI_TELEMETRY_ENABLED": "true" }` to your MCP configuration file (`claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "openmcp": {
      "command": "/absolute/path/to/openmcp/venv/bin/python",
      "args": ["/absolute/path/to/openmcp/mcp_server.py"],
      "env": {
        "OPENDATA_FYI_TELEMETRY_ENABLED": "true"
      }
    }
  }
}
```

#### Method 2: Via Local `.env` File
Create a `.env` file in the project root (or copy `.env.example`) and set:

```env
OPENDATA_FYI_TELEMETRY_ENABLED=true
```

#### Method 3: Via Environment Variable
```bash
export OPENDATA_FYI_TELEMETRY_ENABLED=true
```

The existing `OPENMCP_TELEMETRY_DISABLED=true` and `DISABLE_TELEMETRY=1`
settings remain supported as explicit disable overrides.

## License

MIT
