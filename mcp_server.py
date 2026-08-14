"""
OpenMCP Canada — authoritative Canadian public-data MCP server.

A focused interface over federal, Alberta, and Ontario CKAN portals plus a
reviewed catalog of authoritative non-CKAN sources. CKAN access is GET-only.

Flow:
    search_datasets(query)            -> find datasets (package_search)
    get_dataset(dataset_id)           -> list resources + which are queryable
    query_datastore(resource_id, ...) -> server-side filter/search (no download)
    query_remote_file(url, sql)       -> DuckDB fallback for non-datastore files

Datastore-backed resources (datastore_active: true) are queried server-side via
datastore_search — fast, no full download. Everything else falls back to DuckDB
streaming over the file URL.
"""

import io
import re
import json
import ssl
import zipfile
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import duckdb
from mcp.server.fastmcp import FastMCP

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load .env so telemetry (and any other) config is picked up regardless of launcher
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from semantic.embed import embed_texts
from semantic.store import top_k, DB_PATH, get_by_ids
from telemetry import (
    is_telemetry_disabled,
    log_telemetry,
    register_dataset_resources,
)
from structured_results import (
    SourceReference,
    StructuredToolResult,
    make_error_result,
    make_tool_result,
)
from source_registry import (
    CKAN_SOURCE_IDS,
    INDEX_SOURCE_IDS,
    get_source,
    page_url,
    qualify_dataset_id,
    split_dataset_id,
)
from resource_policy import is_queryable_resource, resource_format
from version import __version__

mcp = FastMCP("OpenMCP Canada — authoritative public data")

if is_telemetry_disabled():
    sys.stderr.write(
        "[OpenMCP] Anonymous telemetry is off. Opt in with "
        "OPENDATA_FYI_TELEMETRY_ENABLED=true.\n"
    )
else:
    sys.stderr.write(
        "[OpenMCP] Anonymous telemetry enabled by user configuration.\n"
    )


# ── CKAN Action APIs (GET-only; pass params in the URL) ──────────────────────
HTTP_TIMEOUT = 20
QUERY_TIMEOUT = 30

STATCAN_WDS_GUIDE_URL = "https://www.statcan.gc.ca/en/developers/wds/user-guide"
STATCAN_WDS_METHODS = frozenset({
    "getChangedSeriesList",
    "getChangedCubeList",
    "getCubeMetadata",
    "getSeriesInfoFromCubePidCoord",
    "getSeriesInfoFromVector",
    "getAllCubesList",
    "getAllCubesListLite",
    "getChangedSeriesDataFromCubePidCoord",
    "getChangedSeriesDataFromVector",
    "getDataFromCubePidCoordAndLatestNPeriods",
    "getDataFromVectorsAndLatestNPeriods",
    "getBulkVectorDataByRange",
    "getDataFromVectorByReferencePeriodRange",
    "getFullTableDownloadCSV",
    "getFullTableDownloadSDMX",
    "getCodeSets",
})
MAX_STATCAN_WDS_BATCH = 25
MAX_STATCAN_WDS_ITEMS = 100

BOC_VALET_DOCS_URL = "https://www.bankofcanada.ca/valet/docs"
BOC_VALET_RESOURCES = frozenset({
    "series",
    "observations",
    "group",
    "group_observations",
    "series_list",
    "group_list",
})
MAX_BOC_VALET_BATCH = 10
MAX_BOC_VALET_RECENT = 1000
MAX_BOC_VALET_ITEMS = 100

MAX_PREVIEW_ROWS = 15
MAX_RESULT_ROWS = 100
MAX_CELL_CHARS = 200
MAX_DESC_CHARS = 300
GEO_COL_KEYWORDS = ("polygon", "geom", "wkt", "shape", "multipolygon", "coordinates")

_WRITE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|copy|pragma|install|load|"
    r"call|export|import|set|reset|vacuum|create\s+table|"
    r"create\s+or\s+replace)\b",
    re.IGNORECASE,
)


def _read_only_sql_error(sql: str) -> Optional[str]:
    """Return validation guidance when SQL is not one read-only SELECT query."""
    statement = sql.strip()
    if not statement:
        return "A SQL query is required."
    statement = statement[:-1].rstrip() if statement.endswith(";") else statement
    if ";" in statement:
        return "Only one SQL statement is permitted."
    if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
        return "Only SELECT queries (including WITH ... SELECT) are permitted."
    if _WRITE_RE.search(statement):
        return "Only read-only SQL is permitted."
    return None


# ── robust remote file download ──────────────────────────────────────────────
# Some government hosts (statcan.gc.ca in particular) intermittently reset
# connections from clients with the default `python-requests/x.y` User-Agent,
# and some also mishandle TLS 1.3 negotiation. Send a browser-ish UA, retry
# transient errors, then fall back to a TLS-1.2-pinned connection.
_UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "OpenMCP-Canada/1.0")}
_retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                status_forcelist=(500, 502, 503, 504))
_dl_session = requests.Session()
_dl_session.headers.update(_UA)
_dl_session.mount("https://", HTTPAdapter(max_retries=_retry))
_dl_session.mount("http://", HTTPAdapter(max_retries=_retry))

# Transient network failures worth retrying over TLS 1.2. ChunkedEncodingError
# is a RequestException but NOT a ConnectionError subclass — a premature body
# termination would otherwise escape the fallback.
_TRANSIENT_ERRORS = (requests.exceptions.ConnectionError,
                     requests.exceptions.SSLError,
                     requests.exceptions.ChunkedEncodingError)


class _TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _robust_get(url: str, timeout: int = None) -> requests.Response:
    """GET a remote file with retries; falls back to TLS 1.2 on handshake resets."""
    timeout = timeout or HTTP_TIMEOUT
    try:
        resp = _dl_session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except _TRANSIENT_ERRORS:
        tls12 = requests.Session()
        tls12.headers.update(_UA)
        tls12.mount("https://", _TLS12Adapter(max_retries=_retry))
        resp = tls12.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp


# ── helpers ────────────────────────────────────────────────────────────────────
def _ckan_get(action: str, *, source_id: str = "canada", **params) -> Dict[str, Any]:
    """Call a CKAN Action API endpoint (GET) and return the `result` payload."""
    source = get_source(source_id)
    if source.source_type != "ckan" or not source.api_base:
        raise ValueError(f"Source '{source_id}' does not expose a CKAN Action API.")
    resp = _dl_session.get(f"{source.api_base}/{action}", params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(f"CKAN error on {action}: {body.get('error')}")
    return body.get("result", {})


def _statcan_product_id(value: str) -> int:
    """Validate a Statistics Canada Product ID, qualified ID, or table URL."""
    text = str(value or "").strip()
    if text.startswith("statcan:") or "statcan.gc.ca" in text:
        source_id, text = split_dataset_id(text)
        if source_id != "statcan":
            raise ValueError("The dataset identifier must refer to Statistics Canada.")
    if not re.fullmatch(r"\d{8}(?:\d{2})?", text):
        raise ValueError("product_id must contain an 8-digit PID (or 10-digit table view PID).")
    return int(text)


def _statcan_vector_ids(value: str) -> List[int]:
    """Parse and bound a comma-separated list of StatCan vector identifiers."""
    raw_ids = [part.strip().strip('"\'') for part in str(value or "").split(",")]
    raw_ids = [part[1:] if part.lower().startswith("v") else part for part in raw_ids if part]
    if not raw_ids:
        raise ValueError("vector_ids must contain at least one vector identifier.")
    if len(raw_ids) > MAX_STATCAN_WDS_BATCH:
        raise ValueError(f"At most {MAX_STATCAN_WDS_BATCH} vector identifiers are permitted per call.")
    if any(not re.fullmatch(r"\d{1,10}", part) for part in raw_ids):
        raise ValueError("Each vector identifier must be V followed by 1-10 digits, or just 1-10 digits.")
    return [int(part) for part in raw_ids]


def _iso_date(value: str, name: str, *, timestamp: bool = False) -> str:
    """Validate an ISO date (optionally with time), shared by WDS and Valet."""
    text = str(value or "").strip()
    pattern = r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?" if timestamp else r"\d{4}-\d{2}-\d{2}"
    if not re.fullmatch(pattern, text):
        suffix = " with an optional THH:MM[:SS] time" if timestamp else ""
        raise ValueError(f"{name} must use YYYY-MM-DD{suffix} format.")
    return text


def _statcan_coordinate(value: str) -> str:
    """Validate a one-to-ten dimension WDS coordinate."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d+(?:\.\d+){0,9}", text):
        raise ValueError("coordinate must contain 1-10 dot-separated numeric member IDs.")
    return text


def _statcan_wds_request(
    method: str,
    *,
    product_id: str = "",
    coordinate: str = "",
    vector_ids: str = "",
    latest_n: int = 3,
    start_date: str = "",
    end_date: str = "",
    language: str = "en",
) -> Tuple[Any, str, str]:
    """Build and execute one allowlisted Statistics Canada WDS request."""
    if method not in STATCAN_WDS_METHODS:
        allowed = ", ".join(sorted(STATCAN_WDS_METHODS))
        raise ValueError(f"Unsupported WDS method '{method}'. Allowed methods: {allowed}")

    source = get_source("statcan")
    url = f"{source.api_base}/{method}"
    request_method = "GET"
    request_kwargs: Dict[str, Any] = {"timeout": HTTP_TIMEOUT}

    no_argument_gets = {
        "getChangedSeriesList", "getAllCubesList", "getAllCubesListLite", "getCodeSets",
    }
    product_posts = {"getCubeMetadata"}
    coordinate_posts = {
        "getSeriesInfoFromCubePidCoord", "getChangedSeriesDataFromCubePidCoord",
    }
    vector_posts = {"getSeriesInfoFromVector", "getChangedSeriesDataFromVector"}

    if method in no_argument_gets:
        pass
    elif method == "getChangedCubeList":
        url += f"/{_iso_date(start_date, 'start_date')}"
    elif method in product_posts:
        request_method = "POST"
        request_kwargs["json"] = [{"productId": _statcan_product_id(product_id)}]
    elif method in coordinate_posts:
        request_method = "POST"
        request_kwargs["json"] = [{
            "productId": _statcan_product_id(product_id),
            "coordinate": _statcan_coordinate(coordinate),
        }]
    elif method in vector_posts:
        request_method = "POST"
        request_kwargs["json"] = [
            {"vectorId": vector_id} for vector_id in _statcan_vector_ids(vector_ids)
        ]
    elif method == "getDataFromCubePidCoordAndLatestNPeriods":
        if not 1 <= latest_n <= MAX_STATCAN_WDS_ITEMS:
            raise ValueError(f"latest_n must be between 1 and {MAX_STATCAN_WDS_ITEMS}.")
        request_method = "POST"
        request_kwargs["json"] = [{
            "productId": _statcan_product_id(product_id),
            "coordinate": _statcan_coordinate(coordinate),
            "latestN": latest_n,
        }]
    elif method == "getDataFromVectorsAndLatestNPeriods":
        if not 1 <= latest_n <= MAX_STATCAN_WDS_ITEMS:
            raise ValueError(f"latest_n must be between 1 and {MAX_STATCAN_WDS_ITEMS}.")
        request_method = "POST"
        request_kwargs["json"] = [
            {"vectorId": vector_id, "latestN": latest_n}
            for vector_id in _statcan_vector_ids(vector_ids)
        ]
    elif method == "getBulkVectorDataByRange":
        request_method = "POST"
        request_kwargs["json"] = {
            "vectorIds": [str(value) for value in _statcan_vector_ids(vector_ids)],
            "startDataPointReleaseDate": _iso_date(start_date, "start_date", timestamp=True),
            "endDataPointReleaseDate": _iso_date(end_date, "end_date", timestamp=True),
        }
    elif method == "getDataFromVectorByReferencePeriodRange":
        vectors = _statcan_vector_ids(vector_ids)
        request_kwargs["params"] = {
            "vectorIds": ",".join(f'"{value}"' for value in vectors),
            "startRefPeriod": _iso_date(start_date, "start_date"),
            "endReferencePeriod": _iso_date(end_date, "end_date"),
        }
    elif method == "getFullTableDownloadCSV":
        lang = str(language or "").strip().lower()
        if lang not in {"en", "fr"}:
            raise ValueError("language must be 'en' or 'fr'.")
        url += f"/{_statcan_product_id(product_id)}/{lang}"
    elif method == "getFullTableDownloadSDMX":
        url += f"/{_statcan_product_id(product_id)}"

    caller = _dl_session.post if request_method == "POST" else _dl_session.get
    response = caller(url, **request_kwargs)
    response.raise_for_status()
    return response.json(), request_method, url


_BOC_NAME_RE = re.compile(r"[A-Za-z0-9_.]{1,64}")
_BOC_RECENT_PARAMS = ("recent", "recent_weeks", "recent_months", "recent_years")


def _boc_series_names(value: str) -> List[str]:
    """Parse and bound a comma-separated list of Valet series names."""
    names = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not names:
        raise ValueError("series_names must contain at least one series name.")
    if len(names) > MAX_BOC_VALET_BATCH:
        raise ValueError(f"At most {MAX_BOC_VALET_BATCH} series names are permitted per call.")
    for name in names:
        if not _BOC_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid series name: {name!r}.")
    return names


def _boc_group_name(value: str) -> str:
    """Validate a single Valet group name."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("group_name is required for this resource.")
    if not _BOC_NAME_RE.fullmatch(text):
        raise ValueError(f"Invalid group name: {text!r}.")
    return text


def _boc_observation_params(start_date: str, end_date: str, recent: int,
                            recent_weeks: int, recent_months: int,
                            recent_years: int, order_dir: str) -> Dict[str, Any]:
    """Validate the mutually exclusive Valet observation range parameters."""
    recents = {
        "recent": recent, "recent_weeks": recent_weeks,
        "recent_months": recent_months, "recent_years": recent_years,
    }
    active = [name for name, val in recents.items() if val]
    if len(active) > 1:
        raise ValueError(f"Only one of {', '.join(_BOC_RECENT_PARAMS)} may be set.")
    if active and (start_date or end_date):
        raise ValueError("Use either a recent_* window or start_date/end_date, not both.")

    params: Dict[str, Any] = {}
    if active:
        name = active[0]
        value = recents[name]
        if not 1 <= value <= MAX_BOC_VALET_RECENT:
            raise ValueError(f"{name} must be between 1 and {MAX_BOC_VALET_RECENT}.")
        params[name] = value
    if start_date:
        params["start_date"] = _iso_date(start_date, "start_date")
    if end_date:
        params["end_date"] = _iso_date(end_date, "end_date")
    order = str(order_dir or "").strip().lower()
    if order:
        if order not in {"asc", "desc"}:
            raise ValueError("order_dir must be 'asc' or 'desc'.")
        params["order_dir"] = order
    return params


def _boc_valet_request(
    resource: str,
    *,
    series_names: str = "",
    group_name: str = "",
    start_date: str = "",
    end_date: str = "",
    recent: int = 0,
    recent_weeks: int = 0,
    recent_months: int = 0,
    recent_years: int = 0,
    order_dir: str = "",
) -> Tuple[Any, str]:
    """Build and execute one allowlisted Bank of Canada Valet API request."""
    if resource not in BOC_VALET_RESOURCES:
        allowed = ", ".join(sorted(BOC_VALET_RESOURCES))
        raise ValueError(f"Unsupported resource '{resource}'. Allowed resources: {allowed}")

    source = get_source("boc")
    params: Dict[str, Any] = {}

    if resource == "series":
        names = _boc_series_names(series_names)
        url = f"{source.api_base}/series/{','.join(names)}/json"
    elif resource == "observations":
        names = _boc_series_names(series_names)
        url = f"{source.api_base}/observations/{','.join(names)}/json"
        params = _boc_observation_params(
            start_date, end_date, recent, recent_weeks, recent_months, recent_years, order_dir
        )
    elif resource == "group":
        name = _boc_group_name(group_name)
        url = f"{source.api_base}/groups/{name}/json"
    elif resource == "group_observations":
        name = _boc_group_name(group_name)
        url = f"{source.api_base}/observations/group/{name}/json"
        params = _boc_observation_params(
            start_date, end_date, recent, recent_weeks, recent_months, recent_years, order_dir
        )
    elif resource == "series_list":
        url = f"{source.api_base}/lists/series/json"
    else:  # group_list
        url = f"{source.api_base}/lists/groups/json"

    response = _dl_session.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json(), url


def _boc_catalog_rows(
    body: Any, key: str, limit: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """Bound a Valet lists/* response — thousands of entries live in a dict
    keyed by name, not a list, so the generic list truncator can't see them.
    """
    catalog = body.get(key, {}) if isinstance(body, dict) else {}
    if not isinstance(catalog, dict):
        return body, _json_rows(body), False
    items = list(catalog.items())
    truncated = len(items) > limit
    kept = items[:limit]
    rows = [
        {"name": name, **(details if isinstance(details, dict) else {"value": details})}
        for name, details in kept
    ]
    bounded = {**{k: v for k, v in body.items() if k != key}, key: dict(kept)}
    return bounded, rows, truncated


def _truncate_json_response(value: Any, limit: int, state: Dict[str, bool]) -> Any:
    """Bound nested API lists and long strings so MCP responses stay usable."""
    if isinstance(value, list):
        if len(value) > limit:
            state["truncated"] = True
        return [_truncate_json_response(item, limit, state) for item in value[:limit]]
    if isinstance(value, dict):
        return {
            str(key): _truncate_json_response(item, limit, state)
            for key, item in value.items()
        }
    if isinstance(value, str) and len(value) > 2000:
        state["truncated"] = True
        return value[:2000] + "…"
    return value


def _json_rows(value: Any) -> List[Dict[str, Any]]:
    """Represent a varying JSON response envelope in the common rows field."""
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"value": item} for item in value]
    if isinstance(value, dict):
        return [value]
    return [{"value": value}]


def _extract_dataset_id(dataset_id: str) -> str:
    """Return the native ID from a qualified ID, legacy ID, or known page URL."""
    return split_dataset_id(dataset_id)[1]


def _plain_english(value: Any) -> str:
    """Normalize bilingual CKAN values for display and embedding records."""
    if isinstance(value, dict):
        value = value.get("en") or value.get("fr") or next(iter(value.values()), "")
    if not value:
        return ""
    text = str(value).strip()
    return text.split("|")[0].strip() if "|" in text else text


def _catalog_record_from_ckan(raw: Dict[str, Any], source_id: str,
                              require_supported: bool = True
                              ) -> Optional[Dict[str, Any]]:
    """Normalize a live CKAN package into the shared catalog record shape."""
    native_id = raw.get("id") or raw.get("name")
    if not native_id:
        return None
    resources = []
    has_supported_resource = False
    for resource in raw.get("resources", []):
        if not is_queryable_resource(resource):
            continue
        fmt = resource_format(resource)
        has_supported_resource = True
        resources.append({
            "id": resource.get("id", ""),
            "name": _plain_english(resource.get("name")),
            "format": fmt,
            "url": resource.get("url", ""),
            "datastore_active": bool(resource.get("datastore_active", False)),
        })
    if require_supported and not has_supported_resource:
        return None
    organization = raw.get("organization") or {}
    title = _plain_english(raw.get("title_translated") or raw.get("title") or raw.get("name"))
    return {
        "id": qualify_dataset_id(source_id, native_id),
        "native_id": native_id,
        "source_id": source_id,
        "source_type": "ckan",
        "title": title or "Untitled",
        "org": _plain_english(
            organization.get("title_translated") or organization.get("title")
        ),
        "notes": _plain_english(raw.get("notes_translated") or raw.get("notes")),
        "topic": _plain_english(raw.get("topic_category") or raw.get("subject")),
        "resources": resources,
        "metadata_modified": raw.get("metadata_modified", ""),
        "page_url": page_url(source_id, native_id),
        "distance": None,
    }


def _canonical_resource_urls(dataset: Dict[str, Any]) -> set[str]:
    """Return normalized resource URLs for federated-record de-duplication."""
    return {
        str(resource.get("url", "")).strip().lower().rstrip("/")
        for resource in dataset.get("resources", [])
        if resource.get("url")
    }


def _truncate_desc(text: str) -> str:
    """Strip the French half of bilingual notes and truncate."""
    if not text:
        return "No description available."
    if "|" in text:
        text = text.split("|")[0]
    text = " ".join(text.split())
    if len(text) > MAX_DESC_CHARS:
        text = text[:MAX_DESC_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop geometry columns, drop CKAN's internal _id, truncate long cells."""
    df = df.drop(columns=["_id", "_full_text"], errors="ignore")
    geo = [c for c in df.columns if any(k in c.lower() for k in GEO_COL_KEYWORDS)]
    df = df.drop(columns=geo, errors="ignore")
    return df.map(
        lambda x: (str(x)[:MAX_CELL_CHARS] + "…")
        if isinstance(x, str) and len(x) > MAX_CELL_CHARS else x
    )


def _df_to_md(df: pd.DataFrame, cap: int = MAX_RESULT_ROWS, offset: int = 0,
              total: Optional[int] = None) -> str:
    """Render a dataframe as markdown, capping rows and noting pagination."""
    shown_total = total if total is not None else len(df)
    df = df.head(cap).map(lambda x: str(x) if pd.notnull(x) else "")
    if df.empty:
        return "_0 rows._"
    md = df.to_markdown(index=False)
    if shown_total > offset + len(df):
        md += (f"\n\n_Showing rows {offset}–{offset + len(df)} of {shown_total}. "
               f"Pass offset={offset + cap} for the next page._")
    return md


def _df_to_structured(df: pd.DataFrame, cap: int = MAX_RESULT_ROWS
                      ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Convert a dataframe into JSON-safe column metadata and capped records."""
    limited = df.head(max(0, cap))
    columns = [
        {"name": str(column), "type": str(dtype)}
        for column, dtype in limited.dtypes.items()
    ]
    # pandas' JSON encoder consistently handles NA, timestamps and numpy scalars.
    rows = json.loads(limited.to_json(orient="records", date_format="iso"))
    return columns, rows


def _new_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create an isolated DuckDB connection for one query."""
    con = duckdb.connect(":memory:")
    try:
        con.execute("LOAD httpfs")
    except Exception:
        # Installation is normally already cached. This fallback supports a clean
        # machine without making every server startup depend on extension loading.
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    return con


def run_query_with_timeout(sql: str, timeout_sec: int = QUERY_TIMEOUT) -> pd.DataFrame:
    """Run a DuckDB query on an isolated connection with a wall-clock timeout."""
    con = _new_duckdb_connection()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(lambda: con.execute(sql).fetchdf())
    try:
        return fut.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        # DuckDB interrupt is safe to call from another thread and prevents the
        # executor context from silently waiting for a timed-out query to finish.
        con.interrupt()
        try:
            fut.result(timeout=1)
        except Exception:
            pass
        raise TimeoutError(
            f"Query timed out after {timeout_sec}s. If this resource has "
            f"datastore_active:true, use query_datastore instead (server-side)."
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if fut.done():
            con.close()
        else:
            fut.add_done_callback(lambda _future: con.close())


# ── Excel workbook cache (url → (bytes, timestamp)) ──────────────────────────
# Multi-sheet workflows (list → preview a sheet → query a sheet) would otherwise
# re-download the whole workbook on every call. Cache the bytes briefly.
import time
_EXCEL_CACHE: Dict[str, Tuple[bytes, float]] = {}
_EXCEL_TTL = 1800        # 30 min
_EXCEL_CACHE_MAX = 8


def _excel_file(url: str) -> "pd.ExcelFile":
    """Return a pandas ExcelFile for a remote workbook, caching the raw bytes."""
    hit = _EXCEL_CACHE.get(url)
    if hit and time.time() - hit[1] < _EXCEL_TTL:
        data = hit[0]
    else:
        resp = _robust_get(url)
        data = resp.content
        if len(_EXCEL_CACHE) >= _EXCEL_CACHE_MAX:
            del _EXCEL_CACHE[min(_EXCEL_CACHE, key=lambda k: _EXCEL_CACHE[k][1])]
        _EXCEL_CACHE[url] = (data, time.time())
    return pd.ExcelFile(io.BytesIO(data))


def _detect_header_row(xl: "pd.ExcelFile", sheet_name: Any, max_scan: int = 20) -> int:
    """Scan the first max_scan rows and return the index of the most likely header row.

    Government Excel files often have a title or metadata block before the real table.
    We pick the row with the most non-empty string-valued cells — that's almost always
    the header row.  Falls back to 0 if nothing looks clearly better.
    """
    try:
        raw = xl.parse(sheet_name, header=None, nrows=max_scan)
    except Exception:
        return 0
    best_row, best_count = 0, 0
    for i, row in raw.iterrows():
        str_count = sum(
            1 for v in row if isinstance(v, str) and v.strip()
        )
        if str_count > best_count:
            best_count = str_count
            best_row = int(i)
    return best_row


def _query_dataframe(sql: str, df: pd.DataFrame,
                     timeout_sec: int = QUERY_TIMEOUT) -> pd.DataFrame:
    """Run read-only SQL against an in-memory DataFrame (registered as `df`)."""
    con = duckdb.connect(":memory:")
    con.register("df", df)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(lambda: con.execute(sql).fetchdf())
    try:
        return fut.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        con.interrupt()
        try:
            fut.result(timeout=1)
        except Exception:
            pass
        raise TimeoutError(f"Query timed out after {timeout_sec}s.")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if fut.done():
            con.close()
        else:
            fut.add_done_callback(lambda _future: con.close())


def _read_tabular(url: str, nrows: Optional[int] = None,
                  sheet_name: Any = 0,
                  header_row: Optional[int] = None) -> pd.DataFrame:
    """Read a remote CSV/JSON/Parquet/XLSX (incl. zipped CSV) into a DataFrame.

    For Excel, `sheet_name` selects the sheet (name or index; default first sheet).
    `header_row` overrides auto-detection; pass None (default) to auto-detect.
    """
    low = url.lower()
    kw = {"nrows": nrows} if nrows else {}

    if low.endswith(".zip"):
        resp = _robust_get(url)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith(".csv") and "__MACOSX" not in n]
            if not names:
                raise ValueError(f"No CSV inside ZIP: {url}")
            name = next((n for n in names if "-eng" in n.lower()), names[0])
            raw = zf.read(name)
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, **kw)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", **kw)

    if low.endswith(".parquet"):
        df = pd.read_parquet(url)
        return df.head(nrows) if nrows else df

    if low.endswith((".xlsx", ".xls")):
        xl = _excel_file(url)
        hdr = header_row if header_row is not None else _detect_header_row(xl, sheet_name)
        df = xl.parse(sheet_name, header=hdr, **kw)
        # Strip unnamed carry-over columns that pandas generates for blank header cells
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed: \d+$")]
        return df

    if low.endswith(".json"):
        resp = _robust_get(url)
        df = pd.read_json(io.BytesIO(resp.content))
        return df.head(nrows) if nrows else df

    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(url, encoding=enc, **kw)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(url, encoding="latin-1", **kw)


# =====================================================================
# DISCOVERY
# =====================================================================
@mcp.tool(structured_output=True)
@log_telemetry("semantic_search_datasets")
def semantic_search_datasets(query: str, limit: int = 10) -> StructuredToolResult:
    """
    Find authoritative Canadian open datasets by meaning or natural-language questions.
    Uses Reciprocal Rank Fusion (RRF) to combine local semantic vector search (bge-small-en-v1.5,
    runs locally — no API key) with concurrent live keyword searches of the federal, Alberta,
    and Ontario CKAN portals.
    
    Examples:
        - "how much did municipalities spend on infrastructure?"
        - "population demographics of alberta cities"
        - "water quality testing records"
        
    Args:
        query: Plain English search term, acronym, or question.
        limit: Max datasets to return (default 8).
    """
    effective_limit = max(1, min(limit, 25))
    query_info = {
        "text": query,
        "limit": effective_limit,
        "mode": "hybrid_rrf",
        "sources": list(INDEX_SOURCE_IDS),
    }
    if not os.path.exists(DB_PATH):
        return make_error_result(
            "Semantic database catalog is missing.",
            code="CatalogMissing",
            query=query_info,
            recovery="Download catalog.duckdb from the latest release or run python semantic/build_index.py.",
        )

    warnings: List[str] = []

    # 1. Retrieve semantic search results (top 25)
    try:
        query_vecs = embed_texts([query], is_query=True)
        if not query_vecs:
            semantic_results = []
        else:
            semantic_results = top_k(query_vecs[0], k=25)
    except Exception as e:
        # Fall back to empty list if local DB search fails
        semantic_results = []
        warnings.append(f"Semantic search was unavailable ({type(e).__name__}); results use keyword search only.")

    # 2. Search all live CKAN portals concurrently. Each portal fails independently.
    keyword_results: Dict[str, List[Dict[str, Any]]] = {}
    live_details: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(CKAN_SOURCE_IDS), thread_name_prefix="CKANSearch"
    ) as executor:
        futures = {
            executor.submit(
                _ckan_get, "package_search", source_id=source_id,
                q=query, rows=25
            ): source_id
            for source_id in CKAN_SOURCE_IDS
        }
        for future in concurrent.futures.as_completed(futures):
            source_id = futures[future]
            try:
                raw_results = future.result().get("results", [])
                normalized = []
                for raw in raw_results:
                    record = _catalog_record_from_ckan(raw, source_id)
                    if record:
                        normalized.append(record)
                        live_details[record["id"]] = record
                keyword_results[source_id] = normalized
            except Exception as error:
                keyword_results[source_id] = []
                warnings.append(
                    f"Live search of {get_source(source_id).name} was unavailable "
                    f"({type(error).__name__})."
                )

    keyword_count = sum(len(results) for results in keyword_results.values())
    if not semantic_results and not keyword_count:
        if len(warnings) >= len(CKAN_SOURCE_IDS) + 1:
            return make_error_result(
                "Semantic search and all live keyword searches failed.",
                code="SearchUnavailable",
                query=query_info,
                retryable=True,
                recovery="Retry the search and verify that catalog.duckdb is readable and the enabled CKAN portals are reachable.",
                warnings=warnings,
            )
        return make_tool_result(
            f"No datasets found for query: '{query}'",
            query=query_info,
            warnings=[*warnings, "No datasets matched the query."],
        )

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    
    # Semantic Rank Fusion
    semantic_map: Dict[str, Dict[str, Any]] = {}
    for rank, ds in enumerate(semantic_results, start=1):
        source_id, native_id = split_dataset_id(ds["id"])
        ds_id = qualify_dataset_id(source_id, native_id)
        ds = {**ds, "id": ds_id}
        semantic_map[ds_id] = ds
        rrf_scores[ds_id] = rrf_scores.get(ds_id, 0.0) + (1.0 / (60.0 + rank))
        
    # Give each portal its own rank list so a large catalog cannot swamp smaller ones.
    keyword_ids = set()
    for source_results in keyword_results.values():
        for rank, ds in enumerate(source_results, start=1):
            ds_id = ds["id"]
            keyword_ids.add(ds_id)
            rrf_scores[ds_id] = rrf_scores.get(ds_id, 0.0) + (1.0 / (60.0 + rank))
            
    # Sort IDs by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 4. Prefer indexed metadata, but retain live results not yet present locally.
    all_details = get_by_ids(sorted_ids)
    for ds_id, record in live_details.items():
        all_details.setdefault(ds_id, record)
    
    fused_results = []
    for ds_id in sorted_ids:
        if ds_id in all_details:
            ds = all_details[ds_id]
            ds["rrf_score"] = rrf_scores[ds_id]
            ds["distance"] = semantic_map[ds_id]["distance"] if ds_id in semantic_map else None
            
            # Check if this hit came from keyword, semantic, or both
            is_semantic = ds_id in semantic_map
            is_keyword = ds_id in keyword_ids
            if is_semantic and is_keyword:
                ds["match_type"] = "Hybrid"
            elif is_semantic:
                ds["match_type"] = "Semantic"
            else:
                ds["match_type"] = "Keyword"
            fused_results.append(ds)

    # Remove the same authoritative dataset when it is federated across portals.
    deduplicated = []
    seen_resource_urls: set[str] = set()
    seen_pages: set[str] = set()
    for dataset in fused_results:
        resource_urls = _canonical_resource_urls(dataset)
        public_page = str(
            dataset.get("page_url") or dataset.get("id", "")
        ).lower().rstrip("/")
        if resource_urls & seen_resource_urls or (
            not resource_urls and public_page in seen_pages
        ):
            continue
        seen_resource_urls.update(resource_urls)
        seen_pages.add(public_page)
        deduplicated.append(dataset)
    fused_results = deduplicated[:effective_limit]

    if not fused_results:
        return make_tool_result(
            f"No queryable tabular datasets found matching: '{query}'",
            query=query_info,
            warnings=[*warnings, "Matches were found, but none were present in the queryable local catalog."],
        )

    md = [f"### Hybrid Search Results (RRF) for '{query}' (showing top {len(fused_results)})\n"]
    datasets: List[Dict[str, Any]] = []
    sources: List[SourceReference] = []
    for ds in fused_results:
        title = ds["title"]
        org = ds["org"] or "Unknown Publisher"
        desc = _truncate_desc(ds["notes"])
        ds_id = ds["id"]
        
        # Resources summary
        resources = ds["resources"]
        csv_count = sum(1 for r in resources if r["format"] == "CSV")
        xlsx_count = sum(1 for r in resources if r["format"] in ("XLSX", "XLS"))
        parquet_count = sum(1 for r in resources if r["format"] == "PARQUET")
        json_count = sum(1 for r in resources if r["format"] == "JSON")
        zip_count = sum(1 for r in resources if r["format"] == "ZIP")
        pdf_count = sum(1 for r in resources if r["format"] == "PDF")
        datastore_count = sum(1 for r in resources if r["datastore_active"])
        
        res_summary_parts = []
        if csv_count:
            res_summary_parts.append(f"{csv_count} CSV")
        if xlsx_count:
            res_summary_parts.append(f"{xlsx_count} Excel")
        if parquet_count:
            res_summary_parts.append(f"{parquet_count} Parquet")
        if json_count:
            res_summary_parts.append(f"{json_count} JSON")
        if zip_count:
            res_summary_parts.append(f"{zip_count} ZIP")
        if pdf_count:
            res_summary_parts.append(f"{pdf_count} PDF")
            
        res_summary = ", ".join(res_summary_parts) if res_summary_parts else "No tabular files"
        if datastore_count:
            res_summary += f" ({datastore_count} API-enabled datastores)"
            
        source_id = ds.get("source_id", split_dataset_id(ds_id)[0])
        source_type = ds.get("source_type", "ckan")
        native_id = ds.get("native_id", split_dataset_id(ds_id)[1])
        page = ds.get("page_url") or page_url(source_id, native_id)

        resource_counts = {
            "csv": csv_count,
            "excel": xlsx_count,
            "parquet": parquet_count,
            "json": json_count,
            "zip": zip_count,
            "pdf": pdf_count,
            "datastore": datastore_count,
        }
        datasets.append({
            "id": ds_id,
            "native_id": native_id,
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "publisher": org,
            "description": desc,
            "page_url": page,
            "metadata_modified": ds.get("metadata_modified", ""),
            "resources": resources,
            "metadata": ds.get("metadata", {}),
            "resource_counts": resource_counts,
            "match_type": ds["match_type"],
            "rrf_score": ds["rrf_score"],
            "cosine_distance": ds["distance"],
        })
        sources.append(SourceReference(
            title=title,
            url=page,
            dataset_id=ds_id,
        ))
        
        md.append(f"**[{title}]({page})**")
        md.append(f"- **Publisher**: {org}")
        md.append(f"- **Catalog source**: {get_source(source_id).name}")
        md.append(f"- **Description**: {desc}")
        md.append(f"- **Resources**: {res_summary}")
        md.append(f"- **Dataset id**: `{ds_id}` → `get_dataset('{ds_id}')`")
        
        match_info = f"Match: {ds['match_type']} (RRF: {ds['rrf_score']:.4f})"
        if ds["distance"] is not None:
            match_info += f" | Cosine Distance: {ds['distance']:.4f}"
        md.append(f"- **Search Info**: {match_info}")
        md.append("")
        
    return make_tool_result(
        "\n".join(md),
        datasets=datasets,
        sources=sources,
        query=query_info,
        warnings=warnings,
    )


@mcp.tool()
@log_telemetry("search_datasets")
def search_datasets(query: str, limit: int = 5, source_id: str = "canada") -> str:
    """
    Search one registered CKAN portal for datasets.
    Uses the CKAN package_search API.

    Args:
        query: Keywords (e.g., "alberta well licences", "population census").
        limit: Max datasets to return (default 5).
        source_id: canada, alberta, or ontario (default canada).

    Returns:
        Markdown list of datasets with title, organization, description, and the
        dataset id to pass to get_dataset.
    """
    try:
        source = get_source(source_id)
        result = _ckan_get(
            "package_search", source_id=source.id,
            q=query, rows=max(1, min(limit, 25))
        )
    except Exception as e:
        return f"Error searching source '{source_id}': {e}"

    results = result.get("results", [])
    if not results:
        return f"No datasets found for '{query}'."

    md = [f"### {source.name} results for '{query}' "
          f"({result.get('count', 0)} total, showing {len(results)})\n"]
    for ds in results:
        title = ds.get("title") or ds.get("name", "Untitled")
        org = (ds.get("organization") or {}).get("title", "Unknown org")
        desc = _truncate_desc(ds.get("notes", ""))
        native_id = ds.get("id", "")
        ds_id = qualify_dataset_id(source.id, native_id)
        n_res = len(ds.get("resources", []))
        page = source.page_url(native_id)
        md.append(f"**[{title}]({page})**")
        md.append(f"- **Publisher**: {org}")
        md.append(f"- **Description**: {desc}")
        md.append(f"- **Resources**: {n_res}")
        md.append(f"- **Dataset id**: `{ds_id}` → `get_dataset('{ds_id}')`")
        md.append("")
    return "\n".join(md)


@mcp.tool(structured_output=True)
@log_telemetry("get_dataset")
def get_dataset(dataset_id: str) -> StructuredToolResult:
    """
    Get a dataset's metadata and resources from its authoritative source.
    Accepts a source-qualified ID, a legacy federal ID, or a known dataset URL.

    For each resource it reports the format and whether it is datastore-backed:
      - datastore_active: true  → query it server-side with query_datastore(resource_id, ...)
      - datastore_active: false → query the file with query_remote_file(url, sql)

    Returns:
        Markdown: title, org, description, and a resource table with ids and formats.
    """
    try:
        source_id, native_id = split_dataset_id(dataset_id)
        source = get_source(source_id)
        ds_id = qualify_dataset_id(source_id, native_id)
    except ValueError as error:
        return make_error_result(
            str(error), code="InvalidDatasetId",
            query={"dataset_id": dataset_id, "operation": "get_dataset"},
            recovery="Use a result ID returned by semantic_search_datasets.",
        )

    query_info = {
        "dataset_id": ds_id,
        "native_id": native_id,
        "source_id": source_id,
        "operation": "package_show" if source.source_type == "ckan" else "catalog_lookup",
    }
    if source.source_type == "boc_valet":
        return make_error_result(
            f"'{ds_id}' is a Bank of Canada Valet identifier, not an indexed dataset.",
            code="UnsupportedDatasetSource",
            query=query_info,
            recovery=(
                f"Call query_boc_valet directly — use resource='series' or "
                f"'observations' with series_names='{native_id}', or "
                f"resource='group'/'group_observations' with group_name='{native_id}' "
                "if this is a group name."
            ),
        )
    try:
        if source.source_type == "ckan":
            raw = _ckan_get("package_show", source_id=source_id, id=native_id)
            ds = _catalog_record_from_ckan(raw, source_id, require_supported=False)
        else:
            ds = get_by_ids([ds_id]).get(ds_id)
        if not ds:
            raise LookupError("Dataset was not found or has no supported resources.")
    except Exception as e:
        return make_error_result(
            f"Could not fetch dataset '{ds_id}': {e}",
            code="DatasetFetchFailed",
            query=query_info,
            retryable=source.source_type == "ckan",
            recovery="Verify the source-qualified dataset ID and rebuild that source's catalog metadata.",
        )

    title = ds.get("title") or "Untitled"
    org = ds.get("org") or "Unknown org"
    desc = _truncate_desc(ds.get("notes", ""))
    page = ds.get("page_url") or source.page_url(native_id)

    md = [f"## [{title}]({page})", f"**Publisher**: {org}",
          f"**Source**: {page}", f"**Description**: {desc}", ""]
    resources = ds.get("resources", [])
    register_dataset_resources(ds_id, resources)
    md.append(f"### Resources ({len(resources)})\n")

    rows = []
    for r in resources:
        rows.append({
            "name": (r.get("name") or "—")[:60],
            "format": (r.get("format") or "?").upper(),
            "datastore": "✅ yes" if r.get("datastore_active") else "no",
            "resource_id": r.get("id", ""),
            "url": r.get("url", ""),
        })
    if rows:
        md.append(pd.DataFrame(rows).to_markdown(index=False))

    queryable = [r for r in rows if r["datastore"].startswith("✅")]
    md.append("")
    if queryable:
        rid = queryable[0]["resource_id"]
        md.append(f"**Tip**: `{queryable[0]['name']}` is datastore-backed → "
                  f"`query_datastore('{rid}', q='...', source_id='{source_id}')` "
                  "(server-side, no download).")
    else:
        md.append("**Tip**: no datastore-backed resources here — use "
                  "`query_remote_file(url, sql)` on a CSV/Parquet resource above.")
    structured_resources = []
    for raw_resource, row in zip(resources, rows):
        structured_resources.append({
            "id": row["resource_id"],
            "name": row["name"],
            "format": row["format"],
            "url": row["url"],
            "datastore_active": bool(raw_resource.get("datastore_active")),
        })

    warnings = [] if resources else ["This dataset does not list any resources."]
    return make_tool_result(
        "\n".join(md),
        datasets=[{
            "id": ds_id,
            "native_id": native_id,
            "source_id": source_id,
            "source_type": source.source_type,
            "title": title,
            "publisher": org,
            "description": desc,
            "page_url": page,
            "metadata_modified": ds.get("metadata_modified", ""),
            "resources": structured_resources,
            "metadata": ds.get("metadata", {}),
        }],
        sources=[SourceReference(title=title, url=page, dataset_id=ds_id)],
        query=query_info,
        warnings=warnings,
    )


@mcp.tool(structured_output=True)
@log_telemetry("query_statcan_wds")
def query_statcan_wds(
    method: str,
    product_id: str = "",
    coordinate: str = "",
    vector_ids: str = "",
    latest_n: int = 3,
    start_date: str = "",
    end_date: str = "",
    language: str = "en",
    max_items: int = 25,
) -> StructuredToolResult:
    """Call an official Statistics Canada Web Data Service method.

    This is a read-only, allowlisted gateway to the WDS methods documented by
    Statistics Canada. Only parameters used by the selected method are read.
    Responses are bounded before being returned to the MCP client.

    Common examples:
      - method="getCubeMetadata", product_id="17100009"
      - method="getDataFromVectorsAndLatestNPeriods",
        vector_ids="V1,V2", latest_n=3
      - method="getDataFromCubePidCoordAndLatestNPeriods",
        product_id="35100003", coordinate="1.12.0.0.0.0.0.0.0.0", latest_n=3
      - method="getChangedCubeList", start_date="2026-07-31"
      - method="getFullTableDownloadCSV", product_id="17100009", language="en"

    Args:
        method: Exact official WDS method name.
        product_id: StatCan PID, statcan-qualified ID, or StatCan table URL.
        coordinate: Dot-separated dimension member coordinate.
        vector_ids: Comma-separated vector IDs, with optional V prefixes.
        latest_n: Latest reference periods to return (1-100).
        start_date: ISO start/release date required by range/change methods.
        end_date: ISO end date required by range methods.
        language: Full CSV language, either en or fr.
        max_items: Maximum items retained in each response list (1-100).
    """
    query_info: Dict[str, Any] = {
        "source_id": "statcan",
        "operation": "wds",
        "method": method,
        "product_id": product_id,
        "coordinate": coordinate,
        "vector_ids": vector_ids,
        "latest_n": latest_n,
        "start_date": start_date,
        "end_date": end_date,
        "language": language,
        "max_items": max_items,
    }
    if not 1 <= max_items <= MAX_STATCAN_WDS_ITEMS:
        return make_error_result(
            f"max_items must be between 1 and {MAX_STATCAN_WDS_ITEMS}.",
            code="InvalidStatCanWDSRequest",
            query=query_info,
            recovery="Choose a max_items value in the supported range.",
        )

    try:
        body, http_method, endpoint = _statcan_wds_request(
            method,
            product_id=product_id,
            coordinate=coordinate,
            vector_ids=vector_ids,
            latest_n=latest_n,
            start_date=start_date,
            end_date=end_date,
            language=language,
        )
    except ValueError as error:
        return make_error_result(
            str(error),
            code="InvalidStatCanWDSRequest",
            query=query_info,
            recovery="Use an official method name and provide the parameters required by that method.",
        )
    except requests.RequestException as error:
        return make_error_result(
            f"Statistics Canada WDS request failed: {error}",
            code="StatCanWDSUnavailable",
            query=query_info,
            retryable=True,
            recovery="Retry the request; tables may be temporarily locked during Statistics Canada updates.",
        )
    except Exception as error:
        return make_error_result(
            f"Statistics Canada WDS returned an invalid response: {error}",
            code="InvalidStatCanWDSResponse",
            query=query_info,
            retryable=True,
            recovery="Retry the request and verify the method parameters against the WDS user guide.",
        )

    state = {"truncated": False}
    bounded = _truncate_json_response(body, max_items, state)
    rows = _json_rows(bounded)
    warnings = []
    if state["truncated"]:
        warnings.append(
            f"The WDS response was truncated to {max_items} items per list; "
            "narrow the request or increase max_items for more."
        )

    query_info.update({"http_method": http_method, "endpoint": endpoint})
    rendered = json.dumps(bounded, ensure_ascii=False, indent=2)
    if len(rendered) > 12000:
        rendered = rendered[:12000] + "\n…"
        warnings.append("The Markdown rendering was shortened; structured rows contain the bounded response.")

    dataset_id = None
    if product_id:
        try:
            pid = str(_statcan_product_id(product_id))
            dataset_id = f"statcan:{pid[:-2] if len(pid) == 10 else pid}"
        except ValueError:
            pass
    return make_tool_result(
        f"## Statistics Canada WDS: `{method}`\n\n```json\n{rendered}\n```",
        rows=rows,
        sources=[SourceReference(
            title=f"Statistics Canada WDS — {method}",
            url=STATCAN_WDS_GUIDE_URL,
            dataset_id=dataset_id,
        )],
        query=query_info,
        warnings=warnings,
    )


@mcp.tool(structured_output=True)
@log_telemetry("query_boc_valet")
def query_boc_valet(
    resource: str,
    series_names: str = "",
    group_name: str = "",
    start_date: str = "",
    end_date: str = "",
    recent: int = 0,
    recent_weeks: int = 0,
    recent_months: int = 0,
    recent_years: int = 0,
    order_dir: str = "",
    max_items: int = 25,
) -> StructuredToolResult:
    """Call the Bank of Canada Valet API for interest rate, FX, and other series.

    A read-only, allowlisted gateway to Valet's REST resources. No API key is
    required. Series and group names are Valet identifiers (e.g. "FXUSDCAD",
    "FX_RATES_DAILY"), not opendata.fyi dataset IDs — use resource="series_list"
    or resource="group_list" to discover them.

    Common examples:
      - resource="series", series_names="FXUSDCAD"
      - resource="observations", series_names="FXUSDCAD,FXEURCAD", recent=10
      - resource="observations", series_names="FXUSDCAD",
        start_date="2026-01-01", end_date="2026-01-31", order_dir="desc"
      - resource="group", group_name="FX_RATES_DAILY"
      - resource="group_observations", group_name="FX_RATES_DAILY", recent_weeks=2
      - resource="series_list"  (full series catalogue, bounded by max_items)
      - resource="group_list"   (full group catalogue, bounded by max_items)

    Args:
        resource: One of series, observations, group, group_observations,
            series_list, group_list.
        series_names: Comma-separated Valet series names (series/observations).
        group_name: A single Valet group name (group/group_observations).
        start_date: ISO start date for observations.
        end_date: ISO end date for observations.
        recent: Most recent N observations. Mutually exclusive with the other
            recent_* parameters and with start_date/end_date.
        recent_weeks: Most recent N weeks of observations.
        recent_months: Most recent N months of observations.
        recent_years: Most recent N years of observations.
        order_dir: Observation sort order, asc or desc.
        max_items: Maximum items retained in each response list (1-100).
    """
    query_info: Dict[str, Any] = {
        "source_id": "boc",
        "operation": "valet",
        "resource": resource,
        "series_names": series_names,
        "group_name": group_name,
        "start_date": start_date,
        "end_date": end_date,
        "recent": recent,
        "recent_weeks": recent_weeks,
        "recent_months": recent_months,
        "recent_years": recent_years,
        "order_dir": order_dir,
        "max_items": max_items,
    }
    if not 1 <= max_items <= MAX_BOC_VALET_ITEMS:
        return make_error_result(
            f"max_items must be between 1 and {MAX_BOC_VALET_ITEMS}.",
            code="InvalidBoCValetRequest",
            query=query_info,
            recovery="Choose a max_items value in the supported range.",
        )

    try:
        body, endpoint = _boc_valet_request(
            resource,
            series_names=series_names,
            group_name=group_name,
            start_date=start_date,
            end_date=end_date,
            recent=recent,
            recent_weeks=recent_weeks,
            recent_months=recent_months,
            recent_years=recent_years,
            order_dir=order_dir,
        )
    except ValueError as error:
        return make_error_result(
            str(error),
            code="InvalidBoCValetRequest",
            query=query_info,
            recovery="Use an allowed resource and provide the parameters it requires.",
        )
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status == 404:
            return make_error_result(
                f"Bank of Canada Valet found no match for this series or group: {error}",
                code="BoCValetNotFound",
                query=query_info,
                recovery="Verify the exact name with resource='series_list' or resource='group_list'.",
            )
        return make_error_result(
            f"Bank of Canada Valet request failed: {error}",
            code="BoCValetUnavailable",
            query=query_info,
            retryable=True,
            recovery="Retry the request; Valet may be temporarily unavailable.",
        )
    except requests.RequestException as error:
        return make_error_result(
            f"Bank of Canada Valet request failed: {error}",
            code="BoCValetUnavailable",
            query=query_info,
            retryable=True,
            recovery="Retry the request; Valet may be temporarily unavailable.",
        )
    except Exception as error:
        return make_error_result(
            f"Bank of Canada Valet returned an invalid response: {error}",
            code="InvalidBoCValetResponse",
            query=query_info,
            retryable=True,
            recovery="Retry the request and verify the resource parameters against the Valet docs.",
        )

    state = {"truncated": False}
    if resource == "series_list":
        bounded, rows, state["truncated"] = _boc_catalog_rows(body, "series", max_items)
    elif resource == "group_list":
        bounded, rows, state["truncated"] = _boc_catalog_rows(body, "groups", max_items)
    else:
        bounded = _truncate_json_response(body, max_items, state)
        rows = _json_rows(bounded)
    warnings = []
    if state["truncated"]:
        warnings.append(
            f"The Valet response was truncated to {max_items} items; "
            "narrow the request or increase max_items for more."
        )

    query_info["endpoint"] = endpoint
    rendered = json.dumps(bounded, ensure_ascii=False, indent=2)
    if len(rendered) > 12000:
        rendered = rendered[:12000] + "\n…"
        warnings.append("The Markdown rendering was shortened; structured rows contain the bounded response.")

    dataset_id = None
    try:
        if resource in ("series", "observations") and series_names:
            dataset_id = f"boc:{_boc_series_names(series_names)[0]}"
        elif resource in ("group", "group_observations") and group_name:
            dataset_id = f"boc:{_boc_group_name(group_name)}"
    except ValueError:
        pass

    return make_tool_result(
        f"## Bank of Canada Valet: `{resource}`\n\n```json\n{rendered}\n```",
        rows=rows,
        sources=[SourceReference(
            title=f"Bank of Canada Valet API — {resource}",
            url=BOC_VALET_DOCS_URL,
            dataset_id=dataset_id,
        )],
        query=query_info,
        warnings=warnings,
    )


# =====================================================================
# SERVER-SIDE QUERY (datastore)  — the fast path, no download
# =====================================================================
@mcp.tool()
@log_telemetry("get_resource_fields")
def get_resource_fields(resource_id: str, source_id: str = "canada") -> str:
    """
    Get the column names and types for a datastore-backed resource — no data download.
    Call this before query_datastore to learn what columns/filters are available.

    Args:
        resource_id: The resource UUID (from get_dataset).
        source_id: CKAN source containing the resource (default canada).

    Returns:
        Markdown table of field id + type, plus the resource's total row count.
    """
    try:
        result = _ckan_get(
            "datastore_search", source_id=source_id,
            resource_id=resource_id, limit=0
        )
    except Exception as e:
        return (f"Error reading fields for '{resource_id}': {e}\n"
                f"(This resource may not be datastore-backed; try query_remote_file.)")
    fields = [f for f in result.get("fields", []) if f.get("id") != "_id"]
    if not fields:
        return "No fields returned (resource may not be datastore-backed)."
    df = pd.DataFrame([{"field": f["id"], "type": f.get("type", "?")} for f in fields])
    return (f"### Fields for `{resource_id}`  (total rows: {result.get('total', '?')})\n\n"
            + df.to_markdown(index=False))


@mcp.tool(structured_output=True)
@log_telemetry("query_datastore")
def query_datastore(resource_id: str, q: str = "", filters: str = "",
                    sort: str = "", limit: int = 50,
                    offset: int = 0, source_id: str = "canada") -> StructuredToolResult:
    """
    Query a datastore-backed resource SERVER-SIDE via CKAN datastore_search.
    No file download — the portal's database does the filtering. Use this for large
    resources (the fast path for "find these rows in a huge table").

    CRITICAL: You must cite the dataset source URL in your response to the user.
    The source URL is included at the end of the tool's return text.

    Args:
        resource_id: Resource UUID (from get_dataset).
        q: Full-text search across all columns (e.g., "Calgary").
        filters: JSON object of exact-match filters, e.g. '{"province": "Alberta"}'.
        sort: Sort spec, e.g. "year desc" or "name asc".
        limit: Max rows to return (default 50, capped at 100).
        offset: Row offset for pagination (default 0).
        source_id: CKAN source containing the resource: canada, alberta, or ontario.

    Returns:
        Markdown table of matching rows with a pagination hint.
    """
    effective_limit = max(1, min(limit, MAX_RESULT_ROWS))
    effective_offset = max(0, offset)
    try:
        source = get_source(source_id)
        if source.source_type != "ckan":
            raise ValueError("Datastore queries require a CKAN source.")
    except ValueError as error:
        return make_error_result(
            str(error), code="InvalidSource",
            query={"operation": "datastore_search", "source_id": source_id},
            recovery="Use canada, alberta, or ontario; use query_remote_file for ordinary files.",
        )
    parsed_filters: Dict[str, Any] = {}
    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "limit": effective_limit,
        "offset": effective_offset,
    }
    if q:
        params["q"] = q
    if sort:
        params["sort"] = sort
    if filters:
        try:
            parsed_filters = json.loads(filters)
            if not isinstance(parsed_filters, dict):
                raise ValueError("filters must be a JSON object")
            params["filters"] = filters
        except (json.JSONDecodeError, ValueError):
            return make_error_result(
                "`filters` must be a valid JSON object, for example {\"province\": \"Alberta\"}.",
                code="InvalidFilters",
                query={
                    "operation": "datastore_search",
                    "resource_id": resource_id,
                    "source_id": source_id,
                    "q": q,
                    "sort": sort,
                    "limit": effective_limit,
                    "offset": effective_offset,
                },
                recovery="Pass exact-match filters as a JSON object encoded as a string.",
            )

    query_info = {
        "operation": "datastore_search",
        "resource_id": resource_id,
        "source_id": source_id,
        "q": q,
        "filters": parsed_filters,
        "sort": sort,
        "limit": effective_limit,
        "offset": effective_offset,
    }

    try:
        result = _ckan_get("datastore_search", source_id=source_id, **params)
    except Exception as e:
        return make_error_result(
            f"Could not query datastore '{resource_id}': {e}",
            code="DatastoreQueryFailed",
            query=query_info,
            retryable=True,
            recovery="If this resource is not datastore-backed, use query_remote_file with its download URL.",
        )

    records = result.get("records", [])

    # Resolve the parent dataset URL for citation
    citation = ""
    sources: List[SourceReference] = []
    warnings: List[str] = []
    try:
        res_info = _ckan_get("resource_show", source_id=source_id, id=resource_id)
        pkg_id = res_info.get("package_id", "")
        if pkg_id:
            catalog_id = qualify_dataset_id(source_id, pkg_id)
            page = source.page_url(pkg_id)
            citation = (f"\n\n---\n**Source**: "
                        f"[{source.name}]({page})")
            sources.append(SourceReference(
                title=f"{source.name} dataset",
                url=page,
                dataset_id=catalog_id,
                resource_id=resource_id,
            ))
    except Exception as e:
        warnings.append(
            f"The parent dataset citation could not be resolved ({type(e).__name__})."
        )

    if not records:
        return make_tool_result(
            f"Query ran, but no rows matched (total in resource: {result.get('total', '?')}).",
            sources=sources,
            query={**query_info, "total": result.get("total")},
            warnings=[*warnings, "The query completed successfully but matched no rows."],
        )

    df = _clean_df(pd.DataFrame(records))
    columns, structured_rows = _df_to_structured(df, cap=effective_limit)

    markdown = (f"### {result.get('total', len(records))} rows matched "
                f"(showing {len(records)} from offset {effective_offset})\n\n"
                + _df_to_md(df, cap=effective_limit, offset=effective_offset,
                            total=result.get("total"))
                + citation)
    return make_tool_result(
        markdown,
        columns=columns,
        rows=structured_rows,
        sources=sources,
        query={**query_info, "total": result.get("total")},
        warnings=warnings,
    )


# =====================================================================
# FILE FALLBACK (DuckDB)  — for resources without a datastore
# =====================================================================
@mcp.tool()
@log_telemetry("preview_remote_file")
def preview_remote_file(file_url: str, max_rows: int = MAX_PREVIEW_ROWS,
                        sheet_name: str = "") -> str:
    """
    Preview the first rows of a remote CSV/JSON/Parquet/XLSX resource via DuckDB.
    Use for resources where datastore_active is false (plain file downloads).

    CRITICAL: You must cite the dataset source URL in your response to the user.
    The source URL is included at the end of the tool's return text.

    Args:
        file_url: Direct resource download URL (from get_dataset).
        max_rows: Rows to preview (default 15).
        sheet_name: For Excel only — which sheet to preview (default: first sheet).
                    Call list_excel_sheets first to see the available sheets.
    """
    if not file_url:
        return "Error: No file URL provided."
    low = file_url.lower()
    citation = f"\n\n---\n**Source**: [{file_url}]({file_url})"
    try:
        if low.endswith((".xlsx", ".xls")):
            df = _read_tabular(file_url, nrows=max_rows,
                               sheet_name=sheet_name or 0)
            label = f" (sheet: {sheet_name})" if sheet_name else ""
            return f"### Preview{label}\n\n" + _df_to_md(_clean_df(df), cap=max_rows) + citation
        if low.endswith((".zip", ".json")):
            df = _read_tabular(file_url, nrows=max_rows)
        else:
            df = run_query_with_timeout(
                f"SELECT * FROM '{file_url}' LIMIT {int(max_rows)}"
            )
        return "### Preview\n\n" + _df_to_md(_clean_df(df), cap=max_rows) + citation
    except Exception as e:
        return f"Error previewing file: {e}"


@mcp.tool()
@log_telemetry("get_file_schema")
def get_file_schema(file_url: str, sheet_name: str = "") -> str:
    """
    Get column names and types for a remote CSV/Parquet/JSON file via DuckDB DESCRIBE
    (minimal download). Use for non-datastore resources before query_remote_file.

    Args:
        file_url: Direct resource download URL.
        sheet_name: For Excel only — which sheet's schema to read (default: first).
    """
    if not file_url:
        return "Error: No file URL provided."
    low = file_url.lower()
    # Excel and ZIP must be read via pandas — DuckDB can't DESCRIBE them remotely.
    if low.endswith((".xlsx", ".xls")) or low.endswith(".zip"):
        try:
            df = _read_tabular(file_url, nrows=5, sheet_name=sheet_name or 0)
            schema = pd.DataFrame(
                [{"column_name": c, "column_type": str(df[c].dtype)} for c in df.columns]
            )
            label = f" (sheet: {sheet_name})" if sheet_name else ""
            suffix = " (extracted from ZIP)" if low.endswith(".zip") else label
            return f"### Schema (sampled){suffix}\n\n" + schema.to_markdown(index=False)
        except Exception as e:
            return f"Error reading file schema: {e}"
    try:
        df = run_query_with_timeout(f"DESCRIBE SELECT * FROM '{file_url}'")
        return "### Schema\n\n" + df[["column_name", "column_type"]].to_markdown(index=False)
    except Exception as e:
        # Fallback for formats DuckDB can't DESCRIBE remotely (json)
        try:
            df = _read_tabular(file_url, nrows=5)
            schema = pd.DataFrame(
                [{"column_name": c, "column_type": str(df[c].dtype)} for c in df.columns]
            )
            return "### Schema (sampled)\n\n" + schema.to_markdown(index=False)
        except Exception as e2:
            return f"Error reading schema: {e} / {e2}"


@mcp.tool()
@log_telemetry("list_excel_sheets")
def list_excel_sheets(file_url: str) -> str:
    """
    List every sheet in a remote Excel workbook with its row/column counts and columns.
    Many open.canada.ca Excel resources are multi-sheet and are NOT datastore-backed,
    so use this to see what's inside before previewing or querying a specific sheet.

    Args:
        file_url: Direct URL to an .xlsx/.xls resource (from get_dataset).
    """
    if not file_url:
        return "Error: No file URL provided."
    if not file_url.lower().endswith((".xlsx", ".xls")):
        return "Error: not an Excel file. Use get_file_schema for CSV/Parquet/JSON."
    try:
        xl = _excel_file(file_url)
    except Exception as e:
        return f"Error opening workbook: {e}"

    rows = []
    for name in xl.sheet_names:
        try:
            hdr = _detect_header_row(xl, name)
            df = xl.parse(name, header=hdr, nrows=200)
            df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed: \d+$")]
            cols = ", ".join(map(str, df.columns[:12]))
            if len(df.columns) > 12:
                cols += ", …"
            hdr_note = f" (header row {hdr})" if hdr > 0 else ""
            rows.append({"sheet": name, "columns": len(df.columns),
                         "rows(sampled)": len(df),
                         "header_row": hdr,
                         "column names": cols[:160] + hdr_note})
        except Exception as e:
            rows.append({"sheet": name, "columns": "?", "rows(sampled)": "?",
                         "header_row": "?", "column names": f"(error: {e})"})
    md = (f"### Workbook sheets ({len(xl.sheet_names)})\n\n"
          + pd.DataFrame(rows).to_markdown(index=False))
    md += ("\n\n_Next: `preview_remote_file(url, sheet_name='<sheet>')` or "
           "`query_excel_sheet(url, '<sheet>', sql)`._")
    return md


@mcp.tool()
@log_telemetry("query_excel_sheet")
def query_excel_sheet(file_url: str, sheet_name: str, sql_query: str) -> str:
    """
    Run a read-only DuckDB SQL query against a single sheet of a remote Excel workbook.
    Use '{sheet}' as the table placeholder. Call list_excel_sheets first to get names.

    CRITICAL: You must cite the dataset source URL in your response to the user.
    The source URL is included at the end of the tool's return text.

    Example:
        query_excel_sheet(url, "2024 Data",
            "SELECT region, SUM(amount) FROM '{sheet}' GROUP BY region")

    Args:
        file_url: Direct URL to an .xlsx/.xls resource.
        sheet_name: Exact sheet name (from list_excel_sheets).
        sql_query: SELECT query using '{sheet}' as the table reference.
    """
    if not (file_url and sheet_name and sql_query):
        return "Error: file_url, sheet_name and sql_query are all required."
    if _WRITE_RE.search(sql_query):
        return "Error: only read-only SELECT queries are permitted."
    try:
        df = _read_tabular(file_url, sheet_name=sheet_name)  # noqa: F841 (used by DuckDB)
    except Exception as e:
        return f"Error reading sheet '{sheet_name}': {e}"

    sql = sql_query.replace("'{sheet}'", "df").replace("{sheet}", "df")
    try:
        out = _query_dataframe(sql, df)
    except Exception as e:
        return f"Error executing query: {e}"
    if out.empty:
        return "Query ran successfully but returned 0 rows."
    citation = f"\n\n---\n**Source**: [{file_url}]({file_url}) — sheet: `{sheet_name}`"
    return _df_to_md(_clean_df(out)) + citation


@mcp.tool(structured_output=True)
@log_telemetry("query_remote_file")
def query_remote_file(file_url: str, sql_query: str) -> StructuredToolResult:
    """
    Run a read-only DuckDB SQL query directly on a remote file (CSV/Parquet/JSON/ZIP).
    Use '{file}' as the table placeholder. For datastore-backed resources prefer
    query_datastore (faster, server-side). ZIP files containing CSV are fully supported.

    CRITICAL: You must cite the dataset source URL in your response to the user.
    The source URL is included at the end of the tool's return text.

    Example:
        query_remote_file(url, "SELECT province, COUNT(*) FROM '{file}' GROUP BY province")

    Args:
        file_url: Direct resource download URL (CSV, Parquet, JSON, or ZIP of CSV).
        sql_query: SELECT query using '{file}' as the table reference.
    """
    query_info = {
        "operation": "remote_file_query",
        "file_url": file_url,
        "sql": sql_query,
        "row_limit": MAX_RESULT_ROWS,
    }
    if not file_url or not sql_query:
        return make_error_result(
            "Both file_url and sql_query are required.",
            code="MissingArgument",
            query=query_info,
            recovery="Pass a direct CSV, Parquet, JSON, or ZIP URL and a SELECT query using {file}.",
        )

    sql_error = _read_only_sql_error(sql_query)
    if sql_error:
        return make_error_result(
            sql_error,
            code="UnsafeOrInvalidSQL",
            query=query_info,
            recovery="Use one read-only SELECT query and refer to the input as '{file}'.",
        )

    citation = f"\n\n---\n**Source**: [{file_url}]({file_url})"
    sources = [SourceReference(title="Remote data resource", url=file_url)]
    low = file_url.lower()
    warnings: List[str] = []

    # ZIP files must be downloaded and extracted first — DuckDB httpfs can't unzip
    if low.endswith(".zip"):
        try:
            df = _read_tabular(file_url)
        except Exception as e:
            return make_error_result(
                f"Could not read ZIP file: {e}",
                code="RemoteFileReadFailed",
                query=query_info,
                retryable=True,
                recovery="Verify that the URL is public and the ZIP contains a supported tabular file.",
            )
        sql = sql_query.replace("'{file}'", "df").replace("{file}", "df")
        try:
            out = _query_dataframe(sql, df)
        except Exception as e:
            return make_error_result(
                f"Could not execute the query on ZIP contents: {e}",
                code="RemoteQueryFailed",
                query=query_info,
                recovery="Inspect the file schema and verify all column names in the SQL query.",
            )
        if out.empty:
            return make_tool_result(
                "Query ran successfully but returned 0 rows." + citation,
                sources=sources,
                query=query_info,
                warnings=["The query completed successfully but returned no rows."],
            )
        clean_out = _clean_df(out)
        columns, rows = _df_to_structured(clean_out)
        return make_tool_result(
            _df_to_md(clean_out) + citation,
            columns=columns,
            rows=rows,
            sources=sources,
            query=query_info,
        )

    # Standard DuckDB httpfs path for CSV / Parquet / JSON
    sql = sql_query.replace("'{file}'", f"'{file_url}'").replace("{file}", f"'{file_url}'")
    try:
        df = run_query_with_timeout(sql)
    except TimeoutError as e:
        return make_error_result(
            str(e),
            code="QueryTimeout",
            query=query_info,
            retryable=True,
            recovery="Add filters or a LIMIT, or use query_datastore when the resource is API-backed.",
        )
    except Exception as e:
        # DuckDB httpfs uses its own HTTP stack (no User-Agent, no TLS 1.2
        # fallback) and gets reset by some government hosts that _robust_get
        # handles fine. Download the bytes ourselves and query the buffer.
        try:
            df_local = _read_tabular(file_url)
            local_sql = sql_query.replace("'{file}'", "df").replace("{file}", "df")
            df = _query_dataframe(local_sql, df_local)
        except Exception as e2:
            return make_error_result(
                f"Remote query failed ({e}); local-download fallback also failed ({e2}).",
                code="RemoteQueryFailed",
                query=query_info,
                retryable=True,
                recovery="Verify the URL and format, then call get_file_schema before retrying the query.",
            )
        warnings.append(
            f"DuckDB HTTP streaming failed ({type(e).__name__}); the query succeeded using a local-download fallback."
        )
    if df.empty:
        return make_tool_result(
            "Query ran successfully but returned 0 rows." + citation,
            sources=sources,
            query=query_info,
            warnings=[*warnings, "The query completed successfully but returned no rows."],
        )
    clean_df = _clean_df(df)
    columns, rows = _df_to_structured(clean_df)
    return make_tool_result(
        _df_to_md(clean_df) + citation,
        columns=columns,
        rows=rows,
        sources=sources,
        query=query_info,
        warnings=warnings,
    )


@mcp.tool()
@log_telemetry("read_pdf")
def read_pdf(file_url: str, pages: str = "1-10") -> str:
    """
    Extract text from a remote PDF resource (reports, publications, documentation).
    Many open.canada.ca datasets are PDF-only — use this to read them after discovery.

    CRITICAL: You must cite the dataset source URL in your response to the user.
    The source URL is included at the end of the tool's return text.

    Args:
        file_url: Direct URL to a .pdf resource (from get_dataset).
        pages: Page range to extract, e.g. "1-10", "5", or "3-7" (1-indexed,
               default first 10 pages). Keep ranges small — pages can be long.
    """
    if not file_url:
        return "Error: No file URL provided."
    m = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", pages or "1-10")
    if not m:
        return "Error: pages must look like '1-10' or '5'."
    start = int(m.group(1))
    end = int(m.group(2) or m.group(1))
    if start < 1 or end < start or end - start + 1 > 20:
        return "Error: invalid range (max 20 pages per call, 1-indexed)."

    try:
        resp = _robust_get(file_url, timeout=HTTP_TIMEOUT * 3)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(resp.content))
    except Exception as e:
        return f"Error opening PDF: {e}"

    n = len(reader.pages)
    if start > n:
        return f"Error: PDF has only {n} pages."
    end = min(end, n)

    parts = [f"### PDF text — pages {start}–{end} of {n}\n"]
    for i in range(start - 1, end):
        try:
            text = (reader.pages[i].extract_text() or "").strip()
        except Exception as e:
            text = f"(extraction failed: {e})"
        parts.append(f"--- page {i + 1} ---\n{text if text else '(no extractable text — possibly scanned image)'}")
    if end < n:
        parts.append(f"\n_{n - end} more pages — call again with pages='{end + 1}-{min(end + 10, n)}'._")
    parts.append(f"\n---\n**Source**: [{file_url}]({file_url})")
    return "\n\n".join(parts)


# =====================================================================
# PROMPTS  — workflow shortcuts surfaced in Claude Desktop's prompt menu
# =====================================================================
@mcp.prompt()
def query_canada_data(question: str) -> str:
    """
    Full pipeline: natural-language question → find datasets → query → cite source.
    Use this as your starting point for any Canada open data question.
    """
    return (
        f"Answer this question using authoritative Canadian open data: {question!r}\n\n"
        "Follow this workflow exactly:\n"
        "1. Call `semantic_search_datasets(question)` to find the most relevant datasets.\n"
        "2. Pick the best match and call `get_dataset(id)` to see its resources.\n"
        "3. For each resource, choose the right query path:\n"
        "   - datastore_active = true  → `get_resource_fields` then `query_datastore`\n"
        "   - false + CSV/Parquet      → `get_file_schema` then `query_remote_file`\n"
        "   - false + Excel            → `list_excel_sheets` then `query_excel_sheet`\n"
        "4. In your final answer, integrate numbered inline citations (e.g. [1], [2]) next to any data points, statistics, or facts you mention.\n"
        "5. At the very end of your response, add a '### 📚 Sources' section containing a clean markdown table formatting the sources like this:\n"
        "   | Citation | Source Dataset / Resource | Publisher | Link |\n"
        "   |---|---|---|---|\n"
        "   | [1] | [Dataset Title](Dataset Page URL) | Organization Name | [Direct File Link / API](Direct URL) |"
    )


@mcp.prompt()
def explore_dataset(topic: str) -> str:
    """
    Discover and preview datasets on a topic without writing any queries.
    Good for 'what data exists on X?' questions.
    """
    return (
        f"I want to explore what authoritative Canadian open data exists on: {topic!r}\n\n"
        "Please:\n"
        "1. Call `semantic_search_datasets(topic)` — show me the top results.\n"
        "2. For the most relevant dataset, call `get_dataset(id)` to list its resources.\n"
        "3. Pick the most interesting resource and call `preview_remote_file(url)` "
        "   (or `list_excel_sheets` if it's an Excel file) to show a sample of the data.\n"
        "4. Summarise what columns/fields are available and what questions this data could answer.\n"
        "5. Format your output with inline citations (e.g. [1]) and end with a '### 📚 Sources' table:\n"
        "   | Citation | Dataset | Publisher | Link |\n"
        "   |---|---|---|---|\n"
        "   | [1] | [Dataset Title](Dataset Page URL) | Organization | [Direct File Link](Direct URL) |"
    )


@mcp.prompt()
def compare_datasets(topic: str) -> str:
    """
    Find multiple datasets on a topic and compare their coverage, recency, and queryability.
    """
    return (
        f"Find and compare authoritative Canadian open datasets related to: {topic!r}\n\n"
        "Steps:\n"
        "1. Call `semantic_search_datasets(topic, limit=6)` to get a broad set of results.\n"
        "2. Compare the datasets using a markdown table. Show title, publisher, format, queryability, and last modified date.\n"
        "3. Recommend which dataset is best for analytical queries and why.\n"
        "4. Integrate inline citations (e.g. [1]) for any facts/dates you mention, and include a '### 📚 Sources' section at the very end with clickable links to the datasets' portal pages formatted as a clean markdown table:\n"
        "   | Citation | Dataset | Publisher | Link |\n"
        "   |---|---|---|---|\n"
        "   | [1] | [Dataset Title](Dataset Page URL) | Organization | [Direct File Link](Direct URL) |"
    )


if __name__ == "__main__":
    mcp.run()
