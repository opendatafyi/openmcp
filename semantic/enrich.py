"""
Content and asset enrichment module for OpenMCP Canada.

Extracts rich child assets from dataset resources:
1. Column schemas & headers from CSV / Datastore resources.
2. PDF documentation text from guides and data dictionaries (via pypdf).
3. Geospatial layer abstracts and titles from WMS / ESRI REST endpoints.

Stores extracted assets in DuckDB `dataset_assets` with EmbeddingGemma 2 vectors.
"""

import io
import os
import csv
import json
import logging
import hashlib
import requests
from typing import Dict, Any, List, Optional, Tuple

import pypdf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from semantic.embed import embed_texts, EMBED_DIM
from semantic.store import save_dataset_assets, DB_PATH
from source_registry import get_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrich_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_session = None


def _get_http_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def _cache_key(url: str, kind: str) -> str:
    h = hashlib.sha256(f"{kind}:{url}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{kind}_{h}.json")


def extract_csv_headers(url: str, timeout: float = 10.0) -> Optional[List[str]]:
    """
    Extract column header names from a remote CSV using an HTTP Range request (first 64 KB).
    """
    if not url or not url.startswith("http"):
        return None

    cache_file = _cache_key(url, "csv_headers")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    session = _get_http_session()
    try:
        # Request only the first 64KB
        headers = {"Range": "bytes=0-65535"}
        resp = session.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code not in (200, 206):
            return None

        content = resp.raw.read(65536)
        # Attempt to decode as UTF-8 or latin-1
        text = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            return None

        # Parse the first few non-empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        reader = csv.reader(io.StringIO(lines[0]))
        headers_list = next(reader, [])
        cleaned_headers = [h.strip() for h in headers_list if h and len(h.strip()) < 100]

        if cleaned_headers:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_headers, f)
            return cleaned_headers
    except Exception as e:
        logger.debug(f"Failed to extract CSV headers from {url}: {e}")
    return None


def extract_datastore_schema(source_id: str, resource_id: str, timeout: float = 8.0) -> Optional[List[str]]:
    """
    Query CKAN datastore_search limit=0 to extract exact column names and types.
    """
    source = get_source(source_id)
    if not source or not source.api_base or not resource_id:
        return None

    cache_file = _cache_key(f"{source_id}:{resource_id}", "ds_schema")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    session = _get_http_session()
    try:
        url = f"{source.api_base}/datastore_search"
        resp = session.get(url, params={"resource_id": resource_id, "limit": 0}, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                fields = data.get("result", {}).get("fields", [])
                field_names = [
                    f["id"] for f in fields 
                    if f.get("id") and not f["id"].startswith("_")
                ]
                if field_names:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(field_names, f)
                    return field_names
    except Exception as e:
        logger.debug(f"Failed to fetch datastore schema for {resource_id}: {e}")
    return None


def extract_pdf_text(url: str, max_pages: int = 5, max_bytes: int = 4 * 1024 * 1024, timeout: float = 12.0) -> Optional[str]:
    """
    Extract readable text from the first N pages of a remote PDF document.
    """
    if not url or not url.startswith("http"):
        return None

    cache_file = _cache_key(url, "pdf_text")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    session = _get_http_session()
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return None

        pdf_bytes = resp.raw.read(max_bytes)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        
        extracted_pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            page_text = page.extract_text() or ""
            cleaned = page_text.strip()
            if cleaned:
                extracted_pages.append(f"--- Page {i+1} ---\n{cleaned}")

        if extracted_pages:
            full_text = "\n\n".join(extracted_pages)
            # Cap text length to ~6,000 chars for embedding
            if len(full_text) > 6000:
                full_text = full_text[:6000]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(full_text, f)
            return full_text
    except Exception as e:
        logger.debug(f"Failed to extract PDF text from {url}: {e}")
    return None


def enrich_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and build child asset records for a single dataset.
    Returns a list of asset dicts ready for embedding.
    """
    dataset_id = dataset["id"]
    source_id = dataset.get("source_id", "canada")
    resources = dataset.get("resources", [])
    assets = []

    for r in resources:
        r_id = r.get("id") or ""
        r_fmt = (r.get("format") or "").upper()
        r_url = r.get("url") or ""
        r_name = r.get("name") or ""
        is_ds = r.get("datastore_active", False)

        # 1. Datastore schema
        if is_ds and r_id:
            fields = extract_datastore_schema(source_id, r_id)
            if fields:
                field_str = ", ".join(fields)
                text = f"Datastore schema for resource '{r_name}': Columns: {field_str}"
                assets.append({
                    "asset_id": f"{dataset_id}:schema:{r_id or 'ds'}",
                    "dataset_id": dataset_id,
                    "kind": "column_schema",
                    "source_url": r_url,
                    "title": f"Table schema: {r_name}",
                    "text": text,
                    "metadata": {"columns": fields, "resource_id": r_id, "format": r_fmt},
                })

        # 2. CSV headers
        elif r_fmt in ("CSV", "TSV") and r_url:
            headers = extract_csv_headers(r_url)
            if headers:
                header_str = ", ".join(headers)
                text = f"CSV column headers for '{r_name}': Columns: {header_str}"
                assets.append({
                    "asset_id": f"{dataset_id}:schema:{r_id or 'csv'}",
                    "dataset_id": dataset_id,
                    "kind": "column_schema",
                    "source_url": r_url,
                    "title": f"CSV Columns: {r_name}",
                    "text": text,
                    "metadata": {"columns": headers, "resource_id": r_id, "format": r_fmt},
                })

        # 3. PDF data dictionaries / methodology reports
        elif r_fmt == "PDF" and r_url:
            pdf_text = extract_pdf_text(r_url, max_pages=3)
            if pdf_text:
                text = f"Document '{r_name}':\n{pdf_text}"
                assets.append({
                    "asset_id": f"{dataset_id}:pdf:{r_id or 'doc'}",
                    "dataset_id": dataset_id,
                    "kind": "pdf_text",
                    "source_url": r_url,
                    "title": f"PDF Guide: {r_name}",
                    "text": text,
                    "metadata": {"resource_id": r_id, "format": "PDF"},
                })

    return assets


def enrich_and_index_assets(datasets: List[Dict[str, Any]], batch_size: int = 8) -> int:
    """
    Enrich a list of datasets, extract their child assets, embed them, and save to DuckDB.
    """
    all_assets = []
    for ds in datasets:
        assets = enrich_dataset(ds)
        all_assets.extend(assets)

    if not all_assets:
        return 0

    logger.info(f"Generated {len(all_assets)} child assets across {len(datasets)} datasets. Generating embeddings...")
    texts_to_embed = [f"title: {a['title']} | text: {a['text']}" for a in all_assets]
    embeddings = embed_texts(texts_to_embed, is_query=False, batch_size=batch_size)

    for asset, emb in zip(all_assets, embeddings):
        asset["embedding"] = emb

    save_dataset_assets(all_assets)
    return len(all_assets)
