import os
import gzip
import json
import argparse
import logging
import requests
import datetime
import sys
from typing import Dict, Any, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .embed import embed_texts, MODEL_NAME, EMBED_DIM, PROMPT_VERSION
    from .store import prune_source_records, save_datasets, set_catalog_meta, create_fts_index
except ImportError:  # Direct execution: python semantic/build_index.py
    from embed import embed_texts, MODEL_NAME, EMBED_DIM, PROMPT_VERSION
    from store import prune_source_records, save_datasets, set_catalog_meta, create_fts_index
from source_registry import (
    CKAN_SOURCE_IDS,
    INDEX_SOURCE_IDS,
    get_source,
    page_url,
    qualify_dataset_id,
)
from resource_policy import is_queryable_resource, resource_format

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CATALOG_URL = "https://open.canada.ca/static/od-do-canada.jsonl.gz"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_GZ_PATH = os.path.join(LOCAL_DIR, "od-do-canada.jsonl.gz")
TABULAR_FORMATS = {"CSV", "XLSX", "XLS", "GEOJSON", "PARQUET", "JSON", "PDF", "TXT", "ZIP"}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
        "(compatible; OpenMCP/1.1; +https://github.com/opendatafyi/openmcp)"
    )
}

def download_catalog() -> None:
    """Download the full government catalog dump if it doesn't already exist locally."""
    if os.path.exists(LOCAL_GZ_PATH):
        logger.info(f"Catalog archive found locally at '{LOCAL_GZ_PATH}'. Skipping download.")
        return
        
    logger.info(f"Downloading catalog archive from {CATALOG_URL}...")
    response = requests.get(CATALOG_URL, stream=True, timeout=60, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    
    with open(LOCAL_GZ_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    logger.info("Download completed successfully.")

def _extract_text(val: Any, lang: str = "en") -> str:
    """Safely extract text in a specific language ('en' or 'fr') from dicts or pipe-separated strings."""
    if not val:
        return ""
    if isinstance(val, dict):
        if lang in val and val[lang]:
            return str(val[lang]).strip()
        # Fallback to other language or first value
        alt_lang = "fr" if lang == "en" else "en"
        return str(val.get(alt_lang) or next(iter(val.values()), "") or "").strip()
    if isinstance(val, str):
        if "|" in val:
            parts = [p.strip() for p in val.split("|")]
            if lang == "fr" and len(parts) > 1:
                return parts[1]
            return parts[0]
        return val.strip()
    return str(val).strip()


def _get_display_text(val: Any, language_priority: Tuple[str, ...] = ("en", "fr")) -> str:
    """Safely extract display text according to source language priority."""
    if not val:
        return ""
    if isinstance(val, dict):
        for lang in language_priority:
            if val.get(lang):
                return str(val[lang]).strip()
        for v in val.values():
            if v:
                return str(v).strip()
        return ""
    if isinstance(val, str):
        if "|" in val:
            parts = [p.strip() for p in val.split("|")]
            if language_priority and language_priority[0] == "fr" and len(parts) > 1:
                return parts[1]
            return parts[0]
        return val.strip()
    return str(val).strip()


def _get_multilingual_list(val: Any) -> List[str]:
    """Extract all multilingual keywords/items from lists or dictionaries."""
    if not val:
        return []
    keywords = set()
    if isinstance(val, list):
        for x in val:
            if isinstance(x, dict):
                for v in x.values():
                    if v and isinstance(v, str):
                        keywords.add(v.strip())
            elif isinstance(x, str):
                if "|" in x:
                    for part in x.split("|"):
                        if part.strip():
                            keywords.add(part.strip())
                elif x.strip():
                    keywords.add(x.strip())
    elif isinstance(val, dict):
        for v in val.values():
            if isinstance(v, list):
                for item in v:
                    if item and isinstance(item, str):
                        keywords.add(item.strip())
            elif isinstance(v, str) and v.strip():
                keywords.add(v.strip())
    return sorted(list(keywords))


def process_dataset_dict(ds: Dict[str, Any], source_id: str = "canada") -> Optional[Dict[str, Any]]:
    """
    Process a parsed dataset dictionary and extract queryable metadata.
    Returns None if the dataset does not contain queryable tabular resources.
    Supports bilingual metadata (EN + FR) and source language priority.
    """
    # Check resources first
    resources = ds.get("resources", [])
    if not resources or not isinstance(resources, list):
        return None
        
    source = get_source(source_id)
    priority = source.language_priority if source else ("en", "fr")

    # Retain only resources with a working datastore or local query path.
    extracted_resources = []
    for r in resources:
        if not is_queryable_resource(r):
            continue
        fmt = resource_format(r)
        desc = _get_display_text(r.get("description_translated") or r.get("description"), priority)
        extracted_resources.append({
            "id": r.get("id", ""),
            "name": _get_display_text(r.get("name_translated") or r.get("name"), priority),
            "format": fmt,
            "description": desc,
            "url": r.get("url", ""),
            "datastore_active": bool(r.get("datastore_active", False))
        })
        
    if not extracted_resources:
        return None
        
    # Extract metadata fields
    native_id = ds.get("id") or ds.get("name")
    if not native_id:
        return None
        
    title = _get_display_text(ds.get("title_translated") or ds.get("title") or ds.get("name"), priority)
    notes = _get_display_text(ds.get("notes_translated") or ds.get("notes"), priority)
    org = _get_display_text((ds.get("organization") or {}).get("title_translated") or (ds.get("organization") or {}).get("title"), priority)
    
    # Extract topic categories/keywords
    topic = _get_display_text(ds.get("topic_category") or ds.get("subject"), priority)
    
    # Bilingual extraction for embedding doc_text
    title_en = _extract_text(ds.get("title_translated") or ds.get("title") or ds.get("name"), "en")
    title_fr = _extract_text(ds.get("title_translated") or ds.get("title") or ds.get("name"), "fr")
    notes_en = _extract_text(ds.get("notes_translated") or ds.get("notes"), "en")
    notes_fr = _extract_text(ds.get("notes_translated") or ds.get("notes"), "fr")
    all_keywords = _get_multilingual_list(ds.get("keywords"))
    
    metadata_modified = ds.get("metadata_modified", "")
    
    # Resource summaries for embedding context
    resource_summaries = []
    for r in extracted_resources:
        r_name = r.get("name") or ""
        r_fmt = r.get("format") or ""
        r_desc = r.get("description") or ""
        item_str = f"- {r_name} ({r_fmt})" if r_name else f"- {r_fmt}"
        if r_desc and r_desc != r_name:
            item_str += f": {r_desc[:200]}"
        resource_summaries.append(item_str)

    # Compose enriched bilingual document text for embedding
    text_parts = []
    if title_en and title_fr and title_en != title_fr:
        text_parts.append(f"{title_en} / {title_fr}")
    elif title:
        text_parts.append(title)
        
    if notes_en:
        text_parts.append(notes_en)
    if notes_fr and notes_fr != notes_en:
        text_parts.append(notes_fr)
        
    if all_keywords:
        text_parts.append(f"Keywords: {', '.join(all_keywords)}")
    if org:
        text_parts.append(f"Publisher: {org}")
    if topic:
        text_parts.append(f"Topic: {topic}")
    if resource_summaries:
        text_parts.append("Resources:\n" + "\n".join(resource_summaries[:30]))
        
    doc_text = "\n\n".join(text_parts)
    if len(doc_text) > 25000:
        doc_text = doc_text[:25000]
    
    return {
        "id": qualify_dataset_id(source_id, native_id),
        "title": title,
        "org": org,
        "notes": notes,
        "topic": topic,
        "resources": extracted_resources,
        "metadata_modified": metadata_modified,
        "source_id": source_id,
        "source_type": "ckan",
        "native_id": native_id,
        "page_url": page_url(source_id, native_id),
        "doc_text": doc_text
    }

def process_dataset_line(line: str, source_id: str = "canada") -> Optional[Dict[str, Any]]:
    """
    Parse a single line from the JSONL export and structure the dataset metadata.
    Returns None if the dataset does not contain queryable tabular resources.
    """
    try:
        ds = json.loads(line)
    except json.JSONDecodeError:
        return None
    return process_dataset_dict(ds, source_id=source_id)


def fetch_ckan_catalog(source_id: str, limit: Optional[int] = None,
                       sort: str = "metadata_modified desc") -> List[Dict[str, Any]]:
    """Fetch and normalize one registered CKAN catalog in bounded pages."""
    source = get_source(source_id)
    if source.source_type != "ckan" or not source.api_base:
        raise ValueError(f"Source '{source_id}' does not expose a CKAN Action API.")
    kept: List[Dict[str, Any]] = []
    start = 0
    page_size = min(limit or 100, 100)
    while limit is None or len(kept) < limit:
        rows = min(page_size, (limit - len(kept)) if limit else page_size)
        response = requests.get(
            f"{source.api_base}/package_search",
            params={"q": "*:*", "sort": sort, "rows": rows, "start": start},
            headers=DEFAULT_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(f"CKAN catalog request failed for {source_id}: {body.get('error')}")
        payload = body.get("result", {})
        raw_results = payload.get("results", [])
        if not raw_results:
            break
        for raw in raw_results:
            processed = process_dataset_dict(raw, source_id=source_id)
            if processed:
                kept.append(processed)
                if limit and len(kept) >= limit:
                    break
        start += len(raw_results)
        if start >= payload.get("count", start):
            break
    logger.info("Fetched %d queryable datasets from %s.", len(kept), source.name)
    return kept


def _code_lookup(items: Any, code_field: str, label_field: str) -> Dict[str, str]:
    """Build a string-keyed WDS code lookup from a code-set array."""
    if not isinstance(items, list):
        return {}
    return {
        str(item[code_field]): str(item[label_field])
        for item in items
        if item.get(code_field) is not None and item.get(label_field)
    }


def process_statcan_cube(cube: Dict[str, Any],
                         code_sets: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one Statistics Canada WDS cube/table for semantic indexing."""
    product_id = str(cube.get("productId") or "").strip()
    title_en = str(cube.get("cubeTitleEn") or "").strip()
    title_fr = str(cube.get("cubeTitleFr") or "").strip()
    title = title_en or title_fr
    if not product_id or not title:
        return None

    subject_lookup = _code_lookup(
        code_sets.get("subject"), "subjectCode", "subjectEn"
    )
    survey_lookup = _code_lookup(
        code_sets.get("survey"), "surveyCode", "surveyEn"
    )
    frequency_lookup = _code_lookup(
        code_sets.get("frequency"), "frequencyCode", "frequencyDescEn"
    )
    subject_codes = [str(value) for value in (cube.get("subjectCode") or [])]
    survey_codes = [str(value) for value in (cube.get("surveyCode") or [])]
    subjects = [subject_lookup.get(code, code) for code in subject_codes]
    surveys = [survey_lookup.get(code, code) for code in survey_codes]
    frequency = frequency_lookup.get(
        str(cube.get("frequencyCode") or ""),
        str(cube.get("frequencyCode") or ""),
    )
    is_archived = str(cube.get("archived", "")) != "2"
    status = "archived" if is_archived else "active"
    cansim_id = str(cube.get("cansimId") or "").strip()
    start_date = str(cube.get("cubeStartDate") or "")[:10]
    end_date = str(cube.get("cubeEndDate") or "")[:10]
    dimensions_en = [
        str(item.get("dimensionNameEn", "")).strip()
        for item in (cube.get("dimensions") or [])
        if item.get("dimensionNameEn")
    ]
    dimensions_fr = [
        str(item.get("dimensionNameFr", "")).strip()
        for item in (cube.get("dimensions") or [])
        if item.get("dimensionNameFr")
    ]

    note_parts = [f"Statistics Canada table {product_id}."]
    if cansim_id:
        note_parts.append(f"Former CANSIM identifier: {cansim_id}.")
    if start_date or end_date:
        note_parts.append(f"Reference-period coverage: {start_date or '?'} to {end_date or '?'}.")
    if frequency:
        note_parts.append(f"Frequency: {frequency}.")
    note_parts.append(f"Table status: {status}.")
    if surveys:
        note_parts.append(f"Surveys/programs: {', '.join(surveys)}.")
    if dimensions_en:
        note_parts.append(f"Dimensions: {', '.join(dimensions_en)}.")
    notes = " ".join(note_parts)
    topic = "; ".join(subjects)
    keywords = [*subjects, *surveys, *dimensions_en, *dimensions_fr, frequency, cansim_id, product_id]
    
    title_line = f"{title_en} / {title_fr}" if title_en and title_fr and title_en != title_fr else title
    doc_text = "\n\n".join(filter(None, (
        title_line,
        notes,
        f"Keywords: {', '.join(value for value in keywords if value)}",
        "Publisher: Statistics Canada",
        f"Topic: {topic}" if topic else "",
    )))
    metadata = {
        **cube,
        "source": "Statistics Canada Web Data Service",
        "status": status,
        "subjectLabelsEn": subjects,
        "surveyLabelsEn": surveys,
        "frequencyLabelEn": frequency,
    }
    return {
        "id": qualify_dataset_id("statcan", product_id),
        "title": title,
        "org": "Statistics Canada",
        "notes": notes,
        "topic": topic,
        "resources": [{
            "id": f"statcan-{product_id}-eng",
            "name": "Full table (English CSV ZIP)",
            "format": "ZIP",
            "url": f"https://www150.statcan.gc.ca/n1/tbl/csv/{product_id}-eng.zip",
            "datastore_active": False,
        }],
        "metadata_modified": cube.get("releaseTime", ""),
        "source_id": "statcan",
        "source_type": "statcan_wds",
        "native_id": product_id,
        "page_url": page_url("statcan", product_id),
        "metadata": metadata,
        "doc_text": doc_text,
    }


def fetch_statcan_catalog(limit: Optional[int] = None,
                          newest_first: bool = False) -> List[Dict[str, Any]]:
    """Fetch Statistics Canada's full table metadata with two bulk calls."""
    source = get_source("statcan")
    inventory_response = requests.get(
        f"{source.api_base}/getAllCubesList", timeout=90, headers=DEFAULT_HEADERS
    )
    inventory_response.raise_for_status()
    inventory = inventory_response.json()
    code_sets = {}
    try:
        codes_response = requests.get(
            f"{source.api_base}/getCodeSets", timeout=60, headers=DEFAULT_HEADERS
        )
        codes_response.raise_for_status()
        codes_body = codes_response.json()
        if codes_body.get("status") == "SUCCESS" and isinstance(codes_body.get("object"), dict):
            code_sets = codes_body.get("object", {})
        else:
            logger.warning("Statistics Canada WDS getCodeSets returned non-success: %s", codes_body.get("object"))
    except Exception as e:
        logger.warning("Failed to fetch Statistics Canada getCodeSets: %s", e)

    if newest_first:
        inventory.sort(key=lambda item: item.get("releaseTime", ""), reverse=True)
    if limit is not None:
        inventory = inventory[:max(0, limit)]
    records = [
        record
        for cube in inventory
        if (record := process_statcan_cube(cube, code_sets)) is not None
    ]
    logger.info("Fetched %d table records from %s.", len(records), source.name)
    return records


def _embed_and_save(records: Sequence[Dict[str, Any]], batch_size: int = 1024) -> None:
    """Embed and upsert records in bounded batches."""
    for start in range(0, len(records), batch_size):
        batch = list(records[start:start + batch_size])
        embeddings = embed_texts([record["doc_text"] for record in batch], is_query=False)
        for record, embedding in zip(batch, embeddings):
            record["embedding"] = embedding
        save_datasets(batch)
        logger.info("Indexed %d/%d records.", min(start + len(batch), len(records)), len(records))

def refresh_index(count: int, sources: Optional[Sequence[str]] = None) -> None:
    """
    Incrementally refresh the index by fetching recently modified packages from CKAN API.
    Does not require downloading the full catalogue dump.
    """
    selected = list(sources or INDEX_SOURCE_IDS)
    processed_datasets: List[Dict[str, Any]] = []
    for source_id in selected:
        if source_id == "statcan":
            processed_datasets.extend(
                fetch_statcan_catalog(limit=count, newest_first=True)
            )
        else:
            processed_datasets.extend(fetch_ckan_catalog(source_id, limit=count))
    if processed_datasets:
        _embed_and_save(processed_datasets)
        set_catalog_meta({
            "embed_model": MODEL_NAME,
            "embed_dim": str(EMBED_DIM),
            "prompt_version": PROMPT_VERSION,
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sources": json.dumps(selected),
        })
        create_fts_index()
    logger.info("Incremental refresh completed.")

def build_index(limit: Optional[int] = None,
                sources: Optional[Sequence[str]] = None) -> None:
    """Read local archive, stream & parse records, embed them, and save to DuckDB."""
    selected = list(sources or INDEX_SOURCE_IDS)
    processed_datasets = []
    if "canada" in selected:
        download_catalog()
        total_lines = 0
        with gzip.open(LOCAL_GZ_PATH, "rt", encoding="utf-8") as handle:
            for line in handle:
                total_lines += 1
                ds_data = process_dataset_line(line, source_id="canada")
                if ds_data:
                    processed_datasets.append(ds_data)
                if limit and len(processed_datasets) >= limit:
                    break
        logger.info("Parsed %d federal records; kept %d.", total_lines,
                    len(processed_datasets))
    for source_id in selected:
        if source_id in ("canada", "statcan"):
            continue
        processed_datasets.extend(fetch_ckan_catalog(source_id, limit=limit))
    if "statcan" in selected:
        processed_datasets.extend(fetch_statcan_catalog(limit=limit))
    if not processed_datasets:
        logger.warning("No datasets match the tabular filter criteria.")
        return
    _embed_and_save(processed_datasets)
    if limit is None:
        for source_id in selected:
            keep_ids = [
                record["id"] for record in processed_datasets
                if record["source_id"] == source_id
            ]
            prune_source_records(source_id, keep_ids)
    set_catalog_meta({
        "embed_model": MODEL_NAME,
        "embed_dim": str(EMBED_DIM),
        "prompt_version": PROMPT_VERSION,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": json.dumps(selected),
    })
    create_fts_index()
    logger.info("Catalog indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh semantic search index over government datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datasets to index (for faster testing).")
    parser.add_argument("--refresh", type=int, default=None, help="Incrementally refresh index with the specified number of recently modified datasets.")
    parser.add_argument(
        "--sources", nargs="+", choices=INDEX_SOURCE_IDS,
        default=list(INDEX_SOURCE_IDS),
        help="Catalog sources to index (default: all registered sources).",
    )
    args = parser.parse_args()
    
    if args.refresh is not None:
        refresh_index(args.refresh, sources=args.sources)
    else:
        build_index(limit=args.limit, sources=args.sources)
