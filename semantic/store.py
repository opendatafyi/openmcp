import os
import json
import logging
from typing import List, Dict, Any, Optional
import duckdb

from source_registry import page_url, qualify_dataset_id, split_dataset_id
from semantic.embed import EMBED_DIM, MODEL_NAME, PROMPT_VERSION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Locate the database relative to this file (project root)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catalog.duckdb")


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize and migrate the datasets table schema in DuckDB."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS datasets (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            org VARCHAR,
            notes VARCHAR,
            topic VARCHAR,
            resources_json VARCHAR,
            metadata_modified VARCHAR,
            source_id VARCHAR,
            source_type VARCHAR,
            native_id VARCHAR,
            page_url VARCHAR,
            metadata_json VARCHAR,
            embedding FLOAT[{EMBED_DIM}]
        );
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS dataset_assets (
            asset_id VARCHAR PRIMARY KEY,
            dataset_id VARCHAR,
            kind VARCHAR,
            source_url VARCHAR,
            title VARCHAR,
            text VARCHAR,
            metadata_json VARCHAR,
            embedding FLOAT[{EMBED_DIM}]
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS catalog_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        );
    """)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info('datasets')").fetchall()
    }
    for name in ("source_id", "source_type", "native_id", "page_url", "metadata_json"):
        if name not in columns:
            conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} VARCHAR")

    # Catalogs built before multi-source support contain only federal IDs.
    conn.execute("""
        UPDATE datasets
        SET source_id = COALESCE(source_id, 'canada'),
            source_type = COALESCE(source_type, 'ckan'),
            native_id = COALESCE(native_id, id),
            page_url = COALESCE(
                page_url,
                'https://open.canada.ca/data/en/dataset/' || id
            )
        WHERE source_id IS NULL OR native_id IS NULL OR page_url IS NULL
    """)
    conn.execute("""
        UPDATE datasets
        SET id = 'canada:' || id
        WHERE source_id = 'canada' AND strpos(id, ':') = 0
    """)
    conn.execute("DELETE FROM datasets WHERE source_id = 'curated'")
    logger.info("DuckDB schema initialized.")


def get_catalog_meta(conn: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, str]:
    """Retrieve catalog metadata (embed_model, embed_dim, prompt_version, etc.)."""
    close_after = False
    if conn is None:
        if not os.path.exists(DB_PATH):
            return {}
        conn = duckdb.connect(DB_PATH, read_only=True)
        close_after = True
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "catalog_meta" not in tables:
            # Infer dimension from datasets schema if catalog_meta does not exist yet
            if "datasets" in tables:
                for col in conn.execute("PRAGMA table_info('datasets')").fetchall():
                    if col[1] == "embedding":
                        type_str = str(col[2])
                        dim = 384
                        if "[" in type_str and "]" in type_str:
                            try:
                                dim = int(type_str.split("[")[-1].split("]")[0])
                            except ValueError:
                                pass
                        return {
                            "embed_model": "BAAI/bge-small-en-v1.5" if dim == 384 else "unknown",
                            "embed_dim": str(dim),
                            "prompt_version": "legacy",
                        }
            return {}
        rows = conn.execute("SELECT key, value FROM catalog_meta").fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        if close_after:
            conn.close()


def set_catalog_meta(meta: Dict[str, str], conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Store or update catalog metadata."""
    close_after = False
    if conn is None:
        conn = duckdb.connect(DB_PATH)
        close_after = True
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS catalog_meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            );
        """)
        for k, v in meta.items():
            conn.execute("""
                INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?);
            """, [k, str(v)])
    finally:
        if close_after:
            conn.close()


def check_catalog_compatibility() -> Optional[Dict[str, Any]]:
    """
    Check if the local catalog.duckdb matches the expected embed_model and embed_dim.
    Returns None if compatible or catalog missing, or a dict with details if mismatched.
    """
    if not os.path.exists(DB_PATH):
        return None
    meta = get_catalog_meta()
    if not meta:
        return None
    catalog_model = meta.get("embed_model")
    catalog_dim = meta.get("embed_dim")
    if (catalog_model and catalog_model != MODEL_NAME) or (catalog_dim and str(catalog_dim) != str(EMBED_DIM)):
        return {
            "expected_model": MODEL_NAME,
            "expected_dim": EMBED_DIM,
            "catalog_model": catalog_model,
            "catalog_dim": catalog_dim,
        }
    return None


def _has_source_columns(conn: duckdb.DuckDBPyConnection) -> bool:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info('datasets')").fetchall()
    }
    return {"source_id", "source_type", "native_id", "page_url"}.issubset(columns)


def _has_metadata_column(conn: duckdb.DuckDBPyConnection) -> bool:
    return any(
        row[1] == "metadata_json"
        for row in conn.execute("PRAGMA table_info('datasets')").fetchall()
    )


def _normalized_record(row: tuple, *, has_sources: bool,
                       has_metadata: bool = False,
                       distance: Any = None) -> Dict[str, Any]:
    if has_sources:
        (stored_id, title, org, notes, topic, resources_json, modified,
         source_id, source_type, native_id, public_page) = row[:11]
        source_id = source_id or "canada"
        source_type = source_type or "ckan"
        native_id = native_id or stored_id.split(":", 1)[-1]
        catalog_id = qualify_dataset_id(source_id, native_id)
        public_page = public_page or page_url(source_id, native_id)
        metadata_json = row[11] if has_metadata else None
    else:
        stored_id, title, org, notes, topic, resources_json, modified = row[:7]
        source_id = "canada"
        source_type = "ckan"
        native_id = stored_id
        catalog_id = qualify_dataset_id(source_id, native_id)
        public_page = page_url(source_id, native_id)
        metadata_json = None

    return {
        "id": catalog_id,
        "title": title,
        "org": org,
        "notes": notes,
        "topic": topic,
        "resources": json.loads(resources_json) if resources_json else [],
        "metadata_modified": modified,
        "distance": float(distance) if distance is not None else None,
        "source_id": source_id,
        "source_type": source_type,
        "native_id": native_id,
        "page_url": public_page,
        "metadata": json.loads(metadata_json) if metadata_json else {},
    }


def save_datasets(datasets_data: List[Dict[str, Any]]) -> None:
    """Save processed datasets with embeddings to DuckDB."""
    if not datasets_data:
        return
        
    conn = duckdb.connect(DB_PATH)
    try:
        init_db(conn)
        conn.execute("BEGIN TRANSACTION;")
        
        rows = []
        for ds in datasets_data:
            source_id, native_id = split_dataset_id(
                ds["id"], default_source=ds.get("source_id", "canada")
            )
            source_type = ds.get("source_type", "ckan")
            public_page = ds.get("page_url", "")
            if not public_page and source_type == "ckan":
                public_page = page_url(source_id, native_id)
            rows.append((
                qualify_dataset_id(source_id, native_id),
                ds["title"],
                ds["org"],
                ds["notes"],
                ds["topic"],
                json.dumps(ds.get("resources", [])),
                ds.get("metadata_modified", ""),
                source_id,
                source_type,
                native_id,
                public_page,
                json.dumps(ds.get("metadata", {})),
                ds["embedding"]
            ))
            
        # Determine embedding dimension from data if present, otherwise default to EMBED_DIM
        record_dim = EMBED_DIM
        for ds in datasets_data:
            if ds.get("embedding"):
                record_dim = len(ds["embedding"])
                break

        # Perform bulk upsert
        conn.executemany(f"""
            INSERT OR REPLACE INTO datasets (
                id, title, org, notes, topic, resources_json, metadata_modified,
                source_id, source_type, native_id, page_url, metadata_json, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{record_dim}]);
        """, rows)
        
        conn.execute("COMMIT;")
        logger.info(f"Successfully saved {len(datasets_data)} records to {DB_PATH}")
    except Exception as e:
        conn.execute("ROLLBACK;")
        logger.error(f"Error saving datasets: {e}")
        raise e
    finally:
        conn.close()


def prune_source_records(source_id: str, keep_ids: List[str]) -> None:
    """Remove stale records for a fully rebuilt source after successful upserts."""
    if not keep_ids:
        raise ValueError(
            f"Refusing to prune source '{source_id}' with an empty completed catalog."
        )
    conn = duckdb.connect(DB_PATH)
    try:
        init_db(conn)
        conn.execute("BEGIN TRANSACTION")
        conn.execute("CREATE TEMP TABLE rebuild_keep_ids (id VARCHAR PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO rebuild_keep_ids VALUES (?)",
            [(dataset_id,) for dataset_id in keep_ids],
        )
        conn.execute("""
            DELETE FROM datasets
            WHERE source_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM rebuild_keep_ids kept WHERE kept.id = datasets.id
              )
        """, [source_id])
        conn.execute("COMMIT")
        logger.info("Pruned stale records for source '%s'.", source_id)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def top_k(query_vec: List[float], k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K datasets closest to the query vector.
    Uses DuckDB's built-in array_cosine_distance.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Catalog database not found at '{DB_PATH}'. "
            "Please run 'python semantic/build_index.py' to generate it."
        )
        
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        has_sources = _has_source_columns(conn)
        has_metadata = _has_metadata_column(conn)
        source_select = (
            ", source_id, source_type, native_id, page_url" if has_sources else ""
        )
        metadata_select = ", metadata_json" if has_sources and has_metadata else ""
        query_dim = len(query_vec) if query_vec else EMBED_DIM
        res = conn.execute(f"""
            SELECT id, title, org, notes, topic, resources_json, metadata_modified
                   {source_select}{metadata_select},
                   array_cosine_distance(embedding, ?::FLOAT[{query_dim}]) AS dist
            FROM datasets
            ORDER BY dist ASC
            LIMIT ?
        """, (query_vec, k)).fetchall()

        distance_index = 11 + int(has_metadata) if has_sources else 7
        return [
            _normalized_record(
                row, has_sources=has_sources, has_metadata=has_metadata,
                distance=row[distance_index]
            )
            for row in res
        ]
    finally:
        conn.close()


def get_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch multiple datasets by their IDs from the DuckDB store in a single query.
    Returns a dictionary mapping id -> dataset.
    """
    if not ids:
        return {}
    if not os.path.exists(DB_PATH):
        return {}
        
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        has_sources = _has_source_columns(conn)
        has_metadata = _has_metadata_column(conn)
        lookup_ids = []
        for dataset_id in ids:
            source_id, native_id = split_dataset_id(dataset_id)
            lookup_ids.append(
                qualify_dataset_id(source_id, native_id) if has_sources else native_id
            )
        placeholders = ",".join(["?"] * len(lookup_ids))
        source_select = (
            ", source_id, source_type, native_id, page_url" if has_sources else ""
        )
        metadata_select = ", metadata_json" if has_sources and has_metadata else ""
        res = conn.execute(f"""
            SELECT id, title, org, notes, topic, resources_json, metadata_modified
                   {source_select}{metadata_select}
            FROM datasets
            WHERE id IN ({placeholders})
        """, lookup_ids).fetchall()

        results = {}
        for row in res:
            record = _normalized_record(
                row, has_sources=has_sources, has_metadata=has_metadata
            )
            results[record["id"]] = record
        return results
    finally:
        conn.close()


def save_dataset_assets(assets_data: List[Dict[str, Any]]) -> None:
    """Save enriched child assets (schemas, PDFs, map context) with embeddings."""
    if not assets_data:
        return
        
    conn = duckdb.connect(DB_PATH)
    try:
        init_db(conn)
        conn.execute("BEGIN TRANSACTION;")
        
        rows = []
        for asset in assets_data:
            rows.append((
                asset["asset_id"],
                asset["dataset_id"],
                asset.get("kind", "schema"),
                asset.get("source_url", ""),
                asset.get("title", ""),
                asset.get("text", ""),
                json.dumps(asset.get("metadata", {}), ensure_ascii=False),
                asset["embedding"]
            ))
            
        record_dim = EMBED_DIM
        for asset in assets_data:
            if asset.get("embedding"):
                record_dim = len(asset["embedding"])
                break

        conn.executemany(f"""
            INSERT OR REPLACE INTO dataset_assets (
                asset_id, dataset_id, kind, source_url, title, text, metadata_json, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{record_dim}]);
        """, rows)
        
        conn.execute("COMMIT;")
        logger.info(f"Successfully saved {len(assets_data)} dataset assets to {DB_PATH}")
    except Exception as e:
        conn.execute("ROLLBACK;")
        logger.error(f"Failed to save dataset assets to DuckDB: {e}")
        raise
    finally:
        conn.close()


def top_k_assets(query_vec: List[float], k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve the top-K child assets closest to the query vector."""
    if not os.path.exists(DB_PATH):
        return []
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "dataset_assets" not in tables:
            return []
        query_dim = len(query_vec) if query_vec else EMBED_DIM
        res = conn.execute(f"""
            SELECT asset_id, dataset_id, kind, source_url, title, text, metadata_json,
                   array_cosine_distance(embedding, ?::FLOAT[{query_dim}]) AS dist
            FROM dataset_assets
            ORDER BY dist ASC
            LIMIT ?
        """, (query_vec, k)).fetchall()

        results = []
        for row in res:
            metadata = json.loads(row[6]) if row[6] else {}
            results.append({
                "asset_id": row[0],
                "dataset_id": row[1],
                "kind": row[2],
                "source_url": row[3],
                "title": row[4],
                "text": row[5],
                "metadata": metadata,
                "distance": float(row[7]) if row[7] is not None else None,
            })
        return results
    finally:
        conn.close()


def bm25_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Perform BM25 full-text keyword search across datasets."""
    if not os.path.exists(DB_PATH) or not query.strip():
        return []
    
    # Sanitize query for DuckDB FTS: keep alphanumeric terms
    clean_query = " ".join([w for w in query.replace("'", " ").replace('"', ' ').split() if w.isalnum()])
    if not clean_query:
        return []

    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "fts_main_datasets" not in tables:
            return []
        has_sources = _has_source_columns(conn)
        has_metadata = _has_metadata_column(conn)
        source_select = (
            ", source_id, source_type, native_id, page_url" if has_sources else ""
        )
        metadata_select = ", metadata_json" if has_sources and has_metadata else ""
        res = conn.execute(f"""
            SELECT id, title, org, notes, topic, resources_json, metadata_modified
                   {source_select}{metadata_select},
                   score
            FROM (
                SELECT *, fts_main_datasets.match_bm25(id, ?) AS score
                FROM datasets
            )
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """, (clean_query, k)).fetchall()

        return [
            _normalized_record(
                row, has_sources=has_sources, has_metadata=has_metadata,
                distance=None
            )
            for row in res
        ]
    except Exception as e:
        logger.debug(f"BM25 search skipped: {e}")
        return []
    finally:
        conn.close()


def create_fts_index(conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    """Create or rebuild the DuckDB FTS index on the datasets table."""
    close_after = False
    if conn is None:
        conn = duckdb.connect(DB_PATH)
        close_after = True
    try:
        conn.execute("INSTALL fts; LOAD fts;")
        conn.execute(
            "PRAGMA create_fts_index('datasets', 'id', 'title', 'notes', 'topic', 'org', overwrite=1);"
        )
        logger.info("DuckDB FTS BM25 index created successfully on datasets.")
    except Exception as e:
        logger.warning(f"Failed to create DuckDB FTS index: {e}")
    finally:
        if close_after:
            conn.close()

