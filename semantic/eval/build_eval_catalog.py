"""Re-index evaluation target datasets and sample records using EmbeddingGemma 2 (768d)."""
import json
import os
import sys
import time
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semantic.embed import embed_texts, MODEL_NAME, EMBED_DIM, PROMPT_VERSION
from semantic.store import save_datasets, set_catalog_meta, init_db

SOURCE_DB = os.path.join(PROJECT_ROOT, "catalog_bge_384.duckdb")
TARGET_DB = os.path.join(PROJECT_ROOT, "catalog.duckdb")
QUERIES_PATH = os.path.join(os.path.dirname(__file__), "queries.jsonl")


def build_doc_text(title: str, notes: str, org: str, topic: str) -> str:
    parts = []
    if notes:
        parts.append(notes)
    if org:
        parts.append(f"Publisher: {org}")
    if topic:
        parts.append(f"Topic: {topic}")
    body = " ".join(parts) if parts else title
    return f"title: {title} | text: {body}"


def main(sample_limit: int = 2000):
    if not os.path.exists(SOURCE_DB):
        raise FileNotFoundError(f"Source database not found at {SOURCE_DB}")

    # 1. Load target IDs from queries.jsonl
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    target_ids = set()
    for q in queries:
        target_ids.update(q["expected_ids"])

    print(f"Loaded {len(queries)} queries with {len(target_ids)} target dataset IDs.")

    # 2. Extract target records + sample background records from SOURCE_DB
    src_conn = duckdb.connect(SOURCE_DB, read_only=True)
    try:
        # Fetch target records
        placeholders = ",".join(["?"] * len(target_ids))
        target_rows = src_conn.execute(f"""
            SELECT id, title, org, notes, topic, resources_json, metadata_modified,
                   source_id, source_type, native_id, page_url, metadata_json
            FROM datasets
            WHERE id IN ({placeholders})
        """, list(target_ids)).fetchall()
        print(f"Found {len(target_rows)}/{len(target_ids)} target records in source catalog.")

        # Fetch sample background records
        sample_rows = src_conn.execute(f"""
            SELECT id, title, org, notes, topic, resources_json, metadata_modified,
                   source_id, source_type, native_id, page_url, metadata_json
            FROM datasets
            WHERE id NOT IN ({placeholders})
            LIMIT ?
        """, list(target_ids) + [sample_limit]).fetchall()
        print(f"Fetched {len(sample_rows)} background sample records.")
    finally:
        src_conn.close()

    all_rows = target_rows + sample_rows
    total_records = len(all_rows)
    print(f"Total records to embed with {MODEL_NAME} ({EMBED_DIM}d): {total_records}")

    # 3. Prepare target catalog file (remove existing target if recreating)
    if os.path.exists(TARGET_DB):
        os.remove(TARGET_DB)

    tgt_conn = duckdb.connect(TARGET_DB)
    init_db(tgt_conn)
    tgt_conn.close()

    # 4. Embed and save in batches
    batch_size = 128
    t_start = time.perf_counter()

    for start_idx in range(0, total_records, batch_size):
        batch_rows = all_rows[start_idx : start_idx + batch_size]
        batch_docs = [
            build_doc_text(
                title=row[1] or "",
                notes=row[3] or "",
                org=row[2] or "",
                topic=row[4] or "",
            )
            for row in batch_rows
        ]

        t0 = time.perf_counter()
        embeddings = embed_texts(batch_docs, is_query=False)
        t_embed = time.perf_counter() - t0

        records_to_save = []
        for row, emb in zip(batch_rows, embeddings):
            resources = json.loads(row[5]) if row[5] else []
            metadata = json.loads(row[11]) if row[11] else {}
            records_to_save.append({
                "id": row[0],
                "title": row[1] or "",
                "org": row[2] or "",
                "notes": row[3] or "",
                "topic": row[4] or "",
                "resources": resources,
                "metadata_modified": row[6] or "",
                "source_id": row[7] or "canada",
                "source_type": row[8] or "ckan",
                "native_id": row[9] or row[0],
                "page_url": row[10] or "",
                "metadata": metadata,
                "embedding": emb,
            })

        save_datasets(records_to_save)
        progress = min(start_idx + len(batch_rows), total_records)
        rate = len(batch_rows) / max(t_embed, 0.001)
        print(f"Indexed {progress}/{total_records} records ({rate:.1f} rec/s)")

    total_time = time.perf_counter() - t_start
    overall_rate = total_records / max(total_time, 0.001)

    import datetime
    set_catalog_meta({
        "embed_model": MODEL_NAME,
        "embed_dim": str(EMBED_DIM),
        "prompt_version": PROMPT_VERSION,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records": str(total_records),
    })

    print("\n" + "=" * 60)
    print(f"Build complete for {total_records} records!")
    print(f"Total time: {total_time:.1f}s ({overall_rate:.1f} records/second)")
    extrapolated_75k = (74728 / overall_rate) / 60
    print(f"Extrapolated full 74,728 catalog build time: ~{extrapolated_75k:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(sample_limit=limit)
