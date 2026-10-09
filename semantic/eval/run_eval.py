"""Evaluation harness for openmcp retrieval quality.

Measures Recall@1, Recall@5, Recall@10, and MRR across query categories and languages.
Can evaluate vector-only search and server hybrid search.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semantic.embed import embed_texts, MODEL_NAME
from semantic.store import top_k


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_retrieval(retrieved_ids: List[str], expected_ids: List[str]) -> Dict[str, float]:
    expected_set = set(expected_ids)
    hit_1 = 1.0 if retrieved_ids and retrieved_ids[0] in expected_set else 0.0
    hit_5 = 1.0 if any(r in expected_set for r in retrieved_ids[:5]) else 0.0
    hit_10 = 1.0 if any(r in expected_set for r in retrieved_ids[:10]) else 0.0

    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids[:10], start=1):
        if rid in expected_set:
            mrr = 1.0 / rank
            break

    return {
        "hit@1": hit_1,
        "hit@5": hit_5,
        "hit@10": hit_10,
        "mrr": mrr,
    }


def run_vector_eval(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = []
    results = []
    
    for q in queries:
        t0 = time.perf_counter()
        query_vecs = embed_texts([q["query"]], is_query=True)
        if query_vecs:
            top_records = top_k(query_vecs[0], k=10)
            retrieved_ids = [r["id"] for r in top_records]
        else:
            retrieved_ids = []
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)

        metrics = evaluate_retrieval(retrieved_ids, q["expected_ids"])
        results.append({
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "lang": q["lang"],
            "expected_ids": q["expected_ids"],
            "retrieved_ids": retrieved_ids,
            "metrics": metrics,
            "latency_ms": latency,
        })

    return summarize_results(results, latencies, "vector_only")


def run_hybrid_eval(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    import unittest.mock as mock
    import mcp_server

    latencies = []
    results = []

    # Mock _ckan_get so hybrid search runs offline and deterministically
    with mock.patch("mcp_server._ckan_get", return_value={"results": []}):
        for q in queries:
            t0 = time.perf_counter()
            tool_result = mcp_server.semantic_search_datasets(query=q["query"], limit=10)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency)

            structured = getattr(tool_result, "structuredContent", {}) or {}
            datasets = structured.get("datasets", [])
            retrieved_ids = [d["id"] for d in datasets]

            metrics = evaluate_retrieval(retrieved_ids, q["expected_ids"])
            results.append({
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "lang": q["lang"],
                "expected_ids": q["expected_ids"],
                "retrieved_ids": retrieved_ids,
                "metrics": metrics,
                "latency_ms": latency,
            })

    return summarize_results(results, latencies, "hybrid_offline")


def summarize_results(results: List[Dict[str, Any]], latencies: List[float], mode: str) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    overall = {
        "hit@1": sum(r["metrics"]["hit@1"] for r in results) / n,
        "hit@5": sum(r["metrics"]["hit@5"] for r in results) / n,
        "hit@10": sum(r["metrics"]["hit@10"] for r in results) / n,
        "mrr": sum(r["metrics"]["mrr"] for r in results) / n,
    }

    # Group by category
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_summary = {}
    for cat, items in sorted(by_category.items()):
        cnt = len(items)
        category_summary[cat] = {
            "count": cnt,
            "hit@1": sum(i["metrics"]["hit@1"] for i in items) / cnt,
            "hit@5": sum(i["metrics"]["hit@5"] for i in items) / cnt,
            "hit@10": sum(i["metrics"]["hit@10"] for i in items) / cnt,
            "mrr": sum(i["metrics"]["mrr"] for i in items) / cnt,
        }

    # Group by language
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["lang"]].append(r)

    lang_summary = {}
    for lang, items in sorted(by_lang.items()):
        cnt = len(items)
        lang_summary[lang] = {
            "count": cnt,
            "hit@1": sum(i["metrics"]["hit@1"] for i in items) / cnt,
            "hit@5": sum(i["metrics"]["hit@5"] for i in items) / cnt,
            "hit@10": sum(i["metrics"]["hit@10"] for i in items) / cnt,
            "mrr": sum(i["metrics"]["mrr"] for i in items) / cnt,
        }

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(n * 0.50)]
    p95 = sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1]

    return {
        "mode": mode,
        "model": MODEL_NAME,
        "total_queries": n,
        "overall": overall,
        "by_category": category_summary,
        "by_lang": lang_summary,
        "latency_ms": {
            "mean": sum(latencies) / n,
            "p50": p50,
            "p95": p95,
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
        },
        "query_details": results,
    }


def print_report(summary: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"RETRIEVAL EVALUATION REPORT — Mode: {summary['mode']}")
    print(f"Model: {summary['model']} | Total Queries: {summary['total_queries']}")
    print("=" * 70)

    ov = summary["overall"]
    lat = summary["latency_ms"]
    print(f"Overall Metrics:")
    print(f"  Recall@1:  {ov['hit@1'] * 100:.1f}%")
    print(f"  Recall@5:  {ov['hit@5'] * 100:.1f}%")
    print(f"  Recall@10: {ov['hit@10'] * 100:.1f}%")
    print(f"  MRR:       {ov['mrr']:.4f}")
    print(f"Latency: Mean={lat['mean']:.1f}ms, p50={lat['p50']:.1f}ms, p95={lat['p95']:.1f}ms")

    print("\nBy Category:")
    print(f"{'Category':<15} {'Count':<6} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8}")
    print("-" * 55)
    for cat, m in summary["by_category"].items():
        print(f"{cat:<15} {m['count']:<6} {m['hit@1']*100:<7.1f}% {m['hit@5']*100:<7.1f}% {m['hit@10']*100:<7.1f}% {m['mrr']:<7.4f}")

    print("\nBy Language:")
    print(f"{'Language':<15} {'Count':<6} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8}")
    print("-" * 55)
    for lang, m in summary["by_lang"].items():
        print(f"{lang:<15} {m['count']:<6} {m['hit@1']*100:<7.1f}% {m['hit@5']*100:<7.1f}% {m['hit@10']*100:<7.1f}% {m['mrr']:<7.4f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation harness.")
    parser.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "queries.jsonl"))
    parser.add_argument("--mode", choices=["vector", "hybrid", "both"], default="both")
    parser.add_argument("--output", default=None, help="Path to save output JSON")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    out_data = {}

    if args.mode in ("vector", "both"):
        print("\nRunning vector-only evaluation...")
        vec_summary = run_vector_eval(queries)
        print_report(vec_summary)
        out_data["vector_only"] = vec_summary

    if args.mode in ("hybrid", "both"):
        print("\nRunning hybrid (offline) evaluation...")
        hyb_summary = run_hybrid_eval(queries)
        print_report(hyb_summary)
        out_data["hybrid_offline"] = hyb_summary

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
