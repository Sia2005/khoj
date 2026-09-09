from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time

import faiss
import numpy as np

from vecs_io import read_fvecs, read_ivecs

CSV_FIELDS = [
    "engine",
    "index",
    "dataset",
    "metric",
    "groundtruth_source",
    "num_base",
    "num_queries",
    "dimension",
    "k",
    "max_neighbors",
    "max_neighbors_layer0",
    "ef_construction",
    "ef_search",
    "threads",
    "runs",
    "recall_at_k",
    "qps_median",
    "qps_min",
    "qps_max",
    "build_seconds",
    "index_rss_bytes",
    "peak_rss_bytes",
]


def read_status_bytes(key: str) -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def current_rss() -> int:
    return read_status_bytes("VmRSS:")


def peak_rss() -> int:
    return read_status_bytes("VmHWM:")


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray, k: int) -> float:
    if truth.shape[1] < k:
        raise ValueError("groundtruth has fewer neighbours per query than k")
    if truth.shape[0] < retrieved.shape[0]:
        raise ValueError("groundtruth has fewer rows than the query set")

    hits = 0
    for row in range(retrieved.shape[0]):
        expected = set(truth[row, :k].tolist())
        hits += sum(1 for label in retrieved[row, :k].tolist() if label in expected)
    return hits / (retrieved.shape[0] * k)


def resolve_metric(name: str) -> int:
    if name == "l2":
        return faiss.METRIC_L2
    if name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"unknown metric {name}")


def exact_groundtruth(base: np.ndarray, queries: np.ndarray, k: int, metric: int) -> np.ndarray:
    index = faiss.IndexFlat(base.shape[1], metric)
    index.add(base)
    _, labels = index.search(queries, k)
    return labels


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--groundtruth", default="")
    parser.add_argument("--dataset", default="sift1m")
    parser.add_argument("--metric", default="l2", choices=["l2", "ip"])
    parser.add_argument("--csv", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", default="10,20,40,80,160,320")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if arguments.runs < 5:
        raise SystemExit("benchmark discipline requires at least 5 runs per configuration")
    if arguments.threads < 1:
        raise SystemExit("--threads must be positive")

    ef_values = [int(item) for item in arguments.ef_search.split(",") if item]
    if not ef_values:
        raise SystemExit("--ef-search must list at least one value")
    if any(ef < arguments.k for ef in ef_values):
        raise SystemExit("ef_search must be at least k")

    metric = resolve_metric(arguments.metric)
    base = read_fvecs(arguments.base, arguments.base_limit)
    queries = read_fvecs(arguments.query, arguments.query_limit)
    if base.shape[1] != queries.shape[1]:
        raise SystemExit("base and query vectors have different dimensions")

    print(
        f"loaded {base.shape[0]} base and {queries.shape[0]} query vectors "
        f"of dimension {base.shape[1]}",
        file=sys.stderr,
    )

    if arguments.groundtruth:
        truth = read_ivecs(arguments.groundtruth, arguments.query_limit)
        truth_source = "file"
    else:
        print("computing groundtruth with faiss.IndexFlat", file=sys.stderr)
        truth = exact_groundtruth(base, queries, arguments.k, metric)
        truth_source = "faiss_flat"

    faiss.omp_set_num_threads(arguments.threads)

    rss_before = current_rss()
    build_start = time.perf_counter()
    index = faiss.IndexHNSWFlat(base.shape[1], arguments.max_neighbors, metric)
    index.hnsw.efConstruction = arguments.ef_construction
    index.add(base)
    build_seconds = time.perf_counter() - build_start
    index_rss = current_rss()

    print(
        f"built faiss hnsw over {index.ntotal} vectors in {build_seconds:.2f}s",
        file=sys.stderr,
    )

    rows: list[dict[str, object]] = []
    for ef in ef_values:
        index.hnsw.efSearch = ef

        rates: list[float] = []
        labels = np.empty((queries.shape[0], arguments.k), dtype=np.int64)
        for _ in range(arguments.runs):
            start = time.perf_counter()
            _, labels = index.search(queries, arguments.k)
            elapsed = time.perf_counter() - start
            rates.append(queries.shape[0] / elapsed if elapsed > 0 else 0.0)

        rows.append(
            {
                "engine": "faiss",
                "index": "hnsw",
                "dataset": arguments.dataset,
                "metric": arguments.metric,
                "groundtruth_source": truth_source,
                "num_base": base.shape[0],
                "num_queries": queries.shape[0],
                "dimension": base.shape[1],
                "k": arguments.k,
                "max_neighbors": arguments.max_neighbors,
                "max_neighbors_layer0": 2 * arguments.max_neighbors,
                "ef_construction": arguments.ef_construction,
                "ef_search": ef,
                "threads": arguments.threads,
                "runs": arguments.runs,
                "recall_at_k": round(recall_at_k(labels, truth, arguments.k), 6),
                "qps_median": round(statistics.median(rates), 3),
                "qps_min": round(min(rates), 3),
                "qps_max": round(max(rates), 3),
                "build_seconds": round(build_seconds, 6),
                "index_rss_bytes": max(index_rss - rss_before, 0),
                "peak_rss_bytes": peak_rss(),
            }
        )

    handle = open(arguments.csv, "w", newline="", encoding="utf-8") if arguments.csv else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if arguments.csv:
            handle.close()

    print("results:", file=sys.stderr)
    for row in rows:
        print(
            f"  hnsw ef_search={row['ef_search']} threads={row['threads']} "
            f"runs={row['runs']} recall@{arguments.k}={row['recall_at_k']:.4f} "
            f"qps_median={row['qps_median']:.1f} "
            f"peak_rss_mib={row['peak_rss_bytes'] / (1024 * 1024):.1f}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
