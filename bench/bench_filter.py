from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT / "build", ROOT):
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import khoj

from plot_filter import plot_filter_results
from service.models import SearchRequest
from service.search import run_search
from vecs_io import read_fvecs

SCHEMA_PATH = ROOT / "sql" / "schema.sql"
LOCAL_DATABASE_URL = "postgresql://khoj:khoj@localhost:5432/khoj"
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
OWNER_ID = 1
LANGUAGE = "en"
GROUND_TRUTH_CHUNK = 50

CSV_FIELDS = [
    "dataset",
    "num_base",
    "num_queries",
    "dimension",
    "k",
    "ef_search",
    "post_widening",
    "max_neighbors",
    "max_neighbors_layer0",
    "ef_construction",
    "strategy",
    "selectivity",
    "matched_documents",
    "threads",
    "runs",
    "recall_at_k",
    "mean_hits_returned",
    "strategy_p50_ms",
    "strategy_p95_ms",
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "qps_median",
    "qps_min",
    "qps_max",
    "index_rss_bytes",
]


@dataclass(frozen=True)
class Selectivity:
    fraction: float
    since: datetime
    allowed_mask: np.ndarray

    @property
    def allowed_count(self) -> int:
        return int(self.allowed_mask.sum())


@dataclass(frozen=True)
class RunSummary:
    recall: float
    mean_hits: float
    strategy_p50_ms: float
    strategy_p95_ms: float
    end_to_end_p50_ms: float
    end_to_end_p95_ms: float
    qps: float


def median_and_p95(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    p95_position = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return statistics.median(ordered), ordered[p95_position]


def resident_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-count", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=80)
    parser.add_argument("--post-widening", type=int, default=10)
    parser.add_argument("--selectivities", default="0.01,0.1,0.5,0.9")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-neighbors-layer0", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-cache")
    parser.add_argument("--database-url", default=LOCAL_DATABASE_URL)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot")
    return parser.parse_args()


def load_or_build_index(arguments: argparse.Namespace, base: np.ndarray) -> khoj.HnswIndex:
    cache = Path(arguments.index_cache) if arguments.index_cache else None
    if cache is not None and cache.exists():
        index = khoj.HnswIndex.load(str(cache))
        if index.size != len(base) or index.dimension != base.shape[1]:
            raise SystemExit(f"{cache} holds {index.size} vectors, expected {len(base)}")
        print(f"loaded cached index from {cache}", flush=True)
        return index

    index = khoj.HnswIndex(
        dimension=base.shape[1],
        max_neighbors=arguments.max_neighbors,
        max_neighbors_layer0=arguments.max_neighbors_layer0,
        ef_construction=arguments.ef_construction,
        seed=arguments.seed,
    )
    started = time.perf_counter()
    index.add_batch(base, labels=np.arange(len(base), dtype=np.uint64))
    elapsed = time.perf_counter() - started
    print(f"built hnsw over {len(base)} vectors in {elapsed:.1f}s", flush=True)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        index.save(str(cache))
    return index


@contextmanager
def scratch_schema(server_url: str) -> Iterator[str]:
    schema = sql.Identifier(f"khoj_bench_{uuid.uuid4().hex[:12]}")
    with psycopg.connect(server_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        try:
            yield psycopg.conninfo.make_conninfo(
                server_url, options=f"-c search_path={schema.as_string(admin)}"
            )
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))


def creation_ranks(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(count)


def load_documents(url: str, ranks: np.ndarray) -> None:
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(SCHEMA_PATH.read_text())
        connection.execute(
            "INSERT INTO owners (id, name, plan) VALUES (%s, 'benchmark', 'pro')", [OWNER_ID]
        )
        with connection.cursor() as cursor:
            with cursor.copy(
                "COPY documents (vector_id, owner_id, language, created_at) FROM STDIN"
            ) as copy:
                for vector_id, rank in enumerate(ranks.tolist()):
                    copy.write_row(
                        (vector_id, OWNER_ID, LANGUAGE, EPOCH + timedelta(seconds=rank))
                    )
        connection.execute("ANALYZE documents")


def selectivity_for(fraction: float, ranks: np.ndarray) -> Selectivity:
    count = len(ranks)
    allowed_count = max(1, round(fraction * count))
    first_allowed_rank = count - allowed_count
    return Selectivity(
        fraction=fraction,
        since=EPOCH + timedelta(seconds=first_allowed_rank),
        allowed_mask=ranks >= first_allowed_rank,
    )


def restricted_ground_truth(
    base: np.ndarray, queries: np.ndarray, selectivities: list[Selectivity], k: int
) -> dict[float, np.ndarray]:
    exact_base = base.astype(np.float64)
    base_norms = np.einsum("ij,ij->i", exact_base, exact_base)
    truth = {
        selectivity.fraction: np.empty((len(queries), k), dtype=np.int64)
        for selectivity in selectivities
    }

    for start in range(0, len(queries), GROUND_TRUTH_CHUNK):
        chunk = queries[start : start + GROUND_TRUTH_CHUNK].astype(np.float64)
        squared = base_norms[None, :] - 2.0 * (chunk @ exact_base.T)
        for selectivity in selectivities:
            masked = np.where(selectivity.allowed_mask[None, :], squared, np.inf)
            nearest = np.argpartition(masked, k, axis=1)[:, :k]
            truth[selectivity.fraction][start : start + len(chunk)] = nearest
    return truth


def recall_of(found: list[list[int]], truth: np.ndarray) -> float:
    hits = sum(len(set(row) & set(expected.tolist())) for row, expected in zip(found, truth))
    return hits / truth.size


def measure_run(
    connection: psycopg.Connection,
    index: khoj.HnswIndex,
    queries: np.ndarray,
    truth: np.ndarray,
    template: SearchRequest,
) -> RunSummary:
    end_to_end_ms: list[float] = []
    strategy_ms: list[float] = []
    found: list[list[int]] = []

    for query in queries:
        request = template.model_copy(update={"vector": query.tolist()})
        started = time.perf_counter()
        response = run_search(connection, index, request)
        end_to_end_ms.append((time.perf_counter() - started) * 1000.0)
        strategy_ms.append(response.latency_ms)
        found.append([hit.vector_id for hit in response.hits])

    strategy_p50, strategy_p95 = median_and_p95(strategy_ms)
    end_to_end_p50, end_to_end_p95 = median_and_p95(end_to_end_ms)
    return RunSummary(
        recall=recall_of(found, truth),
        mean_hits=statistics.fmean(len(row) for row in found),
        strategy_p50_ms=strategy_p50,
        strategy_p95_ms=strategy_p95,
        end_to_end_p50_ms=end_to_end_p50,
        end_to_end_p95_ms=end_to_end_p95,
        qps=len(queries) * 1000.0 / sum(end_to_end_ms),
    )


def main() -> int:
    arguments = parse_arguments()
    if arguments.runs < 5:
        raise SystemExit("--runs must be at least 5: every reported figure is a median of five")

    base = read_fvecs(arguments.base, arguments.base_count)
    queries = read_fvecs(arguments.query, arguments.query_count)
    print(f"loaded {len(base)} base and {len(queries)} query vectors", flush=True)

    index = load_or_build_index(arguments, base)
    index_rss = resident_bytes()
    ranks = creation_ranks(len(base), arguments.seed)
    fractions = [float(value) for value in arguments.selectivities.split(",")]
    selectivities = [selectivity_for(fraction, ranks) for fraction in fractions]

    truth = restricted_ground_truth(base, queries, selectivities, arguments.k)
    print("computed restricted ground truth", flush=True)

    rows: list[dict[str, object]] = []
    with scratch_schema(arguments.database_url) as url:
        load_documents(url, ranks)
        print(f"loaded {len(base)} documents into postgres", flush=True)

        with psycopg.connect(url, autocommit=True) as connection:
            for selectivity in selectivities:
                for strategy in ("pre", "post"):
                    template = SearchRequest(
                        owner_id=OWNER_ID,
                        vector=[0.0],
                        k=arguments.k,
                        ef_search=arguments.ef_search,
                        language=LANGUAGE,
                        since=selectivity.since,
                        strategy=strategy,
                        post_widening=arguments.post_widening,
                    )
                    runs = [
                        measure_run(
                            connection,
                            index,
                            queries,
                            truth[selectivity.fraction],
                            template,
                        )
                        for _ in range(arguments.runs)
                    ]
                    row = summarise(arguments, base, queries, selectivity, strategy, runs)
                    row["index_rss_bytes"] = index_rss
                    rows.append(row)
                    print(
                        f"  {strategy:<4} selectivity={selectivity.fraction:<5} "
                        f"recall@{arguments.k}={row['recall_at_k']:.4f} "
                        f"strategy_p50={row['strategy_p50_ms']:.2f}ms "
                        f"end_to_end_p50={row['end_to_end_p50_ms']:.2f}ms "
                        f"qps={row['qps_median']:.1f}",
                        flush=True,
                    )

    write_csv(arguments.csv, rows)
    print(f"wrote {arguments.csv}")
    if arguments.plot:
        for path in plot_filter_results(rows, arguments.plot):
            print(f"wrote {path}")
    return 0


def summarise(
    arguments: argparse.Namespace,
    base: np.ndarray,
    queries: np.ndarray,
    selectivity: Selectivity,
    strategy: str,
    runs: list[RunSummary],
) -> dict[str, object]:
    recalls = {run.recall for run in runs}
    if len(recalls) != 1:
        raise SystemExit(f"recall varied across runs: {sorted(recalls)}")

    rates = [run.qps for run in runs]
    return {
        "dataset": arguments.dataset,
        "num_base": len(base),
        "num_queries": len(queries),
        "dimension": base.shape[1],
        "k": arguments.k,
        "ef_search": arguments.ef_search,
        "post_widening": arguments.post_widening,
        "max_neighbors": arguments.max_neighbors,
        "max_neighbors_layer0": arguments.max_neighbors_layer0,
        "ef_construction": arguments.ef_construction,
        "strategy": strategy,
        "selectivity": selectivity.fraction,
        "matched_documents": selectivity.allowed_count,
        "threads": 1,
        "runs": len(runs),
        "recall_at_k": runs[0].recall,
        "mean_hits_returned": runs[0].mean_hits,
        "strategy_p50_ms": statistics.median(run.strategy_p50_ms for run in runs),
        "strategy_p95_ms": statistics.median(run.strategy_p95_ms for run in runs),
        "end_to_end_p50_ms": statistics.median(run.end_to_end_p50_ms for run in runs),
        "end_to_end_p95_ms": statistics.median(run.end_to_end_p95_ms for run in runs),
        "qps_median": statistics.median(rates),
        "qps_min": min(rates),
        "qps_max": max(rates),
    }


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
