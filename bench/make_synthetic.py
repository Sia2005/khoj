from __future__ import annotations

import argparse
import os

import numpy as np

from vecs_io import write_fvecs, write_ivecs


def generate_clustered(
    count: int, dimension: int, clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centres = rng.normal(0.0, 6.0, size=(clusters, dimension))
    assignment = rng.integers(0, clusters, size=count)
    noise = rng.normal(0.0, 1.0, size=(count, dimension))
    return (centres[assignment] + noise).astype(np.float32)


def exact_neighbours(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    wide_base = base.astype(np.float64)
    wide_queries = queries.astype(np.float64)
    base_norms = (wide_base**2).sum(axis=1)

    result = np.empty((queries.shape[0], k), dtype=np.int32)
    for start in range(0, wide_queries.shape[0], 64):
        chunk = wide_queries[start : start + 64]
        distances = (
            base_norms[None, :] - 2.0 * (chunk @ wide_base.T) + (chunk**2).sum(axis=1)[:, None]
        )
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        result[start : start + chunk.shape[0]] = order.astype(np.int32)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--base-count", type=int, default=20000)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=40)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    rng = np.random.default_rng(arguments.seed)

    os.makedirs(arguments.out_dir, exist_ok=True)

    base = generate_clustered(
        arguments.base_count, arguments.dimension, arguments.clusters, rng
    )
    queries = generate_clustered(
        arguments.query_count, arguments.dimension, arguments.clusters, rng
    )
    truth = exact_neighbours(base, queries, arguments.k)

    base_path = os.path.join(arguments.out_dir, "base.fvecs")
    query_path = os.path.join(arguments.out_dir, "query.fvecs")
    truth_path = os.path.join(arguments.out_dir, "groundtruth.ivecs")

    write_fvecs(base_path, base)
    write_fvecs(query_path, queries)
    write_ivecs(truth_path, truth)

    print(f"wrote {base_path} ({base.shape[0]} x {base.shape[1]})")
    print(f"wrote {query_path} ({queries.shape[0]} x {queries.shape[1]})")
    print(f"wrote {truth_path} ({truth.shape[0]} x {truth.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
