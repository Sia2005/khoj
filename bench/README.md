# Benchmarks

Measures recall@k, queries per second, and peak resident memory for the khoj flat
index, the khoj HNSW index, and FAISS HNSW as a comparison baseline.

FAISS lives here and only here. The engine never calls it.

## Getting SIFT-1M

Download `ANN_SIFT1M` from <http://corpus-texmex.irisa.fr/> and unpack it into
`data/sift/`. The harness needs three of the four files:

| File | Shape | Format |
|---|---|---|
| `sift_base.fvecs` | 1,000,000 × 128 | `.fvecs` |
| `sift_query.fvecs` | 10,000 × 128 | `.fvecs` |
| `sift_groundtruth.ivecs` | 10,000 × 100 | `.ivecs` |

Both formats store one record per row: a little-endian `int32` dimension followed
by that many `float32` (`.fvecs`) or `int32` (`.ivecs`) values.

## Running

Each index is benchmarked in its own process so that peak RSS is attributable to
one index rather than to whatever else the process had already allocated.

```
cmake --build build -j

./build/bench_khoj --index flat \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --groundtruth data/sift/sift_groundtruth.ivecs \
  --dataset sift1m --k 10 --threads 1 --runs 5 \
  --csv bench/results/khoj_flat.csv

./build/bench_khoj --index hnsw \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --groundtruth data/sift/sift_groundtruth.ivecs \
  --dataset sift1m --k 10 --threads 1 --runs 5 \
  --max-neighbors 16 --max-neighbors-layer0 32 --ef-construction 200 \
  --ef-search 10,20,40,80,160,320 \
  --csv bench/results/khoj_hnsw.csv

PYTHONPATH=bench python bench/bench_faiss.py \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --groundtruth data/sift/sift_groundtruth.ivecs \
  --dataset sift1m --k 10 --threads 1 --runs 5 \
  --max-neighbors 16 --ef-construction 200 \
  --ef-search 10,20,40,80,160,320 \
  --csv bench/results/faiss_hnsw.csv

PYTHONPATH=bench python bench/plot_recall_qps.py \
  khoj-hnsw-baseline=bench/results/khoj_hnsw.csv \
  bench/results/khoj_hnsw_optimized.csv \
  khoj-hnsw-native=bench/results/khoj_hnsw_optimized_native.csv \
  bench/results/faiss_hnsw.csv \
  bench/results/khoj_flat.csv \
  --out bench/results/recall_vs_qps
```

Build in Release. A Debug build makes every timing number meaningless.

## Filtered search

`bench_filter.py` measures the service's two filtering strategies, `pre` and `post`, at several filter selectivities. It needs the service dependencies (`pip install -e '.[test]'` covers them) and PostgreSQL. It connects to `postgresql://khoj:khoj@localhost:5432/khoj` unless you pass `--database-url`, and works in a scratch schema that it creates and drops (see [`sql/README.md`](../sql/README.md) for the local setup).

```
PYTHONPATH=bench python bench/bench_filter.py \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --dataset sift1m --query-count 200 --runs 5 \
  --k 10 --ef-search 80 --post-widening 10 \
  --selectivities 0.01,0.1,0.5,0.9 \
  --index-cache data/sift/sift1m.hnsw \
  --csv bench/results/filter_selectivity.csv \
  --plot bench/results/filter_selectivity
```

What it does:

- Builds the HNSW index once with labels equal to row numbers, and saves it to `--index-cache` so later runs load it instead of rebuilding.
- Loads one `documents` row per vector, all owned by one owner in one language. Each row gets a distinct `created_at` from a seeded random permutation, so the filter is uncorrelated with where the vector sits in space.
- Picks each selectivity *s* by setting `since` so that exactly round(*s*·N) documents pass. That filter pins the full `(owner_id, language, created_at)` prefix, so PostgreSQL range-scans only the matching entries.
- Computes ground truth by exact brute force over only the allowed vectors, in float64, for every selectivity.
- Calls the service's own `run_search` on one connection from one thread, so each query runs the same transaction as `POST /search`: the SQL filter, the HNSW search, and the `search_log` insert and commit. HTTP is the only part left out.

It reports two latencies:

- `strategy_*_ms` is the service's own measurement: SQL filtering plus HNSW search, inside the transaction.
- `end_to_end_*_ms` adds the owner check, the log insert, and the durable commit. The commit costs the same for both strategies and is dominated by WAL fsync on the machine running PostgreSQL.

`qps_*` is derived from end-to-end time. Every figure is the median of `--runs` passes over the same queries, and `recall_at_k` must be identical across passes or the script stops. `index_rss_bytes` is the process's resident memory right after the index is built or loaded, before the ground-truth buffers are allocated.

`plot_filter.py` re-renders the plot from a CSV without re-running anything.

## Smoke testing without SIFT

`make_synthetic.py` writes clustered Gaussian vectors in the same two formats so
the harness can be exercised without the 500 MB download:

```
PYTHONPATH=bench python bench/make_synthetic.py --out-dir data/synthetic
```

Its ground truth is computed in float64 by numpy, independently of the flat index,
so a flat run against it is a genuine correctness check rather than a tautology.
The data is **not** a stand-in for SIFT: well-separated clusters in 128 dimensions
put every point in a cluster at nearly the same distance from the query, which
depresses recall for any graph index. Use it to verify the harness, never to
characterise the engine.

## CSV schema

Both harnesses emit the same columns, one row per configuration:

`engine, index, dataset, metric, groundtruth_source, num_base, num_queries,
dimension, k, max_neighbors, max_neighbors_layer0, ef_construction, ef_search,
threads, runs, recall_at_k, qps_median, qps_min, qps_max, build_seconds,
index_rss_bytes, peak_rss_bytes`

`ef_search`, `max_neighbors`, `max_neighbors_layer0`, and `ef_construction` are
empty for the flat index. Every row carries its own `threads`, `runs`, and
`ef_search`, so no number can be quoted without the settings that produced it.

`groundtruth_source` records where the ground truth came from: `file` for a
supplied `.ivecs`, `flat` when the khoj flat index computed it, `faiss_flat` when
`faiss.IndexFlat` did, and `self` for a flat run with no ground truth supplied,
where recall is 1.0 by construction and carries no information.

## Reading the numbers honestly

`--runs` is rejected below five, and `qps_median` is the median across runs.
`qps_min` and `qps_max` are reported alongside so run-to-run spread stays visible.

`index_rss_bytes` is resident memory measured across index construction.
`peak_rss_bytes` is the process high-water mark from `VmHWM`. When
`groundtruth_source` is not `file`, the ground truth was computed in the same
process, so `peak_rss_bytes` includes a full brute-force index and overstates
what the index under test costs. Supply `--groundtruth` for any memory number you
intend to report.

The two harnesses issue queries differently. `bench_khoj` calls `search` once per
query and spreads queries across `--threads` worker threads. `bench_faiss` hands
FAISS the whole query matrix and lets its OpenMP pool schedule the work, which is
how FAISS is normally measured. At large `ef_search` the per-call difference is
noise against the cost of the search itself; at small `ef_search` it flatters
FAISS, so treat the low-recall end of the curve as indicative rather than exact.
