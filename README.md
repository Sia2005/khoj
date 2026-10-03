# Khoj

A vector search engine written from scratch in C++17 — HNSW and exact flat indexes, no ANN library underneath.

## SIFT-1M

Recall@10 and queries per second on ANN_SIFT1M (1,000,000 base × 10,000 query vectors, 128 dimensions, squared L2), scored against the distributed `sift_groundtruth.ivecs`. Every row is a real run; QPS is the median of five.

| Index | ef_search | recall@10 | QPS | Build | Index RSS |
|---|---:|---:|---:|---:|---:|
| **khoj HNSW** | 10 | **0.7688** | 9,976 | 1,179 s | 680.6 MiB |
| **khoj HNSW** | 20 | **0.8782** | 6,616 | | |
| **khoj HNSW** | 40 | **0.9500** | 4,086 | | |
| **khoj HNSW** | 80 | **0.9843** | 2,364 | | |
| **khoj HNSW** | 160 | **0.9957** | 1,337 | | |
| **khoj HNSW** | 320 | **0.9987** | 735 | | |
| FAISS HNSW | 10 | 0.7157 | 23,140 | 501 s | 632.8 MiB |
| FAISS HNSW | 20 | 0.8457 | 14,326 | | |
| FAISS HNSW | 40 | 0.9337 | 8,215 | | |
| FAISS HNSW | 80 | 0.9775 | 4,825 | | |
| FAISS HNSW | 160 | 0.9937 | 2,800 | | |
| FAISS HNSW | 320 | 0.9982 | 1,428 | | |
| khoj flat (exact) | — | 0.9994 | 23.5 | 0.3 s | 488.5 MiB |

![recall@10 versus queries per second on SIFT-1M](bench/results/recall_vs_qps.png)

All khoj rows are the default build (`KHOJ_NATIVE_ARCH=OFF`, baseline x86-64). The plot also shows khoj HNSW before the optimizations below, and a non-default `-march=native` build.

Sources: [`bench/results/khoj_hnsw_optimized.csv`](bench/results/khoj_hnsw_optimized.csv), [`bench/results/faiss_hnsw.csv`](bench/results/faiss_hnsw.csv), [`bench/results/khoj_flat.csv`](bench/results/khoj_flat.csv); for the plot only, [`bench/results/khoj_hnsw.csv`](bench/results/khoj_hnsw.csv) (before optimization) and [`bench/results/khoj_hnsw_optimized_native.csv`](bench/results/khoj_hnsw_optimized_native.csv) (native build).

The flat index is exhaustive, so its 0.9994 is a property of the ground-truth file rather than of the scan: 137 of the 10,000 queries have an 11th neighbour exactly as close as the 10th, which puts up to 0.00137 of recall@10 beyond reach of any tie-breaking rule. The observed shortfall is 0.00056.

### Optimization

Two changes to `HnswIndex`, measured together: `search_layer` leases a pooled, version-stamped visited list instead of allocating and hashing into a `std::unordered_set` on every call, and HNSW distances go through the flat index's lane-blocked kernels instead of a scalar loop. The graph is unchanged, so **recall@10 is bit-identical across all three builds** at every `ef_search`; only time moved.

| khoj HNSW | Build | QPS @ 10 | 20 | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 1,942 s | 7,219 | 4,629 | 2,355 | 1,197 | 799 | 430 |
| **optimized (default build)** | **1,179 s** | **9,976** | **6,616** | **4,086** | **2,364** | **1,337** | **735** |
| optimized, `KHOJ_NATIVE_ARCH=ON` (non-default) | 838 s | 14,388 | 9,657 | 5,710 | 3,347 | 1,859 | 1,028 |
| recall@10, all three | | 0.7688 | 0.8782 | 0.9500 | 0.9843 | 0.9957 | 0.9987 |

In the default build that is 1.4–2.0× the query throughput and a 1.65× faster build. Compiling with `-march=native` (AVX2 + FMA on this host) takes it to 2.0–2.8× and 2.3×, but the binary then runs only on CPUs with the build host's instruction set, which is why it is not the default. Because both changes landed before either was benchmarked on SIFT-1M, the split between them is not measured.

### Conditions

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-1255U, 6 cores / 12 threads, AVX2 + FMA (no AVX-512) |
| Memory | 7.6 GiB |
| OS | Linux 6.6.87.2 (WSL2), Ubuntu 24.04 |
| Toolchain | GCC 13.3.0, `-O3 -DNDEBUG`, `KHOJ_NATIVE_ARCH=OFF` (baseline x86-64); rows marked native use `KHOJ_NATIVE_ARCH=ON` |
| Baseline | faiss 1.15.1 (CPU), NumPy 2.5.3, Python 3.12.3 |
| Threads | **1**, for search and for index construction, on both engines |
| Runs | 5 per configuration; median reported, min and max kept in the CSV |

HNSW parameters are matched across the two engines: `max_neighbors` (M) = 16, `max_neighbors_layer0` = 32, `ef_construction` = 200, `k` = 10, `ef_search` ∈ {10, 20, 40, 80, 160, 320}. Khoj uses seed 42 and derives its level multiplier as 1/ln(M).

One difference the numbers cannot hide: `bench_khoj` calls `search` once per query, while `bench_faiss` hands FAISS the whole query matrix, which is how FAISS is normally measured. Per-call overhead is noise at high `ef_search` and flatters FAISS at low `ef_search`, so treat the left end of the curve as indicative.

## Where khoj stands against FAISS

**FAISS is 1.9–2.3× faster at matched `ef_search`, and 1.6–1.9× faster at matched recall**, down from 3.1–4.0× and 2.5–3.2× before the optimizations above. Matched-recall ratios interpolate the FAISS curve log-linearly at each khoj recall; `ef_search`=320 is left out because khoj's recall there exceeds anything FAISS reached. FAISS also builds its index in 501 s against khoj's 1,179 s.

The gap that remains is most plausibly instruction set and kernel quality. FAISS ships hand-written SIMD kernels and picks AVX2 at runtime on this host. Khoj's kernels — lane-blocked accumulator loops, shared by both indexes — are portable C++ left to the compiler, which in the default baseline x86-64 build can only emit SSE. The non-default native build puts both engines on AVX2 and narrows the gap to 1.4–1.6× at matched `ef_search` and 1.1–1.3× at matched recall (build: 838 s vs 501 s). What is left after that is unapportioned: hand-written versus compiler-vectorised kernels, other per-query costs in the search loop, and the per-call versus batched harness difference noted above. No profile has been taken to separate them.

**Khoj returns higher recall@10 than FAISS at every `ef_search` tested**, by 5.3 points at `ef_search`=10 narrowing to 0.05 points at 320:

| ef_search | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|---:|---:|
| khoj − FAISS (recall@10, points) | +5.31 | +3.26 | +1.63 | +0.68 | +0.20 | +0.05 |

So khoj extracts more recall per unit of `ef_search`, and FAISS extracts more throughput per unit of work. Matched on recall rather than on `ef_search`, the default-build gap narrows from about 2× to 1.6–1.9×, but it does not close. Closing it in the default build means kernels that use AVX2 without `-march=native` — runtime dispatch, as FAISS does.

Both HNSW indexes are worth their build cost against exhaustive search: at `ef_search`=40 khoj answers queries 174× faster than its own flat index for a 4.9-point recall trade. (The flat index was not re-run; the kernel change only moved its code.)

## Filtered search

`POST /search` restricts results by `documents` metadata in PostgreSQL (`owner_id`, and optionally `language` and `since`) with one of two strategies:

- **`pre`** resolves the allowed `vector_id`s with SQL, then passes them to `HnswIndex.search` as `allowed_ids`. The graph walk visits every node it reaches but admits only allowed ones to the result set, so the graph stays connected however narrow the filter is.
- **`post`** searches unfiltered for `k × post_widening` candidates (with `ef_search` raised to at least that many), then keeps the ones SQL confirms match.

The SQL filter, the search, and the `search_log` insert run in one transaction. [`sql/README.md`](sql/README.md) has the schema, the local PostgreSQL setup, and why the `(owner_id, language, created_at)` index only helps a leftmost prefix.

### Selectivity crossover

SIFT-1M, 200 queries, `k` = 10, `ef_search` = 80, `post_widening` = 10, 1 thread, median of 5 runs. Recall is against exact brute force over only the allowed vectors. The filter is uncorrelated with vector position. Latency is the median time for the SQL filter plus the HNSW search. End-to-end latency also includes the owner check, the log insert, and the durable commit. Both are measured against local PostgreSQL 16 on the same host.

| Selectivity | Allowed docs | pre recall@10 | post recall@10 | pre p50 | post p50 | pre end-to-end p50 | post end-to-end p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 10,000 | **1.0000** | 0.0995 | 46.7 ms | **2.4 ms** | 51.1 ms | **5.5 ms** |
| 10% | 100,000 | **0.9995** | 0.8330 | 104.7 ms | **2.6 ms** | 109.5 ms | **6.0 ms** |
| 50% | 500,000 | **0.9950** | 0.9890 | 294.9 ms | **2.8 ms** | 304.5 ms | **6.8 ms** |
| 90% | 900,000 | 0.9890 | **0.9905** | 508.2 ms | **2.5 ms** | 518.5 ms | **5.7 ms** |

![recall@10 and latency of pre- and post-filtering across selectivities](bench/results/filter_selectivity.png)

**Pre-filtering never wins on latency, and its recall advantage disappears between 10% and 50% selectivity.** Below that point, post-filtering's 100 candidates hold too few matches: an average of 1 hit out of k = 10 at 1% selectivity, and 8.5 at 10%. Pre-filtering stays at 0.9995 recall or better. From 50% up, both strategies fill all ten slots and sit within 0.6 points of each other, with post-filtering ahead at 90%. Post-filtering is faster at every selectivity: 19× at 1%, 40× at 10%, 107× at 50%, and 200× at 90%.

Post-filtering's latency stays flat at 2.4–2.8 ms. Pre-filtering's grows roughly in step with the number of allowed documents, from 47 ms at 10,000 to 508 ms at 900,000. That points to the cost of producing the allowed set: `array_agg` in PostgreSQL, transferring it, and building a hash set of it for every query. The search itself is not the likely cost. No profile has been taken to split that time.

So with this implementation, use `pre` for selective filters (≤10%), where post-filtering loses most of its recall, and use `post` for broad ones. The table doesn't pin the crossover any closer than "between 10% and 50%".

Source: [`bench/results/filter_selectivity.csv`](bench/results/filter_selectivity.csv), which also holds p95s, QPS min/max, and the process RSS of 1.22 GiB. [`bench/README.md`](bench/README.md) has the command line.

## Build

Requires CMake ≥ 3.24, a C++17 compiler, and a Python interpreter with development headers for the bindings (3.12 here). Catch2 and pybind11 are fetched by CMake at configure time.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Anything that produces a timing number must be a Release build; Debug is roughly ten times slower and makes the benchmark meaningless.

```sh
ctest --test-dir build --output-on-failure   # C++ suite (Catch2)
pytest tests/                                # bindings and service (pytest)
```

The service tests need PostgreSQL 16. They connect to `postgresql://khoj:khoj@localhost:5432/khoj` unless `KHOJ_TEST_DATABASE_URL` says otherwise, and skip if the server is unreachable. Each session works in its own scratch schema and drops it afterwards. [`sql/README.md`](sql/README.md) covers the local setup.

```sh
pip install -e '.[test]'                     # service dependencies plus pytest and httpx
```

Build options, all defaulting on except the last: `KHOJ_BUILD_TESTS`, `KHOJ_BUILD_PYTHON`, `KHOJ_BUILD_BENCH`, and `KHOJ_NATIVE_ARCH` (off — set it to compile with `-march=native`; it changes timing numbers, so state it whenever you report one).

Reproducing the benchmarks — including where to get SIFT-1M and how to generate synthetic data without the 500 MB download — is documented in [`bench/README.md`](bench/README.md).

## Architecture

```
include/khoj/     public headers — types, flat_index, hnsw_index
src/              flat_index.cpp, hnsw_index.cpp        (khoj_core, static)
python/           bindings.cpp                          (pybind11 module `khoj`)
service/          FastAPI application, POST /search     (psycopg 3, raw SQL)
sql/              schema.sql, index notes
bench/            harness, FAISS baseline, filter benchmark, plotting
tests/            Catch2 for C++, pytest for the bindings and the service
```

`khoj_core` is a static library with no dependencies beyond the standard library and `Threads::Threads`. Everything above it is optional and can be switched off at configure time.

**`FlatIndex`** stores vectors contiguously in one `std::vector<float>` and scans all of them per query. Exact by construction, and the reference every approximate result is checked against.

**`HnswIndex`** is a flat-array hierarchical graph. Layer 0 adjacency lives in a single `std::vector<InternalId>` of `max_neighbors_layer0` slots per node with a parallel degree array, so a neighbour walk is one indexed load rather than a pointer chase; upper layers are stored per-level in the same shape. Node levels are drawn from an exponential distribution using a seeded xorshift generator, making construction deterministic for a given seed and insertion order. Search descends greedily through the upper layers to find an entry point, then runs a best-first `search_layer` on layer 0 with two heaps bounded by `ef_search` and a visited list leased from a pool on the index; each list holds a 16-bit stamp per node, so starting a query clears it in O(1) by bumping the stamp. Edges are chosen by the relative-neighbourhood heuristic — a candidate is kept only if no already-selected neighbour is closer to it than the base node is, with discarded candidates backfilling to the degree limit — and over-full nodes are re-pruned by the same rule. Indexes serialise to a flat binary file behind a magic number, and `load` restores the graph without rebuilding.

`search` takes an optional label filter. It applies only when a node would enter the result heap: every reached node is still visited, measured, and pushed onto the frontier, so the walk crosses disallowed regions and the graph stays connected. The admission bound is the worst *allowed* result, so a selective filter keeps the search expanding until it has `ef_search` allowed candidates or runs out of graph. Without a filter, search and construction run exactly as before.

`search` is `const`; on HNSW its only shared state is the visited-list pool, which it locks only to lease and return a list. Concurrent queries against a finished index are safe — asserted for the flat index and for filtered HNSW search in the Catch2 suite, and for HNSW with and without a filter in the pytest suite, all by checking that parallel queries reproduce the single-threaded results exactly. Insertion is single-threaded.

The Python module wraps both indexes with NumPy-aware conversion (float64 and non-contiguous inputs are accepted and converted), and releases the GIL around batch search, save, and load. HNSW `search` and `search_batch` take an optional `allowed_ids` (any iterable of labels, or a NumPy integer array), copied into a hash set before the GIL is released.

## Written from scratch

Everything in the search path:

- the HNSW graph — level assignment, greedy descent, best-first layer search, the relative-neighbourhood neighbour-selection heuristic, and degree pruning
- the flat-array adjacency layout, the pooled version-stamped visited list, and the binary save/load format
- L2 and inner-product distance kernels, lane-blocked for compiler auto-vectorisation and shared by the flat and HNSW indexes
- exhaustive search, k-selection, and deterministic tie-breaking by ascending label
- the xorshift PRNG behind level assignment
- the `.fvecs` / `.ivecs` readers, the benchmark harness, and the recall/QPS/RSS measurement

No FAISS, hnswlib, nmslib, Annoy, or ScaNN anywhere in the engine. FAISS appears only in `bench/` as a comparison baseline and is never called by the index. Third-party code is limited to pybind11 (bindings); FastAPI, psycopg 3 and psycopg-pool (service, raw SQL with no ORM); Catch2, pytest and httpx (tests); and NumPy, faiss and matplotlib (benchmark harness only).

Not yet implemented: quantization, deletion or update of indexed vectors, multi-threaded index construction, and keeping the HNSW index in step with `documents` (the service loads a saved index at startup and trusts `vector_id` to match its labels).
