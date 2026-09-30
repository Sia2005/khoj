# Khoj

A vector search engine written from scratch in C++17 — HNSW and exact flat indexes, no ANN library underneath.

## SIFT-1M

Recall@10 and queries per second on ANN_SIFT1M (1,000,000 base × 10,000 query vectors, 128 dimensions, squared L2), scored against the distributed `sift_groundtruth.ivecs`. Every row is a real run; QPS is the median of five.

| Index | ef_search | recall@10 | QPS | Build | Index RSS |
|---|---:|---:|---:|---:|---:|
| **khoj HNSW** | 10 | **0.7688** | 7,219 | 1,942 s | 679.0 MiB |
| **khoj HNSW** | 20 | **0.8782** | 4,629 | | |
| **khoj HNSW** | 40 | **0.9500** | 2,355 | | |
| **khoj HNSW** | 80 | **0.9843** | 1,197 | | |
| **khoj HNSW** | 160 | **0.9957** | 799 | | |
| **khoj HNSW** | 320 | **0.9987** | 430 | | |
| FAISS HNSW | 10 | 0.7157 | 23,140 | 501 s | 632.8 MiB |
| FAISS HNSW | 20 | 0.8457 | 14,326 | | |
| FAISS HNSW | 40 | 0.9337 | 8,215 | | |
| FAISS HNSW | 80 | 0.9775 | 4,825 | | |
| FAISS HNSW | 160 | 0.9937 | 2,800 | | |
| FAISS HNSW | 320 | 0.9982 | 1,428 | | |
| khoj flat (exact) | — | 0.9994 | 23.5 | 0.3 s | 488.5 MiB |

![recall@10 versus queries per second on SIFT-1M](bench/results/recall_vs_qps.png)

Sources: [`bench/results/khoj_hnsw.csv`](bench/results/khoj_hnsw.csv), [`bench/results/faiss_hnsw.csv`](bench/results/faiss_hnsw.csv), [`bench/results/khoj_flat.csv`](bench/results/khoj_flat.csv).

The flat index is exhaustive, so its 0.9994 is a property of the ground-truth file rather than of the scan: 137 of the 10,000 queries have an 11th neighbour exactly as close as the 10th, which puts up to 0.00137 of recall@10 beyond reach of any tie-breaking rule. The observed shortfall is 0.00056.

### Conditions

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-1255U, 6 cores / 12 threads, AVX2 + FMA (no AVX-512) |
| Memory | 7.6 GiB |
| OS | Linux 6.6.87.2 (WSL2), Ubuntu 24.04 |
| Toolchain | GCC 13.3.0, `-O3 -DNDEBUG`, `KHOJ_NATIVE_ARCH=OFF` (baseline x86-64) |
| Baseline | faiss 1.15.1 (CPU), NumPy 2.5.3, Python 3.12.3 |
| Threads | **1**, for search and for index construction, on both engines |
| Runs | 5 per configuration; median reported, min and max kept in the CSV |

HNSW parameters are matched across the two engines: `max_neighbors` (M) = 16, `max_neighbors_layer0` = 32, `ef_construction` = 200, `k` = 10, `ef_search` ∈ {10, 20, 40, 80, 160, 320}. Khoj uses seed 42 and derives its level multiplier as 1/ln(M).

One difference the numbers cannot hide: `bench_khoj` calls `search` once per query, while `bench_faiss` hands FAISS the whole query matrix, which is how FAISS is normally measured. Per-call overhead is noise at high `ef_search` and flatters FAISS at low `ef_search`, so treat the left end of the curve as indicative.

## Where khoj stands against FAISS

**FAISS is 3.1–4.0× faster at matched `ef_search`, and 2.5–3.2× faster at matched recall.** The likely cause is distance-kernel throughput. FAISS ships hand-written SIMD kernels with runtime dispatch. Khoj's are portable C++ left to the compiler: the flat index uses lane-blocked accumulator loops, and the HNSW index — the one benchmarked here — uses a plain scalar loop over the 128 dimensions. The benchmark build also targets baseline x86-64, not the host's AVX2. That khoj reaches higher recall at every matched `ef_search` argues the graph itself is sound and the cost is per distance evaluation, but no profile has been taken to apportion the gap precisely.

**Khoj returns higher recall@10 than FAISS at every `ef_search` tested**, by 5.3 points at `ef_search`=10 narrowing to 0.05 points at 320:

| ef_search | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|---:|---:|
| khoj − FAISS (recall@10, points) | +5.31 | +3.26 | +1.63 | +0.68 | +0.20 | +0.05 |

So khoj extracts more recall per unit of `ef_search`, and FAISS extracts far more throughput per unit of work. Matched on recall rather than on `ef_search`, the throughput gap narrows from 3–4× to roughly 2.5–3×, but it does not close. Closing it means writing the SIMD kernels.

Both HNSW indexes are worth their build cost against exhaustive search: at `ef_search`=40 khoj answers queries 100× faster than its own flat index for a 4.9-point recall trade.

## Build

Requires CMake ≥ 3.24, a C++17 compiler, and a Python interpreter with development headers for the bindings (3.12 here). Catch2 and pybind11 are fetched by CMake at configure time.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Anything that produces a timing number must be a Release build; Debug is roughly ten times slower and makes the benchmark meaningless.

```sh
ctest --test-dir build --output-on-failure   # C++ suite (Catch2)
pytest tests/                                # Python bindings (pytest)
```

Build options, all defaulting on except the last: `KHOJ_BUILD_TESTS`, `KHOJ_BUILD_PYTHON`, `KHOJ_BUILD_BENCH`, and `KHOJ_NATIVE_ARCH` (off — set it to compile with `-march=native`; it changes timing numbers, so state it whenever you report one).

Reproducing the benchmarks — including where to get SIFT-1M and how to generate synthetic data without the 500 MB download — is documented in [`bench/README.md`](bench/README.md).

## Architecture

```
include/khoj/     public headers — types, flat_index, hnsw_index
src/              flat_index.cpp, hnsw_index.cpp        (khoj_core, static)
python/           bindings.cpp                          (pybind11 module `khoj`)
service/          FastAPI application                   (not yet implemented)
bench/            harness, FAISS baseline, plotting
tests/            Catch2 for C++, pytest for the bindings
```

`khoj_core` is a static library with no dependencies beyond the standard library and `Threads::Threads`. Everything above it is optional and can be switched off at configure time.

**`FlatIndex`** stores vectors contiguously in one `std::vector<float>` and scans all of them per query. Exact by construction, and the reference every approximate result is checked against.

**`HnswIndex`** is a flat-array hierarchical graph. Layer 0 adjacency lives in a single `std::vector<InternalId>` of `max_neighbors_layer0` slots per node with a parallel degree array, so a neighbour walk is one indexed load rather than a pointer chase; upper layers are stored per-level in the same shape. Node levels are drawn from an exponential distribution using a seeded xorshift generator, making construction deterministic for a given seed and insertion order. Search descends greedily through the upper layers to find an entry point, then runs a best-first `search_layer` on layer 0 with two heaps and a visited set bounded by `ef_search`. Edges are chosen by the relative-neighbourhood heuristic — a candidate is kept only if no already-selected neighbour is closer to it than the base node is, with discarded candidates backfilling to the degree limit — and over-full nodes are re-pruned by the same rule. Indexes serialise to a flat binary file behind a magic number, and `load` restores the graph without rebuilding.

`search` is `const` and takes no locks, so concurrent queries against a finished index are safe — asserted for the flat index in the Catch2 suite and for HNSW in the pytest suite, both by checking that parallel queries reproduce the single-threaded results exactly. Insertion is single-threaded.

The Python module wraps both indexes with NumPy-aware conversion (float64 and non-contiguous inputs are accepted and converted), and releases the GIL around batch search, save, and load.

## Written from scratch

Everything in the search path:

- the HNSW graph — level assignment, greedy descent, best-first layer search, the relative-neighbourhood neighbour-selection heuristic, and degree pruning
- the flat-array adjacency layout and the binary save/load format
- L2 and inner-product distance kernels (lane-blocked for auto-vectorisation in the flat index, scalar in HNSW)
- exhaustive search, k-selection, and deterministic tie-breaking by ascending label
- the xorshift PRNG behind level assignment
- the `.fvecs` / `.ivecs` readers, the benchmark harness, and the recall/QPS/RSS measurement

No FAISS, hnswlib, nmslib, Annoy, or ScaNN anywhere in the engine. FAISS appears only in `bench/` as a comparison baseline and is never called by the index. Third-party code is limited to pybind11 (bindings), Catch2 and pytest (tests), and NumPy, faiss and matplotlib (benchmark harness only).

Not yet implemented: quantization, deletion or update of indexed vectors, multi-threaded index construction, and the FastAPI service.
