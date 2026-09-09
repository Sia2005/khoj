# Khoj

A vector search engine written from scratch in C++17, exposed to Python through pybind11 and served over FastAPI. Built to be benchmarked against FAISS and exact brute-force search on 1M 128-dimensional vectors.

## Non-negotiable constraints

The HNSW index and the quantizer are implemented from first principles. Do not add FAISS, hnswlib, nmslib, Annoy, ScaNN, or any other ANN library as a dependency of the engine. FAISS may appear only inside `bench/` as a comparison baseline, never as something the engine calls.

## Code style

Write no comments. Not on functions, not inline, not block headers, not TODO markers, not docstrings in the C++ or the Python. Code must be self-documenting through precise naming and small functions. If a line needs a comment to be understood, restructure it until it does not.

C++: C++17, four-space indent, `snake_case` for functions and variables, `PascalCase` for types, trailing underscore on private members. Prefer `std::vector` over raw arrays, `std::unique_ptr` over `new`, and `const` wherever it applies. No `using namespace std` in headers.

Python: PEP 8, type hints on every public function signature, `snake_case` throughout.

## Layout

```
include/khoj/     public headers
src/              implementation
python/           pybind11 bindings
service/          FastAPI application
bench/            benchmark harness and baselines
tests/            catch2 for C++, pytest for Python
CMakeLists.txt
```

## Build and test

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
pytest tests/
```

Always build in Release for anything that produces a timing number. Debug builds are ten times slower and make benchmark results meaningless.

## Testing expectations

Every algorithmic component gets a test before it is considered done. Recall correctness is verified against exact brute-force search on a small subset where ground truth can be computed directly. Concurrency is tested by running parallel queries against a fixed index and asserting results match the single-threaded run.

## Benchmark discipline

Report recall@10, queries per second, and resident memory. Take the median of at least five runs, never the mean of one. State the thread count and the `ef_search` value with every number. Never report a benchmark figure that was not produced by an actual run.

## Working agreement

When a task involves the HNSW graph construction, search, or neighbour selection logic, explain the approach and wait rather than writing the implementation. Those files are written by hand. Everything else in the tree you own end to end.