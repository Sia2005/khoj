from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import khoj

DIMENSION = 16
COUNT = 256
MISSING_LABEL = np.iinfo(np.uint64).max


def sample_vectors(count: int, dimension: int = DIMENSION, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(count, dimension)).astype(np.float32)


def flat_index(vectors: np.ndarray, metric: khoj.Metric = khoj.Metric.L2) -> khoj.FlatIndex:
    index = khoj.FlatIndex(dimension=vectors.shape[1], metric=metric)
    index.add_batch(vectors)
    return index


def hnsw_index(vectors: np.ndarray, metric: khoj.Metric = khoj.Metric.L2) -> khoj.HnswIndex:
    index = khoj.HnswIndex(dimension=vectors.shape[1], metric=metric)
    index.add_batch(vectors)
    return index


def test_flat_index_reports_its_configuration() -> None:
    vectors = sample_vectors(COUNT)
    index = flat_index(vectors)

    assert index.size == COUNT
    assert len(index) == COUNT
    assert index.dimension == DIMENSION
    assert index.metric == khoj.Metric.L2


def test_flat_add_batch_round_trips_through_numpy() -> None:
    vectors = sample_vectors(COUNT)
    index = flat_index(vectors)

    stored = index.to_numpy()

    assert stored.shape == (COUNT, DIMENSION)
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, vectors)


def test_flat_add_assigns_sequential_labels() -> None:
    index = khoj.FlatIndex(dimension=DIMENSION)
    vectors = sample_vectors(3)

    assert index.add(vectors[0]) == 0
    assert index.add(vectors[1]) == 1
    assert index.size == 2


def test_flat_search_batch_matches_per_query_search() -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(8, seed=99)
    index = flat_index(vectors)

    labels, distances = index.search_batch(queries, k=5)

    assert labels.shape == (8, 5)
    assert distances.shape == (8, 5)
    assert labels.dtype == np.uint64
    assert distances.dtype == np.float32

    for row, query in enumerate(queries):
        expected = index.search(query, k=5)
        assert [result.label for result in expected] == labels[row].tolist()
        np.testing.assert_allclose(
            [result.distance for result in expected], distances[row], rtol=1e-6
        )


def test_flat_search_batch_matches_a_numpy_reference() -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(4, seed=1234)
    index = flat_index(vectors)

    labels, distances = index.search_batch(queries, k=10)

    reference = np.linalg.norm(queries[:, None, :] - vectors[None, :, :], axis=2)
    expected_labels = np.argsort(reference, axis=1, kind="stable")[:, :10]

    np.testing.assert_array_equal(labels, expected_labels.astype(np.uint64))
    np.testing.assert_allclose(
        distances, np.take_along_axis(reference, expected_labels, axis=1), rtol=1e-5, atol=1e-5
    )


def test_search_accepts_float64_and_non_contiguous_input() -> None:
    vectors = sample_vectors(COUNT)
    index = flat_index(vectors)

    queries = sample_vectors(4, seed=5).astype(np.float64)
    non_contiguous = np.asfortranarray(queries)

    contiguous_labels, _ = index.search_batch(np.ascontiguousarray(queries), k=4)
    fortran_labels, _ = index.search_batch(non_contiguous, k=4)

    np.testing.assert_array_equal(contiguous_labels, fortran_labels)


def test_search_batch_pads_when_k_exceeds_the_index() -> None:
    vectors = sample_vectors(3)
    index = flat_index(vectors)

    labels, distances = index.search_batch(sample_vectors(1, seed=3), k=5)

    assert labels.shape == (1, 5)
    np.testing.assert_array_equal(labels[0, 3:], np.array([MISSING_LABEL] * 2, dtype=np.uint64))
    assert np.all(np.isinf(distances[0, 3:]))
    assert np.all(distances[0, 3:] > 0)


def test_inner_product_padding_uses_negative_infinity() -> None:
    vectors = sample_vectors(2)
    index = flat_index(vectors, metric=khoj.Metric.InnerProduct)

    _, distances = index.search_batch(sample_vectors(1, seed=3), k=4)

    assert np.all(np.isinf(distances[0, 2:]))
    assert np.all(distances[0, 2:] < 0)


def test_empty_batches_return_empty_arrays() -> None:
    index = flat_index(sample_vectors(COUNT))
    empty = np.empty((0, DIMENSION), dtype=np.float32)

    index.add_batch(empty)
    labels, distances = index.search_batch(empty, k=5)

    assert index.size == COUNT
    assert labels.shape == (0, 5)
    assert distances.shape == (0, 5)


def test_shape_errors_are_reported_as_value_errors() -> None:
    index = khoj.FlatIndex(dimension=DIMENSION)

    with pytest.raises(ValueError, match="two-dimensional"):
        index.add_batch(np.zeros(DIMENSION, dtype=np.float32))

    with pytest.raises(ValueError, match="rows of width 16"):
        index.add_batch(np.zeros((4, DIMENSION + 1), dtype=np.float32))

    with pytest.raises(ValueError, match="one-dimensional"):
        index.search(np.zeros((1, DIMENSION), dtype=np.float32), k=1)

    with pytest.raises(ValueError, match="vector of length 16"):
        index.search(np.zeros(DIMENSION - 1, dtype=np.float32), k=1)


def test_hnsw_index_exposes_its_parameters() -> None:
    index = khoj.HnswIndex(dimension=DIMENSION, max_neighbors=8, ef_construction=64, seed=11)

    assert index.dimension == DIMENSION
    assert index.params.max_neighbors == 8
    assert index.params.ef_construction == 64
    assert index.params.seed == 11
    assert index.size == 0


def test_hnsw_accepts_an_explicit_params_object() -> None:
    params = khoj.HnswParams()
    params.dimension = DIMENSION
    params.ef_construction = 80

    index = khoj.HnswIndex(params)

    assert index.params.ef_construction == 80


def test_hnsw_params_property_is_a_copy() -> None:
    index = khoj.HnswIndex(dimension=DIMENSION, ef_construction=64)

    index.params.ef_construction = 999

    assert index.params.ef_construction == 64


def test_hnsw_rejects_degenerate_parameters() -> None:
    with pytest.raises(ValueError):
        khoj.HnswIndex(dimension=0)

    with pytest.raises(ValueError):
        khoj.HnswIndex(dimension=DIMENSION, max_neighbors=16, ef_construction=4)


def test_hnsw_add_batch_labels_sequentially_by_default() -> None:
    vectors = sample_vectors(COUNT)
    index = hnsw_index(vectors)

    labels, _ = index.search_batch(vectors[:4], k=1, ef_search=32)

    assert index.size == COUNT
    np.testing.assert_array_equal(labels[:, 0], np.arange(4, dtype=np.uint64))


def test_hnsw_add_batch_continues_labels_across_calls() -> None:
    vectors = sample_vectors(COUNT)
    index = khoj.HnswIndex(dimension=DIMENSION)

    index.add_batch(vectors[:100])
    index.add_batch(vectors[100:])

    labels, _ = index.search_batch(vectors[100:104], k=1, ef_search=32)

    assert index.size == COUNT
    np.testing.assert_array_equal(labels[:, 0], np.arange(100, 104, dtype=np.uint64))


def test_hnsw_add_batch_honours_explicit_labels() -> None:
    vectors = sample_vectors(64)
    labels = np.arange(1000, 1064, dtype=np.uint64)

    index = khoj.HnswIndex(dimension=DIMENSION)
    index.add_batch(vectors, labels=labels)

    found, _ = index.search_batch(vectors[:4], k=1, ef_search=32)

    np.testing.assert_array_equal(found[:, 0], labels[:4])


def test_hnsw_add_batch_rejects_mismatched_labels() -> None:
    index = khoj.HnswIndex(dimension=DIMENSION)

    with pytest.raises(ValueError, match="one label per vector"):
        index.add_batch(sample_vectors(4), labels=np.arange(3, dtype=np.uint64))


def test_hnsw_single_add_and_search_agree_with_the_batch_path() -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(8, seed=808)

    index = khoj.HnswIndex(dimension=DIMENSION)
    for label, vector in enumerate(vectors):
        index.add(label, vector)

    labels, distances = index.search_batch(queries, k=5, ef_search=64)

    for row, query in enumerate(queries):
        results = index.search(query, k=5, ef_search=64)
        assert [result.label for result in results] == labels[row].tolist()
        np.testing.assert_allclose(
            [result.distance for result in results], distances[row], rtol=1e-6
        )


def test_hnsw_matches_the_flat_index_at_exhaustive_ef_search() -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(16, seed=2718)

    exact = flat_index(vectors)
    approximate = hnsw_index(vectors)

    expected_labels, expected_distances = exact.search_batch(queries, k=10)
    labels, distances = approximate.search_batch(queries, k=10, ef_search=COUNT)

    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(distances, np.square(expected_distances), rtol=1e-4, atol=1e-4)


def test_hnsw_recall_against_the_flat_index() -> None:
    vectors = sample_vectors(2000, seed=31)
    queries = sample_vectors(100, seed=32)

    expected_labels, _ = flat_index(vectors).search_batch(queries, k=10)
    labels, _ = hnsw_index(vectors).search_batch(queries, k=10, ef_search=100)

    hits = sum(len(set(row) & set(expected)) for row, expected in zip(labels, expected_labels))
    recall = hits / expected_labels.size

    assert recall >= 0.95


def test_hnsw_save_and_load_round_trip(tmp_path: Path) -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(8, seed=451)
    index = hnsw_index(vectors)

    path = tmp_path / "index.hnsw"
    index.save(str(path))
    restored = khoj.HnswIndex.load(str(path))

    assert restored.size == index.size
    assert restored.dimension == index.dimension
    assert restored.max_level == index.max_level

    expected = index.search_batch(queries, k=10, ef_search=64)
    actual = restored.search_batch(queries, k=10, ef_search=64)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_parallel_queries_match_the_single_threaded_run() -> None:
    vectors = sample_vectors(4000, seed=64)
    queries = sample_vectors(200, seed=65)
    index = hnsw_index(vectors)

    expected_labels, expected_distances = index.search_batch(queries, k=10, ef_search=100)
    results: list[tuple[np.ndarray, np.ndarray]] = [(np.empty(0), np.empty(0))] * 4

    def worker(slot: int) -> None:
        results[slot] = index.search_batch(queries, k=10, ef_search=100)

    workers = [threading.Thread(target=worker, args=(slot,)) for slot in range(4)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    for labels, distances in results:
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(distances, expected_distances)


def test_batch_search_releases_the_gil() -> None:
    vectors = sample_vectors(4000, seed=64)
    queries = sample_vectors(500, seed=65)
    index = hnsw_index(vectors)

    ticks = 0
    stop = threading.Event()

    def count_ticks() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    ticker = threading.Thread(target=count_ticks)
    ticker.start()
    try:
        index.search_batch(queries, k=10, ef_search=200)
    finally:
        stop.set()
        ticker.join()

    assert ticks >= 10


def test_distances_are_squared_l2_for_an_indexed_vector() -> None:
    vectors = sample_vectors(COUNT)
    index = hnsw_index(vectors)

    _, distances = index.search_batch(vectors[:1], k=1, ef_search=32)

    assert math.isclose(float(distances[0, 0]), 0.0, abs_tol=1e-6)


def restricted_exact_labels(
    vectors: np.ndarray, queries: np.ndarray, allowed: np.ndarray, k: int
) -> np.ndarray:
    subset = vectors[allowed]
    squared = ((queries[:, None, :] - subset[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(squared, axis=1, kind="stable")[:, :k]
    return allowed[order].astype(np.uint64)


@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.5, 0.9])
def test_filtered_search_batch_matches_restricted_brute_force(fraction: float) -> None:
    vectors = sample_vectors(2000, seed=91)
    queries = sample_vectors(50, seed=92)
    generator = np.random.default_rng(93)
    allowed = np.flatnonzero(generator.random(len(vectors)) < fraction).astype(np.uint64)

    labels, _ = hnsw_index(vectors).search_batch(
        queries, k=10, ef_search=100, allowed_ids=allowed
    )
    expected = restricted_exact_labels(vectors, queries, allowed, k=10)

    returned = labels[labels != MISSING_LABEL]
    assert np.isin(returned, allowed).all()
    hits = sum(len(set(row) & set(truth)) for row, truth in zip(labels, expected))
    assert hits / expected.size >= 0.95


@pytest.mark.parametrize(
    "as_container",
    [
        set,
        list,
        tuple,
        frozenset,
        lambda ids: np.array(ids, dtype=np.uint64),
        lambda ids: np.array(ids, dtype=np.int64),
    ],
    ids=["set", "list", "tuple", "frozenset", "uint64-array", "int64-array"],
)
def test_allowed_ids_accepts_any_iterable_of_labels(as_container) -> None:
    vectors = sample_vectors(COUNT)
    index = hnsw_index(vectors)
    allowed = [3, 17, 42, 99, 200]

    results = index.search(vectors[42], k=10, ef_search=32, allowed_ids=as_container(allowed))

    assert results[0].label == 42
    assert sorted(result.label for result in results) == allowed


def test_no_allowed_ids_means_no_filter() -> None:
    vectors = sample_vectors(COUNT)
    queries = sample_vectors(8, seed=94)
    index = hnsw_index(vectors)

    unfiltered = index.search_batch(queries, k=10, ef_search=64)
    explicit_none = index.search_batch(queries, k=10, ef_search=64, allowed_ids=None)

    np.testing.assert_array_equal(explicit_none[0], unfiltered[0])
    np.testing.assert_array_equal(explicit_none[1], unfiltered[1])


def test_an_empty_allowed_set_pads_every_slot() -> None:
    vectors = sample_vectors(COUNT)
    index = hnsw_index(vectors)

    labels, distances = index.search_batch(vectors[:2], k=5, ef_search=32, allowed_ids=set())

    assert (labels == MISSING_LABEL).all()
    assert np.isinf(distances).all()
    assert index.search(vectors[0], k=5, ef_search=32, allowed_ids=[]) == []


def test_allowed_ids_rejects_negative_and_two_dimensional_labels() -> None:
    vectors = sample_vectors(COUNT)
    index = hnsw_index(vectors)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        index.search(vectors[0], k=5, ef_search=32, allowed_ids=[-1])
    with pytest.raises(ValueError, match="one-dimensional"):
        index.search(vectors[0], k=5, ef_search=32, allowed_ids=np.zeros((2, 2), dtype=np.uint64))


def test_parallel_filtered_queries_match_the_single_threaded_run() -> None:
    vectors = sample_vectors(4000, seed=64)
    queries = sample_vectors(200, seed=65)
    index = hnsw_index(vectors)
    allowed = np.arange(0, 4000, 7, dtype=np.uint64)

    expected = index.search_batch(queries, k=10, ef_search=100, allowed_ids=allowed)
    results: list[tuple[np.ndarray, np.ndarray]] = [(np.empty(0), np.empty(0))] * 4

    def worker(slot: int) -> None:
        results[slot] = index.search_batch(queries, k=10, ef_search=100, allowed_ids=allowed)

    workers = [threading.Thread(target=worker, args=(slot,)) for slot in range(4)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    for labels, distances in results:
        np.testing.assert_array_equal(labels, expected[0])
        np.testing.assert_array_equal(distances, expected[1])
