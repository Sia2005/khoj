from __future__ import annotations

import time
from typing import Callable

import numpy as np
import psycopg

import khoj

from service.filtering import (
    DocumentFilter,
    allowed_vector_ids,
    matching_vector_ids,
    owner_exists,
)
from service.models import SearchHit, SearchRequest, SearchResponse, Strategy


class UnknownOwner(LookupError):
    pass


class InvalidQuery(ValueError):
    pass


StrategyOutcome = tuple[list[SearchHit], int]
StrategyFunction = Callable[
    [psycopg.Connection, khoj.HnswIndex, np.ndarray, SearchRequest, DocumentFilter],
    StrategyOutcome,
]


def as_hits(results: list[khoj.SearchResult]) -> list[SearchHit]:
    return [SearchHit(vector_id=int(result.label), distance=result.distance) for result in results]


def pre_filter_search(
    connection: psycopg.Connection,
    index: khoj.HnswIndex,
    query: np.ndarray,
    request: SearchRequest,
    document_filter: DocumentFilter,
) -> StrategyOutcome:
    allowed = allowed_vector_ids(connection, document_filter)
    if allowed.size == 0:
        return [], 0
    results = index.search(query, request.k, request.ef_search, allowed_ids=allowed)
    return as_hits(results), int(allowed.size)


def post_filter_search(
    connection: psycopg.Connection,
    index: khoj.HnswIndex,
    query: np.ndarray,
    request: SearchRequest,
    document_filter: DocumentFilter,
) -> StrategyOutcome:
    candidate_count = request.k * request.post_widening
    candidates = index.search(query, candidate_count, max(request.ef_search, candidate_count))
    matching = matching_vector_ids(
        connection, document_filter, [int(result.label) for result in candidates]
    )
    survivors = [result for result in candidates if int(result.label) in matching]
    return as_hits(survivors[: request.k]), len(survivors)


STRATEGIES: dict[Strategy, StrategyFunction] = {
    "pre": pre_filter_search,
    "post": post_filter_search,
}


def query_vector(index: khoj.HnswIndex, values: list[float]) -> np.ndarray:
    query = np.asarray(values, dtype=np.float32)
    if query.shape != (index.dimension,):
        raise InvalidQuery(f"vector must have {index.dimension} values, got {query.size}")
    if not np.isfinite(query).all():
        raise InvalidQuery("vector values must be finite")
    return query


def record_search(
    connection: psycopg.Connection,
    request: SearchRequest,
    matched_count: int,
    latency_ms: float,
) -> None:
    connection.execute(
        "INSERT INTO search_log (owner_id, ef_search, k, strategy, latency_ms, matched_count) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        [
            request.owner_id,
            request.ef_search,
            request.k,
            request.strategy,
            latency_ms,
            matched_count,
        ],
    )


def run_search(
    connection: psycopg.Connection, index: khoj.HnswIndex, request: SearchRequest
) -> SearchResponse:
    query = query_vector(index, request.vector)
    document_filter = DocumentFilter(request.owner_id, request.language, request.since)

    with connection.transaction():
        if not owner_exists(connection, request.owner_id):
            raise UnknownOwner(f"owner {request.owner_id} does not exist")

        started = time.perf_counter()
        hits, matched_count = STRATEGIES[request.strategy](
            connection, index, query, request, document_filter
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        record_search(connection, request, matched_count, latency_ms)

    return SearchResponse(
        strategy=request.strategy,
        hits=hits,
        matched_count=matched_count,
        latency_ms=latency_ms,
    )
