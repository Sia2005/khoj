from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("fastapi")
pytest.importorskip("psycopg_pool")

from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

import khoj
import service.search
from service.app import create_app

DIMENSION = 16
COUNT = 600
EXHAUSTIVE_EF = COUNT
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def corpus() -> np.ndarray:
    return np.random.default_rng(1729).normal(size=(COUNT, DIMENSION)).astype(np.float32)


def owner_of(vector_id: int) -> int:
    return 2 if vector_id % 3 == 0 else 1


def language_of(vector_id: int) -> str:
    return "en" if vector_id % 2 == 0 else "fr"


def created_at_of(vector_id: int) -> datetime:
    return EPOCH + timedelta(hours=vector_id)


def matches(
    vector_id: int, owner_id: int, language: str | None, since: datetime | None
) -> bool:
    return (
        owner_of(vector_id) == owner_id
        and (language is None or language_of(vector_id) == language)
        and (since is None or created_at_of(vector_id) >= since)
    )


def exact_order(vectors: np.ndarray, query: np.ndarray) -> list[int]:
    squared = ((vectors - query) ** 2).sum(axis=1)
    return np.argsort(squared, kind="stable").tolist()


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    return corpus()


@pytest.fixture(scope="module")
def index(vectors: np.ndarray) -> khoj.HnswIndex:
    built = khoj.HnswIndex(dimension=DIMENSION)
    built.add_batch(vectors, labels=np.arange(COUNT, dtype=np.uint64))
    return built


@pytest.fixture
def seeded_database(empty_database: str) -> str:
    with psycopg.connect(empty_database) as connection:
        connection.execute(
            "INSERT INTO owners (name, plan) VALUES ('acme', 'pro'), ('globex', 'free')"
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO documents (vector_id, owner_id, language, created_at, title) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (
                        vector_id,
                        owner_of(vector_id),
                        language_of(vector_id),
                        created_at_of(vector_id),
                        f"document {vector_id}",
                    )
                    for vector_id in range(COUNT)
                ],
            )
    return empty_database


@pytest.fixture
def pool(seeded_database: str) -> Iterator[ConnectionPool]:
    with ConnectionPool(seeded_database, min_size=1, max_size=4, open=True) as opened:
        yield opened


@pytest.fixture
def client(index: khoj.HnswIndex, pool: ConnectionPool) -> TestClient:
    return TestClient(create_app(index, pool))


def search_log_rows(url: str) -> list[tuple]:
    with psycopg.connect(url) as connection:
        return connection.execute(
            "SELECT owner_id, ef_search, k, strategy, latency_ms, matched_count "
            "FROM search_log ORDER BY id"
        ).fetchall()


def request_body(query: np.ndarray, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"owner_id": 1, "vector": query.tolist(), "k": 10}
    body.update(overrides)
    return body


def returned_ids(response) -> list[int]:
    assert response.status_code == 200, response.text
    return [hit["vector_id"] for hit in response.json()["hits"]]


def test_schema_rejects_a_plan_outside_free_and_pro(empty_database: str) -> None:
    with psycopg.connect(empty_database) as connection:
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute("INSERT INTO owners (name, plan) VALUES ('initech', 'enterprise')")


def test_schema_enforces_unique_vector_ids_and_known_owners(seeded_database: str) -> None:
    with psycopg.connect(seeded_database) as connection:
        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                "INSERT INTO documents (vector_id, owner_id, language) VALUES (0, 1, 'en')"
            )
    with psycopg.connect(seeded_database) as connection:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                "INSERT INTO documents (vector_id, owner_id, language) VALUES (9999, 77, 'en')"
            )


def test_documents_index_leads_with_owner_then_language_then_created_at(
    empty_database: str,
) -> None:
    with psycopg.connect(empty_database) as connection:
        columns = connection.execute(
            "SELECT attribute.attname "
            "FROM pg_index AS entry "
            "JOIN pg_class AS index_class ON index_class.oid = entry.indexrelid "
            "CROSS JOIN LATERAL unnest(entry.indkey) WITH ORDINALITY AS key(attnum, position) "
            "JOIN pg_attribute AS attribute "
            "  ON attribute.attrelid = entry.indrelid AND attribute.attnum = key.attnum "
            "WHERE index_class.relname = 'documents_owner_language_created_at_idx' "
            "ORDER BY key.position"
        ).fetchall()
    assert [column for (column,) in columns] == ["owner_id", "language", "created_at"]


@pytest.mark.parametrize("strategy", ["pre", "post"])
@pytest.mark.parametrize(
    ("owner_id", "language", "since_hours"),
    [(1, None, None), (1, "en", None), (2, None, 300), (1, "fr", 150)],
)
def test_every_hit_satisfies_the_filter(
    client: TestClient,
    vectors: np.ndarray,
    strategy: str,
    owner_id: int,
    language: str | None,
    since_hours: int | None,
) -> None:
    since = None if since_hours is None else EPOCH + timedelta(hours=since_hours)
    body = request_body(
        vectors[5] + 0.1,
        owner_id=owner_id,
        language=language,
        since=None if since is None else since.isoformat(),
        strategy=strategy,
        ef_search=64,
    )

    hits = returned_ids(client.post("/search", json=body))

    assert hits
    assert all(matches(vector_id, owner_id, language, since) for vector_id in hits)


def test_pre_filter_returns_the_exact_restricted_top_k(
    client: TestClient, vectors: np.ndarray
) -> None:
    query = vectors[11] + 0.05
    since = EPOCH + timedelta(hours=120)
    expected = [
        vector_id
        for vector_id in exact_order(vectors, query)
        if matches(vector_id, 1, "fr", since)
    ][:10]

    body = request_body(
        query, language="fr", since=since.isoformat(), strategy="pre", ef_search=EXHAUSTIVE_EF
    )

    assert returned_ids(client.post("/search", json=body)) == expected


def test_post_filter_keeps_the_allowed_part_of_the_widened_candidates(
    client: TestClient, vectors: np.ndarray
) -> None:
    query = vectors[11] + 0.05
    widening = 3
    candidates = exact_order(vectors, query)[: 10 * widening]
    expected = [vector_id for vector_id in candidates if matches(vector_id, 1, "en", None)][:10]

    body = request_body(
        query,
        language="en",
        strategy="post",
        post_widening=widening,
        ef_search=EXHAUSTIVE_EF,
    )

    assert returned_ids(client.post("/search", json=body)) == expected


def test_a_selective_filter_starves_post_but_not_pre(
    client: TestClient, vectors: np.ndarray
) -> None:
    query = vectors[40]
    since = EPOCH + timedelta(hours=450)
    common = {
        "owner_id": 2,
        "language": "fr",
        "since": since.isoformat(),
        "ef_search": EXHAUSTIVE_EF,
    }
    pre_body = request_body(query, strategy="pre", **common)
    post_body = request_body(query, strategy="post", post_widening=1, **common)

    pre_hits = returned_ids(client.post("/search", json=pre_body))
    post_hits = returned_ids(client.post("/search", json=post_body))

    allowed_count = sum(matches(vector_id, 2, "fr", since) for vector_id in range(COUNT))
    assert allowed_count >= 10
    assert len(pre_hits) == 10
    assert len(post_hits) < 10
    assert set(post_hits) <= set(pre_hits)


@pytest.mark.parametrize("strategy", ["pre", "post"])
def test_each_search_writes_one_log_row(
    client: TestClient, vectors: np.ndarray, seeded_database: str, strategy: str
) -> None:
    body = request_body(vectors[3], language="en", strategy=strategy, ef_search=48, k=7)

    response = client.post("/search", json=body)
    payload = response.json()

    assert response.status_code == 200
    rows = search_log_rows(seeded_database)
    assert len(rows) == 1
    owner_id, ef_search, k, logged_strategy, latency_ms, matched_count = rows[0]
    assert (owner_id, ef_search, k, logged_strategy) == (1, 48, 7, strategy)
    assert latency_ms == pytest.approx(payload["latency_ms"])
    assert latency_ms > 0
    assert matched_count == payload["matched_count"]

    allowed_count = sum(matches(vector_id, 1, "en", None) for vector_id in range(COUNT))
    if strategy == "pre":
        assert matched_count == allowed_count
    else:
        assert len(payload["hits"]) <= matched_count <= 7 * 10


def test_a_filter_matching_nothing_returns_no_hits_and_is_logged(
    client: TestClient, vectors: np.ndarray, seeded_database: str
) -> None:
    response = client.post("/search", json=request_body(vectors[0], language="de"))

    assert response.status_code == 200
    assert response.json()["hits"] == []
    assert response.json()["matched_count"] == 0
    assert len(search_log_rows(seeded_database)) == 1


def test_an_unknown_owner_is_a_404_and_logs_nothing(
    client: TestClient, vectors: np.ndarray, seeded_database: str
) -> None:
    response = client.post("/search", json=request_body(vectors[0], owner_id=99))

    assert response.status_code == 404
    assert search_log_rows(seeded_database) == []


def test_a_failure_after_the_log_insert_rolls_the_insert_back(
    index: khoj.HnswIndex,
    vectors: np.ndarray,
    seeded_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_record_search = service.search.record_search

    def record_then_fail(*arguments: object) -> None:
        real_record_search(*arguments)
        raise RuntimeError("failure after the log insert")

    monkeypatch.setattr(service.search, "record_search", record_then_fail)
    request = service.search.SearchRequest(owner_id=1, vector=vectors[0].tolist())

    with psycopg.connect(seeded_database, autocommit=True) as connection:
        with pytest.raises(RuntimeError, match="after the log insert"):
            service.search.run_search(connection, index, request)

    assert search_log_rows(seeded_database) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"vector": [0.0] * (DIMENSION - 1)},
        {"strategy": "mid"},
        {"k": 0},
        {"ef_search": 0},
        {"since": "not a timestamp"},
    ],
    ids=["wrong-dimension", "unknown-strategy", "zero-k", "zero-ef", "bad-since"],
)
def test_malformed_requests_are_rejected_and_not_logged(
    client: TestClient,
    vectors: np.ndarray,
    seeded_database: str,
    overrides: dict[str, object],
) -> None:
    response = client.post("/search", json=request_body(vectors[0], **overrides))

    assert response.status_code == 422
    assert search_log_rows(seeded_database) == []
