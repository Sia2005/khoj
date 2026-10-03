from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg
from psycopg import sql


@dataclass(frozen=True)
class DocumentFilter:
    owner_id: int
    language: str | None = None
    since: datetime | None = None

    def predicate(self) -> tuple[sql.Composed, list[object]]:
        clauses = [sql.SQL("owner_id = %s")]
        parameters: list[object] = [self.owner_id]
        if self.language is not None:
            clauses.append(sql.SQL("language = %s"))
            parameters.append(self.language)
        if self.since is not None:
            clauses.append(sql.SQL("created_at >= %s"))
            parameters.append(self.since)
        return sql.SQL(" AND ").join(clauses), parameters


def owner_exists(connection: psycopg.Connection, owner_id: int) -> bool:
    row = connection.execute("SELECT 1 FROM owners WHERE id = %s", [owner_id]).fetchone()
    return row is not None


def allowed_vector_ids(
    connection: psycopg.Connection, document_filter: DocumentFilter
) -> np.ndarray:
    predicate, parameters = document_filter.predicate()
    query = sql.SQL(
        "SELECT coalesce(array_agg(vector_id), '{{}}') FROM documents WHERE {}"
    ).format(predicate)
    row = connection.execute(query, parameters).fetchone()
    return np.asarray(row[0], dtype=np.uint64)


def matching_vector_ids(
    connection: psycopg.Connection,
    document_filter: DocumentFilter,
    candidates: list[int],
) -> set[int]:
    if not candidates:
        return set()
    predicate, parameters = document_filter.predicate()
    query = sql.SQL("SELECT vector_id FROM documents WHERE vector_id = ANY(%s) AND {}").format(
        predicate
    )
    rows = connection.execute(query, [candidates, *parameters]).fetchall()
    return {row[0] for row in rows}
