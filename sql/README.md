# SQL metadata

`schema.sql` is the whole schema as raw PostgreSQL DDL: `owners`, `documents` (one row per indexed vector, joined to the HNSW index by `vector_id`, which is the vector's HNSW label), and `search_log` (one row per `/search` call).

## Local setup

Khoj runs against a locally installed PostgreSQL 16; there is no container setup. On Ubuntu:

```sh
sudo apt install postgresql-16
sudo -u postgres psql -c "CREATE ROLE khoj LOGIN PASSWORD 'khoj'"
sudo -u postgres psql -c "CREATE DATABASE khoj OWNER khoj"
PGPASSWORD=khoj psql -h localhost -U khoj -d khoj -f sql/schema.sql
```

The service reads `KHOJ_DATABASE_URL` (for example `postgresql://khoj:khoj@localhost:5432/khoj`) and `KHOJ_INDEX_PATH`, a saved `HnswIndex` whose labels are the `documents.vector_id` values:

```sh
KHOJ_DATABASE_URL=postgresql://khoj:khoj@localhost:5432/khoj \
KHOJ_INDEX_PATH=data/sift/sift1m.hnsw \
uvicorn --factory service.app:create_app_from_environment
```

The pytest suite and `bench/bench_filter.py` connect to `postgresql://khoj:khoj@localhost:5432/khoj` by default. Override it with `KHOJ_TEST_DATABASE_URL` for the tests and `--database-url` for the benchmark. Neither needs `CREATEDB`: each run creates its own scratch schema in that database, applies `schema.sql` inside it, and drops it at the end, so they never touch the tables in `public`.

## The `(owner_id, language, created_at)` index and its leftmost prefix

```sql
CREATE INDEX documents_owner_language_created_at_idx
    ON documents (owner_id, language, created_at);
```

A B-tree on several columns is sorted by the first column, then by the second within equal firsts, then by the third. A query can seek straight to a contiguous slice of it only by pinning a *leftmost prefix* of those columns with equalities, optionally finishing with a range on the next column. For this index:

| Predicate | What the index does |
|---|---|
| `owner_id = ? AND language = ? AND created_at >= ?` | Seeks to exactly the matching slice. Best case. |
| `owner_id = ? AND language = ?` | Seeks to the owner-and-language slice. |
| `owner_id = ?` | Seeks to the owner's slice. |
| `owner_id = ? AND created_at >= ?` | Seeks to the owner's slice, then **reads every entry in it** and checks `created_at` on each, because with `language` unpinned the `created_at` values are not contiguous. |
| `language = ?`, `created_at >= ?`, or both, without `owner_id` | Cannot seek at all. On PostgreSQL 16 the planner falls back to a sequential scan. |

The `/search` service always pins `owner_id`, so it never hits the last row. It does hit the fourth whenever a request passes `since` without `language`. That query is still correct, and it never touches rows belonging to other owners, but its index work grows with the owner's document count rather than with the number of matches.

Measured on PostgreSQL 16.15 by `explain_prefix.sql`, which loads 1,000,000 documents spread evenly over 50 owners, 6 languages, and the year 2026 into a scratch schema, then runs `EXPLAIN (ANALYZE, BUFFERS)` for owner 7, `language = 'fr'`, `created_at >= '2026-07-01'`:

```sh
PGPASSWORD=khoj psql -h localhost -U khoj -d khoj -f sql/explain_prefix.sql
```

| Predicate | Plan | Index buffers | Rows returned |
|---|---|---:|---:|
| owner + language + since | bitmap index scan | 9 | 1,681 |
| owner + since (language skipped) | bitmap index scan | 90 | 10,082 |
| owner only | bitmap index scan | 90 | 20,000 |
| language + since, no owner | sequential scan, 7,353 heap buffers | — | 84,050 |

Owner + since reads the same 90 index pages as owner alone: the `created_at` bound appears in `Index Cond` but does not narrow the scan.

If `since`-without-`language` became the common query, the fix is a second index on `(owner_id, created_at)`, not reordering this one, because reordering to `(owner_id, created_at, language)` would make `language` the column left unpinned. PostgreSQL 18 adds B-tree skip scan, which can jump between the distinct `language` values inside an owner's slice; with only a handful of languages that recovers most of the cost without another index.

## The other indexes the queries rely on

- `documents.vector_id UNIQUE` backs the post-filter strategy's `WHERE vector_id = ANY(%s) AND …`, which checks a few hundred candidate ids by primary lookup.
- `documents.owner_id` needs no index of its own: it is the leading column of the composite index, so the foreign key and owner-only lookups both use it.
- `search_log.owner_id` is not indexed. Nothing in the service reads `search_log`, so the index would only slow down the insert that every search pays for. Add one when something queries the log by owner.
