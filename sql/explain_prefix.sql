CREATE SCHEMA khoj_explain;
SET search_path = khoj_explain;
\i sql/schema.sql

INSERT INTO owners (name, plan)
SELECT 'owner ' || n, CASE WHEN n % 2 = 0 THEN 'pro' ELSE 'free' END
FROM generate_series(1, 50) AS n;

INSERT INTO documents (vector_id, owner_id, language, created_at)
SELECT
    i,
    1 + i % 50,
    (ARRAY['en', 'fr', 'de', 'es', 'hi', 'ja'])[1 + (i / 50) % 6],
    timestamptz '2026-01-01 00:00:00+00' + (i / 1000000.0) * interval '365 days'
FROM generate_series(0, 999999) AS i;

VACUUM ANALYZE documents;

\echo owner + language + since
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF)
SELECT vector_id FROM documents
WHERE owner_id = 7 AND language = 'fr' AND created_at >= '2026-07-01';

\echo owner + since, language skipped
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF)
SELECT vector_id FROM documents
WHERE owner_id = 7 AND created_at >= '2026-07-01';

\echo owner only
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF)
SELECT vector_id FROM documents
WHERE owner_id = 7;

\echo language + since, no owner
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF)
SELECT vector_id FROM documents
WHERE language = 'fr' AND created_at >= '2026-07-01';

DROP SCHEMA khoj_explain CASCADE;
