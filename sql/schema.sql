CREATE TABLE owners (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL CHECK (plan IN ('free', 'pro'))
);

CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    vector_id BIGINT NOT NULL UNIQUE,
    owner_id INT NOT NULL REFERENCES owners (id),
    language TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    title TEXT
);

CREATE INDEX documents_owner_language_created_at_idx
    ON documents (owner_id, language, created_at);

CREATE TABLE search_log (
    id BIGSERIAL PRIMARY KEY,
    owner_id INT NOT NULL REFERENCES owners (id),
    ef_search INT NOT NULL,
    k INT NOT NULL,
    strategy TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    matched_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
