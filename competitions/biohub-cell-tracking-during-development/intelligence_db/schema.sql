-- Biohub - Cell Tracking During Development
-- Experiment intelligence database schema (DuckDB).
--
-- All data is public / our own: Kaggle run logs and diagnostics we produced,
-- and publicly-readable GitHub resources. No hidden competition data, no
-- Kaggle submission API, no private scraping.
--
-- Idempotent init: every table is (re)created with CREATE TABLE IF NOT
-- EXISTS; loaders delete-then-insert by key so re-running never duplicates.

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id            TEXT PRIMARY KEY,
    milestone                TEXT,
    variant                  TEXT,
    label                    TEXT,
    notebook_name            TEXT,
    raw_file                 TEXT,
    command_json             TEXT,
    det_threshold            DOUBLE,
    ilp_edge_weight          DOUBLE,
    ilp_appearance_weight    DOUBLE,
    ilp_disappearance_weight DOUBLE,
    ilp_division_weight      DOUBLE,
    postprocess_ops_json     TEXT,
    public_score             DOUBLE,      -- NULL => pending
    status                   TEXT,        -- scored | pending | failed
    created_at               TEXT,
    notes                    TEXT
);

CREATE TABLE IF NOT EXISTS submission_stats (
    experiment_id  TEXT,
    n_rows         INTEGER,
    n_nodes        INTEGER,
    n_edges        INTEGER,
    max_in_degree  INTEGER,
    max_out_degree INTEGER,
    fallback_used  BOOLEAN,
    valid          BOOLEAN,
    final_source   TEXT
);

CREATE TABLE IF NOT EXISTS postprocess_stats (
    experiment_id         TEXT,
    safe_divisions_added  INTEGER,
    gap1_closed           INTEGER,
    gap2_recovered        INTEGER,
    synthetic_nodes_added INTEGER,
    isolated_nodes_pruned INTEGER,
    edges_removed         INTEGER,
    nodes_before          INTEGER,
    nodes_after           INTEGER,
    edges_before          INTEGER,
    edges_after           INTEGER,
    per_dataset_json      TEXT
);

CREATE TABLE IF NOT EXISTS public_sources (
    source_id            TEXT PRIMARY KEY,
    source_type          TEXT,          -- metric_spec | code | notebook | analysis
    title                TEXT,
    url                  TEXT,
    repo                 TEXT,
    path                 TEXT,
    summary              TEXT,
    extracted_ideas_json TEXT,
    relevance_score      DOUBLE
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id         TEXT PRIMARY KEY,
    after_experiment_id TEXT,
    observation         TEXT,
    recommendation      TEXT,
    next_variant        TEXT,
    risk_level          TEXT,           -- low | medium | high
    created_at          TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id             TEXT PRIMARY KEY,
    topic                 TEXT,
    lesson                TEXT,
    evidence_experiment_ids TEXT,       -- comma-separated experiment_ids
    confidence            TEXT           -- low | medium | high
);
