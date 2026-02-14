-- --- PULIZIA TOTALE ---
DROP TABLE IF EXISTS ingestion_images CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS ingestion_logs CASCADE;

-- 1. ABILITAZIONE ESTENSIONE
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 2. TABELLA LOGS (Standard Postgres Table)
-- NON la convertiamo in Hypertable. Funziona da "Anagrafica" per i caricamenti.
CREATE TABLE ingestion_logs (
    log_id              BIGSERIAL PRIMARY KEY,  -- PK semplice
    source_name         TEXT NOT NULL,
    source_type         TEXT,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status              TEXT,
    total_chunks        INTEGER DEFAULT 0,
    processing_time_ms  INTEGER,
    error_message       TEXT
);
-- Nota: NESSUN 'create_hypertable' qui.

-- 3. TABELLA CHUNKS (Hypertable)
-- Questa diventerà enorme, quindi usiamo TimescaleDB
CREATE TABLE document_chunks (
    chunk_id            BIGSERIAL,
    log_id              BIGINT NOT NULL,        -- FK verso tabella standard
    chunk_index         INTEGER NOT NULL,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    toon_type           TEXT,
    content_raw         TEXT,
    content_semantic    TEXT,
    metadata_json       JSONB,
    
    PRIMARY KEY (chunk_id, ingestion_ts),
    
    -- La FK punta alla tabella Logs (che ora è standard, quindi è permesso)
    CONSTRAINT fk_logs 
        FOREIGN KEY (log_id) 
        REFERENCES ingestion_logs(log_id)
        ON DELETE CASCADE -- Se cancello il log, cancello i chunk
);

-- Convertiamo SOLO questa in Hypertable
SELECT create_hypertable('document_chunks', 'ingestion_ts', migrate_data => true);


-- 4. TABELLA IMMAGINI (Hypertable)
-- Anche questa cresce molto (BLOBs)
CREATE TABLE ingestion_images (
    image_id            BIGSERIAL,
    log_id              BIGINT NOT NULL,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    image_data          BYTEA,
    image_hash          CHAR(64),
    mime_type           TEXT DEFAULT 'image/png',
    description_ai      TEXT,
    
    PRIMARY KEY (image_id, ingestion_ts),
    
    CONSTRAINT fk_logs_img 
        FOREIGN KEY (log_id) 
        REFERENCES ingestion_logs(log_id)
        ON DELETE CASCADE
);

-- Convertiamo anche questa
SELECT create_hypertable('ingestion_images', 'ingestion_ts', migrate_data => true);

-- Indici extra
CREATE INDEX idx_chunks_log ON document_chunks(log_id, chunk_index);
CREATE INDEX idx_images_hash ON ingestion_images(image_hash);


-- -------------------------------------------------------------------

/* ==========================================================
   ADD-ON DDL (NON MODIFICA lo schema attuale)
   Aggiunge:
   - ingestion_runs: tracciamento run batch
   - ingestion_documents: anagrafica documento + versioning
   - ingestion_run_logs: mappa run -> log_id (esistente)
   - chunk_vector_index: mapping chunk -> Qdrant (o altro vector store)
   - doc_graph_index: mapping doc/version -> Neo4j
   - ingestion_consistency_checks: controlli di consistenza cross-store
   ========================================================== */

-- 0) Estensione utile (opzionale) per UUID casuali
-- Non altera nulla, aggiunge solo capacità.
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- 1) RUNS (batch ingestion)
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id              BIGSERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'RUNNING',  -- RUNNING/SUCCESS/FAILED
    host                TEXT,
    git_commit          TEXT,
    params_json         JSONB,      -- parametri run (chunk size, overlap, modelli, ecc.)
    notes               TEXT,

    -- contatori aggregati (facoltativi ma comodi)
    total_docs          INTEGER DEFAULT 0,
    total_chunks        INTEGER DEFAULT 0,
    total_images        INTEGER DEFAULT 0,
    total_vectors       INTEGER DEFAULT 0,
    total_graph_updates INTEGER DEFAULT 0,
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status ON ingestion_runs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at ON ingestion_runs(started_at);


-- 2) DOCUMENTS + VERSIONING
-- Un "documento" logico ha un doc_uid stabile; ogni nuova versione cambia hash/version.
-- log_id collega la versione alla riga di ingestion_logs già esistente.
CREATE TABLE IF NOT EXISTS ingestion_documents (
    doc_uid         UUID        NOT NULL,   -- stabile per sorgente (es. UUIDv5 su source_uri), oppure generato
    doc_version     INTEGER     NOT NULL DEFAULT 1,
    log_id          BIGINT      UNIQUE,      -- 1 versione <-> 1 ingestion_logs row (se vuoi). Può essere NULL.
    
    source_uri      TEXT        NOT NULL,    -- path/URL/email-id, ecc.
    source_name     TEXT,                   -- utile per ricerca (può replicare ingestion_logs.source_name)
    doc_type        TEXT,                   -- pdf/docx/html/email...
    title           TEXT,
    language        TEXT,

    doc_hash        CHAR(64)    NOT NULL,    -- SHA-256 del contenuto (o testo normalizzato)
    size_bytes      BIGINT,
    page_count      INTEGER,

    status          TEXT        NOT NULL DEFAULT 'PENDING', -- PENDING/RUNNING/SUCCESS/FAILED
    error_message   TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- snapshot dei modelli usati (per governance)
    models_json     JSONB,      -- {embedding_model, vision_model, llm_model, ...}
    metadata_json   JSONB,

    PRIMARY KEY (doc_uid, doc_version),

    CONSTRAINT fk_ingestion_documents_log
        FOREIGN KEY (log_id)
        REFERENCES ingestion_logs(log_id)
        ON DELETE SET NULL
);

-- Indici utili per idempotenza / lookup rapido
CREATE INDEX IF NOT EXISTS idx_ingestion_documents_source_uri ON ingestion_documents(source_uri);
CREATE INDEX IF NOT EXISTS idx_ingestion_documents_hash ON ingestion_documents(doc_hash);
CREATE INDEX IF NOT EXISTS idx_ingestion_documents_status ON ingestion_documents(status);

-- Unicità consigliata per prevenire doppioni identici (stessa sorgente + stesso contenuto)
CREATE UNIQUE INDEX IF NOT EXISTS ux_ingestion_documents_uri_hash
ON ingestion_documents(source_uri, doc_hash);


-- 3) RUN <-> LOGS mapping (se una run processa più sorgenti/log_id)
CREATE TABLE IF NOT EXISTS ingestion_run_logs (
    run_id      BIGINT NOT NULL,
    log_id      BIGINT NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (run_id, log_id),

    CONSTRAINT fk_run_logs_run
        FOREIGN KEY (run_id)
        REFERENCES ingestion_runs(run_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_run_logs_log
        FOREIGN KEY (log_id)
        REFERENCES ingestion_logs(log_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ingestion_run_logs_log ON ingestion_run_logs(log_id);


-- 4) Mapping CHUNK -> Vector Store (Qdrant / pgvector / altro)
-- FK sulla PK composta del chunk: (chunk_id, ingestion_ts)
CREATE TABLE IF NOT EXISTS chunk_vector_index (
    chunk_id            BIGINT      NOT NULL,
    ingestion_ts        TIMESTAMPTZ  NOT NULL,

    vector_store        TEXT        NOT NULL DEFAULT 'qdrant', -- qdrant/pgvector/...
    collection_name     TEXT        NOT NULL,
    point_id            TEXT        NOT NULL,                 -- id nel vector store

    embedding_model     TEXT,
    embedding_dim       INTEGER,
    distance_metric     TEXT,                                 -- cosine/dot/euclid...

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload_json        JSONB,                                -- copia payload utile (doc_uid, page_no, etc.)

    PRIMARY KEY (vector_store, collection_name, point_id),

    CONSTRAINT fk_chunk_vector_chunk
        FOREIGN KEY (chunk_id, ingestion_ts)
        REFERENCES document_chunks(chunk_id, ingestion_ts)
        ON DELETE CASCADE
);

-- Lookup tipici
CREATE INDEX IF NOT EXISTS idx_chunk_vector_by_chunk
ON chunk_vector_index(chunk_id, ingestion_ts);

CREATE INDEX IF NOT EXISTS idx_chunk_vector_collection
ON chunk_vector_index(vector_store, collection_name);


-- 5) Mapping DOC/VERSION -> Graph Store (Neo4j)
CREATE TABLE IF NOT EXISTS doc_graph_index (
    doc_uid             UUID        NOT NULL,
    doc_version         INTEGER     NOT NULL,

    graph_store         TEXT        NOT NULL DEFAULT 'neo4j',
    graph_db            TEXT,
    graph_job_id        TEXT,     -- id job/transaction batch lato Neo4j (se lo gestisci)
    
    nodes_upserted      INTEGER DEFAULT 0,
    rels_upserted       INTEGER DEFAULT 0,

    status              TEXT NOT NULL DEFAULT 'PENDING', -- PENDING/SUCCESS/FAILED
    error_message       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details_json        JSONB,

    PRIMARY KEY (doc_uid, doc_version, graph_store),

    CONSTRAINT fk_doc_graph_doc
        FOREIGN KEY (doc_uid, doc_version)
        REFERENCES ingestion_documents(doc_uid, doc_version)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_doc_graph_status ON doc_graph_index(status);


-- 6) Consistency checks cross-store (Postgres chunks/images vs Qdrant vs Neo4j)
CREATE TABLE IF NOT EXISTS ingestion_consistency_checks (
    check_id            BIGSERIAL PRIMARY KEY,
    log_id              BIGINT,
    doc_uid             UUID,
    doc_version         INTEGER,

    checked_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- contatori osservati
    pg_chunks           INTEGER,
    pg_images           INTEGER,
    vector_points       INTEGER,
    graph_nodes         INTEGER,
    graph_rels          INTEGER,

    -- esito
    status              TEXT NOT NULL DEFAULT 'OK',  -- OK/WARN/FAIL
    message             TEXT,
    details_json        JSONB,

    CONSTRAINT fk_checks_log
        FOREIGN KEY (log_id)
        REFERENCES ingestion_logs(log_id)
        ON DELETE SET NULL,

    CONSTRAINT fk_checks_doc
        FOREIGN KEY (doc_uid, doc_version)
        REFERENCES ingestion_documents(doc_uid, doc_version)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_consistency_checks_log ON ingestion_consistency_checks(log_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_consistency_checks_doc ON ingestion_consistency_checks(doc_uid, doc_version, checked_at DESC);


-- 7) VIEW opzionale: stato “KB-grade” per documento
-- (non altera nulla, comoda per monitoraggio)
CREATE OR REPLACE VIEW v_kb_document_status AS
SELECT
    d.doc_uid,
    d.doc_version,
    d.source_uri,
    d.doc_type,
    d.title,
    d.doc_hash,
    d.status AS doc_status,
    d.created_at,
    d.updated_at,
    l.log_id,
    l.ingestion_ts,
    l.status AS log_status,
    l.total_chunks,
    l.processing_time_ms,
    l.error_message AS log_error
FROM ingestion_documents d
LEFT JOIN ingestion_logs l
    ON d.log_id = l.log_id;
