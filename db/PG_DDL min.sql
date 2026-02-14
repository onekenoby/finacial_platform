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


-- ---------------------