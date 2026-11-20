"""PostgreSQL + pgvector storage for the Policy Assistant."""
import psycopg
from pgvector.psycopg import register_vector

from . import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            text PRIMARY KEY,
    title         text NOT NULL,
    category      text NOT NULL,
    version       text NOT NULL,
    status        text NOT NULL,          -- current | superseded
    audience      text NOT NULL,          -- all | managers
    content_hash  text NOT NULL,
    embed_model   text NOT NULL,
    indexed_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_no        int  NOT NULL,
    heading         text NOT NULL,
    content         text NOT NULL,
    status          text NOT NULL,        -- copied from documents for fast filtering
    allowed_groups  text[] NOT NULL,      -- who may see this chunk
    embedding       vector(768) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc_id, chunk_no)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""


def connect():
    conn = psycopg.connect(config.PG_DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def reset(conn):
    conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    conn.execute(SCHEMA)


def groups_for(audience):
    return ["managers"] if audience == "managers" else ["employees", "managers"]
