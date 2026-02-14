import reflex as rx
import torch
import uuid
import os
import time
import re
import json
import hashlib
import psycopg2


from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from openai import OpenAI


def looks_garbled(text: str) -> bool:
    """
    True if text contains typical garbage chars from PDF text layer extraction.
    We should avoid feeding these chunks to the LLM, especially for formulas.
    """
    if not text:
        return False
    bad = ["□", "\ufffd"]  # square box, replacement char
    return any(b in text for b in bad)



# =========================
# ⚙️ CONFIGURAZIONE UTENTE
# =========================
PAGE_TITLE = "Financial AI Analyst 📊"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "financial_docs")

# =========================
# RAG TIER POLICY
# =========================
RAG_DEFAULT_TIERS = os.getenv("RAG_DEFAULT_TIERS", "A,B")  # default prudente
RAG_NEWS_KEYWORDS = os.getenv(
    "RAG_NEWS_KEYWORDS",
    "news,oggi,ieri,ultima,ultime,rumor,breaking,aggiornamenti,recente,recent"
)

# =========================
# 🐘 POSTGRES (Timescale) - RAG ENRICH
# =========================
PG_ENRICH_ENABLED = os.getenv("PG_ENRICH_ENABLED", "1") == "1"
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "ai_ingestion")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin_password")
PG_MIN_CONN = int(os.getenv("PG_MIN_CONN", "1"))
PG_MAX_CONN = int(os.getenv("PG_MAX_CONN", "5"))

# preferisci content_raw (1) o content_semantic (0) quando disponibile
PG_PREFER_RAW = os.getenv("PG_PREFER_RAW", "0") == "1"

pg_pool: Optional[SimpleConnectionPool] = None




# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password_sicura"))

# AI Models
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-3-12b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", LLM_MODEL_NAME)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LM Studio / OpenAI Compatible API
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "6"))  # number of turns (user+assistant)

# Retrieval knobs (RAG v2)
QDRANT_CANDIDATES = int(os.getenv("QDRANT_CANDIDATES", "80"))     # retrieve top-N from qdrant
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))     # rerank at most N (cost control)
FINAL_SOURCES = int(os.getenv("FINAL_SOURCES", "6"))             # final context sources
MAX_PER_PAGE = int(os.getenv("MAX_PER_PAGE", "1"))                # diversify results: max K per page
MAX_PER_DOC = int(os.getenv("MAX_PER_DOC", "3"))                  # diversify results: max K per document

# =========================
# 🎚️ Tier-aware ranking
# =========================
TIER_BOOST_A = float(os.getenv("TIER_BOOST_A", "0.08"))
TIER_BOOST_B = float(os.getenv("TIER_BOOST_B", "0.04"))
TIER_PENALTY_C = float(os.getenv("TIER_PENALTY_C", "0.06"))

# Se la query è news/rumor/recency, NON penalizzare Tier C
TIER_C_PENALTY_IF_NOT_NEWS = os.getenv("TIER_C_PENALTY_IF_NOT_NEWS", "1") == "1"


# Graph expansion knobs
GRAPH_EXPAND_ENABLED = os.getenv("GRAPH_EXPAND_ENABLED", "1") == "1"
GRAPH_MAX_FORMULAS = int(os.getenv("GRAPH_MAX_FORMULAS", "6"))
GRAPH_MAX_NEIGHBOR_CHUNKS = int(os.getenv("GRAPH_MAX_NEIGHBOR_CHUNKS", "4"))

# Prompt limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "22000"))  # prevent prompt blow-ups
MAX_ASSISTANT_CHARS = int(os.getenv("MAX_ASSISTANT_CHARS", "12000"))

AUDIT_ENABLED = True
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "./rag_audit.jsonl")

# =========================
# 🧠 CARICAMENTO RISORSE
# =========================
print("⏳ Init Backend...")

#device_embed = "cuda" if torch.cuda.is_available() else "cpu"
# ...ogica che provi la CPU ma resti flessibile:

device_embed = "cpu" if torch.cuda.is_available() else "cpu" # (In pratica lo forzi sempre)

device_rerank = "cpu"  # IMPORTANT: avoid fighting with Gemma/Vision on the same P5000

embedder = None
reranker = None
llm_client = None
qdrant_client_inst = None
neo4j_driver = None

try:
    print(f"🚀 Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on {device_embed.upper()}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_embed)

    print(f"🚀 Loading Reranker ({RERANKER_MODEL_NAME}) on {device_rerank.upper()}...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device_rerank)

    print(f"🚀 Connecting to LLM ({LLM_MODEL_NAME})...")
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

    qdrant_client_inst = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    neo4j_driver.verify_connectivity()

    if PG_ENRICH_ENABLED:
        pg_pool = SimpleConnectionPool(
            PG_MIN_CONN, PG_MAX_CONN,
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS
        )
        # smoke test
        conn = pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        finally:
            pg_pool.putconn(conn)




    print("✅ Risorse caricate e DB connessi.")
except Exception as e:
    print(f"❌ Errore critico avvio risorse: {e}")
    print("⚠️ L'app partirà ma il retrieval potrebbe fallire.")


# =========================
# 📦 DATA MODELS
# =========================
class GraphEntity(BaseModel):
    name: str
    type: str
    relation: str = "MENTIONED"


class SourceItem(BaseModel):
    id: str
    content: str
    filename: str
    page: int = 0
    type: str = "text"
    score: float = 0.0
    graph_context: List[GraphEntity] = []
    # extra provenance / metadata
    section_hint: str = ""
    image_id: Optional[int] = None
    #NEW
    tier: str = ""
    # ✅ PG canonical provenance
    pg_ingestion_ts: str = ""
    pg_source_name: str = ""
    pg_source_type: str = ""
    pg_log_id: int = 0
    pg_chunk_id: int = 0
    pg_toon_type: str = ""

class RetrievalDebug(BaseModel):
    query: str = ""
    intent: str = "text"

    # Tier logic
    wants_news: bool = False
    default_tiers: List[str] = []

    # Qdrant stats
    qdrant_candidates: int = 0
    kept_after_quality_filters: int = 0
    rerank_candidates: int = 0
    final_sources: int = 0

    # Tier distribution in final set
    tier_counts: Dict[str, int] = {}

    # Scoring (quick summary)
    score_min: float = 0.0
    score_max: float = 0.0
    score_avg: float = 0.0

    # Flags
    reranker_used: bool = False
    graph_expand_used: bool = False

class AuditTrail(BaseModel):
    ts_utc: str = ""
    query: str = ""
    intent: str = ""

    # What we sent to the LLM (hash only, to avoid storing full sensitive context)
    prompt_sha256: str = ""
    context_chars: int = 0

    # Retrieval explainability
    retrieval: RetrievalDebug = RetrievalDebug()

    # Model config snapshot
    llm_model: str = ""
    temperature: float = 0.1
    memory_limit: int = 0

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceItem] = Field(default_factory=list)
    debug_md: str = "" # ✅ NEW: explainability/audit (renderizzato in UI)

# =========================
# 🧰 UTILS
# =========================
def build_alternating_history(messages: List[ChatMessage], max_turns: int) -> List[Dict[str, str]]:
    """Strict alternating user/assistant for LM Studio templates."""
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        content = (m.content or "").strip()
        if not content:
            continue
        if cleaned and cleaned[-1]["role"] == m.role:
            cleaned[-1]["content"] = content
        else:
            cleaned.append({"role": m.role, "content": content})

    limit = max_turns * 2
    cleaned = cleaned[-limit:]
    if cleaned and cleaned[0]["role"] == "assistant":
        cleaned = cleaned[1:]

    alt: List[Dict[str, str]] = []
    for item in cleaned:
        if alt and alt[-1]["role"] == item["role"]:
            alt[-1] = item
        else:
            alt.append(item)

    return alt


def gpu_free_info() -> str:
    """Return free/total VRAM. Works only if CUDA available."""
    if not torch.cuda.is_available():
        return "CPU Mode"
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        name = torch.cuda.get_device_name(0)
        return f"{name} | Free {free_gb:.1f} GB / Total {total_gb:.1f} GB"
    except Exception:
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 1024**3:.1f} GB)"


def detect_intent(query: str) -> str:
    """Very cheap intent router: formula / chart / text."""
    q = (query or "").lower()
    # formula intent
    if any(k in q for k in ["formula","matrix","matrice", "equazione", "equation","derivate", "derivata","integration","integrale","latex", "lift", "support", "confidence", "probabilità", "probability","limit","limite"]):
        return "formula"
    # chart intent
    if any(k in q for k in ["grafico","graph","flow","flowchart","diagramma","diagram","prospect","prospetto", "chart", "figura", "tabella", "table", "asse", "legend", "legenda", "trend","slop", "candela", "candle","ohlc", "volumi", "volume","heatmap"]):
        return "chart"
    return "text"

import re

def is_user_data_analytics(query: str) -> bool:
    """
    Heuristica generale per capire se l'utente ha fornito dati
    e chiede un'analisi (non documentale).
    """
    q = query.lower()

    # pattern forti: liste, array, numeri
    if re.search(r"\[[0-9,\s./-]+\]", q):
        return True

    # molte cifre → probabile dataset
    if len(re.findall(r"\d+", q)) > 10:
        return True

    # parole chiave analitiche (generalissime)
    keywords = [
        "calcola","calculate","stima","estimation","estimate", "analizza","analyze","analyse", "regressione", "regression", "correlazione","correlation",
        "varianza","variance","standard deviation","deviazione standard", "media","mean","average","ave","trend","slop","stagional","sesonal", "decompos",
        "forecast","prediction", "time series", "serie storic", "model"
    ]
    return any(k in q for k in keywords)


def safe_payload_text(payload: Dict[str, Any]) -> str:
    """
    IMPORTANT: align to ingestion payload:
    - most recent ingestion uses 'text_sem'
    - keep fallbacks for older payloads
    """
    return (
        (payload.get("text_sem") or "")
        or (payload.get("content_semantic") or "")
        or (payload.get("content_raw") or "")
        or (payload.get("content") or "")
        or (payload.get("text") or "")
        or ""
    ).strip()


def get_payload_page(payload: Dict[str, Any]) -> int:
    try:
        return int(payload.get("page") or payload.get("page_no") or 0)
    except Exception:
        return 0


def get_payload_type(payload: Dict[str, Any]) -> str:
    return str(payload.get("toon_type") or payload.get("type") or "text")


def get_payload_section(payload: Dict[str, Any]) -> str:
    return str(payload.get("section_hint") or "")


def get_payload_image_id(payload: Dict[str, Any]) -> Optional[int]:
    try:
        v = payload.get("image_id")
        return int(v) if v is not None else None
    except Exception:
        return None

def get_payload_tier(payload: dict) -> str:
    try:
        t = payload.get("tier")
        if not t:
            return ""
        return str(t)
    except Exception:
        return ""

def is_news_query(query: str) -> bool:
    q = (query or "").lower()
    # keyword “recency/news”
    return any(k in q for k in [
        "news", "notizie", "ultime", "oggi", "ieri", "questa settimana",
        "rumor", "gossip", "leak", "unconfirmed", "sources say",
        "breaking", "headline", "annuncio", "earnings", "guidance",
        "fed", "ecb", "inflazione", "tassi", "cpi", "nfp"
    ])

def has_sufficient_ab_sources(sources: List[SourceItem]) -> bool:
    tiers = [(getattr(s, "tier", "") or "").upper() for s in sources]
    for t in tiers:
        if t in ("A", "TIER_A_METHODOLOGY") or t.endswith("_A_METHODOLOGY"):
            return True
        if t in ("B", "TIER_B_REFERENCE") or t.endswith("_B_REFERENCE"):
            return True
    return False


def tier_score_delta(tier: str, query_text: str) -> float:
    t = (tier or "").strip().upper()
    if t.endswith("_A_METHODOLOGY") or t == "TIER_A_METHODOLOGY" or t == "A":
        return TIER_BOOST_A
    if t.endswith("_B_REFERENCE") or t == "TIER_B_REFERENCE" or t == "B":
        return TIER_BOOST_B
    if t.endswith("_C_NEWS") or t == "TIER_C_NEWS" or t == "C":
        if TIER_C_PENALTY_IF_NOT_NEWS and (not is_news_query(query_text)):
            return -TIER_PENALTY_C
        return 0.0
    return 0.0

def diversify(items: List[Dict[str, Any]], max_per_page: int, max_per_doc: int, final_k: int) -> List[Dict[str, Any]]:
    """Keep best-scoring items but limit duplicates by page and document."""
    out = []
    page_count: Dict[Tuple[str, int], int] = {}
    doc_count: Dict[str, int] = {}

    for it in sorted(items, key=lambda x: float(x.get("final_score", x.get("score", 0.0))), reverse=True):
        fname = it.get("filename", "Unknown")
        page = int(it.get("page", 0))
        page_key = (fname, page)

        if doc_count.get(fname, 0) >= max_per_doc:
            continue
        if page_count.get(page_key, 0) >= max_per_page:
            continue

        out.append(it)
        doc_count[fname] = doc_count.get(fname, 0) + 1
        page_count[page_key] = page_count.get(page_key, 0) + 1

        if len(out) >= final_k:
            break
    return out

def append_audit_log(audit: AuditTrail):
    if not AUDIT_ENABLED:
        return
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(audit.model_dump_json() + "\n")
    except Exception as e:
        print(f"⚠️ Audit log write error: {e}")


# =========================
# 🔍 Neo4j Graph Expansion
# =========================
def get_graph_entities(chunk_ids: List[str]) -> Dict[str, List[GraphEntity]]:
    if not chunk_ids or not neo4j_driver:
        return {}
    graph_map: Dict[str, List[GraphEntity]] = {}
    query = """
    MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
    WHERE c.id IN $ids
    RETURN c.id as chunk_id,
           coalesce(e.label, e.id) as name,
           labels(e)[0] as type,
           type(r) as rel
    LIMIT 300
    """
    try:
        with neo4j_driver.session() as session:
            result = session.run(query, ids=chunk_ids)
            for record in result:
                cid = record["chunk_id"]
                entity = GraphEntity(
                    name=record["name"],
                    type=record["type"],
                    relation=record["rel"],
                )
                graph_map.setdefault(cid, []).append(entity)
    except Exception as e:
        print(f"⚠️ Neo4j Query Error (entities): {e}")
    return graph_map


def get_formulas_for_chunks(chunk_ids: List[str], limit: int = GRAPH_MAX_FORMULAS) -> List[str]:
    """Return short strings describing formulas linked to the retrieved chunks."""
    if not chunk_ids or not neo4j_driver:
        return []
    query = """
    MATCH (f:Formula)-[:MENTIONED_IN]->(c:Chunk)
    WHERE c.id IN $ids
    RETURN f.latex AS latex, f.plain AS plain, f.meaning_it AS meaning
    LIMIT $lim
    """
    out = []
    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit)
            for r in res:
                latex = (r["latex"] or "").strip()
                plain = (r["plain"] or "").strip()
                meaning = (r["meaning"] or "").strip()
                s = []
                if latex:
                    s.append(f"LaTeX: {latex}")
                if plain:
                    s.append(f"Plain: {plain}")
                if meaning:
                    s.append(f"Meaning: {meaning}")
                if s:
                    out.append(" | ".join(s))
    except Exception as e:
        print(f"⚠️ Neo4j Query Error (formulas): {e}")
    return out


def get_neighbor_chunk_ids(chunk_ids: List[str], limit: int = GRAPH_MAX_NEIGHBOR_CHUNKS) -> List[str]:
    if not chunk_ids or not neo4j_driver:
        return []
    # Nuova query: trova chunk che condividono le STESSE entità dei chunk trovati da Qdrant
    query = """
    MATCH (c1:Chunk)<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(c2:Chunk)
    WHERE c1.id IN $ids AND NOT c2.id IN $ids
    RETURN c2.id AS cid, count(e) AS common_entities
    ORDER BY common_entities DESC
    LIMIT $lim
    """
    out = []
    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit)
            out = [r["cid"] for r in res if r.get("cid")]
    except Exception as e:
        print(f"⚠️ Neo4j Semantic Neighbors Error: {e}")
    return out


def fetch_chunks_from_qdrant_by_ids(ids: List[str]) -> List[SourceItem]:
    """Fetch Qdrant points by IDs (for graph expansion neighbors)."""
    if not ids or not qdrant_client_inst:
        return []
    out: List[SourceItem] = []
    try:
        # qdrant retrieve works with ids list
        points = qdrant_client_inst.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
        )
        for p in points:
            payload = p.payload or {}
            tier = get_payload_tier(payload)
            content = safe_payload_text(payload)
            if not content:
                continue
            out.append(
                SourceItem(
                    id=str(p.id),
                    content=content,
                    filename=str(payload.get("filename", "Unknown")),
                    page=get_payload_page(payload),
                    type=get_payload_type(payload),
                    score=0.0,
                    graph_context=[],
                    section_hint=get_payload_section(payload),
                    image_id=get_payload_image_id(payload),
                    tier=tier,  # ✅ NEW
                )
            )
    except Exception as e:
        print(f"⚠️ Qdrant retrieve error: {e}")
    return out

def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def wants_news_tier(query_text: str) -> bool:
    q = (query_text or "").lower()
    kws = _parse_csv(RAG_NEWS_KEYWORDS)
    return any(k in q for k in kws)


def tier_qdrant_filter(query_text: str):
    """
    Default: include solo A+B (o quanto definito in RAG_DEFAULT_TIERS).
    Se l'utente chiede esplicitamente news/recente, include anche C (nessun filtro).
    """
    if wants_news_tier(query_text):
        return None  # include A/B/C

    tiers = _parse_csv(RAG_DEFAULT_TIERS)
    if not tiers:
        tiers = ["A", "B"]

    # Qdrant match any (stringhe)
    return {
        "must": [
            {"key": "tier", "match": {"any": tiers}}
        ]
    }

def build_retrieval_audit_md(
    query_text: str,
    intent: str,
    timings: Dict[str, float],
    counts: Dict[str, Any],
    top_sources_preview: List[Dict[str, Any]],
) -> str:
    """Audit compatto in markdown (non entra nel prompt LLM, solo UI)."""
    def ms(x: float) -> str:
        return f"{x*1000:.0f} ms"

    lines = []
    lines.append("### 🔎 Audit Retrieval (Explainability)")
    lines.append(f"- **Intent**: `{intent}`")
    lines.append(f"- **Query**: `{(query_text or '')[:180]}`")

    # timings
    if timings:
        lines.append("\n#### ⏱️ Tempi")
        for k in ["embed", "qdrant_search", "rerank", "diversify", "graph", "total"]:
            if k in timings:
                lines.append(f"- `{k}`: **{ms(timings[k])}**")

    # counts
    if counts:
        lines.append("\n#### 📦 Conteggi")
        if "hits" in counts:
            lines.append(f"- Qdrant hits: **{counts['hits']}**")
        if "candidates" in counts:
            lines.append(f"- Candidates validi: **{counts['candidates']}**")
        if "rerank_used" in counts:
            lines.append(f"- Rerank usati: **{counts['rerank_used']}**")
        if "final_primary" in counts:
            lines.append(f"- Final primary: **{counts['final_primary']}**")
        if "final_neighbors" in counts:
            lines.append(f"- Graph neighbors: **{counts['final_neighbors']}**")
        if "final_formulas" in counts:
            lines.append(f"- Graph formulas: **{counts['final_formulas']}**")

        # tier split (se presente)
        tier_split = counts.get("tier_split", {})
        if isinstance(tier_split, dict) and tier_split:
            lines.append("- Tier split:")
            for t, n in tier_split.items():
                lines.append(f"  - `{t}`: **{n}**")

    # top preview
    if top_sources_preview:
        lines.append("\n#### 🧾 Top selezionati (preview)")
        for i, s in enumerate(top_sources_preview, start=1):
            tier = s.get("tier", "")
            lines.append(
                f"- [{i}] `{s.get('filename','?')}` pag {s.get('page',0)} | "
                f"type `{s.get('type','?')}` | tier `{tier}` | score **{s.get('score',0.0):.3f}**"
            )

    return "\n".join(lines).strip()


def fetch_pg_chunks_by_uuid(chunk_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Ritorna: {chunk_uuid: {"content_raw":..., "content_semantic":..., "metadata_json":..., "ingestion_ts":...}}
    Prende SEMPRE la riga più recente per chunk_uuid.
    """
    if not PG_ENRICH_ENABLED or not pg_pool or not chunk_uuids:
        return {}

    # dedup preservando ordine
    seen = set()
    uuids = []
    for u in chunk_uuids:
        if u and u not in seen:
            uuids.append(u)
            seen.add(u)

    sql = """
    WITH wanted(chunk_uuid) AS (VALUES %s),
    ranked AS (
      SELECT
        d.chunk_uuid::text AS chunk_uuid,
        d.content_raw,
        d.content_semantic,
        d.metadata_json,
        d.ingestion_ts,
        ROW_NUMBER() OVER (PARTITION BY d.chunk_uuid ORDER BY d.ingestion_ts DESC) AS rn
      FROM public.document_chunks d
      JOIN wanted w ON d.chunk_uuid = w.chunk_uuid::uuid
    )
    SELECT chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts
    FROM ranked
    WHERE rn = 1;
    """

    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, [(u,) for u in uuids])
            rows = cur.fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for (chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts) in rows:
            out[str(chunk_uuid)] = {
                "content_raw": content_raw,
                "content_semantic": content_semantic,
                "metadata_json": metadata_json,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            }
        return out
    except Exception as e:
        print(f"⚠️ PG enrich error: {e}")
        return {}
    finally:
        pg_pool.putconn(conn)



# =========================
# 🔍 RAG v2 Retrieval
# =========================
def retrieve_v2(query_text: str) -> Tuple[List[SourceItem], str]:
    if not embedder or not qdrant_client_inst:
        return [SourceItem(id="error", content="Backend non disponibile", filename="System")], "Backend non disponibile"

    t_total0 = time.time()
    timings: Dict[str, float] = {}
    counts: Dict[str, Any] = {}

    intent = detect_intent(query_text)

    # 1) Dense retrieval candidates
    t0 = time.time()
    query_vector = embedder.encode(query_text, normalize_embeddings=True).tolist()
    timings["embed"] = time.time() - t0

    t0 = time.time()
    try:
        hits = qdrant_client_inst.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=QDRANT_CANDIDATES,
            with_payload=True,
            query_filter=tier_qdrant_filter(query_text),  # ✅ AGGIUNGI QUESTO
        )
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
        return [], f"Qdrant error: {e}"
    timings["qdrant_search"] = time.time() - t0

    if not hits:
        timings["total"] = time.time() - t_total0
        debug_md = build_retrieval_audit_md(
            query_text=query_text,
            intent=intent,
            timings=timings,
            counts={"hits": 0, "candidates": 0, "rerank_used": 0},
            top_sources_preview=[],
        )
        return [], debug_md

    counts["hits"] = len(hits)

    # 2) Build candidate list + intent-aware boost + tier capture
    candidates = []
    cross_inputs = []

    tier_split: Dict[str, int] = {}

    for hit in hits:
        payload = hit.payload or {}
        content = safe_payload_text(payload)
        if not content:
            continue

        f = str(payload.get("filename", "Unknown"))
        page = get_payload_page(payload)
        ttype = get_payload_type(payload)
        section_hint = get_payload_section(payload)
        image_id = get_payload_image_id(payload)

        # ✅ tier dal payload (se manca fallback vuoto)
        tier = get_payload_tier(payload)

        base_score = float(getattr(hit, "score", 0.0) or 0.0)

        # Penalize garbled PDF text chunks; for formula intent we can even skip them
        if looks_garbled(content):
            if intent == "formula" and ttype not in ("formula_page", "page_vision_math", "graph_formulas"):
                continue
            if ttype in ("text", "paragraph", "page_text"):
                base_score -= 0.25

        boost = 0.0
        if intent == "formula" and ttype in ("formula_page", "page_vision_math"):
            boost += 0.18
        if intent == "formula" and "latex" in content.lower():
            boost += 0.06
        if intent == "chart" and ttype in ("image_vision", "image_desc"):
            boost += 0.14
        if intent == "chart" and any(k in content.lower() for k in ["axes:", "legend:", "numbers:", "series:"]):
            boost += 0.06

        # ✅ Tier-aware ranking (A > B > C) se presente helper
        if "tier_score_delta" in globals():
            boost += float(tier_score_delta(tier, query_text))

        final_score = base_score + boost

        candidates.append(
            {
                "id": str(hit.id),
                "content": content,
                "filename": f,
                "page": page,
                "type": ttype,
                "score": base_score,
                "final_score": final_score,
                "section_hint": section_hint,
                "image_id": image_id,
                "tier": tier,
            }
        )

        tier_split[tier] = tier_split.get(tier, 0) + 1

        # prepare cross-encoder inputs
        cross_inputs.append((query_text, content))

    counts["candidates"] = len(candidates)
    counts["tier_split"] = tier_split

    if not candidates:
        timings["total"] = time.time() - t_total0
        debug_md = build_retrieval_audit_md(
            query_text=query_text,
            intent=intent,
            timings=timings,
            counts=counts,
            top_sources_preview=[],
        )
        return [], debug_md

    # 3) Cross-encoder rerank (cost-controlled)
    t0 = time.time()
    rerank_used = min(len(candidates), RERANK_CANDIDATES)
    counts["rerank_used"] = rerank_used

    if reranker is not None:
        try:
            scores = reranker.predict(cross_inputs[:rerank_used])
            for i in range(rerank_used):
                candidates[i]["final_score"] += 0.15 * float(scores[i])
        except Exception as e:
            print(f"⚠️ Cross-encoder error: {e}")

    timings["rerank"] = time.time() - t0

    # 4) Sort by final_score desc
    candidates.sort(key=lambda x: float(x.get("final_score", x.get("score", 0.0))), reverse=True)

    # 5) Diversify (page/doc caps)
    t0 = time.time()
    if "diversify_candidates" in globals():
        top = diversify(
                    candidates,
                    max_per_page=MAX_PER_PAGE,
                    max_per_doc=MAX_PER_DOC,
                    final_k=FINAL_SOURCES,
        )
    else:
        # fallback: simple top-k
        top = candidates[:FINAL_SOURCES]
    timings["diversify"] = time.time() - t0

    # ensure at least one formula chunk is present when available
    if intent == "formula":
        formula_candidates = [c for c in candidates if c["type"] in ("formula_page", "page_vision_math")]
        if formula_candidates:
            if not any(t["type"] in ("formula_page", "page_vision_math") for t in top):
                top = top[:-1] + [formula_candidates[0]]

    # 6) Neo4j enrichment: entities + formulas + neighbor chunks
    t0 = time.time()
    chunk_ids = [t["id"] for t in top]
    entities_map = get_graph_entities(chunk_ids)


    # 6.b) Postgres enrichment (content_raw/content_semantic/metadata)
    pg_map = fetch_pg_chunks_by_uuid(chunk_ids)

    # se vuoi sostituire il contenuto Qdrant con quello PG (più affidabile/ricco)
    for t in top:
        pg_row = pg_map.get(t["id"])
        if not pg_row:
            continue

        # scegli quale testo usare
        pg_text = (pg_row.get("content_raw") if PG_PREFER_RAW else pg_row.get("content_semantic")) or ""
        if pg_text.strip():
            t["content"] = pg_text.strip()


    
    sources: List[SourceItem] = []
    for t in top:
        ents = entities_map.get(t["id"], [])
        uniq = []
        seen = set()
        for e in ents:
            if e.name not in seen:
                uniq.append(e)
                seen.add(e.name)

        pg_row = pg_map.get(t["id"], {})

        sources.append(
            SourceItem(
                id=t["id"],
                content=t["content"],
                filename=t["filename"],
                page=t["page"],
                type=t["type"],
                score=t["score"],
                graph_context=t.get("graph_context", []),
                section_hint=t.get("section_hint", ""),
                image_id=t.get("image_id"),
                tier=t.get("tier", ""),

                # ====== PG canonical provenance ======
                pg_ingestion_ts=pg_row.get("ingestion_ts", ""),
                pg_source_name=pg_row.get("source_name", ""),
                pg_source_type=pg_row.get("source_type", ""),
                pg_log_id=pg_row.get("log_id", 0),
                pg_chunk_id=pg_row.get("chunk_id", 0),
                pg_toon_type=pg_row.get("toon_type", ""),
            )
        )


    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        formulas = get_formulas_for_chunks(chunk_ids, limit=GRAPH_MAX_FORMULAS)

        neighbor_ids = get_neighbor_chunk_ids(chunk_ids, limit=GRAPH_MAX_NEIGHBOR_CHUNKS)
        neighbor_sources = fetch_chunks_from_qdrant_by_ids(neighbor_ids)

        for ns in neighbor_sources:
            ns.type = "graph_neighbor"
            ns.score = 0.0
            sources.append(ns)

        if formulas:
            sources.append(
                SourceItem(
                    id="graph_formulas",
                    content="FORMULE (da Neo4j):\n" + "\n".join(f"- {x}" for x in formulas),
                    filename="Neo4j",
                    page=0,
                    type="graph_formulas",
                    score=0.0,
                    graph_context=[],
                )
            )

    timings["graph"] = time.time() - t0

    # 7) final trim (keep prompt bounded)
    primary = [s for s in sources if s.type not in ("graph_neighbor",)]
    neighbors = [s for s in sources if s.type == "graph_neighbor"]
    formulas_src = [s for s in sources if s.type == "graph_formulas"]

    final_sources = primary[:FINAL_SOURCES] + neighbors[:max(0, 3)] + formulas_src[:1]

    counts["final_primary"] = len(primary[:FINAL_SOURCES])
    counts["final_neighbors"] = len(neighbors[:max(0, 3)])
    counts["final_formulas"] = len(formulas_src[:1])

    timings["total"] = time.time() - t_total0

    # 8) Build audit markdown (TOP preview)
    top_preview = []
    for s in final_sources[: min(len(final_sources), FINAL_SOURCES)]:
        top_preview.append(
            {
                "filename": s.filename,
                "page": s.page,
                "type": s.type,
                "tier": getattr(s, "tier", ""),
                "score": float(s.score or 0.0),
            }
        )

    debug_md = build_retrieval_audit_md(
        query_text=query_text,
        intent=intent,
        timings=timings,
        counts=counts,
        top_sources_preview=top_preview,
    )

    return final_sources, debug_md



def build_context_block(sources: List[SourceItem], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context with strong provenance and caps."""
    parts = []
    total = 0

    # IMPORTANT: do not leak technical IDs into the LLM prompt.
    # We number sources as [1], [2], ... and keep IDs only in the UI pop-up.
    for i, s in enumerate(sources, start=1):
        header = f"--- Fonte [{i}] — {s.filename} — Pag {s.page} — ({s.type}) ---\n"
        if s.section_hint:
            header = f"--- Fonte [{i}] — {s.filename} — Pag {s.page} — ({s.type}) — sezione: {s.section_hint} ---\n"

        body = (s.content or "").strip()
        if not body:
            continue

        block = header + body + "\n\n"
        if total + len(block) > max_chars:
            # cut body
            remaining = max(0, max_chars - total - len(header) - 50)
            if remaining <= 200:
                break
            block = header + body[:remaining] + "\n\n"
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "".join(parts).strip()



def build_system_instructions(intent: str) -> str:
    base = (
        "ROLE: Professional Financial and Quantitative Analyst. Your GOAL is to provide accurate, conservative, and well-reasoned answers.\n"
        
        "LANGUAGE RULE (MANDATORY):\n"
        "1. Identify the language of the USER QUESTION.\n"
        "2. You MUST provide the ENTIRE response (Answer, Evidence, Limits, and Sources) in the SAME LANGUAGE as the USER QUESTION.\n"
        "3. DO NOT switch to English if the question is in Italian. DO NOT switch to Italian if the question is in English.\n"
        "\n"
        
        "OPERATIONAL RULES:\n"
        "1) Use document context only when relevant. If citing, use numeric references [n].\n"
        "2) Never output technical IDs or UUIDs.\n"
        "3) If context is insufficient, state it clearly in the user's language.\n"
        "4) For formulas: use LaTeX $...$.\n"
        "5) If the text contains corrupted or unreadable symbols (e.g. □), ignore them and prefer clean or LaTeX-based content.\n"
        "\n"
        
        "OUTPUT STRUCTURE (MANDATORY):\n"
        "A) Answer (Max 15 lines)\n"
        "B) Evidence (Bullet list with citations [n])\n"
        "C) Limits / Assumptions\n"
        "D) Sources: '[1] filename pag X; [2] ...'\n"
    )
    if intent == "formula":
        base += "\nPRIORITY: Give priority to LaTeX blocks and mathematical reasoning.\n"
    elif intent == "chart":
        base += "\nPRIORITY: the question is related to chart/table/graph. Make axes, legenda, trend and visible number description.\n"
    return base



def tier_guardrail_instructions(query_text: str) -> str:
    news = is_news_query(query_text)
    return (
        "GUARDRAILS TIER-FIRST (FINANCE-GRADE):\n"
        "1) Tier A: Primary source for definitions, theory, and mechanisms.\n"
        "2) Tier B: Examples, use cases, and applications.\n"
        "3) Tier C: Temporal context ONLY (news/rumors). If Tier C is used, ALWAYS specify a date.\n"
        "4) Do not invent: every statement must be supported by sources in the context.\n"
        "5) If sources are insufficient, state it explicitly.\n"
        f"6) {'Query news: Tier C allowed for WHAT HAPPENED, with explanations anchored to Tier A/B.' if news else 'Query non-news: Tier C must not drive core statements.'}\n"

        "Language rule:\n"
        "The final answer must always be written in the **SAME LANGUAGE** as the user's **QUESTION**,\n"
        "regardless of the language used in system instructions, guardrails, or document context.\n"
        
    )

# “”

def tier_guardrail_instructions_analytics(query_text: str) -> str:
    return (
        "GUARDRAILS ANALYTICS (DATA-DRIVEN):\n"
        "1) Primary source: data provided directly by the user.\n"
        "2) You may use general knowledge of statistics, mathematics, and data analysis.\n"
        "3) NDo not invent numbers that cannot be derived from the provided data.n"
        "4) Always state assumptions (frequency, model, hypotheses).\n"
        "5) If the analysis is qualitative or methodological, state it explicitly..\n"
        
        "Language rule:\n"
        "The final answer must always be written in the **SAME LANGUAGE** as the user's **QUESTION**,\n"
        "regardless of the language used in system instructions, guardrails, or document context.\n"
    )


def build_system_instructions_analytics(intent: str = "analysis") -> str:
    return f"""
ROLE: Quantitative Analyst and Data Scientist.

LANGUAGE RULE:
- ALWAYS identify the language of the user's question first.
- YOU MUST ANSWER EXCLUSIVELY IN THE LANGUAGE OF THE USER. 
- If the question is in Italian, the output must be 100% Italian.
- If the question is in English, the output must be 100% English.

ANALYTICS RULES:
- User data is the PRIMARY SOURCE.
- You are not limited to the document context
- Use general knowledge of statistics/math only to process user data.
- If exact calculation is impossible, propose  a rigorous methodological analysis and propose procedures/code in Python or R.
- State limitations and assumptions clearly.
- Do not invent numbers that cannot be derived from the provided data.
- DON'T say "I can't answer": always provide the best analysis possible.
- DON'T make up numbers that can't be derived from the data provided.

INTENT: {intent}
""".strip()



def safe_markdown(text: str) -> str:
    """Make markdown safer for frontend rendering."""
    if not text:
        return ""
    t = text

    # limit very long lines (layout killer)
    t = "\n".join(line[:2000] for line in t.splitlines())

    # close unbalanced code fences
    if t.count("```") % 2 == 1:
        t += "\n```"

    return t
def short_text(s: str, n: int = 320) -> str:
    if not s:
        return ""
    return s[:n] + ("..." if len(s) > n else "")


def make_analytics_sources(user_query: str) -> List[SourceItem]:
    """
    In analytics_mode non facciamo retrieval, ma vogliamo comunque
    mostrare nel popup un “provenance” minimo: i dati arrivano dall’utente.
    """
    preview = (user_query or "").strip()
    if len(preview) > 1200:
        preview = preview[:1200] + "…"

    return [
        SourceItem(
            id="user_input",
            content=preview,
            filename="USER_INPUT",
            page=0,
            type="user_data",
            score=1.0,
            graph_context=[],
            section_hint="Dati forniti direttamente dall’utente (analytics_mode)",
            image_id=None,
            tier="USER",
        )
    ]



def strip_id_leaks(text: str) -> str:
    """Safety net: remove accidental id=... leaks from model output."""
    if not text:
        return ""
    text = re.sub(r"\|\s*id=[0-9a-f\-]{8,}\s*\]", "]", text, flags=re.IGNORECASE)
    text = re.sub(r"id=[0-9a-f\-]{8,}", "", text, flags=re.IGNORECASE)
    return text


# =========================
# 🔄 STATE MANAGEMENT
# =========================
class State(rx.State):
    messages: List[ChatMessage] = [
        ChatMessage(
            id="init",
            role="assistant",
            content=f"Ciao! Sono attivo con **{LLM_MODEL_NAME}**. Fammi domande sui tuoi documenti.",
        )
    ]
    input_text: str = ""
    is_processing: bool = False

    inline_open_for: str = ""
    inline_tab: str = "sources"  # "sources" | "audit"

    # Sidebar info
    vram_info: str = "N/A"
    vram_free: str = "N/A"
    backend_status: str = "OK"

    # --- Popup Fonti/Audit ---
    show_sources_modal: bool = False
    modal_sources: List[SourceItem] = []
    modal_debug_md: str = ""
    modal_title: str = ""

    def toggle_inline_sources(self, msg_id: str):
        if self.inline_open_for == msg_id and self.inline_tab == "sources":
            self.inline_open_for = ""
            return
        self.inline_open_for = msg_id
        self.inline_tab = "sources"

    def toggle_inline_audit(self, msg_id: str):
        if self.inline_open_for == msg_id and self.inline_tab == "audit":
            self.inline_open_for = ""
            return
        self.inline_open_for = msg_id
        self.inline_tab = "audit"

    def close_inline_panel(self):
        self.inline_open_for = ""


    def open_sources_audit(self, msg: ChatMessage):
        # titolo utile (puoi cambiare)
        self.modal_title = "Fonti & Audit"
        self.modal_sources = msg.sources
        self.modal_debug_md = msg.debug_md or ""
        self.show_sources_modal = True

    def close_sources_audit(self):
        self.show_sources_modal = False

    def on_load(self):
        self.refresh_gpu()
        self.refresh_backend_status()

    def refresh_backend_status(self):
        ok = True
        if embedder is None or qdrant_client_inst is None or llm_client is None:
            ok = False
        if neo4j_driver is None:
            # neo4j optional, still ok
            pass
        self.backend_status = "OK" if ok else "DEGRADED"

    def refresh_gpu(self):
        self.vram_info = gpu_free_info()
        # keep a short "free only"
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                self.vram_free = f"{free_bytes / (1024**3):.1f} GB free"
            except Exception:
                self.vram_free = "N/A"
        else:
            self.vram_free = "CPU"

    def clear_history(self):
        self.messages = [self.messages[0]]

    def set_input_text(self, text: str):
        self.input_text = text


    async def handle_submit(self):
        if not self.input_text.strip() or self.is_processing:
            return

        user_query = self.input_text.strip()
        self.input_text = ""
        self.is_processing = True



        # Prepariamo un "reminder" finale sulla lingua
        language_reminder = "\n\nREMINDER: Answer EXCLUSIVELY in the same language as the question above."

        try:
            self.refresh_gpu()
            self.refresh_backend_status()

            self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", content=user_query))
            yield rx.scroll_to("chat_bottom")

            intent = detect_intent(user_query)

            # ✅ NEW: decide se bypassare il retrieval (analytics su dati utente)
            analytics_mode = is_user_data_analytics(user_query)

            if analytics_mode:
                # ✅ Fonte sintetica per ripristinare “Fonti” + badge in UI
                sources = make_analytics_sources(user_query)

                # ✅ Audit UI (non retrieval): spiegazione chiara
                debug_md = (
                    "### 🔎 Audit (Analytics Mode)\n"
                    "- retrieval: **bypassed**\n"
                    "- motivo: **l’utente ha fornito un dataset / richiesta di calcolo**\n"
                    "- fonte primaria: **USER_INPUT** (dati dell’utente)\n"
                )
                
                self.inline_open_for = self.messages[-1].id
                self.inline_tab = "sources"

                context_str = ""  # niente contesto documentale
                system_instructions = build_system_instructions_analytics(intent)

                final_prompt_content = (
                    f"### SYSTEM INSTRUCTIONS ###\n"
                    f"{system_instructions}\n\n"
                    f"{tier_guardrail_instructions_analytics(user_query)}\n\n"
                    f"### USER QUESTION ###\n{user_query}"
                )
            else:
                # ✅ retrieve_v2 returns (sources, debug_md)
                sources, debug_md = retrieve_v2(user_query)

                context_str = build_context_block(sources, max_chars=MAX_CONTEXT_CHARS)
                system_instructions = build_system_instructions(intent)

                final_prompt_content = (
                    f"### SYSTEM INSTRUCTIONS ###\n"
                    f"{system_instructions}\n\n"
                    f"{tier_guardrail_instructions(user_query)}\n\n"
                    f"### DOCUMENT CONTEXT ###\n{context_str}\n\n"
                    f"### USER QUESTION ###\n{user_query}"
                )

            if analytics_mode:
                # ... (omissis)
                final_prompt_content = (
                    f"### SYSTEM INSTRUCTIONS ###\n"
                    f"{system_instructions}\n\n"
                    f"### USER QUESTION ###\n{user_query}"
                    f"{language_reminder}" # <--- Aggiunto qui
                )
            else:
                # ... (omissis)
                final_prompt_content = (
                    f"### SYSTEM INSTRUCTIONS ###\n"
                    f"{system_instructions}\n\n"
                    f"{tier_guardrail_instructions(user_query)}\n\n"
                    f"### DOCUMENT CONTEXT ###\n{context_str}\n\n"
                    f"### USER QUESTION ###\n{user_query}"
                    f"{language_reminder}" # <--- Aggiunto qui
                )



            messages_payload = build_alternating_history(self.messages, MEMORY_LIMIT)
            if messages_payload and messages_payload[-1]["role"] == "user":
                messages_payload.pop()
            messages_payload.append({"role": "user", "content": final_prompt_content})

            # ✅ persist debug_md on assistant message
            self.messages.append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="",
                    sources=sources,
                    debug_md=debug_md,
                )
            )
            yield rx.scroll_to("chat_bottom")

            full_resp = ""
            if llm_client:
                try:
                    stream = llm_client.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=messages_payload,
                        temperature=0.1,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            full_resp += delta.content

                            if len(full_resp) > MAX_ASSISTANT_CHARS:
                                full_resp = full_resp[:MAX_ASSISTANT_CHARS] + "\n\n*(Risposta troncata per limiti UI)*"
                                cleaner = globals().get("strip_id_leaks")
                                self.messages[-1].content = cleaner(full_resp) if callable(cleaner) else full_resp
                                yield
                                break

                            self.messages[-1].content = strip_id_leaks(full_resp)
                            yield
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Error from LLM: {error_msg}")
                    self.messages[-1].content = f"⚠️ Errore LLM: {error_msg}. Controlla i log di LM Studio."
            else:
                self.messages[-1].content = "⚠️ Client LLM non inizializzato."

        finally:
            self.refresh_gpu()
            self.is_processing = False


# =========================
# 🎨 UI COMPONENTS
# =========================
def source_badge(text: str, color: str, icon: str):
    return rx.badge(
        rx.hstack(rx.icon(icon, size=12), rx.text(text)),
        color_scheme=color,
        variant="soft",
        radius="full",
        size="1",
    )

def message_ui(msg: ChatMessage):
    is_bot = msg.role == "assistant"
    bg_color = rx.cond(is_bot, rx.color("gray", 3), rx.color("indigo", 9))
    text_color = rx.cond(is_bot, rx.color("gray", 12), "white")
    align_self = rx.cond(is_bot, "start", "end")

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.avatar(
                    fallback=rx.cond(is_bot, "🤖", "👤"),
                    size="2",
                    variant="soft",
                    color_scheme=rx.cond(is_bot, "gray", "indigo"),
                ),
                rx.text(rx.cond(is_bot, "Financial AI", "Tu"), weight="bold", size="2"),
                rx.spacer(),
                rx.cond(
                    is_bot & ((msg.sources.length() > 0) | (msg.debug_md.length() > 0)),
                    rx.button(
                        rx.hstack(
                            rx.icon("book-open", size=14),
                            rx.text("Fonti/Audit", size="1"),
                            rx.cond(
                                msg.sources.length() > 0,
                                rx.badge(rx.text(msg.sources.length()), color_scheme="green", variant="soft"),
                                rx.box(),
                            ),
                            spacing="2",
                            align_items="center",
                        ),
                        variant="ghost",
                        size="1",
                        on_click=lambda: State.open_sources_audit(msg),
                    ),
                    rx.box(),
                ),
                width="100%",
                align_items="center",
                spacing="2",
            ),
            rx.markdown(
                msg.content,
                width="100%",
                overflow_wrap="anywhere",
                word_break="break-word",
            ),
            spacing="2",
            width="100%",
        ),
# ---- Inline popup "Fonti + Audit" sotto la risposta LLM ----
rx.cond(
    is_bot & ((msg.sources.length() > 0) | (msg.debug_md.length() > 0)),
    rx.box(
        # barra azioni
        rx.hstack(
            rx.button(
                rx.hstack(
                    rx.icon("book-open", size=14),
                    rx.text("Fonti", size="1"),
                    rx.badge(rx.text(msg.sources.length()), color_scheme="green", variant="soft"),
                    spacing="2",
                    align_items="center",
                ),
                size="1",
                variant="soft",
                on_click=lambda: State.toggle_inline_sources(msg.id),
            ),
            rx.button(
                rx.hstack(
                    rx.icon("shield-check", size=14),
                    rx.text("Audit", size="1"),
                    spacing="2",
                    align_items="center",
                ),
                size="1",
                variant="soft",
                on_click=lambda: State.toggle_inline_audit(msg.id),
            ),
            rx.spacer(),
            spacing="2",
            width="100%",
            margin_top="0.6em",
        ),

        # pannello espanso: FONTI
        rx.cond(
            State.inline_open_for == msg.id,
            rx.box(
                rx.cond(
                    State.inline_tab == "sources",
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(
                                msg.sources,
                                lambda s: rx.card(
                                    rx.vstack(
                                        rx.hstack(
                                            rx.icon("file-text", size=14),
                                            rx.text(s.filename, weight="bold", size="2"),
                                            rx.spacer(),
                                            rx.badge(f"Pag {s.page}", color_scheme="gray", variant="soft"),
                                            rx.badge(s.type, color_scheme="blue", variant="soft"),
                                            rx.cond(
                                                (s.tier != "") & (s.tier != None),
                                                source_badge(s.tier, "green", "layers"),
                                                rx.box(),
                                            ),
                                            align_items="center",
                                            width="100%",
                                        ),
                                        rx.text(
                                            rx.cond(
                                                s.content.length() > 700,
                                                s.content[:700] + "…",
                                                s.content,
                                            ),
                                            size="1",
                                            color="gray",
                                            width="100%",
                                        ),

                                        # provenance PG
                                        rx.flex(
                                            rx.badge(f"chunk_uuid: {s.id}", variant="soft"),
                                            rx.cond(s.pg_log_id > 0, rx.badge(f"log_id: {s.pg_log_id}", variant="soft"), rx.box()),
                                            rx.cond(s.pg_chunk_id > 0, rx.badge(f"chunk_id: {s.pg_chunk_id}", variant="soft"), rx.box()),
                                            rx.cond(s.pg_ingestion_ts != "", rx.badge(f"ts: {s.pg_ingestion_ts}", variant="soft"), rx.box()),
                                            rx.cond(s.pg_source_name != "", rx.badge(f"source: {s.pg_source_name}", color_scheme="blue", variant="soft"), rx.box()),
                                            rx.cond(s.pg_source_type != "", rx.badge(f"type: {s.pg_source_type}", color_scheme="blue", variant="soft"), rx.box()),
                                            rx.cond(s.pg_toon_type != "", rx.badge(f"toon: {s.pg_toon_type}", color_scheme="purple", variant="soft"), rx.box()),
                                            spacing="2",
                                            width="100%",
                                            wrap="wrap",
                                        ),
                                        spacing="2",
                                        width="100%",
                                        align_items="start",
                                    ),
                                    size="2",
                                    width="100%",
                                ),
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        height="260px",
                        type="always",
                    ),
                    rx.box(
                        rx.heading("Audit", size="3"),
                        rx.markdown(
                            msg.debug_md,
                            width="100%",
                            overflow_wrap="anywhere",
                            word_break="break-word",
                        ),
                        width="100%",
                    ),
                ),

                rx.hstack(
                    rx.spacer(),
                    rx.button(
                        "Chiudi",
                        size="1",
                        variant="ghost",
                        on_click=State.close_inline_panel,
                    ),
                    width="100%",
                    margin_top="0.5em",
                ),

                border=f"1px solid {rx.color('gray', 5)}",
                border_radius="12px",
                padding="0.8em",
                margin_top="0.6em",
                bg=rx.color("gray", 1),
                width="100%",
            ),
            rx.box(),
        ),
        width="100%",
    ),
    rx.box(),
),


        bg=bg_color,
        color=text_color,
        padding="1em",
        border_radius="12px",
        max_width="85%",
        align_self=align_self,
        box_shadow="sm",
        margin_y="0.5em",
    )






def index():
    return rx.flex(
        # Sidebar
        rx.vstack(
            rx.heading("System Status", size="3"),
            rx.divider(),
            rx.hstack(rx.icon("cpu"), rx.text(State.vram_info, size="1")),
            rx.hstack(rx.icon("hard-drive"), rx.text(f"GPU free: {State.vram_free}", size="1")),
            rx.hstack(
                rx.icon("activity"),
                rx.text(f"Backend: {State.backend_status}", size="1"),
            ),
            rx.text(f"LLM: {LLM_MODEL_NAME}", size="1", color="gray"),
            rx.spacer(),
            rx.button(
                "Refresh GPU",
                on_click=State.refresh_gpu,
                color_scheme="gray",
                variant="soft",
                width="100%",
            ),
            rx.button(
                "Clear Chat",
                on_click=State.clear_history,
                color_scheme="red",
                variant="soft",
                width="100%",
            ),
            width="260px",
            height="100%",
            padding="1.5em",
            bg=rx.color("gray", 2),
            display=["none", "none", "flex"],
            flex_shrink="0",
            min_height="0",
            overflow="hidden",
        ),

        # Main
        rx.vstack(
            # Header
            rx.box(
                rx.heading(PAGE_TITLE, size="6", align="center"),
                rx.text(
                    f"Powered by {LLM_MODEL_NAME} + Qdrant + Neo4j",
                    color="gray",
                    size="2",
                    align="center",
                ),
                padding_y="1em",
                width="100%",
                text_align="center",
                flex_shrink="0",
            ),
            
            # --- Popup Fonti/Audit (una sola volta, fuori dal foreach dei messaggi) ---
            rx.dialog.root(
                rx.dialog.content(
                    rx.dialog.title(State.modal_title),
                    rx.dialog.description("Fonti e audit della risposta."),
                    rx.divider(),

                    # ====== FONTI ======
                    rx.cond(
                        State.modal_sources.length() > 0,
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    State.modal_sources,
                                    lambda s: rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("file-text", size=14),
                                                rx.text(s.filename, weight="bold", size="2"),
                                                rx.spacer(),
                                                rx.badge(f"Pag {s.page}", color_scheme="gray", variant="soft"),
                                                rx.badge(s.type, color_scheme="blue", variant="soft"),
                                                rx.cond(
                                                    (s.tier != "") & (s.tier != None),
                                                    source_badge(s.tier, "green", "layers"),
                                                    rx.box(),
                                                ),
                                                align_items="center",
                                                width="100%",
                                            ),
                                            rx.text(
                                                rx.cond(
                                                    s.content.length() > 900,
                                                    s.content[:900] + "…",
                                                    s.content,
                                                ),
                                                size="1",
                                                color="gray",
                                                width="100%",
                                            ),

                                            # ====== Provenance PG (se valorizzata) ======
                                            rx.flex(
                                                rx.badge(f"chunk_uuid: {s.id}", variant="soft"),
                                                rx.cond(s.pg_log_id > 0, rx.badge(f"log_id: {s.pg_log_id}", variant="soft"), rx.box()),
                                                rx.cond(s.pg_chunk_id > 0, rx.badge(f"chunk_id: {s.pg_chunk_id}", variant="soft"), rx.box()),
                                                rx.cond(s.pg_ingestion_ts != "", rx.badge(f"ts: {s.pg_ingestion_ts}", variant="soft"), rx.box()),
                                                rx.cond(s.pg_source_name != "", rx.badge(f"source: {s.pg_source_name}", color_scheme="blue", variant="soft"), rx.box()),
                                                rx.cond(s.pg_source_type != "", rx.badge(f"type: {s.pg_source_type}", color_scheme="blue", variant="soft"), rx.box()),
                                                rx.cond(s.pg_toon_type != "", rx.badge(f"toon: {s.pg_toon_type}", color_scheme="purple", variant="soft"), rx.box()),
                                                spacing="2",
                                                width="100%",
                                                wrap="wrap",
                                            ),

                                            # Graph context (opzionale)
                                            rx.cond(
                                                s.graph_context.length() > 0,
                                                rx.flex(
                                                    rx.text("Graph:", size="1", weight="bold", color="gray"),
                                                    rx.foreach(
                                                        s.graph_context,
                                                        lambda e: rx.badge(
                                                            e.name,
                                                            color_scheme="purple",
                                                            variant="surface",
                                                            size="1",
                                                        ),
                                                    ),
                                                    spacing="1",
                                                    wrap="wrap",
                                                    margin_top="4px",
                                                ),
                                                rx.box(),
                                            ),

                                            spacing="2",
                                            width="100%",
                                            align_items="start",
                                        ),
                                        size="2",
                                        width="100%",
                                    ),
                                ),
                                spacing="3",
                                width="100%",
                            ),
                            height="320px",
                            type="always",
                        ),
                        rx.text("Nessuna fonte disponibile.", color="gray"),
                    ),

                    rx.divider(),

                    # ====== AUDIT ======
                    rx.cond(
                        State.modal_debug_md.length() > 0,
                        rx.box(
                            rx.heading("Audit", size="3"),
                            rx.markdown(
                                State.modal_debug_md,
                                width="100%",
                                overflow_wrap="anywhere",
                                word_break="break-word",
                            ),
                            width="100%",
                        ),
                        rx.text("Nessun audit disponibile.", color="gray"),
                    ),

                    rx.hstack(
                        rx.spacer(),
                        rx.button("Chiudi", variant="soft", on_click=State.close_sources_audit),
                        width="100%",
                        margin_top="1em",
                    ),

                    max_width="900px",
                    width="90vw",
                ),
            ),

            # Chat scroll area
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(State.messages, message_ui),
                    rx.box(id="chat_bottom"),
                    width="100%",
                    padding="1em",
                    max_width="900px",
                    margin="0 auto",
                    spacing="4",
                    min_height="0",
                ),
                width="100%",
                flex="1",
                min_height="0",
                type="always",
                scrollbars="vertical",
                id="chat_scroll_area",
                overflow_x="hidden",   # <-- IMPORTANT: niente overflow orizzontale nella scroll area
            ),

            # Input area
            rx.box(
                rx.hstack(
                    rx.input(
                        placeholder="Chiedi informazioni sui documenti...",
                        value=State.input_text,
                        on_change=State.set_input_text,
                        on_key_down=lambda k: rx.cond(k == "Enter", State.handle_submit(), None),
                        radius="full",
                        size="3",
                        flex="1",
                    ),
                    rx.button(
                        rx.icon("send"),
                        on_click=State.handle_submit,
                        loading=State.is_processing,
                        radius="full",
                        size="3",
                    ),
                    width="100%",
                    max_width="900px",
                    padding="1em",
                ),
                width="100%",
                display="flex",
                justify_content="center",
                bg=rx.color("gray", 1),
                border_top="1px solid #e5e5e5",
                flex_shrink="0",
            ),

            height="100%",
            width="100%",
            spacing="0",
            overflow="hidden",
            overflow_x="hidden",  # <-- IMPORTANT
            min_height="0",
        ),

        # ROOT: ancorato alla viewport (impedisce al body di "prendere lo scroll")
        width="100%",
        height="100dvh",       # più robusto di 100vh
        position="fixed",
        top="0",
        left="0",
        right="0",
        bottom="0",
        overflow="hidden",
        overflow_x="hidden",   # <-- IMPORTANT: elimina la scrollbar orizzontale del body
        min_height="0",
    )




app = rx.App(theme=rx.theme(appearance="light", accent_color="indigo", radius="large"))
app.add_page(index, on_load=State.on_load)
