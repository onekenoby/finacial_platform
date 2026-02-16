import reflex as rx
import torch
import os
import time
import re
import json
import hashlib
import psycopg2
import requests
from collections import Counter

from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from neo4j import GraphDatabase
from openai import OpenAI

from qdrant_client import QdrantClient, models  # <--- Serve per il filtro Tier A/C
from sentence_transformers import SentenceTransformer, CrossEncoder # <--- Per i vettori della GUI
import uuid # <--- Per generare gli ID dei messaggi

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
RAG_DEFAULT_TIERS = os.getenv("RAG_DEFAULT_TIERS", "A,B,C")  # default prudente

##### NB DA GENERALIZZARE
RAG_NEWS_KEYWORDS = os.getenv(
    "RAG_NEWS_KEYWORDS",
    "news,oggi,ieri,cina,china,geopolitica,ultima,ultime,rumor,breaking,aggiornamenti,recente"
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
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", LLM_MODEL_NAME)

#EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MODEL_NAME = "E:/Modelli/bge-m3"

#RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_MODEL_NAME = "E:/Modelli/ms-marco-reranker"


# LM Studio / OpenAI Compatible API
#LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
#LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy key, Ollama non la valida


MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "3"))  # number of turns (user+assistant)

# Retrieval knobs (RAG v2)
QDRANT_CANDIDATES = int(os.getenv("QDRANT_CANDIDATES", "60"))     # retrieve top-N from qdrant
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))     # Aumentato per catturare più sfumature
FINAL_SOURCES = int(os.getenv("FINAL_SOURCES", "4"))             # Aumentato per dare più contesto
MAX_PER_PAGE = int(os.getenv("MAX_PER_PAGE", "2"))                # ✅ FONDAMENTALE: Consente più chunk per la stessa pagina
MAX_PER_DOC = int(os.getenv("MAX_PER_DOC", "3"))                  # ✅ FONDAMENTALE: Consente Deep-Dive su un singolo documento

# =========================
# 🎚️ Tier-aware ranking
# =========================
TIER_BOOST_A = float(os.getenv("TIER_BOOST_A", "0.08"))
TIER_BOOST_B = float(os.getenv("TIER_BOOST_B", "0.04"))
TIER_PENALTY_C = float(os.getenv("TIER_PENALTY_C", "0.015"))

# Se la query è news/rumor/recency, NON penalizzare Tier C
TIER_C_PENALTY_IF_NOT_NEWS = os.getenv("TIER_C_PENALTY_IF_NOT_NEWS", "1") == "1"


# Graph expansion knobs
GRAPH_EXPAND_ENABLED = os.getenv("GRAPH_EXPAND_ENABLED", "1") == "1"
GRAPH_MAX_FORMULAS = int(os.getenv("GRAPH_MAX_FORMULAS", "6"))
GRAPH_MAX_NEIGHBOR_CHUNKS = int(os.getenv("GRAPH_MAX_NEIGHBOR_CHUNKS", "4"))

# Prompt limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))  # prevent prompt blow-ups
MAX_ASSISTANT_CHARS = int(os.getenv("MAX_ASSISTANT_CHARS", "12000"))

AUDIT_ENABLED = True
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "./rag_audit.jsonl")

# =========================
# 🧠 CARICAMENTO RISORSE
# =========================
print("⏳ Init Backend...")

#device_embed = "cuda" if torch.cuda.is_available() else "cpu"
# ...ogica che provi la CPU ma resti flessibile:

device_embed = "cuda" if torch.cuda.is_available() else "cpu"

device_rerank = "cpu"  # IMPORTANT: avoid fighting with Gemma/Vision on the same P5000

embedder = None
reranker = None
llm_client = None
qdrant_client_inst = None
neo4j_driver = None

try:
    #print(f"🚀 Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on {device_embed.upper()}...")
    #embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_embed)    
    print(f"🚀 Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on {device_embed.upper()}...")
    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME, 
        device=device_embed, 
        local_files_only=True # Fondamentale per la modalità offline
    )    
       

    #print(f"🚀 Loading Reranker ({RERANKER_MODEL_NAME}) on {device_rerank.upper()}...")
    #reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device_rerank)
    
    print(f"🚀 Loading Reranker ({RERANKER_MODEL_NAME}) on {device_rerank.upper()}...")
    reranker = CrossEncoder(
        RERANKER_MODEL_NAME, 
        device=device_rerank,
        # Se la libreria CrossEncoder lo supporta direttamente (dipende dalla versione), 
        # altrimenti il percorso locale è sufficiente.
    )


    print(f"🚀 Connecting to LLM via Ollama ({LLM_MODEL_NAME}) at {OLLAMA_URL}...")
    llm_client = OpenAI(base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY)



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
    tier: str = "C"
    # ✅ PG canonical provenance
    pg_ingestion_ts: str = ""
    pg_source_name: str = ""
    pg_source_type: str = ""
    pg_log_id: int = 0
    pg_chunk_id: int = 0
    pg_toon_type: str = ""
    db_origin: str = "Unknown"
    
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
    """Very cheap intent router: formula / table / chart / text."""
    q = (query or "").lower()

    # formula intent
    if any(k in q for k in [
        "formula","matrix","matrice","equazione","equation","derivate","derivata",
        "integration","integrale","latex","limit","limite","lift","support","confidence",
        "probabilità","probability"
    ]):
        return "formula"

    # table intent (prima di chart)
    if any(k in q for k in [
        "tabella","table","tabulation","righe","colonne","row","rows","column","columns",
        "indice","indici","valuta","currency","legenda valute","legend"
    ]):
        return "table"

    # chart intent
    if any(k in q for k in [
        "grafico","graph","flow","flowchart","diagramma","diagram","prospect","prospetto",
        "chart","figura","asse","legend","legenda","trend","slop","candela","candle",
        "ohlc","volumi","volume","heatmap"
    ]):
        return "chart"

    return "text"



def extract_requested_pages(query: str):
    import re
    if not query:
        return []

    q = query.lower().strip()
    # "pag 8-9", "pagina 8/9", "page 10-12"
    pattern = r"\b(?:pag(?:ina)?|page|p)\.?\s*[:=]?\s*(\d{1,4})(?:\s*[-/]\s*(\d{1,4}))?\b"
    m = re.search(pattern, q, flags=re.IGNORECASE)
    if not m:
        return []

    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else None

    if b is None:
        return [a] if a > 0 else []
    if a <= 0 or b <= 0:
        return []

    lo, hi = (a, b) if a <= b else (b, a)
    # clamp max span to avoid huge expansions
    if hi - lo > 20:
        return [lo, hi]
    return list(range(lo, hi + 1))


# ------------------------------------------------------------
# TABLE-FIRST RETRIEVAL REORDERING (ANTI-GENERIC ANSWERS)
# ------------------------------------------------------------

'''
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
'''

def is_user_data_analytics(query: str) -> bool:
    """
    Rileva se l'utente ha fornito dati nel prompt.
    Attiva Analytics Mode solo se ci sono evidenze di dataset (numeri o parentesi).
    """
    q = query.lower()
    # Verifica presenza di array [1, 2, 3] o molti numeri (più di 10)
    has_data_structure = bool(re.search(r"\[[0-9,\s./-]+\]", q))
    digit_count = len(re.findall(r"\d+", q))
    
    # Parole chiave analitiche
    keywords = [
        # Calcolo e Stima (Base)
        "calcola", "calculate", "stima", "estimate", "totale", "total", "somma", "sum","analizza","analyse","analyze"
        
        # Statistica e Analisi Dati
        "regressione", "regression", "correlazione", "correlation", "media", "mean", 
        "average", "varianza", "variance", "deviazione", "std dev", "distribuzione", "distribution","standard deviation","ave","decompos",
        
        # Forecasting e Serie Storiche
        "prevedi", "forecast", "proiezione", "projection", "trend", "stagionalità", "seasonality","slop","prediction","time series", "serie storic", "model",
        
        # Metriche Finanziarie (Se l'utente fornisce i dati)
        "volatilità", "volatility", "sharpe", "beta", "alpha", "rendimento", "return", "drawdown","stocks","bound","azioni"
    ]
    has_keywords = any(k in q for k in keywords)

    # TRIGGER: Attiva solo se ci sono dati forniti (strutture o molti numeri) E parole chiave
    # Questo permette a "Analizza i documenti" di andare correttamente al RAG.
    return (has_data_structure or digit_count > 10) and has_keywords

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
    # Se la stringa contiene "A", assegna il boost di metodologia
    if "A" in t:
        return TIER_BOOST_A
    if "B" in t:
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
    # FIX ANTI-RUMORE: Richiede correlazione forte (minimo 2 entità in comune)
    # o esclude entità troppo generiche (Type='Entity' generico)
    query = """
    MATCH (c1:Chunk)<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(c2:Chunk)
    WHERE c1.id IN $ids 
      AND NOT c2.id IN $ids
      AND NOT e.type IN ['Generic', 'Year', 'Date'] -- Filtro Stop-Nodes opzionale
    
    WITH c2, count(DISTINCT e) as strength, collect(e.label) as overlaps
    WHERE strength >= 2  -- <--- FILTRO CRITICO: Almeno 2 concetti in comune
    
    RETURN c2.id AS cid
    ORDER BY strength DESC
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
#    if is_news_query(query_text):
#        return None  # include A/B/C
#
#    return {
#        "must": [
#            {"key": "tier", "match": {"any": ["A", "B", "C"]}}
#        ]
#    }
    return

def build_retrieval_audit_md(
    query_text: str,
    intent: str,
    timings: Dict[str, float],
    counts: Dict[str, Any],
    top_sources_preview: List[Dict[str, Any]],
) -> str:
    """Audit avanzato che scompone l'attività di Qdrant, Postgres e Neo4j."""
    def ms(x: float) -> str:
        return f"{x*1000:.0f} ms"

    lines = []
    lines.append("### 🔎 Audit Retrieval (Multi-Database Analysis)")
    lines.append(f"- **Intent**: `{intent}`")
    lines.append(f"- **Query**: `{(query_text or '')[:180]}`")

    # 🌌 SEZIONE QDRANT (Vettoriale)
    lines.append("\n#### 🌌 Qdrant (Vector Search)")
    if "qdrant_search" in timings:
        lines.append(f"- Tempo: **{ms(timings['qdrant_search'])}**")
    lines.append(f"- Hits vettoriali: **{counts.get('qdrant_hits', 0)}**")

    # 🐘 SEZIONE POSTGRES (BM25)
    lines.append("\n#### 🐘 Postgres (Keyword Search)")
    if "bm25_search" in timings:
        lines.append(f"- Tempo: **{ms(timings['bm25_search'])}**")
    lines.append(f"- Match testuali: **{counts.get('bm25_hits', 0)}**")

    # 🕸️ SEZIONE NEO4J (Grafo)
    if counts.get('final_formulas', 0) > 0 or "graph" in timings:
        lines.append("\n#### 🕸️ Neo4j (Graph Expansion)")
        if "graph" in timings:
            lines.append(f"- Tempo: **{ms(timings['graph'])}**")
        lines.append(f"- Formule/Relazioni: **{counts.get('final_formulas', 0)}**")

    # ⚖️ SEZIONE PERFORMANCE & RERANK
    lines.append("\n#### ⚖️ Fusione & Reranking")
    if "rerank" in timings:
        lines.append(f"- Tempo Reranker: **{ms(timings['rerank'])}**")
    lines.append(f"- Candidati totali: **{counts.get('qdrant_hits', 0) + counts.get('bm25_hits', 0)}**")
    if "total" in timings:
        lines.append(f"- **Tempo Totale Retrieval**: **{ms(timings['total'])}**")

    # 📦 DISTRIBUZIONE TIER
    tier_split = counts.get("tier_split", {})
    if tier_split:
        lines.append("\n#### 📦 Tier Distribution")
        for t, n in tier_split.items():
            lines.append(f"- `{t}`: **{n}**")

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
    JOIN wanted w ON d.chunk_uuid::text = w.chunk_uuid::text
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


def search_pg_bm25(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not PG_ENRICH_ENABLED or not pg_pool: return []
    if not query_text.strip(): return []

    # Usiamo websearch_to_tsquery che è il più robusto (gestisce "virgolette", OR, ecc.)
    sql = """
    SELECT chunk_uuid::text, content_raw, content_semantic, metadata_json,
        ts_rank_cd(
            to_tsvector('simple', content_semantic || ' ' || COALESCE(metadata_json::text, '')), 
            websearch_to_tsquery('simple', %s)
        ) AS rank
    FROM public.document_chunks
    WHERE 
        to_tsvector('simple', content_semantic || ' ' || COALESCE(metadata_json::text, '')) 
        @@ websearch_to_tsquery('simple', %s)
    ORDER BY rank DESC LIMIT %s;
    """
    
    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (query_text, query_text, limit))
            rows = cur.fetchall()
            return [{
                "id": r[0], "content": r[2] or r[1], "metadata": r[3] or {}, "score": float(r[4])
            } for r in rows]
    except Exception as e:
        print(f"⚠️ BM25 Error: {e}")
        return []
    finally:
        pg_pool.putconn(conn)

# =========================
# 🔍 RAG v2 Retrieval
# =========================

def fetch_pg_chunks_by_doc_and_index(pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Return latest PG chunk rows for each (doc_id, chunk_index)."""
    if not PG_ENRICH_ENABLED or not pg_pool or not pairs:
        return {}

    # dedup preserving order
    seen = set()
    uniq: List[Tuple[str, int]] = []
    for d, i in pairs:
        if not d:
            continue
        key = (str(d), int(i))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    sql = """
    WITH wanted(doc_id, chunk_index) AS (VALUES %s),
    ranked AS (
      SELECT
        d.doc_id::text AS doc_id,
        d.chunk_index::int AS chunk_index,
        d.chunk_uuid::text AS chunk_uuid,
        d.content_raw,
        d.content_semantic,
        d.metadata_json,
        d.ingestion_ts,
        ROW_NUMBER() OVER (
          PARTITION BY d.doc_id, d.chunk_index
          ORDER BY d.ingestion_ts DESC
        ) AS rn
        FROM public.document_chunks d
        JOIN wanted w ON d.chunk_uuid::text = w.chunk_uuid::text
       AND d.chunk_index::int = w.chunk_index
    )
    SELECT doc_id, chunk_index, chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts
    FROM ranked
    WHERE rn = 1;
    """

    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, [(d, i) for d, i in uniq])
            rows = cur.fetchall()

        out: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for (doc_id, chunk_index, chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts) in rows:
            out[(str(doc_id), int(chunk_index))] = {
                "chunk_uuid": chunk_uuid,
                "content_raw": content_raw,
                "content_semantic": content_semantic,
                "metadata_json": metadata_json,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            }
        return out
    except Exception as e:
        print(f"⚠️ PG enrich (doc_id+chunk_index) error: {e}")
        return {}
    finally:
        pg_pool.putconn(conn)

def apply_rrf_scoring(candidates: List[Dict[str, Any]], k: int = 60):
    """
    Applica Reciprocal Rank Fusion (RRF) direttamente alla lista di candidati.
    Modifica la lista in-place.
    """
    # 1. Inizializzazione sicura: assicuriamoci che tutti abbiano il campo rrf_score
    for c in candidates:
        c["rrf_score"] = 0.0

    # 2. RRF su Vettori (Qdrant)
    # Creiamo una lista temporanea ordinata per score vettoriale
    vec_sorted = sorted(
        [c for c in candidates if c.get("score_vec", 0) > 0], 
        key=lambda x: x["score_vec"], 
        reverse=True
    )
    
    # Assegniamo i punti basati sul rango
    for rank, item in enumerate(vec_sorted):
        # item è un riferimento al dizionario originale, quindi la modifica è persistente
        item["rrf_score"] += (1.0 / (k + rank + 1))

    # 3. RRF su Keyword (BM25 Postgres)
    bm25_sorted = sorted(
        [c for c in candidates if c.get("score_bm25", 0) > 0], 
        key=lambda x: x["score_bm25"], 
        reverse=True
    )
    
    for rank, item in enumerate(bm25_sorted):
        item["rrf_score"] += (1.0 / (k + rank + 1))

def retrieve_v2(query_text: str) -> Tuple[List[SourceItem], str]:
    """
    Retrieval V4: DEBUG ESTREMO + FILENAME FORCING.
    """
    print(f"\n\n{'='*40}")
    print(f"🔎 DEBUG RETRIEVAL START")
    print(f"❓ Query: '{query_text}'")
    
    if not embedder or not qdrant_client_inst:
        return [SourceItem(id="error", content="Backend OFF", filename="System")], "Backend OFF"

    t_total0 = time.time()
    timings: Dict[str, float] = {}
    counts: Dict[str, Any] = {}
    intent = detect_intent(query_text)

    # 1) Embedding
    t0 = time.time()
    query_vector = embedder.encode(query_text, normalize_embeddings=True).tolist()
    timings["embed"] = time.time() - t0

    # 2) Qdrant (Vettoriale)
    t0 = time.time()
    hits = []
    try:
        # Prendiamo pochi candidati ma buoni
        hits = qdrant_client_inst.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=20, 
            with_payload=True,
        )
        counts["qdrant_hits"] = len(hits)
        print(f"🌌 Qdrant ha trovato {len(hits)} chunk.")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
    timings["qdrant_search"] = time.time() - t0

    # 3) Postgres (Keyword)
    t0 = time.time()
    # Usiamo un limite alto per essere sicuri di pescare il file se c'è
    bm25_hits = search_pg_bm25(query_text, limit=40) 
    counts["bm25_hits"] = len(bm25_hits)
    print(f"🐘 Postgres ha trovato {len(bm25_hits)} chunk.")
    timings["bm25_search"] = time.time() - t0

    # 4) Unificazione Candidati
    candidates_dict = {}

    # -- Import Qdrant --
    for hit in hits:
        uid = str(hit.id)
        p = hit.payload or {}
        fname = str(p.get("filename", "Unknown"))
        candidates_dict[uid] = {
            "id": uid,
            "content": safe_payload_text(p),
            "filename": fname,
            "page": get_payload_page(p),
            "type": get_payload_type(p),
            "tier": str(p.get("tier", "C")),
            "score_base": float(hit.score or 0.0),
            "origin": "Qdrant",
            "section_hint": get_payload_section(p)
        }

    # -- Import Postgres --
    for b in bm25_hits:
        uid = b["id"]
        meta = b.get("metadata", {})
        fname = meta.get("filename", "Unknown")
        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": b["content"],
                "filename": fname,
                "page": int(meta.get("page_no") or 0),
                "type": meta.get("toon_type", "text"),
                "tier": meta.get("tier", "C"),
                "score_base": 0.0, # Verrà aggiornato dopo
                "origin": "Postgres",
                "section_hint": meta.get("section_hint", "")
            }

    candidates = list(candidates_dict.values())
    if not candidates:
        print("❌ NESSUN CANDIDATO TROVATO!")
        return [], "Nessun risultato."


    # 5) SCORING INTELLIGENTE & FILENAME BOOST (HARD MODE)
    # Prepariamo token query (parole > 3 lettere) per catturare anche "WACC", "Table", ecc.
    # RIMOSSO IL LIMITE [:2] per analizzare tutta la frase, non solo le parole più lunghe.
    query_tokens = [w.lower() for w in query_text.split() if len(w) > 3]
    
    print(f"🎯 Target Tokens (Filename Match): {query_tokens}")

    for c in candidates:
        fname_lower = (c.get("filename") or "").lower()
        
        # LOGICA BRUTALE: Se il nome file è nella query, questo chunk DEVE vincere.
        boost = 0.0
        hits_fname = 0
        
        # 1. Match Esatto Parziale (es. "Formulae_Table.pdf" contiene "formulae")
        for token in query_tokens:
            if token in fname_lower:
                # Escludiamo parole troppo comuni per evitare falsi positivi
                if token not in ["della", "delle", "file", "documento", "page", "pagina"]:
                    hits_fname += 1
        
        # 2. Assegnazione Boost: +5.0 punti portano il documento in Cima Assoluta
        if hits_fname > 0:
            boost = 5.0 * hits_fname  
            c["origin"] += " [TARGET FILE]"
            print(f"   🚀 SUPER BOOST per {c.get('filename')} (Match: {hits_fname} token)")

        # Score base + SUPER BOOST
        c["final_score"] = float(c.get("score_base", 0.0)) + boost

    # 6) RERANKING SELETTIVO
    # Mandiamo al reranker SOLO chi non ha già vinto per filename match
    # Questo salva un sacco di tempo e previene che il reranker abbassi il target
    
    # Separiamo i "Vincitori sicuri" (Target File) dagli altri
    winners = [c for c in candidates if c["final_score"] > 400]
    others = [c for c in candidates if c["final_score"] <= 400]
    
    # Rerankiamo solo gli "others" (i top 15)
    others.sort(key=lambda x: x["final_score"], reverse=True)
    top_others = others[:15]
    
    if reranker and top_others:
        t0 = time.time()
        pairs = [(query_text, c["content"]) for c in top_others]
        try:
            scores = reranker.predict(pairs)
            for i, score in enumerate(scores):
                top_others[i]["final_score"] = float(score) # Score normale del reranker
        except Exception as e:
            print(f"⚠️ Reranker Error: {e}")
        timings["rerank"] = time.time() - t0
    
    # Riuniamo tutto: Vincitori (Score 500+) + Altri Rerankati
    final_pool = winners + top_others
    final_pool.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 7) Selezione Finale (DIVERSIFY)
    # Riduciamo a 5 fonti massime per evitare blocchi LLM
    final_selection = diversify(final_pool, MAX_PER_PAGE, MAX_PER_DOC, 5)

    print("-" * 20)
    print("🏆 CLASSIFICA FINALE (Top 3):")
    for i, s in enumerate(final_selection[:3]):
        print(f"  {i+1}. {s['filename']} (Score: {s['final_score']:.1f}) - {s['origin']}")
    print("="*40 + "\n")

    # 8) Output Object Construction
    sources = []
    for t in final_selection:
        sources.append(SourceItem(
            id=t["id"], 
            content=t["content"],
            filename=t["filename"], 
            page=t["page"], 
            type=t["type"],
            score=t.get("final_score", 0.0), 
            tier=t["tier"],
            db_origin=t.get("origin", "Unknown"), 
            section_hint=t.get("section_hint", "")
        ))

    # Graph (opzionale, teniamolo leggero)
    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        chunk_ids = [s.id for s in sources]
        formulas = get_formulas_for_chunks(chunk_ids, limit=2) # Solo 2 formule max
        if formulas:
            sources.append(SourceItem(
                id="graph", 
                content="Formule collegate:\n" + "\n".join(formulas), 
                filename="KG", type="formula", tier="GRAPH"
            ))

    return sources, build_retrieval_audit_md(query_text, intent, timings, counts, [])

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
    """
    Core system prompt for the LLM.
    v2.2: Stronger grounding + Table-first reconstruction.
    """
    base = """
        ROLE: Senior Quantitative Financial Analyst.

        CORE OBJECTIVE: Answer the user's question using ONLY the provided context snippets.

        CONTEXT STRUCTURE:
        The context is provided in chunks with headers:
        `--- Fonte [n] — Filename — Pag X — (Type) ---`
        You MUST use these headers to locate page-specific info (e.g., "page 9", "Pag 9").

        ### NON-NEGOTIABLE GROUNDING RULES (ANTI-HALLUCINATION)
        0) You are NOT allowed to invent details.
        - If a detail (row/column/value/name) is not explicitly present in ANY chunk, you must say it is NOT available.
        - Do NOT “complete” tables from memory or common knowledge.

        1) **METADATA SENSITIVITY (CRITICAL)**:
        - If the user asks about a specific Page or Document, focus on chunks whose header matches that page/document.
        - If a table spans multiple chunks on the same page, merge them mentally.

        2) **TRUTH HIERARCHY**:
        - [TIER A - Methodology]: Highest priority.
        - [TIER B - Reference]: Operational details.
        - [TIER C - News/Rumors]: Only for temporal context (dates/recent events).
        - If Tier C contradicts Tier A, warn about conflict.

        3) **VISUAL/TABLE DATA HANDLING**:
        - Chunks labeled `(immagine)` / `(table)` / `(chart)` are AI-extracted descriptions of visual assets.
        - Treat them as factual ONLY within what they explicitly state.
        - If the user asks for “data in the table”, you MUST extract the rows/columns that are explicitly listed and present them.

        4) **TABLE RECONSTRUCTION (WHEN ASKED ABOUT A TABLE)**:
        - If the context includes a Markdown table, reproduce it (or the relevant subset).
        - If the context describes table rows in bullet form (“ASX 200 → AUD …”), you MUST reconstruct a Markdown table with columns.
        - If one chunk says “table not shown” but another chunk contains the table rows, prefer the chunk with the actual rows.
        - NO generic summaries: you must enumerate the table content.

        5) **FRAGMENT REASSEMBLY**:
        - If text continues across chunks (same page), treat it as continuous.
        - Do not complain about partial data unless the info is missing from ALL chunks.

        OUTPUT STRUCTURE:
        You MUST structure your response in two parts.
        
        PART 1: INTERNAL REASONING (Hidden from user, but crucial)
        Enclose this in <reasoning> tags.
        1. Analyze User Intent: Definition vs Calculation vs Lookup.
        2. Verify Tiers: Do I have Tier A chunks? If yes, prioritize them over Tier B/C.
        3. Check Conflicts: Does News (Tier C) contradict Methodology (Tier A)?
        4. Plan Citations: Ensure every assertion has a [Source ID].
        </reasoning>

        PART 2: FINAL RESPONSE (Visible to user)
        **A) Risposta**
        - Direct, technical answer in the USER'S LANGUAGE.
        - If the user asks about a table, include a reconstructed Markdown table.

        **B) Evidenze**
        - Bullet points citing the source ID(s), e.g. "[2] Pag 9 lists 10 indices and currencies."

        **C) Limiti**
        - State strictly what is missing (e.g., “the PDF table header is present but row values are not in the retrieved chunks”).

        **D) Fonti**
        - List filenames used.

        LANGUAGE RULE:
        You MUST respond EXCLUSIVELY in the same language as the user's question.
        """

    # Dynamic Intent Injection
    if intent == "formula":
        base += "\nINTENT: FORMULA. Prioritize formula fidelity. Reconstruct split formulas across chunks.\n"
    elif intent == "table":
        base += "\nINTENT: TABLE. The user wants the FULL DATA. Do NOT summarize. Output the complete Markdown table even if it is long.\n"
    elif intent == "chart":
        base += "\nINTENT: CHART/DATA. Extract explicit numbers/trends only. No estimation.\n"

    return base


def tier_guardrail_instructions(query_text: str) -> str:
    news = is_news_query(query_text)
    return (
        "GUARDRAILS TIER-FIRST (FINANCE-GRADE):\n"
        "1) Tier A: Primary source for definitions, theory, and mechanisms.\n"
        "2) Tier B: Examples, use cases, and applications.\n"
        "3) Tier C: Temporal context and recent events. ALWAYS specify dates if available.\n"
        "4) Grounding: Every statement must be supported by the provided context. Do not hallucinate.\n"
        "5) Gap Analysis: If sources are insufficient, state it explicitly in section C.\n"
        f"6) {'Query news: Use Tier C as the primary source for facts.' if news else 'Query standard: Use Tier C to provide updated context to Tier A/B data.'}\n"

        "Language rule:\n"
        "The final answer must always be written in the **SAME LANGUAGE** as the user's **QUESTION**.\n"
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

    ANALYTICS RULES:
    - User data provided in the prompt is your PRIMARY SOURCE.
    - Use rigorous mathematical/statistical logic.
    - If calculation is impossible, propose Python/R code.

    OUTPUT STRUCTURE (MANDATORY):
    Use ONLY the bold titles as headers.
    **A) Risposta**
    [Analisi dettagliata dei dati forniti]

    **B) Evidenze**
    [Passaggi logici o calcoli effettuati]

    **C) Limiti e Assunzioni**
    [Ipotesi statistiche o limiti dei dati forniti]

    **D) Fonti**
    [Indica 'Dati forniti dall'utente']

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
    """
    Rimuove artefatti tecnici se l'LLM ripete per errore i metadati nel testo.
    """
    if not text:
        return ""
    
    # 1. Rimuove pattern tipo "SourceID: 1" o "File: report.pdf" se compaiono nel testo
    text = re.sub(r"\[SourceID:\s*\d+.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r">>> SOURCE \[\d+\].*?\n", "", text, flags=re.IGNORECASE)
    
    # 2. Rimuove UUID tecnici residui (es. 42f22b...)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "", text)
    
    # 3. Pulisce eventuali tag rimasti orfani
    text = text.replace("Tier: A", "").replace("Tier: B", "").replace("Tier: C", "")
    
    return text.strip()


# =========================
# 🔄 STATE MANAGEMENT
# =========================
class State(rx.State):
    messages: List[ChatMessage] = [
        ChatMessage(
            id="init",
            role="assistant",
            content=f"Ciao! Sono attivo con **{LLM_MODEL_NAME}**. Metodologia Tier A, Ricerca Tier B e News Tier C caricate. Fammi domande sui tuoi documenti.",
        )
    ]
    input_text: str = ""
    is_processing: bool = False

    inline_open_for: str = ""
    inline_tab: str = "sources"

    vram_info: str = "N/A"
    vram_free: str = "N/A"
    backend_status: str = "OK"

    show_sources_modal: bool = False
    modal_sources: List[SourceItem] = []
    modal_debug_md: str = ""
    modal_title: str = ""

    def get_context_by_tier(self, query: str, tier: str) -> str:
        try:
            # Usa l'embedder globale già caricato per risparmiare RAM
            query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

            search_result = qdrant_client_inst.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="tier", match=models.MatchValue(value=tier))]
                ),
                limit=3
            )
            # Fondamentale: usa safe_payload_text che prova tutte le chiavi (text_sem, raw, ecc.)
            texts = []
            for res in search_result:
                p = res.payload or {}
                content = safe_payload_text(p)
                if content:
                    texts.append(content)

            return "\n".join(texts)
        except Exception as e:
            print(f"⚠️ Errore recupero Tier {tier}: {e}")
            return ""

    # --- Metodi di gestione UI ---
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

    def open_sources_audit(self, msg_id: str):

        self.modal_title = "Fonti & Audit"

        found = next((m for m in self.messages if m.id == msg_id), None)

        self.modal_sources = found.sources if found else []

        self.modal_debug_md = (found.debug_md or "") if found else ""

        self.show_sources_modal = True

    def close_sources_audit(self):
        self.show_sources_modal = False

    def on_load(self):
        self.refresh_gpu()
        self.refresh_backend_status()

    def refresh_backend_status(self):
        self.backend_status = "OK" if llm_client else "DEGRADED"

    def refresh_gpu(self):
        self.vram_info = gpu_free_info()
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                self.vram_free = f"{free_bytes / (1024**3):.1f} GB free"
            except: self.vram_free = "N/A"
        else: self.vram_free = "CPU"

    def clear_history(self):
        self.messages = [self.messages[0]]

    def set_input_text(self, text: str):
        self.input_text = text

    # ✅ ORA INDENTATO CORRETTAMENTE DENTRO LA CLASSE

    async def handle_submit(self):
        # Import necessario per la gestione asincrona della UI
        import asyncio 

        if not self.input_text.strip() or self.is_processing:
            return

        user_query = self.input_text.strip()
        self.input_text = ""
        self.is_processing = True
        
        # English instructions for the model
        language_reminder = "\n\nCRITICAL: You MUST detect the language of the user's question and answer EXCLUSIVELY in that same language."

        try:
            self.refresh_gpu()
            # 1. Mostra subito il messaggio dell'utente nella chat
            self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", content=user_query))
            yield rx.scroll_to("chat_bottom")
            
            # --- FIX CRITICO: Pausa per aggiornare la UI ---
            # Senza questo, l'app sembra bloccata finché il RAG non finisce i calcoli.
            # 0.1 secondi sono sufficienti a Reflex per renderizzare il messaggio a video.
            await asyncio.sleep(0.1) 
            # -----------------------------------------------

            intent = detect_intent(user_query)
            analytics_mode = is_user_data_analytics(user_query)

            # Variabili per il payload
            system_instructions = ""
            final_user_content = ""
            debug_md = ""
            sources = []

            if analytics_mode:
                sources = make_analytics_sources(user_query)
                debug_md = "### 🔎 Audit (Analytics Mode)\n- retrieval: **bypassed**\n- source: **USER_INPUT**"
                system_instructions = build_system_instructions_analytics(intent)
                
                # In Analytics Mode, i dati sono nella domanda stessa
                final_user_content = f"### QUESTION ###\n{user_query}{language_reminder}"
            else:
                # 1. RECUPERO DATI (Hybrid Search + Rerank)
                # Qui avviene il calcolo pesante che prima bloccava tutto
                sources, debug_md = retrieve_v2(user_query)
                
                # 2. RAGGRUPPAMENTO FONTI
                c_a_list, c_b_list, c_c_list, c_g_list = [], [], [], []

                for i, s in enumerate(sources, start=1):
                    header = f"--- Fonte [{i}] — {s.filename} — Pag {s.page} — ({s.type}) ---\n"
                    meta = f"(tier={s.tier} | db={s.db_origin})\n"
                    body = (s.content or "").strip()

                    if not body:
                        continue

                    snippet = header + meta + body + "\n\n"

                    if s.tier == "A":
                        c_a_list.append(snippet)
                    elif s.tier == "B":
                        c_b_list.append(snippet)
                    elif s.tier == "C":
                        c_c_list.append(snippet)
                    elif s.tier == "GRAPH":
                        c_g_list.append(snippet)

                c_a = "".join(c_a_list).strip()
                c_b = "".join(c_b_list).strip()
                c_c = "".join(c_c_list).strip()
                c_g = "".join(c_g_list).strip()

                # 3. PROMPT DI SISTEMA
                system_instructions = build_system_instructions(intent)

                # Aggiunta audit nel debug visivo
                debug_md += (
                    f"\n\n### 🛡️ Tier Context Check\n"
                    f"- Tier A (Methodology): {'✅ Presente' if c_a else '❌ Assente'}\n"
                    f"- Tier B (Research): {'✅ Presente' if c_b else '❌ Assente'}\n"
                    f"- Tier C (News): {'✅ Presente' if c_c else '❌ Assente'}"
                )

                # 4. ASSEMBLAGGIO CONTENUTO UTENTE
                final_user_content = (
                    f"### METHODOLOGY [TIER A] ###\n{c_a if c_a else 'No specific methodology found.'}\n\n"
                    f"### RESEARCH [TIER B] ###\n{c_b if c_b else 'No specific research found.'}\n\n"
                    f"### NEWS & EVENTS [TIER C] ###\n{c_c if c_c else 'No recent news found.'}\n\n"
                    f"### KNOWLEDGE GRAPH [NEO4J] ###\n{c_g if c_g else 'No relational/formula data.'}\n\n"
                    f"### USER QUESTION ###\n{user_query}\n"
                    f"{language_reminder}"
                )
            
            # --- COSTRUZIONE PAYLOAD CHAT ---
            messages_payload = build_alternating_history(self.messages, MEMORY_LIMIT)
            
            if messages_payload and messages_payload[-1]["role"] == "user":
                messages_payload.pop()
            
            messages_payload = [m for m in messages_payload if m["role"] != "system"]

            final_messages = [
                {"role": "system", "content": system_instructions}
            ] + messages_payload + [
                {"role": "user", "content": final_user_content}
            ]


            # Aggiunge subito un messaggio "placeholder" (senza fonti) così la UI non sembra bloccata
            assistant_id = str(uuid.uuid4())
            self.messages.append(
                ChatMessage(
                    id=assistant_id,
                    role="assistant",
                    content="⏳ Sto generando la risposta…",
                    sources=[],          # ✅ NON mostrare fonti subito
                    debug_md=""          # ✅ audit dopo
                )
            )
            yield rx.scroll_to("chat_bottom")
            yield  # ✅ forza refresh UI

            # --- BLOCCO UNICO DI GENERAZIONE CORRETTO ---
            # --- BLOCCO UNICO DI GENERAZIONE (FIXATO) ---
            full_resp = ""
            if llm_client:
                # Usiamo extra_body per passare i parametri OLLAMA (memoria estesa)
                stream = llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME, 
                    messages=final_messages, 
                    temperature=0.0, # Temperatura 0 per precisione sui numeri
                    stream=True,
                    extra_body={
                        "options": {
                            "num_ctx": 8192,       # <--- ESTENDE LA MEMORIA (Evita tagli documenti)
                            "num_predict": 4096,   # Lunghezza massima risposta
                            "repeat_penalty": 1.1  # Riduce le ripetizioni
                        }
                    }
                )
                
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        full_resp += delta.content
                        self.messages[-1].content = strip_id_leaks(full_resp)
                        yield

                # ✅ SOLO ALLA FINE agganciamo le fonti e l'audit
                # Questo evita sfarfallii o popup vuoti durante la generazione
                self.messages[-1].sources = sources
                self.messages[-1].debug_md = debug_md
                yield
            else:
                self.messages[-1].content = "⚠️ LLM non inizializzato. Verifica che Ollama sia attivo."
                self.messages[-1].sources = sources
                self.messages[-1].debug_md = debug_md
                yield

        finally:
            self.is_processing = False
            self.refresh_gpu()

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
                # Pulsante "Info" in alto a destra nel messaggio
                rx.cond(
                    is_bot & (msg.sources.length() > 0),
                    rx.button(
                        rx.hstack(
                            rx.icon("info", size=14),
                            rx.text("Dettagli Ricerca", size="1"),
                            spacing="2",
                        ),
                        variant="soft",
                        color_scheme="gray",
                        size="1",
                        on_click=lambda: State.open_sources_audit(msg.id),
                    ),
                    rx.box(),
                ),
                width="100%",
                align_items="center",
                spacing="2",
            ),
            # Contenuto del Messaggio
            rx.markdown(
                msg.content,
                width="100%",
                overflow_wrap="anywhere",
                word_break="break-word",
            ),
            
            # Badge rapidi sotto il testo (Opzionale, richiama la funzione helper)
            rx.cond(
                is_bot & (msg.sources.length() > 0),
                render_inline_sources(msg)
            ),

            spacing="2",
            width="100%",
        ),

        # ---- Inline popup "Fonti + Audit" sotto la risposta LLM ----
        rx.cond(
            is_bot & ((msg.sources.length() > 0) | (msg.debug_md.length() > 0)),
            rx.box(
                # barra azioni (Pulsanti Fonti / Audit)
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

                # --- PANNELLO ESPANSO ---
                rx.cond(
                    State.inline_open_for == msg.id,
                    rx.box(
                        rx.cond(
                            State.inline_tab == "sources",
                            
                            # === SEZIONE FONTI (FIXATA: NESSUN LOOP SU STATE.MESSAGES) ===
                            rx.scroll_area(
                                rx.vstack(
                                    rx.text("📚 Fonti Documentali correlate:", font_weight="bold", size="2", margin_bottom="0.5em"),
                                    rx.foreach(
                                        msg.sources,
                                        lambda s: rx.card(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.badge(s.tier, color_scheme="red", variant="soft"),
                                                    rx.badge(s.db_origin, color_scheme="violet", variant="outline"),
                                                    rx.text(f"{s.filename}", size="1", weight="bold"),
                                                    rx.spacer(),
                                                    rx.text(f"Pag. {s.page}", size="1"),
                                                    width="100%",
                                                ),
                                                rx.text(s.content, size="1", line_clamp=3, font_style="italic", color_scheme="gray"),
                                                spacing="1",
                                                width="100%",
                                            ),
                                            variant="ghost",
                                            width="100%",
                                            margin_bottom="0.5em",
                                        )
                                    ),
                                    spacing="2",
                                    width="100%",
                                ),
                                height="260px",
                                type="always",
                            ),
                            
                            # === SEZIONE AUDIT ===
                            rx.box(
                                rx.heading("Audit & Reasoning", size="3", margin_bottom="0.5em"),
                                rx.scroll_area(
                                    rx.markdown(
                                        msg.debug_md,
                                        width="100%",
                                        overflow_wrap="anywhere",
                                        word_break="break-word",
                                    ),
                                    height="260px",
                                    type="always",
                                ),
                                width="100%",
                            ),
                        ),

                        # Footer del pannello (Pulsante Chiudi)
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
                    rx.box(), # Else block del pannello espanso (vuoto)
                ),
                width="100%",
            ),
            rx.box(), # Else block del pulsante espansione (vuoto)
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


def render_inline_sources(msg: ChatMessage):
    """Visualizza i badge sintetici delle fonti sotto il messaggio."""
    return rx.flex(
        rx.foreach(
            msg.sources,
            lambda s: rx.badge(
                rx.hstack(
                    rx.icon("database", size=12),
                    # FIX: Passiamo i valori come argomenti separati a rx.text
                    # invece di usare una f-string che può causare errori su Var
                    rx.text(s.db_origin, ": ", s.filename, " (p.", s.page, ")", size="1"),
                    align_items="center",
                    spacing="1",
                ),
                variant="soft",
                color_scheme="indigo",
                margin_right="0.5em",
                margin_bottom="0.2em",
                cursor="pointer",
                # Cliccando sul badge si apre il pannello dettagli
                on_click=lambda: State.toggle_inline_sources(msg.id), 
            )
        ),
        wrap="wrap",
        margin_top="0.5em",
    )

def render_inline_audit(msg: ChatMessage):
    """Visualizza il log di ragionamento (Audit) sotto il messaggio."""
    return rx.box(
        rx.markdown(msg.debug_md),
        background_color="#FFFBEB",
        padding="1rem",
        border_radius="md",
        margin_top="0.5rem",
        border_left="4px solid #F6AD55",
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

                    # ====== FONTI (Visualizzazione pulita e mirata) ======
                    rx.cond(
                        State.modal_sources.length() > 0,
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    State.modal_sources,
                                    lambda s: rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.badge(s.tier, color_scheme="tomato", variant="surface"),
                                                rx.badge(s.db_origin, color_scheme="plum", variant="outline"),
                                                rx.text(f"Doc: {s.filename}", weight="bold", size="2"),
                                                width="100%",
                                                justify="between",
                                            ),
                                            rx.text(s.content, size="1", line_clamp=3),
                                            rx.hstack(
                                                rx.text(f"Pagina: {s.page}", size="1", color_scheme="gray"),
                                                rx.spacer(),
                                                rx.text(f"Score: {s.score}", size="1", color_scheme="gray"),
                                                width="100%",
                                            ),
                                            spacing="2",
                                        ),
                                        width="100%",
                                        margin_bottom="2",
                                    )
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            height="400px",
                            type="always",
                        ),
                        rx.center(rx.text("Nessuna fonte trovata per questo messaggio.", color="gray")),
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
