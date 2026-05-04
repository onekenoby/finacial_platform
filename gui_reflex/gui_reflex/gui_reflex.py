from gitdb import stream
from pydantic_settings import sources
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

import threading
_init_lock = threading.Lock()


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
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma4:latest")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-vl:8b")

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

# In UI conviene partire con evaluation disabilitata.
# La faithfulness può essere eseguita dopo, offline o con un bottone dedicato.
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "0") == "1"

# Può essere lo stesso modello, ma idealmente sarebbe un modello diverso usato come judge.
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", LLM_MODEL_NAME)

# =========================
# 🧾 LOG PATHS - fuori dalla cartella progetto Reflex
# =========================

LOG_DIR = os.getenv(
    "RAG_LOG_DIR",
    os.path.join(os.path.expanduser("~"), "ai_rag_logs")
)

os.makedirs(LOG_DIR, exist_ok=True)

AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1") == "1"
AUDIT_LOG_PATH = os.getenv(
    "AUDIT_LOG_PATH",
    os.path.join(LOG_DIR, "rag_audit.jsonl")
)

EVAL_LOG_PATH = os.getenv(
    "EVAL_LOG_PATH",
    os.path.join(LOG_DIR, "rag_eval_log.jsonl")
)

EVAL_MAX_CONTEXT_CHARS = int(os.getenv("EVAL_MAX_CONTEXT_CHARS", "12000"))

# Soglie KPI
EVAL_MIN_FAITHFULNESS = float(os.getenv("EVAL_MIN_FAITHFULNESS", "0.75"))
EVAL_MIN_ANSWER_RELEVANCE = float(os.getenv("EVAL_MIN_ANSWER_RELEVANCE", "0.70"))

# Se 1, blocca/sostituisce risposte giudicate non fedeli.
# Per iniziare ti consiglio 0: prima osservi le metriche, poi eventualmente blocchi.
EVAL_STRICT_BLOCK = os.getenv("EVAL_STRICT_BLOCK", "0") == "1"



# ============================================================
# 🧠 CARICAMENTO RISORSE AI & DB (SINGLETON PATTERN)
# ============================================================

# Inizializzazione variabili globali a None per caricamento Lazy/Controllato
embedder = None
reranker = None
llm_client = None
qdrant_client_inst = None
neo4j_driver = None
pg_pool = None

# Device selection (già definiti nel tuo script, ma assicurati siano accessibili)
device_embed = "cuda" if torch.cuda.is_available() else "cpu"
device_rerank = "cpu" 

def init_resources():
    """
    Inizializza i modelli e le connessioni ai database in un unico passaggio.
    Previene il caricamento duplicato durante la compilazione del frontend Reflex.
    """
    global embedder, reranker, llm_client, qdrant_client_inst, neo4j_driver, pg_pool

    # Se l'embedder è già istanziato, saltiamo per non saturare la VRAM
    if embedder is not None:
        return

    print("\n" + "═"*60)
    print("⏳ [BACKEND] Avvio inizializzazione modelli e database...")
    print("═"*60)

    try:
        # 1. Embedding Model (BGE-M3) - Caricato su CUDA se disponibile
        print(f"🚀 Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on {device_embed.upper()}...")
        embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            device=device_embed, 
            local_files_only=True
        )

        # 2. Reranker Model - Forzato su CPU per non competere con l'LLM
        print(f"🚀 Loading Reranker ({RERANKER_MODEL_NAME}) on {device_rerank.upper()}...")
        reranker = CrossEncoder(
            RERANKER_MODEL_NAME, 
            device=device_rerank
        )

        # 3. LLM Connection (Ollama / OpenAI Compatible)
        print(f"🚀 Connecting to LLM via Ollama ({LLM_MODEL_NAME}) at {OLLAMA_URL}...")
        llm_client = OpenAI(base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY)

        # 4. Qdrant (Vector DB)
        print(f"🌌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        qdrant_client_inst = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # 5. Neo4j (Graph DB)
        print(f"🕸️ Connecting to Neo4j Graph at {NEO4J_URI}...")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        neo4j_driver.verify_connectivity()

        # 6. Postgres Pool (TimescaleDB)
        if PG_ENRICH_ENABLED:
            print(f"🐘 Initializing Postgres Pool ({PG_HOST})...")
            pg_pool = SimpleConnectionPool(
                PG_MIN_CONN, PG_MAX_CONN,
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS
            )
            # Smoke test per validare la connessione
            conn = pg_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            finally:
                pg_pool.putconn(conn)

        print("✅ [BACKEND] Risorse caricate con successo.")
        print("═"*60 + "\n")

    except Exception as e:
        print(f"❌ [ERRORE] Fallimento inizializzazione: {e}")
        # Reset variabili per permettere retry se necessario
        embedder = None
        # Non blocchiamo l'esecuzione dell'intera app, ma il RAG non funzionerà
        
# --- ESECUZIONE SELETTIVA ---
# REFLEX_RELOAD viene impostato durante l'hot-reload del server di sviluppo.
# Questo controllo assicura che il caricamento pesante avvenga solo nel processo worker.
if not os.environ.get("REFLEX_RELOAD"):
    init_resources()


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

class RagEvalResult(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_support: float = 0.0
    hallucination_risk: float = 1.0
    source_scope_violation: bool = False
    verdict: str = "UNKNOWN"
    unsupported_claims: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    reason: str = ""


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
        "calcola", "calculate", "stima", "estimate", "totale", "total", "somma", "sum","analizza","analyse","analyze",
        
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


def normalize_tier_value(tier: str) -> str:
    """
    Normalizza i tier in valori canonici:
    A, B, C, GRAPH, USER oppure C come fallback.
    Evita bug tipo: 'GRAPH' contiene la lettera 'A' e viene scambiato per Tier A.
    """
    t = (tier or "").strip().upper()

    if not t:
        return "C"

    if t == "GRAPH" or t.startswith("GRAPH"):
        return "GRAPH"

    if t == "USER" or t.startswith("USER"):
        return "USER"

    if t == "A" or t == "TIER_A_METHODOLOGY" or t.endswith("_A_METHODOLOGY"):
        return "A"

    if t == "B" or t == "TIER_B_REFERENCE" or t.endswith("_B_REFERENCE"):
        return "B"

    if t == "C" or t == "TIER_C_NEWS" or t.endswith("_C_NEWS") or "NEWS" in t:
        return "C"

    return t

def tier_score_delta(tier: str, query_text: str) -> float:
    """
    Applica boost/penalty in modo sicuro sui tier normalizzati.
    Nota importante:
    - non usare mai 'if "A" in tier', perché 'GRAPH' contiene la lettera A.
    """
    t = normalize_tier_value(tier)

    if t == "A":
        return TIER_BOOST_A

    if t == "B":
        return TIER_BOOST_B

    if t == "C":
        if TIER_C_PENALTY_IF_NOT_NEWS and not is_news_query(query_text):
            return -TIER_PENALTY_C
        return 0.0

    # GRAPH, USER, UNKNOWN: nessun boost metodologico
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
    """
    Recupera le entità collegate ai chunk.
    Coerente con ingestion:
    - Entity -> Chunk usa PRESENT_IN
    - Mantiene compatibilità anche con vecchie ingestion che usavano MENTIONED_IN
    """
    if not chunk_ids or not neo4j_driver:
        return {}

    graph_map: Dict[str, List[GraphEntity]] = {}

    query = """
    MATCH (e:Entity)-[r:PRESENT_IN|MENTIONED_IN]->(c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) IN $ids
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(e.name, e.label, e.id) AS name,
        coalesce(e.category, labels(e)[0], 'Entity') AS type,
        type(r) AS rel
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
    """
    Recupera formule collegate ai chunk.
    Coerente con ingestion:
    - Formula -> Chunk usa MENTIONED_IN
    """
    if not chunk_ids or not neo4j_driver:
        return []

    query = """
    MATCH (f:Formula)-[:MENTIONED_IN]->(c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) IN $ids
    RETURN
        f.latex AS latex,
        f.plain AS plain,
        f.meaning_it AS meaning
    LIMIT $lim
    """

    out: List[str] = []

    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit)

            for r in res:
                latex = (r["latex"] or "").strip()
                plain = (r["plain"] or "").strip()
                meaning = (r["meaning"] or "").strip()

                parts = []

                if latex:
                    parts.append(f"LaTeX: {latex}")

                if plain:
                    parts.append(f"Plain: {plain}")

                if meaning:
                    parts.append(f"Meaning: {meaning}")

                if parts:
                    out.append(" | ".join(parts))

    except Exception as e:
        print(f"⚠️ Neo4j Query Error (formulas): {e}")

    return out


def get_neighbor_chunk_ids(chunk_ids: List[str], limit: int = GRAPH_MAX_NEIGHBOR_CHUNKS) -> List[str]:
    """
    Espande semanticamente i chunk usando entità condivise nel grafo.
    Coerente con ingestion:
    - Entity -> Chunk usa PRESENT_IN
    - Compatibile anche con MENTIONED_IN per vecchi dati
    """
    if not chunk_ids or not neo4j_driver:
        return []

    query = """
    MATCH (c1:Chunk)<-[:PRESENT_IN|MENTIONED_IN]-(e:Entity)-[:PRESENT_IN|MENTIONED_IN]->(c2:Chunk)
    WHERE coalesce(c1.chunk_id, c1.id) IN $ids
      AND NOT coalesce(c2.chunk_id, c2.id) IN $ids
      AND NOT coalesce(e.type, e.category, labels(e)[0], '') IN ['Generic', 'Year', 'Date']

    WITH c2, count(DISTINCT e) AS strength
    WHERE strength >= 2

    RETURN coalesce(c2.chunk_id, c2.id) AS cid
    ORDER BY strength DESC
    LIMIT $lim
    """

    out: List[str] = []

    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit)
            out = [str(r["cid"]) for r in res if r.get("cid")]

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

    # 📄 SEZIONE DOCUMENT SCOPE
    if counts.get("requested_doc"):
        lines.append("\n#### 📄 Document Scope")
        lines.append(f"- Documento richiesto: `{counts.get('requested_doc')}`")
        lines.append(f"- Chunk trovati nel documento: **{counts.get('doc_scope_hits', 0)}**")
        lines.append(f"- Prima del filtro documento: **{counts.get('doc_scope_before', 0)}**")
        lines.append(f"- Dopo il filtro documento: **{counts.get('doc_scope_after', 0)}**")

    # 🕸️ SEZIONE NEO4J (Grafo)
    neo4j_direct = counts.get("neo4j_direct_hits", 0)
    neo4j_expanded = counts.get("neo4j_hits", 0)
    final_formulas = counts.get("final_formulas", 0)

    if (
        neo4j_direct > 0
        or neo4j_expanded > 0
        or final_formulas > 0
        or "graph" in timings
        or "neo4j_direct_search" in timings
    ):
        lines.append("\n#### 🕸️ Neo4j (Graph Search / Expansion)")

        if "neo4j_direct_search" in timings:
            lines.append(f"- Tempo direct search: **{ms(timings['neo4j_direct_search'])}**")

        if "graph" in timings:
            lines.append(f"- Tempo graph expansion: **{ms(timings['graph'])}**")

        lines.append(f"- Chunk trovati da Neo4j direct search: **{neo4j_direct}**")
        lines.append(f"- Chunk aggiunti da graph expansion: **{neo4j_expanded}**")
        lines.append(f"- Formule collegate recuperate: **{final_formulas}**")

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
    Recupera da Postgres i chunk usando l'ID corretto: chunk_uuid.

    Ritorna:
    {
        chunk_uuid: {
            "chunk_uuid": ...,
            "content_raw": ...,
            "content_semantic": ...,
            "metadata_json": ...,
            "ingestion_ts": ...
        }
    }

    Nota:
    - chunk_uuid corrisponde all'id usato in Qdrant.
    - chunk_uuid corrisponde al chunk_id usato in Neo4j.
    - prende sempre la versione più recente del chunk in base a ingestion_ts.
    """
    if not PG_ENRICH_ENABLED or not pg_pool or not chunk_uuids:
        return {}

    # Dedup preservando l'ordine
    seen = set()
    uuids: List[str] = []

    for u in chunk_uuids:
        if not u:
            continue

        key = str(u).strip()
        if not key or key in seen:
            continue

        seen.add(key)
        uuids.append(key)

    if not uuids:
        return {}

    sql = """
    WITH wanted(chunk_uuid) AS (
        VALUES %s
    ),
    ranked AS (
        SELECT
            d.chunk_uuid::text AS chunk_uuid,
            d.content_raw,
            d.content_semantic,
            d.metadata_json,
            d.ingestion_ts,
            ROW_NUMBER() OVER (
                PARTITION BY d.chunk_uuid
                ORDER BY d.ingestion_ts DESC
            ) AS rn
        FROM public.document_chunks d
        JOIN wanted w
          ON d.chunk_uuid::text = w.chunk_uuid::text
    )
    SELECT
        chunk_uuid,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts
    FROM ranked
    WHERE rn = 1;
    """

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                sql,
                [(u,) for u in uuids]
            )
            rows = cur.fetchall()

        out: Dict[str, Dict[str, Any]] = {}

        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts in rows:
            # metadata_json può arrivare già come dict oppure come stringa JSON
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out[str(chunk_uuid)] = {
                "chunk_uuid": str(chunk_uuid),
                "content_raw": content_raw or "",
                "content_semantic": content_semantic or "",
                "metadata_json": metadata_json,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            }

        return out

    except Exception as e:
        print(f"⚠️ PG enrich by chunk_uuid error: {e}")
        return {}

    finally:
        pg_pool.putconn(conn)


def search_pg_bm25(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca keyword/BM25-like su Postgres usando full-text search.
    Ritorna chunk identificati da chunk_uuid, coerenti con Qdrant e Neo4j.
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    if not query_text or not query_text.strip():
        return []

    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", query_text.lower())
    tokens = [t for t in tokens if len(t) > 3]

    if not tokens:
        return []

    pg_query = " | ".join(tokens)

    sql = """
    WITH q AS (
        SELECT to_tsquery('simple', %s) AS tsq
    )
    SELECT
        chunk_uuid::text,
        content_raw,
        content_semantic,
        metadata_json,
        ts_rank_cd(
            to_tsvector(
                'simple',
                COALESCE(content_semantic, '') || ' ' ||
                COALESCE(content_raw, '') || ' ' ||
                COALESCE(metadata_json::text, '')
            ),
            q.tsq
        ) AS rank
    FROM public.document_chunks, q
    WHERE
        to_tsvector(
            'simple',
            COALESCE(content_semantic, '') || ' ' ||
            COALESCE(content_raw, '') || ' ' ||
            COALESCE(metadata_json::text, '')
        ) @@ q.tsq
    ORDER BY rank DESC
    LIMIT %s;
    """

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (pg_query, limit))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []

        for chunk_uuid, content_raw, content_semantic, metadata_json, rank in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out.append({
                "id": str(chunk_uuid),
                "content": content_semantic or content_raw or "",
                "metadata": metadata_json,
                "score": float(rank or 0.0),
            })

        return out

    except Exception as e:
        print(f"⚠️ BM25 Error: {e}")
        return []

    finally:
        pg_pool.putconn(conn)

# =========================
# 🔍 RAG v2 Retrieval
# =========================


def apply_rrf_scoring(candidates: List[Dict[str, Any]], k: int = 60):
    """
    Reciprocal Rank Fusion tra:
    - Qdrant vector rank
    - Postgres BM25 rank
    - Neo4j graph rank
    """

    for c in candidates:
        c["rrf_score"] = 0.0

    vec_sorted = sorted(
        [c for c in candidates if c.get("score_vec", c.get("score_base", 0.0)) > 0],
        key=lambda x: x.get("score_vec", x.get("score_base", 0.0)),
        reverse=True,
    )

    bm25_sorted = sorted(
        [c for c in candidates if c.get("score_bm25", 0.0) > 0],
        key=lambda x: x.get("score_bm25", 0.0),
        reverse=True,
    )

    graph_sorted = sorted(
        [c for c in candidates if c.get("score_graph", 0.0) > 0],
        key=lambda x: x.get("score_graph", 0.0),
        reverse=True,
    )

    for rank, item in enumerate(vec_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(graph_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)


RAG_STOPWORDS = {
    # italian
    "della", "delle", "degli", "dello", "dalla", "dalle", "dagli",
    "nella", "nelle", "negli", "nello", "alla", "alle", "agli",
    "sono", "presenti", "presente", "ciascuna", "ciascuno",
    "quale", "quali", "cosa", "come", "dove", "quando",
    "spiega", "spiegami", "riporta", "riportale", "mostra",
    "documento", "file", "fonte", "fonti",

    # english
    "what", "which", "where", "when", "explain", "show",
    "report", "document", "file", "source", "sources",
    "present", "available", "each", "about",

    # rag/formula generic terms
    "formula", "formule", "matematica", "matematiche",
    "latex", "concetto", "riferisce"
}


def extract_rag_tokens(query_text: str) -> List[str]:
    """
    Estrae token utili per:
    - filename matching
    - Neo4j direct search
    - formula lookup
    Evita stopword tipo 'sono', 'presenti', 'ciascuna'.
    """
    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", query_text or "")
        if len(t) > 3
    ]

    return [t for t in tokens if t not in RAG_STOPWORDS]


def search_neo4j_entities(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca diretta nel grafo Neo4j sui nodi Entity.

    Versione pulita:
    - non usa proprietà Neo4j inesistenti come source_name/content_semantic/content_raw
    - ritorna chunk_id coerente con chunk_uuid usato da Qdrant/Postgres
    - lascia a Postgres il compito di arricchire il contenuto completo
    """
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)

    if not tokens:
        return []

    cypher = """
    MATCH (e:Entity)
    WHERE any(tok IN $tokens WHERE
        toLower(coalesce(e.name, '')) CONTAINS tok OR
        toLower(coalesce(e.id, '')) CONTAINS tok OR
        toLower(coalesce(e.label, '')) CONTAINS tok OR
        toLower(coalesce(e.description, '')) CONTAINS tok OR
        any(s IN coalesce(e.synonyms, []) WHERE toLower(s) CONTAINS tok)
    )
    MATCH (e)-[r]-(c:Chunk)
    WITH e, c, count(r) AS rel_count
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.chunk_index, 0) AS chunk_index,
        coalesce(c.text, c.content, '') AS content,
        labels(e)[0] AS entity_label,
        coalesce(e.name, e.label, e.id) AS entity_name,
        rel_count
    ORDER BY rel_count DESC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []

    try:
        with neo4j_driver.session() as session:
            rows = session.run(cypher, tokens=tokens, limit=limit)

            for r in rows:
                cid = r.get("chunk_id")

                if not cid:
                    continue

                out.append({
                    "id": str(cid),
                    "content": r.get("content") or "",
                    "filename": r.get("filename") or "Neo4j",
                    "page": int(r.get("page") or 0),
                    "type": "graph",
                    "tier": "GRAPH",
                    "score_graph": float(r.get("rel_count") or 1.0),
                    "origin": f"Neo4j Entity Search: {r.get('entity_name')}",
                    "section_hint": f"Entity: {r.get('entity_name')}",
                })

    except Exception as e:
        print(f"⚠️ Neo4j direct search error: {e}")

    return out

def search_neo4j_formulas(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca diretta delle formule nel Knowledge Graph.

    Utile per domande come:
    - quali formule sono presenti nel documento?
    - riportale in LaTeX
    - quali equazioni usa il documento?
    """
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)

    if not tokens:
        return []

    cypher = """
    MATCH (f:Formula)-[:MENTIONED_IN]->(c:Chunk)
    WHERE any(tok IN $tokens WHERE
        toLower(coalesce(c.filename, '')) CONTAINS tok OR
        toLower(coalesce(f.latex, '')) CONTAINS tok OR
        toLower(coalesce(f.plain, '')) CONTAINS tok OR
        toLower(coalesce(f.meaning_it, '')) CONTAINS tok
    )
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.chunk_index, 0) AS chunk_index,
        coalesce(f.latex, '') AS latex,
        coalesce(f.plain, '') AS plain,
        coalesce(f.meaning_it, '') AS meaning,
        count(*) AS rel_count
    ORDER BY page ASC, chunk_index ASC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []

    try:
        with neo4j_driver.session() as session:
            rows = session.run(cypher, tokens=tokens, limit=limit)

            for r in rows:
                cid = r.get("chunk_id")

                if not cid:
                    continue

                latex = (r.get("latex") or "").strip()
                plain = (r.get("plain") or "").strip()
                meaning = (r.get("meaning") or "").strip()

                formula_parts = []

                if latex:
                    formula_parts.append(f"LaTeX: {latex}")

                if plain:
                    formula_parts.append(f"Plain: {plain}")

                if meaning:
                    formula_parts.append(f"Meaning: {meaning}")

                if not formula_parts:
                    continue

                out.append({
                    "id": str(cid),
                    "content": "Formula from Knowledge Graph:\n" + "\n".join(formula_parts),
                    "filename": r.get("filename") or "Neo4j",
                    "page": int(r.get("page") or 0),
                    "type": "formula",
                    "tier": "GRAPH",
                    "score_graph": float(r.get("rel_count") or 5.0),
                    "origin": "Neo4j Formula Search",
                    "section_hint": "Formula node",
                })

    except Exception as e:
        print(f"⚠️ Neo4j formula search error: {e}")

    return out


def normalize_doc_name(value: str) -> str:
    """
    Normalizza un nome documento per confronti robusti:
    - lowercase
    - rimuove estensioni
    - rimuove caratteri non alfanumerici
    - rimuove suffissi tecnici comuni tipo _out / output
    """
    if not value:
        return ""

    v = os.path.basename(str(value).lower().strip())

    v = re.sub(r"\.(pdf|md|txt|docx|html)$", "", v)
    v = re.sub(r"[_\-\s]+out$", "", v)
    v = re.sub(r"[_\-\s]+output$", "", v)
    v = re.sub(r"[^a-z0-9]+", "", v)

    return v


def extract_requested_document(query_text: str) -> str:
    """
    Estrae il documento richiesto dalla query in modo sicuro.
    Evita falsi positivi come "il documento consiglia...".
    """
    q = query_text or ""

    patterns = [
        # 1. Nome tra virgolette o apici: nel documento "Trading_Tesi"
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+["\']([^"\']+)["\']',
        
        # 2. Nome con estensione esplicita: file report.pdf
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+([A-Za-z0-9_\-\.]+\.(?:pdf|md|txt|docx|csv|html))\b',
        
        # 3. Nome tecnico con underscore o trattini: documento TRADING_ALGORITMICO
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+([A-Za-z0-9]+[_\-][A-Za-z0-9_\-\.]+)\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,:;!?\"'")

    return ""

def candidate_matches_requested_doc(candidate: Dict[str, Any], requested_doc: str) -> bool:
    """
    Verifica se un candidato appartiene al documento richiesto.
    """
    if not requested_doc:
        return True

    wanted = normalize_doc_name(requested_doc)
    if not wanted:
        return True

    filename = normalize_doc_name(candidate.get("filename", ""))

    # Match robusto nei due versi
    return wanted in filename or filename in wanted

def search_pg_by_document_scope(
    requested_doc: str,
    query_text: str,
    limit: int = 80
) -> List[Dict[str, Any]]:
    """
    Recupera chunk da Postgres appartenenti al documento richiesto,
    indipendentemente dal fatto che siano entrati nei primi risultati BM25 generici.

    Serve per evitare falsi negativi quando l'utente chiede:
    "nel documento X..."
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    wanted_norm = normalize_doc_name(requested_doc)

    if not wanted_norm:
        return []

    sql = """
    WITH q AS (
        SELECT plainto_tsquery('simple', %s) AS tsq
    ),
    ranked AS (
        SELECT
            d.chunk_uuid::text AS chunk_uuid,
            d.content_raw,
            d.content_semantic,
            d.metadata_json,
            d.ingestion_ts,

            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(
                            coalesce(
                                d.metadata_json->>'filename',
                                d.metadata_json->>'source_name',
                                ''
                            )
                        ),
                        '\\.(pdf|md|txt|docx|html)$',
                        '',
                        'g'
                    ),
                    '[_\\-\\s]+(out|output)$',
                    '',
                    'g'
                ),
                '[^a-z0-9]+',
                '',
                'g'
            ) AS filename_norm,

            ts_rank_cd(
                to_tsvector(
                    'simple',
                    coalesce(d.content_semantic, '') || ' ' ||
                    coalesce(d.content_raw, '') || ' ' ||
                    coalesce(d.metadata_json::text, '')
                ),
                q.tsq
            ) AS rank,

            row_number() OVER (
                PARTITION BY d.chunk_uuid
                ORDER BY d.ingestion_ts DESC
            ) AS rn

        FROM public.document_chunks d, q
    )
    SELECT
        chunk_uuid,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts,
        rank
    FROM ranked
    WHERE rn = 1
      AND length(filename_norm) > 0
      AND (
            filename_norm LIKE %s
            OR %s LIKE ('%%' || filename_norm || '%%')
      )
    ORDER BY rank DESC, ingestion_ts DESC
    LIMIT %s;
    """

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_text,
                    f"%{wanted_norm}%",
                    wanted_norm,
                    limit,
                )
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []

        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts, rank in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out.append({
                "id": str(chunk_uuid),
                "content": content_semantic or content_raw or "",
                "metadata": metadata_json,
                "score": float(rank or 0.001),
                "origin": "PostgresDocScope",
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            })

        return out

    except Exception as e:
        print(f"⚠️ PG document scope search error: {e}")
        return []

    finally:
        pg_pool.putconn(conn)

def retrieve_v2(query_text: str, active_doc: str = "") -> Tuple[List[SourceItem], str]:
    """
    Retrieval V5:
    - Qdrant vector search
    - Postgres BM25 keyword search
    - Neo4j entity/formula search
    - Neo4j graph expansion
    - RRF fusion
    - CrossEncoder reranking
    - Final Postgres enrichment by chunk_uuid
    """
    print(f"\n\n{'=' * 40}")
    print("🔎 DEBUG RETRIEVAL START")
    print(f"❓ Query: '{query_text}'")

    if not embedder or not qdrant_client_inst:
        return [SourceItem(id="error", content="Backend OFF", filename="System")], "Backend OFF"

    t_total0 = time.time()
    timings: Dict[str, float] = {}
    counts: Dict[str, Any] = {}
    intent = detect_intent(query_text)


    # LOGICA DI MEMORIA:
    extracted_doc = extract_requested_document(query_text)
    
    # Se l'utente nomina un file ora, usa quello. 
    # Altrimenti usa quello che abbiamo in memoria (active_doc).
    requested_doc = extracted_doc if extracted_doc else active_doc
    requested_doc_norm = normalize_doc_name(requested_doc)

    if requested_doc:
        print(f"📄 Requested document scope: {requested_doc} -> {requested_doc_norm}")
        counts["requested_doc"] = requested_doc


    # 1) Embedding query
    t0 = time.time()
    query_vector = embedder.encode(query_text, normalize_embeddings=True).tolist()
    timings["embed"] = time.time() - t0

    # 2) Qdrant vector search
    t0 = time.time()
    hits = []

    try:
        hits = qdrant_client_inst.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=QDRANT_CANDIDATES,
            with_payload=True,
        )
        counts["qdrant_hits"] = len(hits)
        print(f"🌌 Qdrant ha trovato {len(hits)} chunk.")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
        counts["qdrant_hits"] = 0

    timings["qdrant_search"] = time.time() - t0

    # 3) Postgres BM25 search
    t0 = time.time()
    bm25_hits = search_pg_bm25(query_text, limit=40)
    counts["bm25_hits"] = len(bm25_hits)
    print(f"🐘 Postgres ha trovato {len(bm25_hits)} chunk.")
    timings["bm25_search"] = time.time() - t0


    # 3B) Postgres document-scope search
    # Se l'utente chiede un documento specifico, recuperiamo chunk direttamente da quel documento.
    t0 = time.time()
    doc_scope_hits = []

    if requested_doc:
        doc_scope_hits = search_pg_by_document_scope(
            requested_doc=requested_doc,
            query_text=query_text,
            limit=80,
        )

    counts["doc_scope_hits"] = len(doc_scope_hits)

    if requested_doc:
        print(
            f"📄 Postgres document-scope search ha trovato "
            f"{len(doc_scope_hits)} chunk per documento '{requested_doc}'."
        )

    timings["doc_scope_search"] = time.time() - t0

    # 4) Neo4j direct entity/formula search
    t0 = time.time()

    neo4j_entity_hits = search_neo4j_entities(query_text, limit=20)

    formula_query = (
        intent == "formula"
        or any(k in (query_text or "").lower() for k in [
            "formula", "formule", "latex", "equazione", "equazioni"
        ])
    )

    neo4j_formula_hits = (
        search_neo4j_formulas(query_text, limit=GRAPH_MAX_FORMULAS)
        if formula_query
        else []
    )

    neo4j_direct_hits = neo4j_entity_hits + neo4j_formula_hits

    counts["neo4j_entity_hits"] = len(neo4j_entity_hits)
    counts["neo4j_formula_direct_hits"] = len(neo4j_formula_hits)
    counts["neo4j_direct_hits"] = len(neo4j_direct_hits)

    print(
        f"🕸️ Neo4j direct search ha trovato {len(neo4j_direct_hits)} chunk "
        f"({len(neo4j_entity_hits)} entity, {len(neo4j_formula_hits)} formule)."
    )

    timings["neo4j_direct_search"] = time.time() - t0

    # 5) Candidate merge
    candidates_dict: Dict[str, Dict[str, Any]] = {}

    # 5A) Import Qdrant candidates
    for hit in hits:
        uid = str(hit.id)
        payload = hit.payload or {}

        content = safe_payload_text(payload)
        if not content:
            continue

        candidates_dict[uid] = {
            "id": uid,
            "content": content,
            "filename": str(payload.get("filename", "Unknown")),
            "page": get_payload_page(payload),
            "type": get_payload_type(payload),
            "tier": normalize_tier_value(str(payload.get("tier", "C"))),
            "score_base": float(hit.score or 0.0),
            "score_vec": float(hit.score or 0.0),
            "score_bm25": 0.0,
            "score_graph": 0.0,
            "origin": "Qdrant",
            "section_hint": get_payload_section(payload),
        }

    # 5A-BIS) Import Postgres document-scope candidates
    for d in doc_scope_hits:
        uid = str(d.get("id", "")).strip()

        if not uid:
            continue

        meta = d.get("metadata", {}) or {}

        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        fname = meta.get("filename") or meta.get("source_name") or requested_doc or "Unknown"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        toon_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": d.get("content", ""),
                "filename": fname,
                "page": page,
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(d.get("score", 0.001)),
                "score_graph": 0.0,
                "score_doc_scope": 1.0,
                "origin": "PostgresDocScope",
                "section_hint": meta.get("section_hint", ""),
            }
        else:
            candidates_dict[uid]["score_bm25"] = max(
                float(candidates_dict[uid].get("score_bm25", 0.0)),
                float(d.get("score", 0.001)),
            )
            candidates_dict[uid]["score_doc_scope"] = 1.0

            # Se Qdrant/Neo4j avevano filename Unknown o Neo4j,
            # correggiamo usando i metadati Postgres.
            if candidates_dict[uid].get("filename") in ("", "Unknown", "Neo4j"):
                candidates_dict[uid]["filename"] = fname

            if not candidates_dict[uid].get("page"):
                candidates_dict[uid]["page"] = page

            if "PostgresDocScope" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + PostgresDocScope"

    # 5B) Import Postgres BM25 candidates
    for b in bm25_hits:
        uid = str(b.get("id", "")).strip()
        if not uid:
            continue

        meta = b.get("metadata", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        fname = meta.get("filename") or meta.get("source_name") or "Unknown"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        toon_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": b.get("content", ""),
                "filename": fname,
                "page": page,
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(b.get("score", 0.0)),
                "score_graph": 0.0,
                "origin": "Postgres",
                "section_hint": meta.get("section_hint", ""),
            }
        else:
            candidates_dict[uid]["score_bm25"] = max(
                float(candidates_dict[uid].get("score_bm25", 0.0)),
                float(b.get("score", 0.0)),
            )
            if "Postgres" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + Postgres"

    # 5C) Import Neo4j direct candidates
    for g in neo4j_direct_hits:
        uid = str(g.get("id", "")).strip()
        if not uid:
            continue

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": g.get("content", ""),
                "filename": g.get("filename", "Neo4j"),
                "page": int(g.get("page") or 0),
                "type": g.get("type", "graph"),
                "tier": "GRAPH",
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": 0.0,
                "score_graph": float(g.get("score_graph", 0.0)),
                "origin": g.get("origin", "Neo4j"),
                "section_hint": g.get("section_hint", ""),
            }
        else:
            candidates_dict[uid]["score_graph"] = max(
                float(candidates_dict[uid].get("score_graph", 0.0)),
                float(g.get("score_graph", 0.0)),
            )
            if "Neo4j" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + Neo4j"

    # 6) Neo4j graph expansion
    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        t0_graph = time.time()

        seed_ids = [str(hit.id) for hit in hits][:10]
        graph_sources = []

        try:
            neighbor_ids = get_neighbor_chunk_ids(
                seed_ids,
                limit=GRAPH_MAX_NEIGHBOR_CHUNKS,
            )
        except Exception as e:
            print(f"⚠️ Neo4j neighbor search error: {e}")
            neighbor_ids = []

        if neighbor_ids:
            graph_sources = fetch_chunks_from_qdrant_by_ids(neighbor_ids)

            for gs in graph_sources:
                if gs.id not in candidates_dict:
                    candidates_dict[gs.id] = {
                        "id": gs.id,
                        "content": gs.content,
                        "filename": gs.filename,
                        "page": gs.page,
                        "type": gs.type,
                        "tier": normalize_tier_value(getattr(gs, "tier", "C")),
                        "score_base": 0.0,
                        "score_vec": 0.0,
                        "score_bm25": 0.0,
                        "score_graph": 1.0,
                        "origin": "Neo4j_Expansion",
                        "section_hint": getattr(gs, "section_hint", ""),
                    }

            print(f"🕸️ Neo4j ha aggiunto {len(graph_sources)} chunk semanticamente collegati.")

        counts["neo4j_hits"] = len(graph_sources)
        timings["graph"] = time.time() - t0_graph
    else:
        counts["neo4j_hits"] = 0

    # 7) Final candidate list
    candidates = list(candidates_dict.values())

    if not candidates:
        print("❌ NESSUN CANDIDATO TROVATO!")
        timings["total"] = time.time() - t_total0
        return [], build_retrieval_audit_md(query_text, intent, timings, counts, [])

    # 7B) HARD DOCUMENT SCOPE FILTER
    # Se l'utente chiede un documento specifico, NON permettere fonti di altri documenti.
    if requested_doc:
        before_doc_scope = len(candidates)

        scoped_candidates = [
            c for c in candidates
            if candidate_matches_requested_doc(c, requested_doc)
        ]

        counts["doc_scope_before"] = before_doc_scope
        counts["doc_scope_after"] = len(scoped_candidates)

        print(
            f"📄 Document scope filter: {before_doc_scope} -> {len(scoped_candidates)} "
            f"for requested_doc='{requested_doc}'"
        )

        if not scoped_candidates:
            timings["total"] = time.time() - t_total0
            audit = build_retrieval_audit_md(query_text, intent, timings, counts, [])
            audit += (
                f"\n\n#### 📄 Document Scope\n"
                f"- Documento richiesto: `{requested_doc}`\n"
                f"- Nessun chunk trovato appartenente al documento richiesto.\n"
            )
            return [], audit

        candidates = scoped_candidates



    # 8) RRF scoring
    apply_rrf_scoring(candidates)

    query_tokens = extract_rag_tokens(query_text)

    print(f"🎯 Target Tokens (Filename Match): {query_tokens}")

    for c in candidates:
        fname_lower = (c.get("filename") or "").lower()
        hits_fname = sum(1 for token in query_tokens if token in fname_lower)
        filename_boost = 0.03 * hits_fname

        if hits_fname > 0:
            c["origin"] += " [TARGET FILE]"
            print(f"   🚀 Filename boost per {c.get('filename')} (match={hits_fname})")

        tier_delta = tier_score_delta(c.get("tier", ""), query_text)

        doc_scope_boost = 0.20 if c.get("score_doc_scope", 0.0) > 0 else 0.0

        c["pre_rerank_score"] = (
            float(c.get("rrf_score", 0.0))
            + filename_boost
            + tier_delta
            + doc_scope_boost
        )

    # 9) Reranking
    candidates.sort(key=lambda x: x.get("pre_rerank_score", 0.0), reverse=True)
    top_candidates = candidates[:RERANK_CANDIDATES]

    if reranker and top_candidates:
        t0 = time.time()

        pairs = [
            (query_text, c.get("content", "") or "")
            for c in top_candidates
        ]

        try:
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                top_candidates[i]["final_score"] = (
                    float(score)
                    + float(top_candidates[i].get("pre_rerank_score", 0.0))
                )

        except Exception as e:
            print(f"⚠️ Reranker Error: {e}")

            for c in top_candidates:
                c["final_score"] = float(c.get("pre_rerank_score", 0.0))

        timings["rerank"] = time.time() - t0

    else:
        for c in top_candidates:
            c["final_score"] = float(c.get("pre_rerank_score", 0.0))

    top_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # 10) Diversification
    final_selection = diversify(
        top_candidates,
        MAX_PER_PAGE,
        MAX_PER_DOC,
        FINAL_SOURCES,
    )

    # 11) Final Postgres enrichment by chunk_uuid
    pg_rows = fetch_pg_chunks_by_uuid(
        [str(t.get("id")) for t in final_selection if t.get("id")]
    )

    counts["pg_enriched_hits"] = len(pg_rows)

    for t in final_selection:
        uid = str(t.get("id", ""))
        pg_row = pg_rows.get(uid)

        if not pg_row:
            continue

        pg_meta = pg_row.get("metadata_json", {}) or {}
        if isinstance(pg_meta, str):
            try:
                pg_meta = json.loads(pg_meta)
            except Exception:
                pg_meta = {}

        preferred_content = (
            pg_row.get("content_raw", "")
            if PG_PREFER_RAW
            else (pg_row.get("content_semantic", "") or pg_row.get("content_raw", ""))
        )

        if preferred_content:
            t["content"] = preferred_content

        t["filename"] = (
            t.get("filename")
            or pg_meta.get("filename")
            or pg_meta.get("source_name")
            or "Unknown"
        )

        t["page"] = int(
            t.get("page")
            or pg_meta.get("page_no")
            or pg_meta.get("page")
            or 0
        )

        t["type"] = (
            t.get("type")
            or pg_meta.get("toon_type")
            or pg_meta.get("type")
            or "text"
        )

        t["tier"] = normalize_tier_value(
            t.get("tier")
            or pg_meta.get("tier")
            or "C"
        )

        t["pg_ingestion_ts"] = pg_row.get("ingestion_ts", "")
        t["pg_source_name"] = pg_meta.get("source_name", "")
        t["pg_source_type"] = pg_meta.get("source_type", "")
        t["pg_log_id"] = int(pg_meta.get("log_id") or 0)
        t["pg_chunk_id"] = int(pg_meta.get("chunk_index") or 0)
        t["pg_toon_type"] = pg_meta.get("toon_type", "")

        if "PG_Enrich" not in t["origin"]:
            t["origin"] += " + PG_Enrich"

    counts["tier_split"] = dict(
        Counter(normalize_tier_value(str(s.get("tier", "UNKNOWN"))) for s in final_selection)
    )
    counts["final_sources"] = len(final_selection)
    timings["total"] = time.time() - t_total0

    print("-" * 20)
    print("🏆 CLASSIFICA FINALE (Top 3):")

    for i, s in enumerate(final_selection[:3]):
        print(
            f"  {i + 1}. {s.get('filename')} "
            f"(Score: {float(s.get('final_score', 0.0)):.3f}) - {s.get('origin')}"
        )

    print("=" * 40 + "\n")

    # 12) Output SourceItem construction
    sources: List[SourceItem] = []

    for t in final_selection:
        sources.append(
            SourceItem(
                id=str(t.get("id", "")),
                content=t.get("content", ""),
                filename=t.get("filename", "Unknown"),
                page=int(t.get("page") or 0),
                type=t.get("type", "text"),
                score=float(t.get("final_score", 0.0)),
                tier=normalize_tier_value(t.get("tier", "C")),
                db_origin=t.get("origin", "Unknown"),
                section_hint=t.get("section_hint", ""),
                pg_ingestion_ts=t.get("pg_ingestion_ts", ""),
                pg_source_name=t.get("pg_source_name", ""),
                pg_source_type=t.get("pg_source_type", ""),
                pg_log_id=int(t.get("pg_log_id") or 0),
                pg_chunk_id=int(t.get("pg_chunk_id") or 0),
                pg_toon_type=t.get("pg_toon_type", ""),
            )
        )

    # 13) Final formulas from Neo4j
    counts["final_formulas"] = 0

    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        chunk_ids = [s.id for s in sources if s.id and s.id != "graph"]
        formulas = get_formulas_for_chunks(chunk_ids, limit=GRAPH_MAX_FORMULAS)

        counts["final_formulas"] = len(formulas)

        if formulas:
            sources.append(
                SourceItem(
                    id="graph",
                    content="Formule collegate:\n" + "\n".join(formulas),
                    filename="KG",
                    page=0,
                    type="formula",
                    tier="GRAPH",
                    score=0.0,
                    db_origin="Neo4j Formula Lookup",
                )
            )

    return sources, build_retrieval_audit_md(
        query_text,
        intent,
        timings,
        counts,
        [],
    )


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
    v2.3: Strong grounding + Tier A non-contradiction principle + table-first reconstruction.
    """
    base = """
        ROLE:
        You are a Senior Quantitative Financial Analyst.

        CORE OBJECTIVE:
        Answer the user's question using ONLY the provided context snippets.

        CONTEXT STRUCTURE:
        The context is provided in chunks with headers like:
        `--- Source [n] — Filename — Page X — (Type) ---`

        You MUST use these headers to locate page-specific information.
        If the user asks about a specific page, document, table, chart, formula, or section,
        prioritize chunks whose headers match that page or document.

        ### NON-NEGOTIABLE GROUNDING RULES

        0) NO INVENTION:
        - You are NOT allowed to invent details.
        - If a detail, row, column, value, name, date, formula, definition, or relationship
          is not explicitly present in ANY retrieved chunk, you must say it is not available.
        - Do NOT complete tables from memory, assumptions, common knowledge, or external knowledge.

        SOURCE SUFFICIENCY RULE:
        If the retrieved sources do not contain explicit evidence for the answer, you MUST reply exactly:

        "I did not find sufficient evidence in the retrieved documents."
        You MUST NOT answer from general knowledge.
        You MUST NOT infer missing facts.
        You MUST NOT complete missing data from assumptions, memory, or external knowledge.
        Use only the retrieved context.

        SOURCE USAGE POLICY:
        Default mode is OPEN_CORPUS.
        If the user asks a conceptual question, answer the core concept using all relevant retrieved sources.

        Use sources as evidence, not as semantic limitations, unless the user explicitly requests:
        - a specific document;
        - a specific source;
        - a specific file;
        - a specific PDF;
        - a specific version;
        - a comparison between specific documents or versions.

        If the user explicitly names one document, file, PDF, source, or version, use only evidence from that named source.
        If the user explicitly names multiple documents, files, PDFs, sources, or versions, compare or synthesize only across those named sources.
        If the user asks a general conceptual question, you MAY consolidate evidence from multiple retrieved sources.

        When the same formula, concept, indicator, method, or relationship appears in multiple sources, consolidate the concept and cite all supporting sources.
        If retrieved sources disagree, highlight the conflict instead of forcing a single answer.

        Do NOT reject useful evidence only because it comes from multiple documents, unless the user explicitly constrained the answer to one document or one version.

        1) **METADATA SENSITIVITY (CRITICAL)**:
        - If the user asks about a specific page, prioritize chunks whose header matches that page.
        - If the user explicitly names a specific document/file/PDF/source/version, use only chunks from that named source.
        - If the user asks a conceptual question without naming a specific source, use all relevant retrieved chunks across the corpus.
        - If a table spans multiple chunks on the same page, merge them mentally.

        2) TRUTH HIERARCHY:
        - [TIER A - Methodology]&#58; Highest priority. It defines authoritative methodology, formulas, definitions, validation rules, ontology, schema, and interpretation criteria.
        - [TIER B - Reference]&#58; Operational details, examples, factual references, and domain documents.
        - [TIER C - News/Rumors]&#58; Temporal context only. It cannot override Tier A or Tier B.

        3) TIER A NON-CONTRADICTION PRINCIPLE:
        - The final answer MUST NOT contain any statement that contradicts Tier A.
        - If Tier B contradicts Tier A, prefer Tier A and explicitly mention the conflict.
        - If Tier C contradicts Tier A, reject the Tier C claim in the final answer and explicitly mention the conflict.
        - If multiple Tier A chunks contradict each other, do NOT decide arbitrarily. State that the retrieved Tier A sources are inconsistent and explain the conflict.
        - Do NOT reconcile contradictions by inventing assumptions.
        - Do NOT average, merge, or soften contradictory claims unless the sources explicitly provide a reconciliation rule.
        - A contradiction exists when two retrieved claims cannot both be true for the same entity, metric, formula, methodology, date, scope, or definition.
        - Before answering, silently verify that every final statement is compatible with Tier A.

        4) VISUAL AND TABLE DATA HANDLING:
        - Chunks labeled `(image)`, `(table)`, or `(chart)` are AI-extracted descriptions of visual assets.
        - Treat them as factual ONLY within what they explicitly state.
        - If the user asks for data in a table, you MUST extract the rows and columns that are explicitly listed.
        - Do NOT invent missing rows, missing columns, missing values, or missing units.

        5) TABLE RECONSTRUCTION:
        - If the context includes a Markdown table, reproduce it or the relevant subset.
        - If the context describes table rows in bullet form, reconstruct a Markdown table with columns.
        - If one chunk says “table not shown” but another chunk contains the actual rows, prefer the chunk with the actual rows.
        - Do NOT provide generic summaries when the user asks for table data.
        - Enumerate the available table content.

        6) FORMULA HANDLING:
        - Preserve formula fidelity.
        - Do NOT alter mathematical symbols, variables, operators, or definitions.
        - If a formula is split across chunks on the same page, reconstruct it only if the continuation is explicit.
        - If the formula is incomplete, say that it is incomplete in the retrieved context.
        - The final formula must not contradict Tier A methodology.

        7) FRAGMENT REASSEMBLY:
        - If text continues across chunks from the same page, treat it as continuous.
        - Do not complain about partial data unless the information is missing from ALL relevant chunks.
        - Do not combine fragments from unrelated documents unless the relationship is explicit.

        INTERNAL CHECKLIST:
        Before answering, silently check:
        1. Is every claim supported by the retrieved sources?
        2. Does any claim contradict Tier A?
        3. Are there conflicts between Tier A, Tier B, and Tier C?
        4. Are page and document references respected?
        5. If evidence is insufficient, is the limitation clearly stated?

        Do not output the checklist.

OUTPUT STRUCTURE:
        You MUST structure your response in EXACTLY these four sections, using these EXACT Italian headers:

        **A) Risposta**
        - Provide a direct, technical answer in the USER'S LANGUAGE.
        - If the user asks about a table, include a reconstructed Markdown table.
        - Do not include unsupported facts.
        - Do not include claims that contradict Tier A.

        **B) Evidenze**
        - Use bullet points citing the source ID(s), for example: "[2] Pagina 9 elenca 10 indici."
        - If Tier A was used to resolve a conflict, explicitly state which Tier A source controlled the answer.

        **C) Limiti / Conflitti**
        - Strictly state what is missing.
        - If there is a contradiction, explain it clearly.
        - If Tier B or Tier C contradicts Tier A, state that Tier A prevails.
        - If Tier A sources contradict each other, state that the retrieved methodology sources are inconsistent.

        **D) Fonti**
        - List the filenames used.
        - Do not expose internal chunk IDs, UUIDs, database IDs, or technical metadata unless explicitly requested.

        FORMATTING RESTRICTIONS (CRITICAL):
        - Do NOT include internal system headers (like 'NEWS & EVENTS', 'KNOWLEDGE GRAPH', or 'USER QUESTION') in your final output.
        - ONLY output the exact four requested headers verbatim: "**A) Risposta**", "**B) Evidenze**", "**C) Limiti / Conflitti**", "**D) Fonti**". Do NOT create new sections (like 'E', 'Grafici', or 'Note') and do NOT comment on empty context blocks.

        LANGUAGE RULE:
        You MUST detect the language of the user's question.
        You MUST respond EXCLUSIVELY in the same language as the user's question.
        Do not switch language, even if the retrieved sources or system instructions are written in another language.
        """

    # Dynamic Intent Injection
    if intent == "formula":
        base += """
        INTENT: FORMULA.
        Prioritize formula fidelity.
        Reconstruct split formulas only when the continuation is explicitly present.
        Do not infer missing mathematical terms.
        The final formula must not contradict Tier A methodology.
        """
    elif intent == "table":
        base += """
        INTENT: TABLE.
        The user wants the full available data.
        Do NOT summarize.
        Output the complete Markdown table when the data is present.
        Do not invent missing rows or columns.
        If table interpretation conflicts with Tier A, Tier A prevails.
        """
    elif intent == "chart":
        base += """
        INTENT: CHART / DATA.
        Extract explicit numbers, labels, axes, and trends only.
        No estimation.
        No interpolation.
        If the chart interpretation conflicts with Tier A methodology, Tier A prevails.
        """

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

    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?reasoning>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\[SourceID:\s*\d+.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r">>> SOURCE \[\d+\].*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "", text)
    text = text.replace("Tier: A", "").replace("Tier: B", "").replace("Tier: C", "")

    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Estrae un oggetto JSON da una risposta LLM.
    Serve perché alcuni modelli locali possono aggiungere testo prima/dopo il JSON.
    """
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def build_eval_context(sources: List[SourceItem], max_chars: int = EVAL_MAX_CONTEXT_CHARS) -> str:
    """
    Costruisce il contesto da passare al judge.
    Qui NON servono chunk_id tecnici: bastano fonte, pagina, tier e contenuto.
    """
    parts = []
    total = 0

    for i, s in enumerate(sources, start=1):
        if not s.content:
            continue

        header = (
            f"--- SOURCE [{i}] ---\n"
            f"filename: {s.filename}\n"
            f"page: {s.page}\n"
            f"type: {s.type}\n"
            f"tier: {normalize_tier_value(s.tier)}\n"
            f"origin: {s.db_origin}\n"
        )

        body = (s.content or "").strip()
        block = header + body + "\n\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total - len(header) - 100
            if remaining <= 300:
                break
            block = header + body[:remaining] + "\n\n"

        parts.append(block)
        total += len(block)

        if total >= max_chars:
            break

    return "".join(parts).strip()


def append_rag_eval_log(
    query_text: str,
    answer: str,
    sources: List[SourceItem],
    eval_result: RagEvalResult,
    requested_doc: str = "",
):
    """
    Salva le metriche KPI in JSONL.
    Non salva necessariamente tutto il contesto, ma salva abbastanza per audit tecnico.
    """
    if not EVAL_ENABLED:
        return

    try:
        row = {
            "ts_utc": datetime.utcnow().isoformat(),
            "query": query_text,
            "requested_doc": requested_doc,
            "answer_sha256": hashlib.sha256((answer or "").encode("utf-8")).hexdigest(),
            "sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "type": s.type,
                    "tier": normalize_tier_value(s.tier),
                    "db_origin": s.db_origin,
                    "score": s.score,
                }
                for s in sources
            ],
            "metrics": eval_result.model_dump(),
            "llm_model": LLM_MODEL_NAME,
            "eval_model": EVAL_MODEL_NAME,
        }

        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"⚠️ RAG eval log write error: {e}")


def evaluate_rag_answer(
    query_text: str,
    answer: str,
    sources: List[SourceItem],
    requested_doc: str = "",
) -> RagEvalResult:
    """
    Valuta la risposta rispetto ai documenti recuperati.

    Metriche:
    - faithfulness: quanto la risposta è supportata dalle fonti
    - answer_relevance: quanto risponde alla domanda
    - context_support: quanto il contesto contiene evidenza sufficiente
    - hallucination_risk: rischio di allucinazione
    - source_scope_violation: True se usa fonti fuori scope documentale
    """
    if not EVAL_ENABLED:
        return RagEvalResult(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_support=1.0,
            hallucination_risk=0.0,
            verdict="DISABLED",
            reason="Evaluation disabled.",
        )

    if not llm_client:
        return RagEvalResult(
            verdict="ERROR",
            reason="LLM client not initialized for evaluation.",
        )

    if not answer or not answer.strip():
        return RagEvalResult(
            verdict="FAIL",
            reason="Empty answer.",
        )

    if not sources:
        return RagEvalResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_support=0.0,
            hallucination_risk=1.0,
            verdict="FAIL",
            reason="No retrieved sources available.",
        )

    eval_context = build_eval_context(sources)

    if not eval_context:
        return RagEvalResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_support=0.0,
            hallucination_risk=1.0,
            verdict="FAIL",
            reason="Retrieved sources have no usable textual content.",
        )

    scope_rule = ""
    if requested_doc:
        scope_rule = (
            f"The user explicitly requested the document/source/version: {requested_doc}. "
            "Mark source_scope_violation=true if the answer relies on other documents."
        )

    judge_system = """
You are a strict RAG faithfulness evaluator.

You must evaluate whether the ANSWER is supported ONLY by the provided SOURCES.

Return ONLY valid JSON with this schema:

{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_support": 0.0,
  "hallucination_risk": 1.0,
  "source_scope_violation": false,
  "verdict": "PASS|WARN|FAIL",
  "unsupported_claims": [],
  "supported_claims": [],
  "reason": ""
}

Scoring rules:
- faithfulness = 1.0 only if all factual claims in the answer are explicitly supported by the sources.
- answer_relevance = 1.0 only if the answer directly addresses the user question.
- context_support = 1.0 only if the retrieved sources contain enough evidence to answer.
- hallucination_risk = 1.0 when the answer contains unsupported facts.
- source_scope_violation = true if the answer uses evidence outside the requested document/source/version.
- Do not use external knowledge.
- Do not reward plausible but unsupported claims.
- If the answer correctly says that evidence is insufficient, faithfulness can be high.
"""

    judge_user = f"""
### USER QUESTION
{query_text}

### REQUESTED SOURCE SCOPE
{scope_rule if scope_rule else "No explicit document/source/version constraint."}

### SOURCES
{eval_context}

### ANSWER TO EVALUATE
{answer}
"""

    try:
        resp = llm_client.chat.completions.create(
            model=EVAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user},
            ],
            temperature=0.0,
            stream=False,
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 768,
                    "repeat_penalty": 1.05,
                }
            },
        )

        raw = resp.choices[0].message.content or ""
        data = _extract_json_object(raw)

        result = RagEvalResult(
            faithfulness=_clamp01(data.get("faithfulness"), 0.0),
            answer_relevance=_clamp01(data.get("answer_relevance"), 0.0),
            context_support=_clamp01(data.get("context_support"), 0.0),
            hallucination_risk=_clamp01(data.get("hallucination_risk"), 1.0),
            source_scope_violation=bool(data.get("source_scope_violation", False)),
            verdict=str(data.get("verdict", "UNKNOWN")).upper(),
            unsupported_claims=list(data.get("unsupported_claims", []) or []),
            supported_claims=list(data.get("supported_claims", []) or []),
            reason=str(data.get("reason", "") or ""),
        )

        if result.verdict not in ("PASS", "WARN", "FAIL"):
            if (
                result.faithfulness >= EVAL_MIN_FAITHFULNESS
                and result.answer_relevance >= EVAL_MIN_ANSWER_RELEVANCE
                and not result.source_scope_violation
            ):
                result.verdict = "PASS"
            elif result.faithfulness >= 0.55:
                result.verdict = "WARN"
            else:
                result.verdict = "FAIL"

        return result

    except Exception as e:
        print(f"⚠️ RAG evaluation error: {e}")
        return RagEvalResult(
            verdict="ERROR",
            reason=str(e),
        )


def format_eval_debug_md(eval_result: RagEvalResult) -> str:
    """
    Formatta le metriche nel pannello Audit della UI.
    """
    unsupported = eval_result.unsupported_claims[:5]
    supported = eval_result.supported_claims[:5]

    lines = []
    lines.append("### 🧪 RAG Faithfulness Evaluation")
    lines.append(f"- **Verdict**: `{eval_result.verdict}`")
    lines.append(f"- **Faithfulness**: **{eval_result.faithfulness:.2f}**")
    lines.append(f"- **Answer relevance**: **{eval_result.answer_relevance:.2f}**")
    lines.append(f"- **Context support**: **{eval_result.context_support:.2f}**")
    lines.append(f"- **Hallucination risk**: **{eval_result.hallucination_risk:.2f}**")
    lines.append(f"- **Source scope violation**: **{eval_result.source_scope_violation}**")

    if eval_result.reason:
        lines.append(f"- **Reason**: {eval_result.reason}")

    if unsupported:
        lines.append("\n#### Unsupported claims")
        for c in unsupported:
            lines.append(f"- {c}")

    if supported:
        lines.append("\n#### Supported claims")
        for c in supported:
            lines.append(f"- {c}")

    return "\n".join(lines).strip()

# =========================
# 🛡️ UI SAFETY HELPERS
# =========================

MAX_UI_SOURCES = int(os.getenv("MAX_UI_SOURCES", "8"))
MAX_UI_SOURCE_CONTENT_CHARS = int(os.getenv("MAX_UI_SOURCE_CONTENT_CHARS", "900"))
MAX_UI_DEBUG_CHARS = int(os.getenv("MAX_UI_DEBUG_CHARS", "6000"))


def ui_safe_text(value, max_chars: int) -> str:
    """
    Versione minimale e compatibile con Reflex.
    Serve solo a evitare testi enormi o caratteri di controllo nella UI.
    Non altera il contenuto usato dal RAG/LLM.
    """
    if value is None:
        return ""

    try:
        text = str(value)
    except Exception:
        text = ""

    # Rimuove caratteri di controllo problematici per JSON/React.
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[contenuto troncato per la UI]"

    return text


def ui_safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def ui_safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:  # NaN
            return default
        if v == float("inf") or v == float("-inf"):
            return default
        return round(v, 4)
    except Exception:
        return default


def prepare_sources_for_ui(sources: List[SourceItem]) -> List[SourceItem]:
    """
    Crea una copia ridotta delle fonti SOLO per la UI.
    Evita crash o sparizione schermata quando i chunk sono troppo lunghi.
    """
    out: List[SourceItem] = []

    for s in (sources or [])[:MAX_UI_SOURCES]:
        out.append(
            SourceItem(
                id=ui_safe_text(getattr(s, "id", ""), 200),
                content=ui_safe_text(getattr(s, "content", ""), MAX_UI_SOURCE_CONTENT_CHARS),
                filename=ui_safe_text(getattr(s, "filename", "Unknown"), 240),
                page=ui_safe_int(getattr(s, "page", 0), 0),
                type=ui_safe_text(getattr(s, "type", "text"), 80),
                score=ui_safe_float(getattr(s, "score", 0.0), 0.0),
                graph_context=[],
                section_hint=ui_safe_text(getattr(s, "section_hint", ""), 300),
                image_id=getattr(s, "image_id", None),
                tier=ui_safe_text(getattr(s, "tier", "C"), 40),
                pg_ingestion_ts=ui_safe_text(getattr(s, "pg_ingestion_ts", ""), 80),
                pg_source_name=ui_safe_text(getattr(s, "pg_source_name", ""), 160),
                pg_source_type=ui_safe_text(getattr(s, "pg_source_type", ""), 80),
                pg_log_id=ui_safe_int(getattr(s, "pg_log_id", 0), 0),
                pg_chunk_id=ui_safe_int(getattr(s, "pg_chunk_id", 0), 0),
                pg_toon_type=ui_safe_text(getattr(s, "pg_toon_type", ""), 80),
                db_origin=ui_safe_text(getattr(s, "db_origin", "Unknown"), 160),
            )
        )

    return out


def prepare_debug_for_ui(debug_md: str) -> str:
    """
    Riduce l'audit solo per visualizzazione.
    """
    return ui_safe_text(debug_md or "", MAX_UI_DEBUG_CHARS)

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
    
    current_active_doc: str = ""
    inline_open_for: str = ""
    
    
    inline_tab: str = "sources"

    vram_info: str = "N/A"
    vram_free: str = "N/A"
    backend_status: str = "OK"

    show_sources_modal: bool = False
    modal_sources: List[SourceItem] = []
    modal_debug_md: str = ""
    modal_title: str = ""
    '''
    def on_load(self):
        """Eseguito all'apertura della pagina."""
        # 1. Acquisiamo il lucchetto
        with _init_lock:
            # 2. Controlliamo se i modelli sono già stati caricati da qualcun altro
            if embedder is None:
                init_resources()
                
        # 3. Aggiorniamo la UI
        self.refresh_gpu()
        self.refresh_backend_status()
    '''
    
    def set_sources_modal_open(self, value: bool):
        self.show_sources_modal = value
    
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
                # --- INIZIO NUOVA LOGICA: MEMORIA DI CONTESTO ---
                # Estraiamo il documento dalla query. Se c'è, lo salviamo in memoria.
                extracted_doc = extract_requested_document(user_query)
                if extracted_doc:
                    self.current_active_doc = extracted_doc

                # 1. RECUPERO DATI (Hybrid Search + Rerank)
                # Passiamo il documento in memoria (active_doc) alla funzione di ricerca
                sources, debug_md = retrieve_v2(user_query, active_doc=self.current_active_doc)
                # --- FINE NUOVA LOGICA ---

                if not sources:
                    self.messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=(
                                "**A) Risposta**\n\n"
                                "Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                                "**B) Evidenze**\n\n"
                                "- Nessuna fonte pertinente recuperata per il documento richiesto.\n\n"
                                "**C) Limiti**\n\n"
                                "- Il sistema non deve usare formule provenienti da altri documenti.\n\n"
                                "**D) Fonti**\n\n"
                                "- Nessuna fonte utilizzabile."
                            ),
                            sources=[],
                            debug_md=prepare_debug_for_ui(debug_md),
                        )
                    )
                    self.is_processing = False
                    yield rx.scroll_to("chat_bottom")
                    return
                
                # --- INIZIO NUOVA LOGICA: PROMPT ANTI-CONTAMINAZIONE IN INGLESE ---
                # Subito dopo il blocco "if not sources:", creiamo le istruzioni di sistema
                # e aggiungiamo il guardrail robusto.
                system_instructions = build_system_instructions(intent)
                system_instructions += """
                
                8) ANTI-CONTAMINATION AND DISAMBIGUATION (CRITICAL):
                - If the user specifies or implies a specific document context, you MUST STRICTLY IGNORE retrieved chunks from other documents that define the same variables (e.g., 'alpha', 'D') differently.
                - Mathematical variables are highly context-dependent. Do not mix formulas from 'algorithmic trading' with those from 'asset allocation' or other topics.
                - If you see conflicting definitions for a variable across different sources, ALWAYS prioritize the definitions from the active requested document context.
                """
                # --- FINE NUOVA LOGICA ---

                if not is_news_query(user_query):
                    has_tier_a = any((s.tier or "").upper() == "A" for s in sources)

                    if not has_tier_a:
                        tier_a_context = self.get_context_by_tier(user_query, "A")

                        if tier_a_context:
                            sources.insert(
                                0,
                                SourceItem(
                                    id="forced_tier_a",
                                    content=tier_a_context,
                                    filename="TIER_A_METHODOLOGY",
                                    page=0,
                                    type="methodology",
                                    score=1.0,
                                    tier="A",
                                    db_origin="Qdrant Forced Tier A",
                                    section_hint="Forced methodology context"
                                )
                            )
                                
                # 2. RAGGRUPPAMENTO FONTI
                c_a_list, c_b_list, c_c_list, c_g_list = [], [], [], []

                for i, s in enumerate(sources, start=1):
                    tier_norm = normalize_tier_value(s.tier)

                    # FIX: Usa "Source" e "Page" per allinearsi perfettamente al System Prompt
                    header = f"--- Source [{i}] — {s.filename} — Page {s.page} — ({s.type}) ---\n"
                    meta = f"(tier={tier_norm} | db={s.db_origin})\n"
                    body = (s.content or "").strip()

                    if not body:
                        continue

                    snippet = header + meta + body + "\n\n"

                    if tier_norm == "A":
                        c_a_list.append(snippet)
                    elif tier_norm == "B":
                        c_b_list.append(snippet)
                    elif tier_norm == "GRAPH":
                        c_g_list.append(snippet)
                    else:
                        # FIX CRITICO: Qualsiasi tier non riconosciuto finisce qui. 
                        # Nessun chunk recuperato verrà mai più perso.
                        c_c_list.append(snippet)

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
                requested_doc = extract_requested_document(user_query)

                doc_scope_block = ""
                if requested_doc:
                    doc_scope_block = (
                        f"### REQUESTED DOCUMENT SCOPE ###\n"
                        f"The user explicitly requested this document: {requested_doc}\n"
                        f"You MUST answer using ONLY sources whose filename matches this requested document.\n"
                        f"If the retrieved context does not contain sources from this document, answer only:\n"
                        f"Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                    )

                final_user_content = (
                    doc_scope_block +
                    f"### PROVIDED CONTEXT SNIPPETS ###\n\n"
                    f"### METHODOLOGY [TIER A] ###\n{c_a if c_a else 'No specific methodology found.'}\n\n"
                    f"### RESEARCH [TIER B] ###\n{c_b if c_b else 'No specific research found.'}\n\n"
                    f"### NEWS & EVENTS [TIER C] ###\n{c_c if c_c else 'No recent news found.'}\n\n"
                    f"### KNOWLEDGE GRAPH [NEO4J] ###\n{c_g if c_g else 'No relational/formula data.'}\n\n"
                    f"### USER QUESTION ###\n{user_query}\n"
                    f"{language_reminder}\n\n"
                    f"CRITICAL REMINDER: You MUST output EXACTLY these four headers and nothing else: "
                    f"**A) Risposta**, **B) Evidenze**, **C) Limiti / Conflitti**, **D) Fonti**."
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
                    temperature=0.15, # Leggermente alzata per evitare loop ripetitivi
                    stream=True,
                    extra_body={
                        "options": {
                            "num_ctx": 8192,       # <--- ESTENDE LA MEMORIA (Evita tagli documenti)
                            "num_predict": 4096,   # Lunghezza massima risposta
                            "repeat_penalty": 1.15 # <--- Aumentata per disincentivare la copia esatta
                        }
                    }
                )


                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        full_resp += delta.content
                        self.messages[-1].content = strip_id_leaks(full_resp)
                        try:
                            yield
                        except Exception as e:
                            print(f"⚠️ Client disconnected during stream: {e}")
                            break # Interrompe l'aggiornamento UI se l'utente ha chiuso la pagina

                
              
                # ✅ SOLO ALLA FINE agganciamo fonti, audit e KPI di faithfulness
                answer_clean = strip_id_leaks(full_resp)

                requested_doc = ""
                try:
                    requested_doc = extract_requested_document(user_query)
                except Exception:
                    requested_doc = ""

                eval_result = evaluate_rag_answer(
                    query_text=user_query,
                    answer=answer_clean,
                    sources=sources,
                    requested_doc=requested_doc,
                )

                debug_md += "\n\n" + format_eval_debug_md(eval_result)

                append_rag_eval_log(
                    query_text=user_query,
                    answer=answer_clean,
                    sources=sources,
                    eval_result=eval_result,
                    requested_doc=requested_doc,
                )

                # Modalità osservabilità: mostra la risposta ma segnala il rischio nell'audit.
                self.messages[-1].content = answer_clean

                # Modalità blocco severo: sostituisce risposte non fedeli.
                if EVAL_STRICT_BLOCK:
                    bad_faithfulness = eval_result.faithfulness < EVAL_MIN_FAITHFULNESS
                    bad_relevance = eval_result.answer_relevance < EVAL_MIN_ANSWER_RELEVANCE
                    bad_scope = eval_result.source_scope_violation

                    if bad_faithfulness or bad_relevance or bad_scope:
                        self.messages[-1].content = (
                            "**A) Risposta**\n\n"
                            "Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                            "**B) Evidenze**\n\n"
                            "- La risposta generata non ha superato il controllo automatico di faithfulness.\n\n"
                            "**C) Limiti**\n\n"
                            f"- Faithfulness: {eval_result.faithfulness:.2f}\n"
                            f"- Answer relevance: {eval_result.answer_relevance:.2f}\n"
                            f"- Source scope violation: {eval_result.source_scope_violation}\n\n"
                            "**D) Fonti**\n\n"
                            "- Vedi pannello Fonti/Audit."
                        )

                # ✅ SOLO ALLA FINE agganciamo fonti e audit in versione UI-safe
                # Il RAG usa sources/debug_md completi; la UI riceve una versione ridotta.
                self.messages[-1].sources = prepare_sources_for_ui(sources)
                self.messages[-1].debug_md = prepare_debug_for_ui(debug_md)
                yield
            else:
                self.messages[-1].content = "⚠️ LLM non inizializzato. Verifica che Ollama sia attivo."
                self.messages[-1].sources = prepare_sources_for_ui(sources)
                self.messages[-1].debug_md = prepare_debug_for_ui(debug_md)
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
                        on_click=State.open_sources_audit(msg.id),
                    ),
                    rx.box(),
                ),
                width="100%",
                align_items="center",
                spacing="2",
            ),
            # Contenuto del Messaggio
            rx.box(
                rx.markdown(
                    msg.content,
                    width="100%",
                    overflow_wrap="anywhere",
                    word_break="break-word",
                ),
                width="100%",
                min_width="0",
                overflow_x="auto",
                overflow_y="visible",
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
                        on_click=State.toggle_inline_sources(msg.id),
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
                        on_click=State.toggle_inline_audit(msg.id),
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
                                                    rx.text(s.filename, size="1", weight="bold"),
                                                    rx.spacer(),
                                                    rx.text("Pag. ", s.page, size="1"),
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
        width="85%",
        align_self=align_self,
        box_shadow="sm",
        margin_y="0.5em",
        min_width="280px",
        flex_shrink="0",
        overflow="visible",
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
                on_click=State.toggle_inline_sources(msg.id),
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

            # Popup Fonti/Audit
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
                                                rx.badge(
                                                    s.tier,
                                                    color_scheme="tomato",
                                                    variant="surface",
                                                ),
                                                rx.badge(
                                                    s.db_origin,
                                                    color_scheme="plum",
                                                    variant="outline",
                                                ),
                                                rx.text(
                                                    "Doc: ",
                                                    s.filename,
                                                    weight="bold",
                                                    size="2",
                                                ),
                                                width="100%",
                                                justify="between",
                                            ),
                                            rx.text(
                                                s.content,
                                                size="1",
                                                line_clamp=3,
                                            ),
                                            rx.hstack(
                                                rx.text(
                                                    "Pagina: ",
                                                    s.page,
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                rx.spacer(),
                                                rx.text(
                                                    "Score: ",
                                                    s.score,
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                width="100%",
                                            ),
                                            spacing="2",
                                        ),
                                        width="100%",
                                        margin_bottom="2",
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            height="400px",
                            type="always",
                        ),
                        rx.center(
                            rx.text(
                                "Nessuna fonte trovata per questo messaggio.",
                                color="gray",
                            )
                        ),
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
                        rx.button(
                            "Chiudi",
                            variant="soft",
                            on_click=State.close_sources_audit,
                        ),
                        width="100%",
                        margin_top="1em",
                    ),

                    max_width="900px",
                    width="90vw",
                ),
                open=State.show_sources_modal,
                on_open_change=State.set_sources_modal_open,
            ),

            # Chat scroll area
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(State.messages, message_ui),
                    rx.box(id="chat_bottom", height="1px", flex_shrink="0"),
                    width="100%",
                    padding="1em",
                    max_width="900px",
                    margin="0 auto",
                    spacing="4",
                    min_height="0",
                    flex_shrink="0",
                    align_items="stretch",
                ),
                width="100%",
                flex="1",
                min_height="0",
                min_width="0",
                type="always",
                scrollbars="vertical",
                id="chat_scroll_area",
                overflow_x="hidden",
            ),

            # Input area
            rx.box(
                rx.hstack(
                    rx.input(
                        placeholder="Chiedi informazioni sui documenti...",
                        value=State.input_text,
                        on_change=State.set_input_text,
                        #on_key_down=lambda k: rx.cond(
                        #    k == "Enter",
                        #    State.handle_submit(),
                        #    None,
                        #),
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
            overflow_x="hidden",
            min_height="0",
        ),

        # ROOT
        width="100%",
        height="100dvh",
        position="fixed",
        top="0",
        left="0",
        right="0",
        bottom="0",
        overflow="hidden",
        overflow_x="hidden",
        min_height="0",
    )




app = rx.App(theme=rx.theme(appearance="light", accent_color="indigo", radius="large"))
app.add_page(index, on_load=State.on_load)
