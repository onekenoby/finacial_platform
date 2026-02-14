
"""
set EMBED_BATCH_SIZE=16
set DB_FLUSH_SIZE=96
set VISION_PARALLEL_WORKERS=3
set VISION_DPI=150
set MAX_KG_CHUNKS_PER_DOC=10
set PG_COMMIT_EVERY_N_PAGES=25

Ingestion Engine - v2.3 FAST + DETERMINISTIC FORMULAS
✅ Vision: chart/table/diagram (fact-only) + formule (LaTeX/plain/meaning) -> chunk dedicati
✅ Neo4j: struttura ricca + Formula nodes DETERMINISTICI (no LLM)
✅ Speed: Vision cache (hash), Vision parallel su embedded images, meno commit, KG ridotto e selettivo
"""

import os
import re
import json
import time
import uuid
import shutil
import hashlib
import base64
from typing import List, Dict, Tuple, Optional, Any

from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import SimpleConnectionPool

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from openai import OpenAI


# =========================
# TIERS / TAXONOMY (DOC ORGANIZATION) - NEW
# =========================

TIER_FOLDERS = {
    "TIER_A_METHODOLOGY": {"tier": "A", "content_type": "methodology", "source_kind": "internal"},
    "TIER_B_REFERENCE":   {"tier": "B", "content_type": "reference",   "source_kind": "internal"},
    "TIER_C_NEWS":        {"tier": "C", "content_type": "news",        "source_kind": "scraping"},
}
DEFAULT_TIER_META = {"tier": "B", "content_type": "reference", "source_kind": "internal"}

# Ontology layer (2° livello cartella): esempi -> financial, risk, legal, educational, strategy, sustainability, generic...
DEFAULT_ONTOLOGY = "generic"

# opzionale: topic keyword -> topics (solo best-effort su filename; puoi estendere più avanti)
import re

TOPIC_PATTERNS = {
    "time_series": [
        r"\btime\s*series\b",
        r"\b(arima|arma|sarima|sarimax)\b",
        r"\b(garch|egarch|tgarch|gjr-garch)\b",
        r"\b(prophet)\b",
        r"\b(acf|pacf)\b",
        r"\b(seasonality|seasonal)\b",
        r"\b(unit\s*root|adf\s*test|augmented\s*dickey\s*fuller)\b",
        r"\b(cointegration|cointegrated|johansen)\b",

        # forme tipo AR(1), MA(2), ARMA(2,1), SARIMA(1,1,1)
        r"\bAR\(\s*\d+\s*\)\b",
        r"\bMA\(\s*\d+\s*\)\b",
        r"\bARMA\(\s*\d+\s*,\s*\d+\s*\)\b",
        r"\bSARIMA\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*\d+\s*)?\)\b",
    ],

    "statistics": [
        r"\b(regression|linear\s*regression|logistic\s*regression)\b",
        r"\b(bayes|bayesian)\b",
        r"\b(p-?value|p\s*value)\b",
        r"\b(anova)\b",
        r"\b(distribution|normal|gaussian|poisson|binomial)\b",
        r"\b(variance|standard\s*deviation|std\.?\s*dev\.?)\b",
        r"\b(confidence\s*interval|ci\b)\b",
        r"\b(hypothesis\s*test|t-?test|chi-?square)\b",
    ],

    "risk": [
        r"\b(var|value\s*at\s*risk)\b",
        r"\b(cvar|conditional\s*var|expected\s*shortfall|es)\b",
        r"\b(drawdown|max\s*drawdown)\b",
        r"\b(stress\s*test|stress\s*testing)\b",
        r"\b(duration|modified\s*duration|convexity)\b",
        r"\b(hedge|hedging|delta\s*hedge)\b",
        r"\b(volatility|implied\s*volatility|realized\s*volatility)\b",
    ],

    "trading": [
        r"\b(order\s*book)\b",
        r"\b(market\s*making)\b",
        r"\b(slippage)\b",
        r"\b(spread|bid-ask|bid\s*ask)\b",
        r"\b(stop\s*loss|stop-loss|take\s*profit|take-profit)\b",
        r"\b(backtest|back-testing)\b",
        r"\b(position\s*sizing|risk\s*parity)\b",
    ],

    "macro": [
        r"\b(inflation|cpi|pce)\b",
        r"\b(rates|interest\s*rate|yield|yields)\b",
        r"\b(fed|fomc|ecb|bce)\b",
        r"\b(gdp|pil)\b",
        r"\b(unemployment|jobless|disoccupazione)\b",
        r"\b(recession|soft\s*landing|hard\s*landing)\b",
    ],

    "rumor": [
        r"\b(rumou?r|gossip|leak|leaked)\b",
        r"\b(unconfirmed|not\s*confirmed)\b",
        r"\b(sources\s*say|according\s*to\s*sources)\b",
        r"\b(speculation|speculative)\b",
    ],
}

def infer_topics_regex(text: str, max_topics: int = 6) -> list[str]:
    """
    Tag 'regex-safe': conta i match per topic e ritorna i top N.
    - Usa IGNORECASE
    - Evita falsi positivi (word boundary)
    """
    if not text:
        return []

    # Normalizzazione leggera (non distruttiva)
    t = text

    scores = {}
    for topic, patterns in TOPIC_PATTERNS.items():
        topic_score = 0
        for pat in patterns:
            try:
                topic_score += len(re.findall(pat, t, flags=re.IGNORECASE))
            except re.error:
                # pattern sbagliato non deve rompere ingestion
                continue
        if topic_score > 0:
            scores[topic] = topic_score

    if not scores:
        return []

    # Ordina per score decrescente
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:max_topics]]


def _safe_read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def read_sidecar_meta(file_path: str) -> dict:
    """
    Supporta: <file>.meta.json affiancato (override puntuale, consigliato per casi speciali).
    """
    sidecar = file_path + ".meta.json"
    if os.path.exists(sidecar):
        return _safe_read_json(sidecar)
    return {}

def dispatch_document(file_path: str, root_dir: str) -> dict:
    """
    Decide tier/content_type/source_kind deterministici da cartelle:
      INBOX/TIER_X_.../<ontology>/file.ext

    + ontology: 2° livello cartella (se presente)
    + topics: best-effort (keyword su filename)
    + effective_date: auto per Tier C se mancante (meglio se arriva da scraping/manifest)
    + merge con sidecar .meta.json (override)
    """
    rel = os.path.relpath(root_dir, INBOX_DIR).replace("\\", "/")
    parts = [p for p in rel.split("/") if p and p != "."]

    tier_key = parts[0] if len(parts) >= 1 else ""
    ontology = parts[1] if len(parts) >= 2 else DEFAULT_ONTOLOGY
    base = dict(TIER_FOLDERS.get(tier_key, DEFAULT_TIER_META))

    # topics best-effort da filename
    fname = os.path.basename(file_path)
    topics = infer_topics_regex(fname)
    base["topics"] = topics[:6]
    
    base["ontology"] = ontology or DEFAULT_ONTOLOGY

    # effective_date (fondamentale per Tier C/news)
    if base.get("tier") == "C" and not base.get("effective_date"):
        base["effective_date"] = time.strftime("%Y-%m-%d")

    # override da sidecar manifest
    side = read_sidecar_meta(file_path)
    if isinstance(side, dict) and side:
        # sidecar può includere: tier, content_type, source_kind, source, source_quality, ontology, topics, effective_date, tags...
        base.update(side)

        # normalizza topics se arriva come stringa
        if isinstance(base.get("topics"), str):
            base["topics"] = [base["topics"]]
        if base.get("topics") is None:
            base["topics"] = []

    return base

def ensure_inbox_structure(inbox_dir: str):
    # struttura minima + sottocartelle ontology (puoi estendere)
    # TIER_A = contenuti stabili e “canonici” che vuoi privilegiare nelle risposte -> come corpus principale
    # TIER_B = contenuti più operativi/contestuali che vuoi usare come supporto

    structure = {
        "TIER_A_METHODOLOGY": [],
        "TIER_B_REFERENCE":   [],
        "TIER_C_NEWS":        [],
    }

    for tier_folder, ontologies in structure.items():
        tier_path = os.path.join(inbox_dir, tier_folder)
        os.makedirs(tier_path, exist_ok=True)

        for onto in ontologies:
            os.makedirs(os.path.join(tier_path, onto), exist_ok=True)


# =========================# =========================# =========================# 

# =========================
# CONFIG
# =========================
BASE_DATA_DIR = "./data_ingestion"
INBOX_DIR = os.path.join(BASE_DATA_DIR, "INBOX")
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "PROCESSED")
FAILED_DIR = os.path.join(BASE_DATA_DIR, "FAILED")

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "40"))

CONTEXT_WINDOW_CHARS = int(os.getenv("CONTEXT_WINDOW_CHARS", "260"))
INCLUDE_CONTEXT_IN_KG = os.getenv("INCLUDE_CONTEXT_IN_KG", "1") == "1"

DB_FLUSH_SIZE = int(os.getenv("DB_FLUSH_SIZE", "200"))          # un po' più alto per meno flush - 
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))    # se la tua VRAM regge, aumenta velocità

# Vision switches
PDF_VISION_ENABLED = os.getenv("PDF_VISION_ENABLED", "1") == "1"
PDF_VISION_ONLY_IF_TEXT_SCARSO = os.getenv("PDF_VISION_ONLY_IF_TEXT_SCARSO", "0") == "1"
PDF_MIN_TEXT_LEN_FOR_NO_VISION = int(os.getenv("PDF_MIN_TEXT_LEN_FOR_NO_VISION", "450"))

VISION_DPI = int(os.getenv("VISION_DPI", "160"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "2000000"))

VISION_MAX_FORMULAS_PER_PAGE = int(os.getenv("VISION_MAX_FORMULAS_PER_PAGE", "10"))

PDF_EXTRACT_EMBEDDED_IMAGES = os.getenv("PDF_EXTRACT_EMBEDDED_IMAGES", "1") == "1"
PDF_VISION_ON_EMBEDDED_IMAGES = os.getenv("PDF_VISION_ON_EMBEDDED_IMAGES", "1") == "1"
PDF_MAX_IMAGES_PER_PAGE = int(os.getenv("PDF_MAX_IMAGES_PER_PAGE", "3"))
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", "20000"))

# Speed: Vision parallel + cache
VISION_PARALLEL_WORKERS = int(os.getenv("VISION_PARALLEL_WORKERS", "4"))  # 4-6 di solito ok
VISION_CACHE_MAX = int(os.getenv("VISION_CACHE_MAX", "5000"))             # entries in-memory

# Commit policy
PG_COMMIT_EVERY_N_PAGES = int(os.getenv("PG_COMMIT_EVERY_N_PAGES", "25"))

# KG extraction (solo dove serve)
KG_ENABLED = os.getenv("KG_ENABLED", "1") == "1"
KG_MIN_LEN = int(os.getenv("KG_MIN_LEN", "250")) #
MAX_KG_CHUNKS_PER_DOC = int(os.getenv("MAX_KG_CHUNKS_PER_DOC", "10"))

KG_KEYWORDS = [
    # --- Finanza & Performance ---
    "risk", "rischio", "yield", "rendimento", "revenue", "ricavi", "profit", "profitto",
    "earnings", "utili", "ebitda", "margine", "margin", "debt", "debito", "cash", "cassa",
    "forecast", "previsione", "guidance", "outlook", "dividend", "dividendo",
    
    # --- Strategia & Mercato ---
    "merger", "fusione", "acquisition", "acquisizione", "partnership", "accordo", 
    "agreement", "subsidiary", "controllata", "stakeholder", "competitor", "concorrente",
    "market", "mercato", "share", "quota", "strategy", "strategia", "ceo", "management",
    
    # --- Macro & Regulation ---
    "inflation", "inflazione", "rate", "tasso", "fed", "fomc", "ecb", "bce", "gdp", "pil","deflaction","deflazione"
    "regulation", "regolamento", "compliance", "normativa", "contract", "contratto",
    "clause", "clausola", "policy", "politica", "esg", "sustainability", "sostenibilità",
    
    # --- Analisi Dati & Visual (già nel tuo script) ---
    "grafico", "graph", "tabella", "trend", "asse", "legenda", "chart", "table", "axis", "legend",
    "regression", "regressione", "model", "modello", "algorithm", "algoritmo", "correlation", "correlazione"
]

# Compilazione Regex per KG_KEYWORDS (massima efficienza)
# Il prefisso \b assicura il match di parole intere (es. "rate" non matcha "pirate")
_KG_PAT = re.compile(r'\b(' + '|'.join(KG_KEYWORDS) + r')\b', re.IGNORECASE)



# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_docs")

# Postgres
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "ai_ingestion")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin_password")
PG_MIN_CONN = int(os.getenv("PG_MIN_CONN", "1"))
PG_MAX_CONN = int(os.getenv("PG_MAX_CONN", "5"))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password_sicura")
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "1") == "1"

# LM Studio / OpenAI-compatible
LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234/v1")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-3-12b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", LLM_MODEL_NAME)

# Embeddings
#EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MODEL_NAME = "E:/Modelli/bge-m3"

QDRANT_TEXT_MAX_CHARS = int(os.getenv("QDRANT_TEXT_MAX_CHARS", "2500"))

# LLM reliability
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1300"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))


# =========================
# CLIENTS INIT (LAZY)
# =========================
openai_client = None
embedder = None
qdrant_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
    return openai_client

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedder

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

neo4j_driver = None
if NEO4J_ENABLED:
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except Exception as e:
        print(f"⚠️ Neo4j disabled (driver init failed): {e}")
        NEO4J_ENABLED = False

pg_pool = SimpleConnectionPool(
    PG_MIN_CONN, PG_MAX_CONN,
    host=PG_HOST, port=PG_PORT, dbname=PG_DB,
    user=PG_USER, password=PG_PASS
)


# =========================
# PROMPTS
# =========================
FORMULA_VISION_PROMPT = f"""
You are a careful scientific parser. The image is a PDF page (or part of a page).
Extract ONLY formulas that are clearly visible. Do not invent anything.

Return ONLY valid JSON (no markdown):

{{
  "has_formulas": true/false,
  "formulas": [
    {{
      "latex": "LaTeX if possible, without surrounding $",
      "plain": "normalized plain text",
      "meaning_it": "short conservative explanation in Italian (2-4 lines)",
      "keywords": ["..."],
      "variables": [{{"name":"X","meaning":"..."}}, {{"name":"Y","meaning":"..."}}]
    }}
  ],
  "notes": ["unreadable parts or uncertainties"]
}}

Rules:
- If no formulas: {{"has_formulas": false, "formulas": [], "notes": []}}
- Max {VISION_MAX_FORMULAS_PER_PAGE} formulas.
- If unsure about LaTeX: leave 'latex' empty and fill 'plain' best-effort.
"""

CHART_VISION_PROMPT = """
You are a strict financial data extraction engine.
The image is a chart/table/diagram/photo extracted from a PDF.

IMPORTANT:
- Extract ONLY what is explicitly visible.
- Do NOT infer trends, causes, or interpretations unless directly stated on the image.
- If text is unreadable, state it in "unreadable_parts".
- Never invent numbers/series.

Return ONLY valid JSON (no markdown):

{
  "kind": "table|chart|diagram|photo|other",
  "title": "visible title or empty",
  "source": "visible source label or empty",
  "timeframe": "visible timeframe or empty",
  "axes": { "x": "label or empty", "y": "label or empty" },
  "legend": ["series names as visible"],
  "data_points": [
    { "series": "series", "x": "x category/date", "y": "y value as visible", "unit": "%|USD|EUR|bps|..." }
  ],
  "key_points": ["short factual statements describing what is visible"],
  "numbers": [
    { "label": "as visible", "value": "as visible", "unit": "%|USD|EUR|bps|...", "period": "if visible" }
  ],
  "entities": [ { "type": "Company|Index|Metric|Instrument|Other", "label": "as visible" } ],
  "unreadable_parts": ["..."]
}

Rules:
- data_points: include only clearly readable points (max ~40).
- If decorative/empty: kind="other" and empty arrays.
"""

CHART_RECONCILE_PROMPT = """
You receive:
(A) PAGE_TEXT (raw text from PDF layer)
(B) VISION_JSON (chart/table extraction)

Task:
- Merge ONLY factual, consistent information.
- Never add numbers/series not present in VISION_JSON.
- You may add labels from PAGE_TEXT only if explicitly stated.

Return ONLY valid JSON with the SAME schema as VISION_JSON.
"""

KG_PROMPT = """
You are a Financial Knowledge Engineer.
Extract a HIGH-FIDELITY Knowledge Graph from the provided text.

You MUST extract:
1) NODES: {id,label,type,properties(flat json)}
2) EDGES: {source,target,relation,properties(flat json)}

Return ONLY JSON:
{"nodes":[...],"edges":[...]}
"""


# =========================
# UTILS
# =========================
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def deterministic_chunk_id(
    doc_id: str,
    page_no: int,
    chunk_index: int,
    toon_type: str,
    text_sem: str,
    image_id: Optional[int] = None,
) -> str:
    """
    ID deterministico per chunk.
    Se re-ingestisci lo stesso documento con lo stesso chunking -> stesso chunk_id -> niente duplicazioni Neo4j/Qdrant/PG.
    """
    text_hash = sha256_hex((text_sem or "").encode("utf-8"))[:16]
    base = f"{doc_id}::p{page_no}::i{chunk_index}::{toon_type}::{text_hash}"
    if image_id is not None:
        base += f"::img{image_id}"
    # 32 hex chars: stabile, corto, compatibile come string ID
    return sha256_hex(base.encode("utf-8"))[:32]

def normalize_ws(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_entity_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s)
    return s[:180]

def safe_json_extract(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        s = raw.strip()
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
        if not s.startswith("{"):
            m = re.search(r"\{.*\}", s, flags=re.DOTALL)
            if m:
                s = m.group(0).strip()
        return json.loads(s)
    except Exception:
        return None

def split_text_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text]
    out = []
    i = 0
    step = max(1, max_chars - overlap)
    while i < len(text):
        out.append(text[i:i + max_chars])
        i += step
    return out

def split_paragraphs(text: str, max_chars: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    final_chunks = []
    for m in merged:
        final_chunks.extend(split_text_with_overlap(m, max_chars, overlap))
    return final_chunks

def find_section_hint(page_text: str) -> str:
    if not page_text:
        return ""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for ln in lines[:15]:
        if 4 <= len(ln) <= 90 and ln.count(".") <= 1 and not ln.endswith("."):
            return ln[:90]
    return ""

def add_context_windows(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    texts = [c.get("text_sem", "") for c in chunks]
    for i, c in enumerate(chunks):
        prev_txt = texts[i - 1] if i > 0 else ""
        next_txt = texts[i + 1] if i + 1 < len(texts) else ""
        c["context_prev"] = prev_txt[-CONTEXT_WINDOW_CHARS:] if prev_txt else ""
        c["context_next"] = next_txt[:CONTEXT_WINDOW_CHARS] if next_txt else ""
    return chunks

def extract_facts(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text[:20000]
    perc = re.findall(r"\b\d+(?:[\.,]\d+)?\s?%\b", t)
    currency = re.findall(r"(?:€\s?\d[\d\.,]*|\$\s?\d[\d\.,]*|\b\d[\d\.,]*\s?(?:EUR|USD)\b)", t)
    facts: Dict[str, Any] = {}
    if perc: facts["percentages"] = list(set(perc[:20]))
    if currency: facts["amounts"] = list(set(currency[:20]))
    return facts



# Candidate patterns
MATH_CANDIDATE_PAT = re.compile(
    r"("
    r"[∑∏∫√=≈≠≤≥→↔∩∪∞±×÷]"
    r"|[_^]\{?"
    r"|\bP\s*\("
    r"|\bPr\s*\("
    r"|\blift\b|\bsupport\b|\bconfidence\b"
    r"|\bformula\b|\bequation\b"
    r"|\bprobabilit[aà]\b|\bprobability\b"
    r")",
    re.IGNORECASE
)



# 3. Pattern per Elementi Visuali (Bilingue)
CHART_CANDIDATE_PAT = re.compile(
    r"\b("
    r"chart|graph|figure|table|diagram|heatmap|candlestick|ohlc|volume|axis|legend|"
    r"grafico|tabella|figura|diagramma|candela|volumi|asse|legenda|trend"
    r")\b",
    re.IGNORECASE
)


#def is_keyword_candidate(text: str) -> bool:
#    t = (text or "").lower()
#    if len(t) < KG_MIN_LEN:
#        return False
#    return any(k in t for k in KG_KEYWORDS)



def is_keyword_candidate(text: str) -> bool:
    """
    Determina se un chunk è un candidato per l'estrazione KG.
    Verifica la lunghezza minima e la presenza di keyword strategiche bilingue.
    """
    if not text:
        return False
        
    # Controllo lunghezza minima (definito nelle variabili di CONFIG)
    if len(text) < KG_MIN_LEN:
        return False
        
    # Match bilingue con parole intere
    return bool(_KG_PAT.search(text))

def is_formula_candidate_page(page_text: str) -> bool:
    return bool(page_text and MATH_CANDIDATE_PAT.search(page_text))

def is_chart_candidate_page(page_text: str) -> bool:
    return bool(page_text and CHART_CANDIDATE_PAT.search(page_text))



# TIME CHECKER START
def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)

def log_phase(filename: str, label: str, ms: int):
    print(f"   ⏱️ {filename} | {label}: {ms} ms")
# TIME CHECKER END


# =========================
# Postgres helpers
# =========================
def pg_get_conn():
    return pg_pool.getconn()

def pg_put_conn(conn):
    pg_pool.putconn(conn)

def pg_start_log(source_name: str, source_type: str) -> int:
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingestion_logs (source_name, source_type, ingestion_ts, status) "
                "VALUES (%s, %s, NOW(), %s) RETURNING log_id",
                (source_name, source_type, "RUNNING")
            )
            log_id = cur.fetchone()[0]
        conn.commit()
        return log_id
    finally:
        pg_put_conn(conn)

def pg_close_log(log_id: int, status: str, total_chunks: int, processing_ms: int, error_msg: str = None):
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_logs SET status = %s, total_chunks = %s, processing_time_ms = %s, error_message = %s "
                "WHERE log_id = %s",
                (status, total_chunks, processing_ms, error_msg, log_id)
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        pg_put_conn(conn)

def pg_get_image_by_hash(image_hash: str, cur) -> Optional[Tuple[int, str]]:
    cur.execute(
        "SELECT image_id, description_ai FROM ingestion_images WHERE image_hash = %s LIMIT 1",
        (image_hash,)
    )
    return cur.fetchone()

def pg_save_image(log_id: int, image_bytes: bytes, mime_type: str, description: str, cur) -> int:
    img_hash = sha256_hex(image_bytes)
    cached = pg_get_image_by_hash(img_hash, cur)
    if cached:
        return cached[0]
    cur.execute(
        "INSERT INTO ingestion_images (log_id, image_data, image_hash, mime_type, description_ai, ingestion_ts) "
        "VALUES (%s, %s, %s, %s, %s, NOW()) RETURNING image_id",
        (log_id, psycopg2.Binary(image_bytes), img_hash, mime_type, description)
    )
    return cur.fetchone()[0]

def flush_postgres_chunks_batch(batch_data: List[Tuple]):
    if not batch_data:
        return
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO document_chunks (log_id, chunk_index, toon_type, content_raw, content_semantic, metadata_json, chunk_uuid)
                VALUES %s
                """,
                batch_data
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ⚠️ Postgres Batch Error: {e}")
    finally:
        pg_put_conn(conn)


# =========================
# Qdrant helpers
# =========================
def ensure_qdrant_collection():
    dim = embedder.get_sentence_embedding_dimension()
    try:
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        if info.config.params.vectors.size != dim:
            print(f"⚠️ Vector dim mismatch! {info.config.params.vectors.size} vs {dim}")
    except Exception:
        print(f"🆕 Creating Qdrant collection '{QDRANT_COLLECTION}' (dim={dim})")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

NEO4J_BATCH_QUERY = """
UNWIND $rows AS r

// 1) Documento (Radice)
MERGE (d:Document {doc_id: r.doc_id})
SET d.filename = r.filename,
    d.doc_type = r.doc_type,
    d.log_id = r.log_id,
    d.ingested_at = datetime()

// 2) Pagina (Contesto intermedio)
WITH d, r
MERGE (p:Page {pid: r.doc_id + "::" + toString(r.page_no)})
SET p.doc_id = r.doc_id,
    p.page_no = r.page_no
MERGE (d)-[:HAS_PAGE]->(p)

// 3) Chunk (Il perno della granularità)
WITH p, r
MERGE (c:Chunk {id: r.chunk_id})
SET c.chunk_index = r.chunk_index,
    c.toon_type = r.toon_type,
    c.page = r.page_no,
    c.text = left(r.text_sem, 1000), 
    c.section_hint = coalesce(r.section_hint, ""),
    c.ontology = r.ontology
MERGE (p)-[:HAS_CHUNK]->(c)

// 4) Esplosione Entità: MENTIONED_IN verso il CHUNK
WITH r, c
CALL (r, c) {
  UNWIND coalesce(r.nodes, []) AS n
  WITH n, c
  WHERE n.id IS NOT NULL AND n.id <> ""

  MERGE (e:Entity {id: n.id})
  ON CREATE SET e.label = n.label,
                e.kinds = [coalesce(n.type, "Entity")]
  SET e += coalesce(n.props, {}),
      e.label = coalesce(e.label, n.label)

  // Aggiorna i tipi di entità senza duplicati
  FOREACH (_ IN CASE
    WHEN NOT coalesce(n.type,"Entity") IN coalesce(e.kinds, [])
    THEN [1] ELSE []
  END |
    SET e.kinds = coalesce(e.kinds, []) + [coalesce(n.type,"Entity")]
  )

  // RELAZIONE GRANULARE: L'entità appartiene a QUESTO specifico frammento di testo
  MERGE (e)-[:PRESENT_IN]->(c)
  RETURN count(*) AS node_count
}

// 5) Relazioni tra entità (Knowledge Graph puro)
WITH r
CALL (r) {
  UNWIND coalesce(r.edges, []) AS rel
  WITH rel
  WHERE rel.source IS NOT NULL AND rel.target IS NOT NULL
    AND rel.source <> rel.target
    AND rel.relation IS NOT NULL AND rel.relation <> ""

  MERGE (s:Entity {id: rel.source})
  MERGE (t:Entity {id: rel.target})
  WITH s, t, rel

  CALL apoc.merge.relationship(s, rel.relation, {}, {}, t, {}) YIELD rel AS r_out
  SET r_out += coalesce(rel.props, {}),
      r_out.last_seen = datetime(),
      r_out.count = coalesce(r_out.count, 0) + 1
  RETURN count(*) AS rel_count
}

RETURN count(*) AS processed
"""






# Formula nodes deterministici
NEO4J_FORMULA_QUERY = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk_id})
MERGE (f:Formula {fid: r.fid})
SET f.latex = r.latex, f.plain = r.plain, f.meaning_it = r.meaning_it, f.keywords = r.keywords,
    f.page = r.page_no, f.source = r.filename
MERGE (f)-[:MENTIONED_IN]->(c)
"""

def _flat_props(props) -> Dict[str, Any]:
    if not isinstance(props, dict):
        return {}
    out = {}
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool)):
            out[k[:60]] = v
        elif isinstance(v, list) and len(v) <= 12 and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k[:60]] = v
    return out

def _clean_type(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"[^a-zA-Z0-9]", "", t[:60].capitalize())
    return t[:50] or "Entity"

def _clean_rel(r: str) -> str:
    r = (r or "").strip().upper()
    r = re.sub(r"[^A-Z0-9_]+", "_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    return r[:50] or "RELATED_TO"

def _sanitize_graph(graph: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(graph, dict):
        return [], []
    nodes = []
    seen = set()
    for n in graph.get("nodes", []) or []:
        nid = normalize_entity_id(n.get("id") or n.get("label"))
        if nid and nid not in seen:
            seen.add(nid)
            nodes.append({
                "id": nid,
                "label": (n.get("label") or nid)[:200],
                "type": _clean_type(n.get("type") or "Entity"),
                "props": _flat_props(n.get("properties", {}) or {})
            })
    edges = []
    for e in graph.get("edges", []) or []:
        s = normalize_entity_id(e.get("source"))
        t = normalize_entity_id(e.get("target"))
        if s and t:
            edges.append({
                "source": s,
                "target": t,
                "relation": _clean_rel(e.get("relation")),
                "props": _flat_props(e.get("properties", {}) or {})
            })
    return nodes, edges

def flush_neo4j_rows_batch(rows: List[Dict[str, Any]]):
    if not NEO4J_ENABLED or not rows:
        return
    try:
        with neo4j_driver.session() as session:
            session.run(NEO4J_BATCH_QUERY, rows=rows)
    except Exception as e:
        print(f"   ⚠️ Neo4j Batch Error: {e}")

def flush_neo4j_formulas_batch(rows: List[Dict[str, Any]]):
    if not NEO4J_ENABLED or not rows:
        return
    try:
        with neo4j_driver.session() as session:
            session.run(NEO4J_FORMULA_QUERY, rows=rows)
    except Exception as e:
        print(f"   ⚠️ Neo4j Formula Batch Error: {e}")


# =========================
# LLM / Vision
# =========================
def llm_chat(prompt: str, text: str, model: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    for _ in range(LLM_RETRIES + 1):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content or ""
        except Exception:
            time.sleep(0.35)
    return ""

def llm_chat_multimodal(prompt: str, image_bytes: bytes, model: str, max_tokens: int = 900) -> str:
    if not image_bytes:
        return ""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    for _ in range(LLM_RETRIES + 1):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content or ""
        except Exception:
            time.sleep(0.35)
    return ""

def _downscale_and_compress_for_vision(img_bytes: bytes, max_side: int = 1200, max_bytes: int = VISION_MAX_IMAGE_BYTES) -> Optional[bytes]:
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

        q = 85
        out = io.BytesIO()
        im.save(out, "JPEG", quality=q, optimize=True)
        data = out.getvalue()

        while len(data) > max_bytes and q >= 50:
            q -= 7
            out = io.BytesIO()
            im.save(out, "JPEG", quality=q, optimize=True)
            data = out.getvalue()
        return data
    except Exception:
        return None

def render_full_page_jpeg(page: fitz.Page, dpi: int = VISION_DPI) -> bytes:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("jpg")

# --- Vision cache (in-memory) ---
# chiave: sha256(image_bytes) / valore: json vision
_vision_cache: Dict[str, Dict[str, Any]] = {}
_vision_cache_order: List[str] = []

def _vision_cache_get(key: str) -> Optional[Dict[str, Any]]:
    return _vision_cache.get(key)

def _vision_cache_put(key: str, val: Dict[str, Any]):
    if key in _vision_cache:
        return
    _vision_cache[key] = val
    _vision_cache_order.append(key)
    if len(_vision_cache_order) > VISION_CACHE_MAX:
        old = _vision_cache_order.pop(0)
        _vision_cache.pop(old, None)

def extract_formulas_via_vision(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    vbytes = _downscale_and_compress_for_vision(img_bytes)
    if not vbytes:
        return None
    key = sha256_hex(vbytes) + "::formula"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    raw = llm_chat_multimodal(FORMULA_VISION_PROMPT, vbytes, VISION_MODEL_NAME, max_tokens=900)
    js = safe_json_extract(raw)
    if isinstance(js, dict) and "has_formulas" in js and "formulas" in js:
        if not isinstance(js.get("formulas"), list):
            js["formulas"] = []
        _vision_cache_put(key, js)
        return js
    return None

def extract_chart_via_vision(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    vbytes = _downscale_and_compress_for_vision(img_bytes)
    if not vbytes:
        return None
    key = sha256_hex(vbytes) + "::chart"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    raw = llm_chat_multimodal(CHART_VISION_PROMPT, vbytes, VISION_MODEL_NAME, max_tokens=900)
    js = safe_json_extract(raw)
    if isinstance(js, dict) and "kind" in js and "key_points" in js and "numbers" in js:
        js.setdefault("title", "")
        js.setdefault("source", "")
        js.setdefault("timeframe", "")
        js.setdefault("axes", {"x": "", "y": ""})
        js.setdefault("legend", [])
        js.setdefault("data_points", [])
        js.setdefault("entities", [])
        js.setdefault("unreadable_parts", [])
        _vision_cache_put(key, js)
        return js
    return None

def reconcile_chart_with_page_text(page_text: str, vision_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # opzionale: è un'altra chiamata LLM, quindi la facciamo solo se serve
    # euristica: se title/timeframe vuoti e page_text contiene parole chiave, tentiamo reconcile
    need = (not vision_json.get("title") and not vision_json.get("timeframe") and len(page_text) > 200)
    if not need:
        return None
    payload = f"PAGE_TEXT:\n{normalize_ws(page_text)[:6000]}\n\nVISION_JSON:\n{json.dumps(vision_json, ensure_ascii=False)[:12000]}"
    raw = llm_chat(CHART_RECONCILE_PROMPT, payload, LLM_MODEL_NAME, max_tokens=800)
    js = safe_json_extract(raw)
    if isinstance(js, dict) and "kind" in js:
        return js
    return None


# =========================
# Chunk builders
# =========================
def build_formula_semantic_chunk(page_no: int, formulas_json: Dict[str, Any]) -> str:
    lines = [f"FORMULAS (Page {page_no}):"]
    for f in (formulas_json.get("formulas") or [])[:VISION_MAX_FORMULAS_PER_PAGE]:
        latex = (f.get("latex") or "").strip()
        plain = (f.get("plain") or "").strip()
        meaning = (f.get("meaning_it") or "").strip()
        if latex:
            lines.append(f"- LaTeX: {latex}")
        if plain:
            lines.append(f"  Plain: {plain}")
        if meaning:
            lines.append(f"  Meaning: {meaning}")
    notes = formulas_json.get("notes") or []
    if notes:
        lines.append("Notes: " + "; ".join(str(x) for x in notes[:6]))
    return normalize_ws("\n".join(lines))

def build_chart_semantic_chunk(page_no: int, chart_json: Dict[str, Any], prefix: str = "VISUAL") -> str:
    kind = chart_json.get("kind", "other")
    title = chart_json.get("title", "") or ""
    timeframe = chart_json.get("timeframe", "") or ""
    axes = chart_json.get("axes", {}) or {}
    legend = chart_json.get("legend", []) or []
    kps = chart_json.get("key_points", []) or []
    nums = chart_json.get("numbers", []) or []

    lines = [f"{prefix} (Page {page_no}) kind={kind}"]
    if title:
        lines.append(f"Title: {title}")
    if timeframe:
        lines.append(f"Timeframe: {timeframe}")
    if axes.get("x") or axes.get("y"):
        lines.append(f"Axes: x={axes.get('x','')} | y={axes.get('y','')}")
    if legend:
        lines.append("Legend: " + ", ".join(str(x) for x in legend[:12]))
    if nums:
        lines.append("Numbers:")
        for n in nums[:12]:
            label = (n.get("label") or "").strip()
            val = (n.get("value") or "").strip()
            unit = (n.get("unit") or "").strip()
            period = (n.get("period") or "").strip()
            row = f"- {label}: {val}"
            if unit:
                row += f" {unit}"
            if period:
                row += f" ({period})"
            lines.append(row)
    if kps:
        lines.append("Key points: " + "; ".join(str(x) for x in kps[:8]))
    unread = chart_json.get("unreadable_parts") or []
    if unread:
        lines.append("Unreadable: " + "; ".join(str(x) for x in unread[:5]))
    return normalize_ws("\n".join(lines))


# =========================
# KG extraction (LLM) - SOLO dove serve
# =========================
import time

def llm_extract_kg(filename: str, page_no: int, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    t0 = time.perf_counter() # TIME CHECHER

    kg_text = f"SOURCE: {filename} p.{page_no}\nCONTENT:\n{normalize_ws(text)[:11000]}"
    raw = llm_chat(
        KG_PROMPT,
        kg_text,
        LLM_MODEL_NAME,
        max_tokens=LLM_MAX_TOKENS
    )

    dt = time.perf_counter() - t0 # TIME CHECHER
    print(f"   ⏱️ {filename} | KG LLM p{page_no}: {dt:.2f}s")

    js = safe_json_extract(raw)
    if not js:
        return [], []

    return _sanitize_graph(js)



# =========================
# PDF extraction (FAST)
# =========================
def extract_pdf_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    - testo: chunk sempre
    - formule: full-page vision se candidate (priorità)
    - immagini embedded: vision PARALLELA + cache
    """
    out_chunks: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)

    pg_conn = pg_get_conn()
    pg_conn.autocommit = False
    pg_cur = pg_conn.cursor()

    try:
        total_pages = len(doc)
        print(f"   🔍 PDF pages: {total_pages}")

        # ThreadPool per vision su embedded images
        executor = ThreadPoolExecutor(max_workers=VISION_PARALLEL_WORKERS) if (PDF_VISION_ON_EMBEDDED_IMAGES and PDF_VISION_ENABLED) else None

        for page_idx in range(total_pages):
            page_no = page_idx + 1
            page = doc[page_idx]

            page_text = normalize_ws(page.get_text("text"))
            section_hint = find_section_hint(page_text)
            
            # Fast check: se la pagina non ha immagini embedded e non è candidata chart/formule,
            # evita lavori extra (soprattutto su tesi/slide pesanti)
            has_embedded_imgs = False
            if PDF_EXTRACT_EMBEDDED_IMAGES:
                try:
                    has_embedded_imgs = len(page.get_images(full=True)) > 0
                except Exception:
                    has_embedded_imgs = False



            # 1) Text chunks
            if page_text:
                for ch in split_paragraphs(page_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                    out_chunks.append({
                        "text_raw": ch,
                        "text_sem": ch,
                        "page_no": page_no,
                        "toon_type": "text",
                        "section_hint": section_hint
                    })

            # 2) Formula full-page vision
            do_formula_vision = PDF_VISION_ENABLED and is_formula_candidate_page(page_text)
            if PDF_VISION_ONLY_IF_TEXT_SCARSO and len(page_text) >= PDF_MIN_TEXT_LEN_FOR_NO_VISION:
                do_formula_vision = False

            if do_formula_vision:
                try:
                    full_jpg = render_full_page_jpeg(page, dpi=VISION_DPI)
                    formulas_json = extract_formulas_via_vision(full_jpg)
                    if formulas_json and formulas_json.get("has_formulas") and formulas_json.get("formulas"):
                        _ = pg_save_image(log_id, full_jpg, "image/jpeg", f"PDF page {page_no} formulas", cur=pg_cur)
                        sem = build_formula_semantic_chunk(page_no, formulas_json)
                        out_chunks.append({
                            "text_raw": sem,
                            "text_sem": sem,
                            "page_no": page_no,
                            "toon_type": "formula_page",
                            "section_hint": section_hint,
                            "metadata_override": formulas_json
                        })
                except Exception as e:
                    print(f"   ⚠️ Formula vision failed p.{page_no}: {e}")

            # 3) Embedded images -> save + vision parallel
            
            
            # Se non ci sono immagini embedded, salta direttamente il blocco immagini
            future_to_meta = {}
            if PDF_EXTRACT_EMBEDDED_IMAGES and PDF_VISION_ENABLED and has_embedded_imgs:
                try:
                    img_list = page.get_images(full=True)[:PDF_MAX_IMAGES_PER_PAGE]
                    for k, img in enumerate(img_list):
                        xref = img[0]
                        base = doc.extract_image(xref)
                        ibytes = base.get("image", b"")
                        ext = base.get("ext", "bin")
                        if len(ibytes) < MIN_IMAGE_BYTES:
                            continue

                        iid = pg_save_image(log_id, ibytes, f"image/{ext}", f"Embedded image p.{page_no} idx={k}", cur=pg_cur)

                        # Vision chart/table (parallel)
                        if executor is not None:
                            fut = executor.submit(extract_chart_via_vision, ibytes)
                            future_to_meta[fut] = (page_no, section_hint, iid, page_text)
                        else:
                            # solo ref
                            out_chunks.append({
                                "text_raw": f"EMBEDDED_IMAGE_REF (p.{page_no}) image_id={iid}",
                                "text_sem": f"EMBEDDED_IMAGE_REF (p.{page_no}) image_id={iid}",
                                "page_no": page_no,
                                "toon_type": "image_ref",
                                "section_hint": section_hint,
                                "image_id": iid
                            })

                except Exception as e:
                    print(f"   ⚠️ Embedded images failed p.{page_no}: {e}")

            # Raccogli risultati vision embedded
            if future_to_meta:
                for fut in as_completed(list(future_to_meta.keys())):
                    try:
                        vis = fut.result()
                        if not vis:
                            continue

                        page_no2, section_hint2, iid2, page_text2 = future_to_meta[fut]

                        rec = reconcile_chart_with_page_text(page_text2, vis)
                        final = rec if rec else vis

                        sem = build_chart_semantic_chunk(page_no2, final, prefix="EMBEDDED_VISUAL")
                        out_chunks.append({
                            "text_raw": sem,
                            "text_sem": sem,
                            "page_no": page_no2,
                            "toon_type": "image_vision",
                            "section_hint": section_hint2,
                            "metadata_override": final,
                            "image_id": iid2
                        })
                    except Exception:
                        pass

            if page_no % PG_COMMIT_EVERY_N_PAGES == 0:
                pg_conn.commit()

        pg_conn.commit()
        if executor:
            executor.shutdown(wait=True)
        return add_context_windows(out_chunks)

    finally:
        try:
            pg_cur.close()
        except Exception:
            pass
        try:
            pg_put_conn(pg_conn)
        except Exception:
            pass
        try:
            doc.close()
        except Exception:
            pass


# =========================
# FILE DISPATCH (PDF only here)
# =========================
def extract_file_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_chunks(file_path, log_id)
    return []


# =========================
# MAIN: embed + store + neo4j
# =========================
def process_single_file(file_path: str, doc_type: str, doc_meta: dict):

    # Lazy init clients (solo quando serve davvero)
    _ = get_openai_client()
    _ = get_qdrant_client()
    _ = get_embedder()
    ensure_qdrant_collection()

    filename = os.path.basename(file_path)
    # --- Doc organization (Tier + Ontology) ---
    doc_meta = doc_meta or {}

    tier = doc_meta.get("tier", "B")
    content_type = doc_meta.get("content_type", "reference")
    source_kind = doc_meta.get("source_kind", "internal")
    source = doc_meta.get("source", "")
    source_quality = doc_meta.get("source_quality", "")
    ontology = (doc_meta.get("ontology") or DEFAULT_ONTOLOGY).strip().lower()
    topics = doc_meta.get("topics", []) or []
    effective_date = doc_meta.get("effective_date")
    
    start_time = time.time()
    print(f"   ⚙️ Engine Start: {filename}")

    doc_sha = ""
    try:
        doc_sha = sha256_file(file_path)
    except Exception:
        pass
    doc_id = f"{filename}::{doc_sha[:16] if doc_sha else 'nohash'}"
    log_id = pg_start_log(filename, doc_type)

# TIME CHECKER START 
    t_extract = time.time()
    chunks = extract_file_chunks(file_path, log_id)
    log_phase(filename, "extract_file_chunks", _ms(t_extract))
# TIME CHECKER END


    try:
        valid_chunks = [c for c in chunks if len(c.get("text_sem", "")) >= MIN_CHUNK_LEN]
        kg_pages_done: set[int] = set()

        
        if not valid_chunks:
            pg_close_log(log_id, "SKIPPED_EMPTY", 0, 0)
            return

        print(f"   🚀 Chunks: {len(valid_chunks)}. Embedding + flush...")

        qdrant_points: List[models.PointStruct] = []
        pg_rows: List[Tuple] = []
        neo4j_rows: List[Dict[str, Any]] = []
        neo4j_formula_rows: List[Dict[str, Any]] = []

        kg_chunks_used = 0

        for i in range(0, len(valid_chunks), EMBED_BATCH_SIZE):
            batch = valid_chunks[i:i + EMBED_BATCH_SIZE]
            texts = [b["text_sem"][:QDRANT_TEXT_MAX_CHARS] for b in batch]

            # 1) EMBEDDING (una sola volta, prima di usare vecs)
            t_embed = time.time()
            vecs = get_embedder().encode(texts, normalize_embeddings=True)
            embed_ms = _ms(t_embed)
            log_phase(filename, f"embed_batch_{len(texts)}", embed_ms)

            # 2) Costruzione payload e righe DB usando vecs
            for j, ch in enumerate(batch):
                c_idx = i + j
                toon_type = ch.get("toon_type", "text")
                page_no = ch.get("page_no", 1)

                image_id = ch.get("image_id")
                c_uuid = deterministic_chunk_id(
                    doc_id=doc_id,
                    page_no=page_no,
                    chunk_index=c_idx,
                    toon_type=toon_type,
                    text_sem=ch.get("text_sem", ""),
                    image_id=image_id,
                )

                vec = vecs[j].tolist()
                section_hint = ch.get("section_hint", "")
                meta_override = ch.get("metadata_override", {}) or {}

                payload = {
                    "tier": tier,
                    "content_type": content_type,
                    "source_kind": source_kind,
                    "source": source,
                    "source_quality": source_quality,
                    "ontology": ontology,
                    "topics": topics,
                    "effective_date": effective_date,
                    "filename": filename,
                    "doc_id": doc_id,
                    "type": doc_type,
                    "page": page_no,
                    "chunk_index": c_idx,
                    "toon_type": toon_type,
                    "section_hint": section_hint,
                    "text_sem": ch["text_sem"][:QDRANT_TEXT_MAX_CHARS],
                    "facts": extract_facts(ch["text_sem"]),
                    "vision_json": meta_override,
                    "image_id": ch.get("image_id"),
                }

                qdrant_points.append(models.PointStruct(id=c_uuid, vector=vec, payload=payload))
                pg_rows.append((log_id, c_idx, toon_type, ch["text_raw"], ch["text_sem"], Json(payload), c_uuid))

                # ===== Neo4j part =====
                kg_nodes, kg_edges = [], []

                if toon_type == "formula_page":
                    formulas = (meta_override.get("formulas") or [])
                    for f in formulas[:VISION_MAX_FORMULAS_PER_PAGE]:
                        latex = (f.get("latex") or "").strip()
                        plain = (f.get("plain") or "").strip()
                        meaning = (f.get("meaning_it") or "").strip()
                        keywords = f.get("keywords") or []
                        fid_src = f"{filename}::{page_no}::{latex}::{plain}"
                        fid = sha256_hex(fid_src.encode("utf-8"))[:24]
                        neo4j_formula_rows.append({
                            "chunk_id": c_uuid,
                            "fid": fid,
                            "latex": latex,
                            "plain": plain,
                            "meaning_it": meaning,
                            "keywords": keywords[:12] if isinstance(keywords, list) else [],
                            "page_no": page_no,
                            "filename": filename
                        })

                if KG_ENABLED and toon_type in ("image_vision", "text"):
                    if page_no in kg_pages_done:
                        kg_nodes, kg_edges = [], []
                    else:
                        rich = (
                            toon_type == "image_vision"
                            or (toon_type == "text" and is_keyword_candidate(ch["text_sem"]))
                        )
                        if rich and kg_chunks_used < MAX_KG_CHUNKS_PER_DOC:
                            kg_nodes, kg_edges = llm_extract_kg(filename, page_no, ch["text_sem"])
                            kg_chunks_used += 1
                            kg_pages_done.add(page_no)

                if NEO4J_ENABLED:
                    neo4j_rows.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "doc_type": doc_type,
                        "log_id": log_id,
                        "chunk_id": c_uuid,
                        "chunk_index": c_idx,
                        "toon_type": toon_type,
                        "page_no": page_no,
                        "section_hint": section_hint,
                        "text_sem": ch["text_sem"],
                        "nodes": kg_nodes,
                        "edges": kg_edges
                    })


            # flush parziale
            if len(qdrant_points) >= DB_FLUSH_SIZE:
                t_flush = time.time()

                # 1) Qdrant
                if qdrant_points:
                    qdrant_client.upsert(QDRANT_COLLECTION, points=qdrant_points)

                # 2) Postgres
                if pg_rows:
                    flush_postgres_chunks_batch(pg_rows)

                # 3) Neo4j (KG + formule)
                if neo4j_rows:
                    flush_neo4j_rows_batch(neo4j_rows)

                if neo4j_formula_rows:
                    flush_neo4j_formulas_batch(neo4j_formula_rows)

                # 4) Timing flush
                log_phase(filename, "flush_all", _ms(t_flush))

                # 5) Reset buffer
                qdrant_points.clear()
                pg_rows.clear()
                neo4j_rows.clear()
                neo4j_formula_rows.clear()

                print(".", end="", flush=True)

        # flush finale
        if qdrant_points:
            qdrant_client.upsert(QDRANT_COLLECTION, points=qdrant_points)

        flush_postgres_chunks_batch(pg_rows)
        flush_neo4j_rows_batch(neo4j_rows)
        flush_neo4j_formulas_batch(neo4j_formula_rows)

        elapsed = int((time.time() - start_time) * 1000)
        pg_close_log(log_id, "COMPLETED", len(valid_chunks), elapsed)
        print(f"\n✅ Done: {filename} in {elapsed}ms")

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))

    except Exception as e:
        print(f"❌ Error: {e}")
        pg_close_log(log_id, "FAILED", 0, 0, str(e))
        os.makedirs(FAILED_DIR, exist_ok=True)
        try:
            shutil.move(file_path, os.path.join(FAILED_DIR, filename))
        except Exception:
            pass


def main():

    os.makedirs(INBOX_DIR, exist_ok=True)
    ensure_inbox_structure(INBOX_DIR)
    print("=== Ingestion Engine v2.3 (FAST + Formulas deterministic + Charts Vision) ===")

    supported = {".pdf"}  # lasciamo invariato per ora (solo PDF), come nel tuo script

    # 1) Raccogli i file supportati (senza processarli subito)
    pdf_files = []
    for root, _, files in os.walk(INBOX_DIR):
        for fname in files:
            if fname.lower().endswith(".meta.json"):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue

            pdf_files.append((root, os.path.join(root, fname)))

    # 2) Early-exit: se INBOX è vuota, non inizializzare nulla (no HF / no embedder / no Qdrant)
    if not pdf_files:
        print("✅ INBOX vuota: create solo le directory. Nessuna ingestion eseguita.")
        return

    # 3) Processa i file
    for root, file_path in pdf_files:
        # Dispatcher: Tier (da cartella) + Ontology (2° livello) + (opzionale) override sidecar
        doc_meta = dispatch_document(file_path, root)

        process_single_file(file_path, "pdf", doc_meta)


if __name__ == "__main__":
    main()
