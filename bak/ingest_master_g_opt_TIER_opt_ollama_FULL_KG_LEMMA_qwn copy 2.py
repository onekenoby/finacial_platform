
"""
set EMBED_BATCH_SIZE=16
set DB_FLUSH_SIZE=96
set VISION_PARALLEL_WORKERS=3
set VISION_DPI=150
set MAX_KG_CHUNKS_PER_DOC=10

set PG_COMMIT_EVERY_N_PAGES=25
"""

"""
Ingestion Engine - v2.4 HYPER-FAST (Virtual Markdown + Asset Parking)
✅ Strategy: PDF -> Virtual MD + Image Asset Park (RAM)
✅ Vision: Surgical AI on parked assets only (Gemma 3 12B)
✅ Value Hunter: Tier-based selective KG + AI Gatekeeper (num_predict: 2)
✅ Hardware: Optimized for P5000 (16GB VRAM, num_ctx: 3072)
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
import concurrent.futures as cf
import subprocess
import requests
from threading import Lock

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import unicodedata

import shutil # Aggiungi questo import in cima al file

import fitz  # PyMuPDF
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import errors as pg_errors

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from openai import OpenAI

# OLLAMA
from ollama import chat
from ollama import ChatResponse

from pdfminer.layout import LAParams

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
except Exception:
    extract_pages = None
    LTTextContainer = None




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
    rel = os.path.relpath(root_dir, INBOX_DIR).replace("\\", "/")
    parts = [p for p in rel.split("/") if p and p != "."]

    tier_key = parts[0].upper() if len(parts) >= 1 else ""
    
    # Se esiste una sottocartella (es. INBOX/TIER_A/FINANCE), usa quella come ontology.
    # Altrimenti, usa il 'content_type' del Tier (es. methodology) invece di 'generic'.
    if len(parts) >= 2:
        ontology = parts[1].lower()
    else:
        ontology = TIER_FOLDERS.get(tier_key, {}).get("content_type", DEFAULT_ONTOLOGY)
    
    base = dict(TIER_FOLDERS.get(tier_key, DEFAULT_TIER_META))
    base["ontology"] = ontology
    
    # topics e data effettiva (invariati)
    fname = os.path.basename(file_path)
    base["topics"] = infer_topics_regex(fname)[:6]
    if base.get("tier") == "C" and not base.get("effective_date"):
        base["effective_date"] = time.strftime("%Y-%m-%d")

    side = read_sidecar_meta(file_path)
    if isinstance(side, dict) and side:
        base.update(side)
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

# -------------------------
# KG LIMITS (coherent names)
# -------------------------
# These are the ONLY canonical names used by the pipeline.
# A few aliases are kept for backward-compat with older snippets.
MIN_ENTITY_DENSITY = int(os.getenv("MIN_ENTITY_DENSITY", "1"))
MIN_FIN_KEYWORDS = int(os.getenv("MIN_FIN_KEYWORDS", "1"))

KG_TEXT_MAX_CHARS = int(os.getenv("KG_TEXT_MAX_CHARS", "4000"))   # chars sent to KG model per page
KG_MAX_TRIPLES = int(os.getenv("KG_MAX_TRIPLES", "30"))           # soft cap (sanitize already caps)
KG_TIMEOUT = int(os.getenv("KG_TIMEOUT", "120"))                   # seconds per KG task/page

# Backward-compat aliases (do NOT use in new code)
KG_CHARS_LIMIT = KG_TEXT_MAX_CHARS
KG_MAX_CHARS = KG_TEXT_MAX_CHARS
KG_MAX_TRIPLES_PER_PAGE = KG_MAX_TRIPLES
MAX_TRIPLES_KG = KG_MAX_TRIPLES
KG_TASK_TIMEOUT = KG_TIMEOUT
KG_TASK_TIMEOUT_PER_PAGE = KG_TIMEOUT
KG_TIMEOUT_PER_PAGE = KG_TIMEOUT





BASE_DATA_DIR = "./data_ingestion"
INBOX_DIR = os.path.join(BASE_DATA_DIR, "INBOX")
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "PROCESSED")
FAILED_DIR = os.path.join(BASE_DATA_DIR, "FAILED")

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "40"))

CONTEXT_WINDOW_CHARS = int(os.getenv("CONTEXT_WINDOW_CHARS", "260"))
INCLUDE_CONTEXT_IN_KG = os.getenv("INCLUDE_CONTEXT_IN_KG", "1") == "1"

DB_FLUSH_SIZE = int(os.getenv("DB_FLUSH_SIZE", "200"))          # un po' più alto per meno flush - 
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))    # se la tua VRAM regge, aumenta velocità

# Vision switches
PDF_VISION_ENABLED = os.getenv("PDF_VISION_ENABLED", "1") == "1"
PDF_VISION_ONLY_IF_TEXT_SCARSO = False #= os.getenv("PDF_VISION_ONLY_IF_TEXT_SCARSO", "0") == "1"
PDF_MIN_TEXT_LEN_FOR_NO_VISION = 0 #= int(os.getenv("PDF_MIN_TEXT_LEN_FOR_NO_VISION", "450"))

VISION_DPI = int(os.getenv("VISION_DPI", "160"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "2000000"))

VISION_MAX_FORMULAS_PER_PAGE = int(os.getenv("VISION_MAX_FORMULAS_PER_PAGE", "10"))

# --- SOGLIE ASSET VISUALI ---
# Immagini più piccole di questo valore (in byte) verranno ignorate.
# 7000 = icone/loghi | 2000 = molto permissivo | 15000 = molto sever
PDF_EXTRACT_EMBEDDED_IMAGES = True #= os.getenv("PDF_EXTRACT_EMBEDDED_IMAGES", "1") == "1"
PDF_VISION_ON_EMBEDDED_IMAGES = True # = os.getenv("PDF_VISION_ON_EMBEDDED_IMAGES", "1") == "1"
PDF_MAX_IMAGES_PER_PAGE = int(os.getenv("PDF_MAX_IMAGES_PER_PAGE", "8"))
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", "1"))
MIN_ASSET_SIZE = int(os.getenv("MIN_ASSET_SIZE", "1"))


# Speed: Vision parallel + cache
VISION_PARALLEL_WORKERS = 2 #int(os.getenv("VISION_PARALLEL_WORKERS", "4"))  # 4-6 di solito ok
OLLAMA_NUM_PARALLEL=4
VISION_CACHE_MAX = int(os.getenv("VISION_CACHE_MAX", "5000"))             # entries in-memory

# Commit policy
PG_COMMIT_EVERY_N_PAGES = int(os.getenv("PG_COMMIT_EVERY_N_PAGES", "25"))

# KG extraction (solo dove serve)
KG_ENABLED = os.getenv("KG_ENABLED", "1") == "1"
KG_MIN_LEN = int(os.getenv("KG_MIN_LEN", "400")) #
MAX_KG_CHUNKS_PER_DOC = int(os.getenv("MAX_KG_CHUNKS_PER_DOC", "1000000"))



PDF_TEXT_EXTRACTOR = "fitz"
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "pdfminer").lower()
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "fitz").lower() #<--- OPZIONALE PIù VELOCE
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "fitz").lower()  # fitz | pdfminer

# --- Nella sezione A. GESTIONE FORMULE ---
FULLPAGE_DPI = 120 
CROP_DPI = 160 
KG_WORKERS = 3  # Forza l'elaborazione seriale per non saturare la VRAM
kg_executor = ThreadPoolExecutor(max_workers=KG_WORKERS)

CID_RE = re.compile(r"\(cid:\d+\)")

REL_CANON_CACHE_PATH = os.getenv("REL_CANON_CACHE_PATH", "relation_canon_cache.json")
REL_CANON_MAX_TOKENS = int(os.getenv("REL_CANON_MAX_TOKENS", "700"))

RELTYPE_OK = re.compile(r"^[A-Z][A-Z_]{2,60}$")


KG_KEYWORDS = [
    # --- Finanza & Performance ---
    "risk", "rischio", "yield", "rendimento", "revenue", "ricavi", "profit", "profitto",
    "earnings", "utili", "ebitda", "margine", "margin", "debt", "debito", "cash", "cassa",
    "forecast", "previsione", "guidance", "outlook", "dividend", "dividendo", "equity",
    "capitale", "loss", "perdita",
    
    # --- Strategia & Mercato ---
    "merger", "fusione", "acquisition", "acquisizione", "partnership", "accordo", 
    "agreement", "subsidiary", "controllata", "stakeholder", "competitor", "concorrente",
    "market", "mercato", "share", "quota", "strategy", "strategia", "ceo", "management",
    
    # --- Macro & Regulation ---
    "inflation", "inflazione", "rate", "tasso", "fed", "fomc", "ecb", "bce", "gdp", "pil","deflaction","deflazione",
    "regulation", "regolamento", "compliance", "normativa", "contract", "contratto",
    "clause", "clausola", "policy", "politica", "esg", "sustainability", "sostenibilità",
    
    # --- Analisi Dati & Visual (già nel tuo script) ---
    "grafico", "graph", "tabella", "trend", "asse", "legenda", "chart", "table", "axis", "legend",
    "regression", "regressione", "model", "modello", "algorithm", "algoritmo", "correlation", "correlazione",
    "inference", "inferenza", "variance", "varianza","slope", "mean", "average", "moda", "mode", "modale", "modal"
]

###################
# --- NUOVO: Filtro Pagine Strutturali (Junk Filter) ---
STRUCTURAL_PAT = re.compile(
    r"\b(Contents|Index|Bibliography|Acknowledgements|Glossary|Appendix|Reference|Table of Contents|"
    r"Indice|Sommario|Bibliografia|Ringraziamenti|Appendice|Riferimenti)\b",
    re.IGNORECASE
)

def is_structural_page(text: str, p_no: int = 1) -> bool:
    """Rileva se la pagina è junk (indice/sommario), ma risparmia la Pagina 1."""
    if not text: return False
    # La Pagina 1 non è MAI considerata puramente strutturale (contiene il titolo/intro)
    if p_no == 1: return False
    return bool(STRUCTURAL_PAT.search(text[:400]))

def get_entity_density(text: str) -> int:
    """
    Heuristica semplice: conta "candidate entità" (Title Case) che non risultano
    essere parole comuni (lowercase) presenti nello stesso testo.

    Serve solo come gating, quindi deve essere:
    - veloce
    - deterministica
    - stabile
    """
    if not text or len(text) < 20:
        return 0

    clean_text = re.sub(r'# PAGE \d+', '', text)

    # Parole in Title Case (min 3 char)
    potential_entities = re.findall(r'\b[A-ZÀ-Ú][a-zà-ú]{2,}\b', clean_text)

    # Vocabolario "comune" in lowercase (min 3 char)
    common_vocab = set(re.findall(r'\b[a-zà-ú]{3,}\b', clean_text))

    true_entities = {w for w in set(potential_entities) if w.lower() not in common_vocab}
    return len(true_entities)


def count_unique_keywords(text: str) -> int:
    """Conta quanti concetti finanziari diversi sono presenti nel chunk."""
    found = set(_KG_PAT.findall(text))
    return len(found)

def ai_gatekeeper_decision(text: str) -> bool:
    """Gemma 3 12B agisce come 'buttafuori' ultra-veloce (2 token max)."""
    try:
        resp = chat(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "user", "content": f"Is this financial text worth a KG? Answer ONLY YES or NO.\n\n{text[:500]}"}],
            options={"temperature": 0.0, "num_predict": 2, "num_ctx": 1024}
        )
        return "YES" in resp['message']['content'].upper()
    except:
        return True
#---------------------

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
#LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234/v1")
#LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-3-12b")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-8b-gguf")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b-finstudio")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-vl-8b-instruct")
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

#def get_openai_client():
#    global openai_client
#    if openai_client is None:
#        openai_client = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
#    return openai_client

def get_embedder():
    global embedder
    if embedder is None:
        # device="cpu" evita conflitti di memoria con Ollama
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
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

FORMULA_VISION_PROMPT = """
You are a STRICT scientific OCR engine for formula sheets and math tables.

RULES (NO HALLUCINATIONS):
- Transcribe ONLY what is visually present.
- Do NOT invent charts/graphs. If it is a grid of numbers -> it's a table.
- If unreadable, write 'NOT_READABLE' (do not guess).

OUTPUT: Return ONLY valid JSON with this EXACT schema:

{
  "summary_it": "Breve sintesi in italiano di ciò che contiene la pagina",
  "formulas": [
    {
      "latex": "LaTeX without $$ delimiters",
      "meaning_it": "Descrizione in italiano (se leggibile) oppure 'NOT_READABLE'",
      "variables": [
        {"name": "r", "meaning": "discount rate"},
        {"name": "n", "meaning": "number of periods"}
      ]
    }
  ],
  "tables_md": [
    {
      "caption": "Titolo tabella o contesto (o 'NOT_READABLE')",
      "markdown": "| ... | ... |\\n|---|---|\\n| ... | ... |"
    }
  ],
  "confidence": 0.0
}
"""


CHART_VISION_PROMPT = """
You are a STRICT BUT USEFUL VISUAL EXTRACTOR for RAG.
Your goal: Extract visual insights (Image description) AND structured data (Vector-ready data) if visible.

ABSOLUTE PROHIBITIONS:
- Do NOT invent numbers. If values are not explicit on bars/lines, do NOT estimate.

Return ONLY valid JSON (no markdown), EXACT schema:

{
  "kind": "chart|table|diagram|photo|other",
  "title": "MAIN title text or 'NOT READABLE'",
  "subtitle": "subtitle text or 'NOT READABLE'",
  "source": "source text or 'NOT READABLE'",
  "timeframe": "years/period or 'NOT READABLE'",

  "what_is_visible_it": "2–6 lines in ITALIAN describing visible elements (visual layer)",
  
  "data_table_md": "| Year | Value |\n|---|---|\n| 2020 | 100 |",  

  "observations_it": ["visible structural facts - NO guesses"],
  "visual_trends_it": ["purely visual comparisons (higher/lower)"],

  "legend_it": {
    "is_readable": true,
    "mapping": [{"label": "...", "color_or_style": "..."}]
  },
  "numbers": [
    {"label":"...","value":"...","unit":"...","period":"..."}
  ],
  "confidence": 0.0
}
"""

CHART_ANALYST_PROMPT = """
You are an expert financial analyst for a RAG system.
Your task is to analyze structured data (JSON) from a chart/table and the surrounding page context to generate a discursive description in ENGLISH.

INPUT:
A JSON object containing:
1. "vision_json": Visually extracted data (title, values, trends, data table).
2. "page_text": The surrounding PDF page text (for context).

INSTRUCTIONS:
1. Synthesize in ITALIAN what the chart/table demonstrates.
2. Explicitly integrate numbers and dates found in "vision_json" (e.g., "Revenue in 2022 reached 5M").
3. If "data_table_md" is present, use it to describe key data points.
4. Describe visual trends (growth, decline, volatility) mentioned in "visual_trends_it".
5. Be concise but information-dense (keywords) to facilitate semantic search.

DO NOT invent numbers. If data is scarce, write: "Visual analysis limited due to low resolution."
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
You are a High-Fidelity Financial Knowledge Engineer. Your task is to extract a hyper-detailed and atomic Knowledge Graph from the provided text.
Extract a concise Knowledge Graph from the text.

RULES:
- ALWAYS extract a graph if the text mentions people, countries, organizations, policies, events, assets, indicators, or dates.
- Prefer simple, generic types if unsure: Person, Country, Organization, Policy, Event, Asset, Indicator.
- Use clear relations only: AFFECTS, CAUSES, ASSOCIATED_WITH, INCREASES, DECREASES, PART_OF, ANNOUNCED_BY, OCCURS_IN, REFERENCES.
- Keep the graph SMALL and focused.

HARD LIMITS:
- Max 20 nodes
- Max 30 edges
- Minimal properties (empty object {} is fine)

OUTPUT:
- Return ONLY valid JSON
- NO markdown, NO explanations
- EXACT schema:
{
  "nodes": [
    {"id": "...", "label": "...", "type": "...", "properties": {}}
  ],
  "edges": [
    {"source": "...", "target": "...", "relation": "...", "properties": {}}
  ]
}

If nothing can be extracted, return:
{"nodes": [], "edges": []}
"""




REL_CANON_PROMPT = """
You are a relation-type canonicalizer for a knowledge graph.

INPUT: a JSON array of relation types (UPPERCASE, snake_case, may be Italian or English).

OUTPUT: ONLY valid JSON, no markdown, no comments. Schema:
{
  "map": {
    "<RAW>": {
      "verb": "<ENGLISH_VERB_LEMMA>",
      "object": "<ENGLISH_OBJECT_OR_EMPTY>",
      "qualifier": "<QUALIFIER_OR_EMPTY>"
    }
  }
}

RULES:
- verb MUST be a SINGLE ENGLISH VERB in UPPERCASE (lemma), e.g. VISIT, MEET, REDUCE, RESPOND, SEE, SUPPORT, ACCUSE, SIGN, ANNOUNCE.
- object MUST be a short ENGLISH noun (or noun phrase with underscores) in UPPERCASE, e.g. TARIFFS, DUTIES, TAXES, AGREEMENT, ELECTIONS, COUNTRY, CITY.
- If RAW encodes an object (e.g. RIDUCE_DAZI, REDUCE_TARIFFS), extract it into object.
- If RAW contains extra trailing tokens (e.g. _IN, _DURING, _TO, adverbs, etc.), put ONLY the last token into qualifier and keep verb/object unchanged.
- Convert Italian/English forms to the same ENGLISH verb/object (VISITA/VISITS -> verb VISIT; DAZI -> object TARIFFS or DUTIES depending on context; if unsure use TARIFFS).
- If unsure about object, leave object empty.
- Never output inflected verb forms (e.g. VISITED, VISITS, MEETS, DECLARED). Always output the lemma (VISIT, MEET, DECLARE).
- If object is implicit, use a generic object: DISCUSS->TOPIC, DECLARE->STATEMENT, APPROACH->TARGET, AGREE->AGREEMENT, RESPOND->REQUEST.

Return JSON only.
"""

# =========================
# UTILS
# =========================
# --- Vision stats (thread-safe) ---
VISION_STATS = {
    "pages_total": 0,
    "pages_with_imgs": 0,
    "pages_crop_only": 0,
    "pages_fullpage": 0,
}
_VISION_STATS_LOCK = Lock()

# --- UTILITY DI PULIZIA E FILTRAGGIO MD ---
'''
def ai_vision_gatekeeper(image_bytes: bytes) -> bool:
    """Modificato: include foto e contesti informativi news."""
    if not image_bytes: return False
    prompt = "Is this image a financial chart, a data diagram, or a photograph with informative content? Answer ONLY YES or NO."
    try:
        raw = llm_chat_multimodal(prompt, image_bytes, VISION_MODEL_NAME, max_tokens=2)
        return "YES" in raw.upper()
    except: return True
'''


def ai_vision_gatekeeper(image_bytes: bytes) -> bool:
    # filtro dimensione (icone/loghi)
    if not image_bytes or len(image_bytes) < 9000:
        return False
    return True

    

def clean_markdown_structure(md_content: str) -> str:
    """Rimuove sezioni strutturali pesanti (Indici, Sommari) e rumore web."""
    sections = re.split(r'(\n#+ .*)', md_content)
    cleaned_sections = []
    start_idx = 0
    if len(sections) > 1 and len(sections[0].strip()) < 300:
        start_idx = 2 
        
    for i in range(start_idx, len(sections)):
        section_text = sections[i]
        if bool(STRUCTURAL_PAT.search(section_text[:400])):
            continue
        section_text = re.sub(r'https?://\S+', '', section_text)
        section_text = re.sub(r'\S+@\S+', '', section_text)
        cleaned_sections.append(section_text)
    return "".join(cleaned_sections)


def extract_markdown_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Estrattore ad alte prestazioni per file Markdown.
    Gestisce l'auto-cleaning e la Vision AI chirurgica.
    """
    out_chunks = []
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # 1. AUTO-CLEANING: Rimuoviamo indici, sommari e copertine
        content = clean_markdown_structure(raw_content)

        # 2. CHUNKING STRUTTURATO: Dividiamo per titoli o doppi a capo
        raw_paras = re.split(r'\n(?=# )|\n\n', content)
        
        for idx, para in enumerate(raw_paras):
            para = para.strip()
            if len(para) < MIN_CHUNK_LEN:
                continue
                
            # Identificazione asset visuali: ![desc](percorso/immagine.png)
            img_matches = re.findall(r'!\[.*?\]\((.*?)\)', para)
            
            chunk_data = {
                "text_raw": para,
                "text_sem": f"Doc: {filename} | Sezione: {para[:60]}...\n{para}",
                "page_no": 1, # Markdown è un flusso continuo
                "toon_type": "text",
                "section_hint": para[:80] if para.startswith("#") else ""
            }
            
            # VISION AI CHIRURGICA
            if img_matches and PDF_VISION_ENABLED:
                img_path = img_matches[0]
                full_img_path = os.path.join(os.path.dirname(file_path), img_path)
                
                # Attiviamo la Vision solo se l'immagine esiste e non è un'icona (< 5KB)
                if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 5000:
                    with open(full_img_path, "rb") as img_f:
                        img_bytes = img_f.read()
                    
                    
                    # Analisi con Gemma 3 12B (Vision)

                    CHART_MIN_CONF = float(os.getenv("CHART_MIN_CONF", "0.55"))

                    c_js = extract_chart_via_vision(img_bytes)

                    conf = 0.0
                    try:
                        conf = float((c_js or {}).get("confidence") or 0.0)
                    except Exception:
                        conf = 0.0

                    # ✅ accetta SOLO se è davvero un chart (kind != other) e confidence sufficiente
                    if c_js and c_js.get("kind") != "other" and conf >= CHART_MIN_CONF:
                        chunk_data["text_sem"] = build_chart_semantic_chunk(chunk_data["page_no"], c_js)
                        chunk_data["metadata_override"] = c_js
                        print(f"   📝 Analisi Semantica generata (chart conf={conf:.2f})")
                    else:
                        # opzionale: log di debug
                        if c_js:
                            print(f"   ⚠️ Chart scartato (kind={c_js.get('kind')}, conf={conf:.2f})")
   
                        # Salvataggio binario in Postgres (Asset Management)
                        conn = pg_get_conn()
                        try:
                            with conn.cursor() as cur:
                                chunk_data["image_id"] = pg_save_image(log_id, img_bytes, "image/jpeg", f"MD_{filename}_{idx}", cur)
                            conn.commit()
                        finally:
                            pg_put_conn(conn)
            
            out_chunks.append(chunk_data)
            
        print(f"   📄 Markdown Ingested: {len(out_chunks)} chunk validi dopo pulizia.")
        return out_chunks

    except Exception as e:
        print(f"   ❌ Errore durante l'estrazione Markdown: {e}")
        return []

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def force_restart_ollama(num_parallel="4"):
    """
    Chiude ogni istanza di Ollama e la riavvia con il parallelismo attivo.
    """
    print(f"🔄 Resetting Ollama Server (NUM_PARALLEL={num_parallel})...")
    
    # 1. Kill dei processi esistenti (Invariato)
    try:
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/f", "/im", "ollama_app.exe"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    except Exception:
        pass


    # 2. Configura l'ambiente
    env = os.environ.copy()
    #num_parallel=4
    env["OLLAMA_NUM_PARALLEL"] = num_parallel
    env["OLLAMA_MAX_LOADED_MODELS"] = "2"
    
    # 3. Individua il percorso di Ollama
    # Cerchiamo ollama nel PATH; se non lo trova, usiamo il percorso standard di installazione su Windows
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        ollama_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")

    print(f"   🚀 Starting Ollama from: {ollama_path}")

    # 4. Avvio del server con shell=True per risolvere i problemi di PATH su Windows
    try:
        subprocess.Popen(
            [ollama_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            shell=True # Fondamentale per evitare il WinError 2
        )
    except Exception as e:
        print(f"   ❌ Errore critico nell'avvio di Ollama: {e}")
        return False

    # 5. Verifica disponibilità (Invariato)
    for i in range(15):
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=1)
            if res.status_code == 200:
                print(f"   ✨ Ollama is READY (Parallel={num_parallel})")
                return True
        except:
            time.sleep(2)
            print(f"   ...waiting for server ({i+1}/15)")
    
    return False



def ensure_ollama_parallel(num_parallel="4"):
    """
    Imposta le variabili d'ambiente e avvia il server Ollama.
    """
    # 1. Imposta la variabile d'ambiente per il processo Python e i suoi figli
    os.environ["OLLAMA_NUM_PARALLEL"] = num_parallel
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2" # Ottimizza la VRAM
    
    print(f"🚀 Configurazione Ollama: NUM_PARALLEL={num_parallel}")

    # 2. Verifica se Ollama è già attivo
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("   ✅ Ollama è già in esecuzione. Nota: se non è stato avviato con NUM_PARALLEL, i thread saranno sequenziali.")
            return
    except requests.exceptions.ConnectionError:
        print("   ⚠️ Server Ollama non trovato. Avvio in corso...")

    # 3. Avvio del server come processo in background (subprocess.Popen)
    # 'ollama serve' rimarrà attivo mentre lo script prosegue
    subprocess.Popen(
        ["ollama", "serve"],
        env=os.environ, # Passa le variabili d'ambiente impostate sopra
        stdout=subprocess.DEVNULL, # Nasconde i log del server per pulizia terminale
        stderr=subprocess.DEVNULL
    )

    # 4. Attesa che il server sia pronto
    for _ in range(10):
        try:
            if requests.get("http://localhost:11434/api/tags").status_code == 200:
                print("   ✨ Server Ollama pronto!")
                return
        except:
            time.sleep(2)
    print("   ❌ Errore: Impossibile avviare Ollama automaticamente.")



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
    """
    Normalizza gli spazi ma preserva la struttura 'visiva' delle tabelle.
    Non schiaccia più spazi > 2 in uno solo, perché nei PDF significano 'colonna'.
    """
    text = (text or "").replace("\x00", " ")
    
    # Sostituisce i tab con spazi per uniformità
    text = text.replace("\t", "    ")
    
    # 1. Rimuove spazi eccessivi SOLO se sono singoli (normale testo)
    # Se ci sono più di 2 spazi consecutivi, li manteniamo (potrebbe essere una tabella)
    # Regex: Sostituisce 1 spazio ripetuto, ma rispetta '   ' (3+) come separatore colonna
    # text = re.sub(r"[ ]{2,}", " ", text) <--- VECCHIO CODICE CHE ROMPEVA LE TABELLE
    
    # NUOVO: Collassa spazi enormi (>10) ma lascia il respiro per le colonne (2-10 spazi)
    text = re.sub(r" {10,}", "    ", text) 
    
    # Gestione a capo: rimuove solo i tripli a capo inutili
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

def normalize_entity_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s)
    return s[:180]

import json
import re

RECONCILE_FORMULAS_PROMPT = r"""
You are a strict LaTeX normalizer for OCR-extracted finance/maths formulas.

RULES (MANDATORY):
- DO NOT add new formulas.
- DO NOT remove formulas.
- DO NOT change the order.
- DO NOT change meanings; only fix OCR artifacts.
- Keep LaTeX concise and standard.
- If something is unclear, keep it as-is (do NOT guess).
- Replace OCR noise like "( )" empty parentheses where appropriate ONLY if it is obviously noise.
- Normalize symbols: × -> \times, –/− -> -, · used as multiply may become \cdot ONLY if it is clearly multiply (not decimal dot).
- Remove stray spaces inside tokens: "E ri( )" -> "E[r_i]" only if it is obviously CAPM expectation formatting; otherwise keep minimal safe fix like "E( r_i )".

OUTPUT JSON ONLY:
{
  "confidence": 0.0,
  "formulas": [
    {
      "id": 1,
      "latex": "...",
      "meaning_it": "..."
    }
  ]
}

IMPORTANT:
- The output must contain exactly the same number of formulas as input.
- 'meaning_it' must be copied from input unchanged (or empty if missing).
"""

def _extract_formula_list(f_js: dict):
    """Ritorna lista formule se presente, altrimenti []"""
    if not isinstance(f_js, dict):
        return []
    lst = f_js.get("formulas")
    return lst if isinstance(lst, list) else []

def _needs_reconcile(latex: str) -> bool:
    if not latex:
        return False
    s = latex
    # euristiche semplici: parentesi vuote, token spezzati, troppo rumore OCR
    if "( )" in s or "ri( )" in s or "rm ( )" in s:
        return True
    if re.search(r"[A-Za-z]\s+\(", s):  # tipo "E ri( )"
        return True
    if "" in s or "" in s or "" in s or "�" in s:
        return True
    return False

def reconcile_formulas_with_strict_rules(
    f_js: dict,
    page_image_bytes: bytes,
    model: str,
    max_tokens: int = 650,
) -> dict:
    """
    Second pass: ripulisce LaTeX senza inventare.
    Richiede: llm_chat_multimodal(prompt, image_bytes, model, max_tokens) + safe_json_extract(str)
    Ritorna un dict aggiornato (o f_js originale se fallisce).
    """
    if not isinstance(f_js, dict):
        return f_js

    formulas = _extract_formula_list(f_js)
    if not formulas:
        return f_js

    # decide se serve: basta che UNA formula appaia rotta
    if not any(_needs_reconcile((x or {}).get("latex", "")) for x in formulas if isinstance(x, dict)):
        return f_js

    # prepara input minimal: id + latex + meaning_it (immutabile)
    compact = []
    for i, frm in enumerate(formulas, start=1):
        if not isinstance(frm, dict):
            compact.append({"id": i, "latex": "", "meaning_it": ""})
            continue
        compact.append({
            "id": frm.get("id") or i,
            "latex": (frm.get("latex") or ""),
            "meaning_it": (frm.get("meaning_it") or frm.get("description") or "")
        })

    payload = json.dumps({"formulas": compact}, ensure_ascii=False)

    prompt = RECONCILE_FORMULAS_PROMPT + "\n\nINPUT_FORMULAS_JSON:\n" + payload

    try:
        out_str = llm_chat_multimodal(
            prompt=prompt,
            image_bytes=page_image_bytes,
            model=model,
            max_tokens=max_tokens,
        )
    except Exception:
        return f_js

    out_js = safe_json_extract(out_str)
    if not isinstance(out_js, dict):
        return f_js

    out_list = out_js.get("formulas")
    if not isinstance(out_list, list):
        return f_js

    # vincolo: stesso numero di formule
    if len(out_list) != len(compact):
        return f_js

    # ricostruisci mantenendo meaning_it originale se il modello la cambia
    fixed = []
    for idx, frm_out in enumerate(out_list):
        orig = compact[idx]
        if not isinstance(frm_out, dict):
            frm_out = {}
        fixed.append({
            "id": orig.get("id") or (idx + 1),
            "latex": (frm_out.get("latex") or orig.get("latex") or "").strip(),
            "meaning_it": orig.get("meaning_it", "")  # BLOCCATA: non permettiamo variazioni
        })

    # aggiorna conf (se presente)
    conf = 0.0
    try:
        conf = float(out_js.get("confidence") or f_js.get("confidence") or 0.0)
    except Exception:
        conf = float(f_js.get("confidence") or 0.0) if isinstance(f_js.get("confidence"), (int, float, str)) else 0.0

    # scrivi su f_js senza cambiare altre parti (tables_md, summary_it ecc.)
    f_js2 = dict(f_js)
    f_js2["formulas"] = fixed
    f_js2["confidence"] = conf
    return f_js2




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


def is_text_layer_corrupt(text: str) -> bool:
    """
    Rileva layer testuale rotto. (Versione Hardened per '2C0D')
    """
    if not text: return False
    
    # 1. Pattern inequivocabili di formule rotte
    bad_markers = [
        "2C0D",          # EOQ Formula rotta
        "•", "–", "—", "˜", # Bullet point corrotti
        "× ×",           # Operatori duplicati
        "( ) ( )"        # Parentesi vuote
    ]
    
    for marker in bad_markers:
        if marker in text:
            return True # Trovato marcatore di corruzione -> Butta il testo!

    # 2. Densità numeri
    lines = text.split('\n')
    numeric_lines = 0
    total_lines = len(lines)
    if total_lines > 5:
        for line in lines:
            if re.fullmatch(r'[\d\s\Wμ]+', line.strip()) and len(line) > 5:
                numeric_lines += 1
        
        if (numeric_lines / total_lines) > 0.4:
            return True
            
    return False


def smart_clean_text(text: str) -> str:
    """
    Interpreta e pulisce il testo del PDF:
    1. Rimuove simboli di font corrotti (artefatti matematici).
    2. Rimuove righe che sono solo sequenze di numeri spaziati (indici di formule esplosi).
    3. Normalizza spaziature.
    """
    if not text: 
        return ""
    
    # 1. Rimuovi i caratteri "Ghost" specifici che hai mostrato nel log
    # Questi codici (x95, x96...) sono spesso bullet point o simboli matematici mappati male
    text = re.sub(r'[\x95\x96\x97\x98\x81\x80\x8d\uf0b7\uf020]', ' ', text)
    
    lines = text.split('\n')
    valid_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 2. Rilevamento Header/Footer ripetitivi
        # Se la riga è identica a intestazioni note, saltala
        if line.lower() in ["formulae sheet", "maths tables", "financial management (fm)", "s17", "think ahead acca"]:
            continue
            
        # 3. Rilevamento "Numeri Esplosi" (es. "3 3 4" o "1 1 1")
        # Le formule matematiche nel layer testo spesso appaiono come numeri spaziati senza senso.
        # Se una riga ha solo numeri e spazi e meno di 2 lettere, è spazzatura del layer testo.
        if re.match(r'^[\d\s\W]+$', line) and len(re.findall(r'[a-zA-Z]', line)) < 2:
            continue

        valid_lines.append(line)
    
    # Ricostruisci il testo
    text = " ".join(valid_lines)
    
    # 4. Collassa spazi multipli generati dalla rimozione
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_normalize_text(text: str) -> str:
    """
    Normalizzazione universale per testi estratti da PDF.
    - Gestisce legature standard (fi, fl, ff) tramite NFKC.
    - Rimuove caratteri di controllo e 'garbage' non-ASCII sospetti.
    - Preserva i simboli matematici necessari per le formule.
    """
    if not text:
        return ""

    # 1. Normalizzazione NFKC: Decompone le legature standard (es. 'ﬁ' -> 'fi')
    # e normalizza i caratteri simili in un'unica forma standard.
    text = unicodedata.normalize('NFKC', text)

    # 2. Identificazione contesto matematico
    # Se il testo sembra una formula, siamo più conservativi nella rimozione dei simboli.
    is_math = bool(MATH_CANDIDATE_PAT.search(text))

    # 3. Pulizia dei caratteri "Phantom" o "Private Use Area"
    # Molti glitch dei PDF finiscono nel range Unicode E000-F8FF (Private Use Area)
    # o sono caratteri di controllo non stampabili.
    if not is_math:
        # Rimuove caratteri non stampabili e simboli strani (non-latin, non-punctuation)
        # mantenendo però lettere accentate e punteggiatura standard.
        text = re.sub(r'[^\x20-\x7E\u00A0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
    else:
        # In contesto matematico, preserviamo i simboli LaTeX comuni
        # ma puliamo comunque i null byte e i caratteri di controllo.
        text = text.replace('\x00', ' ')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)

    # 4. Normalizzazione degli spazi bianchi
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def _load_rel_canon_cache() -> dict:
    try:
        with open(REL_CANON_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _save_rel_canon_cache(cache: dict) -> None:
    try:
        with open(REL_CANON_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _get_rel_canon_map(rel_types: set[str]) -> dict:
    """
    Ritorna cache completa {RAW: {"verb":..., "object":..., "qualifier":...}}
    e aggiorna la cache per i missing con UNA chiamata LLM per batch/doc.
    """
    cache = _load_rel_canon_cache()

    missing = [rt for rt in sorted(rel_types) if rt and rt not in cache]
    if not missing:
        return cache

    raw = llm_chat(
        REL_CANON_PROMPT,
        json.dumps(missing, ensure_ascii=False),
        LLM_MODEL_NAME,
        max_tokens=REL_CANON_MAX_TOKENS
    )

    js = safe_json_extract(raw) or {}
    mapping = (js.get("map") or {}) if isinstance(js, dict) else {}

    for k, v in mapping.items():
        if not isinstance(v, dict):
            continue

        kk = (k or "").strip().upper()
        verb = (v.get("verb") or "").strip().upper()
        obj = (v.get("object") or "").strip().upper()
        qual = (v.get("qualifier") or "").strip().upper()

        if kk and verb:
            cache[kk] = {"verb": verb, "object": obj, "qualifier": qual}

    _save_rel_canon_cache(cache)
    return cache


def canonicalize_edges_to_verb_object(edges: list[dict]) -> list[dict]:
    """
    VERBO-ONLY:
    - ee["relation"] = VERB lemma in EN (uppercase)
    - props tiene audit: raw_relation, canon_verb, canon_object (se presente), qualifier (se presente)
    """
    if not edges:
        return edges

    rel_types = set(
        (e.get("relation") or "").strip().upper()
        for e in edges
        if e.get("relation")
    )
    canon_map = _get_rel_canon_map(rel_types)

    out: list[dict] = []

    for e in edges:
        raw_rel = (e.get("relation") or "").strip().upper()
        raw_rel = raw_rel.replace("__", "_").strip("_")
        if not raw_rel:
            out.append(e)
            continue

        m = canon_map.get(raw_rel) or {}
        verb = (m.get("verb") or raw_rel).strip().upper()
        obj = (m.get("object") or "").strip().upper()
        qual = (m.get("qualifier") or "").strip().upper()

        # lemma cheap (en)
        verb = _cheap_lemma_en(verb)

        ee = dict(e)
        props = dict(ee.get("props") or {})

        # TYPE Neo4j: SOLO VERB
        canon_type = _safe_reltype(verb)

        # audit/provenance
        raw_audit = raw_rel.replace("__", "_").strip("_")
        if not RELTYPE_OK.match(raw_audit):
            raw_audit = raw_rel

        props.setdefault("raw_relation", raw_audit)
        props.setdefault("canon_verb", verb)

        # object come proprietà (non nel type)
        if obj:
            props.setdefault("canon_object", obj)

        # fallback opzionale (se vuoi): VISIT senza oggetto -> PLACE
        if not obj and canon_type == "VISIT":
            props.setdefault("canon_object", "PLACE")

        if qual:
            props.setdefault("qualifier", qual)

        ee["props"] = props
        ee["relation"] = canon_type
        out.append(ee)

    return out



def _safe_reltype(t: str) -> str:
    t = (t or "").strip().upper()
    if RELTYPE_OK.match(t):
        return t
    return "RELATES_TO"

def _cheap_lemma_en(verb: str) -> str:
    v = (verb or "").strip().upper()

    if len(v) <= 4:
        return v

    # forme comuni
    if v.endswith("ING") and len(v) > 6:
        v = v[:-3]
    elif v.endswith("ED") and len(v) > 5:
        v = v[:-2]
    elif v.endswith("S") and len(v) > 5:
        v = v[:-1]

    # fix "troncamenti" frequenti prodotti da LLM/JSON cleaning:
    # ESTABLISHE -> ESTABLISH
    if v.endswith("ISHE") and len(v) >= 8:
        v = v[:-1]  # drop final 'E'

    # DISCUS -> DISCUSS
    if v.endswith("CUS") and len(v) >= 6:
        v = v + "S"

    return v



def canonicalize_edges_by_base_presence(edges: list[dict]) -> list[dict]:
    """
    NO whitelist, data-driven:
    se nello stesso batch esistono BASE e BASE_SUFFIX, collassa BASE_SUFFIX -> BASE
    e salva raw_relation + qualifier nelle props.
    """
    if not edges:
        return edges

    rel_types = set((e.get("relation") or "").strip().upper() for e in edges if e.get("relation"))
    out = []

    for e in edges:
        rel = (e.get("relation") or "").strip().upper()
        rel = rel.replace("__", "_").strip("_")
        if not rel or "_" not in rel:
            out.append(e)
            continue

        parts = rel.split("_")
        if parts:
            parts[0] = _cheap_lemma_en(parts[0])
            rel = "_".join(parts)


        base, suffix = rel.rsplit("_", 1)

        if base in rel_types and len(base) >= 6 and 1 <= len(suffix) <= 12:
            ee = dict(e)
            ee["relation"] = base
            props = dict(ee.get("props") or {})
            props.setdefault("raw_relation", rel)
            props.setdefault("qualifier", suffix)
            ee["props"] = props
            out.append(ee)
        else:
            out.append(e)

    return out

def is_garbage_text(text: str, threshold: float = 0.10) -> bool:
    """
    Rileva se il testo estratto è corrotto (es. glitch di fitz).
    Soglia 0.10 significa che se più del 10% dei caratteri sono □ o , il chunk viene scartato.
    """
    if not text or len(text) < 5: 
        return True
    # Identifica caratteri "garbage" comuni nei PDF estratti male
    bad_chars = len(re.findall(r"[□\ufffd]", text))
    
    if bad_chars / len(text) > threshold:
        return True
    return False


def sanitize_chart_for_analysis(chart_json: dict) -> dict:
    if not isinstance(chart_json, dict):
        return {}

    cj = dict(chart_json)

    # 1) Periodo: accetta solo anni a 4 cifre
    tf = str(cj.get("timeframe", "") or "")
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", tf)
    if years:
        cj["timeframe"] = " vs ".join(sorted(set(years)))
    else:
        cj["timeframe"] = "NOT READABLE"

    # 2) Categorie: rimuovi qualificatori sospetti
    cats = cj.get("categories_it", [])
    if isinstance(cats, list):
        cleaned = []
        for c in cats:
            low = str(c).lower()
            if " sud" in low or "south" in low:
                continue
            cleaned.append(c)
        cj["categories_it"] = cleaned

    return cj



def generate_chart_analysis_it(chart_json: dict, page_text: str = "") -> str:
    if not isinstance(chart_json, dict):
        return ""
    try:
        payload = {
            "vision_json": chart_json,
            "page_text": (page_text or "")[:2500]
        }
        resp = chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": CHART_ANALYST_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            options={"temperature": 0.2, "num_predict": 450, "num_ctx": 2048},
        )
        return (resp.get("message", {}).get("content", "") or "").strip()
    except Exception:
        return ""



MATH_CANDIDATE_PAT = re.compile(
    r"(?i)("
    r"formulae\s+sheet|"           # Keyword forte
    r"maths\s+tables|"
    r"economic\s+order\s+quantity|" 
    r"2c0d|"                       # <--- IL TUO ERRORE SPECIFICO (diventa trigger)
    r"miller[-.\s]?orr|"
    r"capm|"
    r"wacc|"
    r"asset\s+beta|"
    r"growth\s+model|"
    r"fisher\s+formula|"
    r"purchasing\s+power\s+parity|"
    r"\bbeta\b|"
    r"standard\s+deviation|"
    r"[\u2200-\u22FF]|"            # Operatori matematici
    r"[∑∏∫√=≈≠≤≥→↔∩∪∞±×÷]|"       # Simboli
    r"[•–—˜]"                      # <--- SE VEDI SPAZZATURA, È MATEMATICA!
    r")"
)

# 3. Pattern per Elementi Visuali (Bilingue)
CHART_CANDIDATE_PAT = re.compile(
    r"\b("
    r"chart|graph|figure|plot|diagram|heatmap|candlestick|ohlc|volume|axis|legend|"
    r"grafico|grafica|figura|plot|diagramma|candela|ohlc|volumi|asse|legenda"
    r")\b",
    re.IGNORECASE
)


# Rileva parole che iniziano con la maiuscola (entità potenziali)
_ENTITY_PROPER_NOUNS = re.compile(r'\b[A-Z][a-zà-ù]{1,}\b')

def is_keyword_candidate_hybrid(text: str) -> bool:
    """
    Trigger SELETTIVO: attiva l'estrazione KG solo se il chunk è denso.
    Richiede: Almeno 2 Nomi Propri DIVERSI (es. Soggetto e Oggetto) 
    E almeno 1 keyword finanziaria (es. Azione/Contesto).
    """
    if not text or len(text) < KG_MIN_LEN: 
        return False
    
    clean_text = safe_normalize_text(text)
    
    # 1. Conta entità uniche (Nomi Propri)
    proper_nouns = set(_ENTITY_PROPER_NOUNS.findall(clean_text))
    
    # 2. Verifica presenza di concetti finanziari/tecnici
    has_finance_key = bool(_KG_PAT.search(clean_text))

    # Attiva solo se c'è ALTA probabilità di trovare una relazione significativa
    if len(proper_nouns) >= 2 and has_finance_key:
        return True
    
    # Fallback: attiva se ci sono moltissime keyword finanziarie (almeno 4 diverse)
    # indicando un paragrafo puramente tecnico.
    finance_keywords_count = len(set(_KG_PAT.findall(clean_text)))
    if finance_keywords_count >= 4:
        return True

    return False



def is_keyword_candidate(text: str) -> bool:
    """Valida se il chunk contiene concetti chiave per il Knowledge Graph."""
    if not text or len(text) < KG_MIN_LEN: #
        return False
    return bool(_KG_PAT.search(text)) #

def is_formula_candidate_page(page_text: str) -> bool:
    """Valida se la pagina contiene elementi matematici per la Vision."""
    return bool(page_text and MATH_CANDIDATE_PAT.search(page_text)) #

def is_chart_candidate_page(page_text: str) -> bool:
    """Valida se la pagina contiene riferimenti visivi (grafici/tabelle)."""
    return bool(page_text and CHART_CANDIDATE_PAT.search(page_text)) #


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
    if not batch_data: return
    
    # Prepariamo i dati aggiungendo il timestamp richiesto dall'indice
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    # normalized = [row + (now,) for row in batch_data]

    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO document_chunks (
                    log_id, chunk_index, toon_type, content_raw, 
                    content_semantic, metadata_json, chunk_uuid, ingestion_ts
                )
                VALUES %s
                ON CONFLICT (chunk_uuid, ingestion_ts) DO UPDATE SET
                    content_semantic = EXCLUDED.content_semantic,
                    metadata_json = EXCLUDED.metadata_json
                """,
                [row + (now,) for row in batch_data]
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

// 5) Relazioni tra entità (Versione Nativa senza APOC)
WITH r
CALL (r) {
  UNWIND coalesce(r.edges, []) AS rel
  WITH rel
  WHERE rel.source IS NOT NULL AND rel.target IS NOT NULL
    AND rel.source <> rel.target
    AND rel.relation IS NOT NULL AND rel.relation <> ""

  MERGE (s:Entity {id: rel.source})
  MERGE (t:Entity {id: rel.target})
  
  // Sostituzione di APOC con MERGE nativo
  // Usiamo RELATES_TO come tipo base e salviamo l'azione specifica nelle proprietà
  MERGE (s)-[r_out:RELATES_TO]->(t)
  SET r_out.type = rel.relation,
      r_out += coalesce(rel.props, {}),
      r_out.last_seen = datetime(),
      r_out.count = coalesce(r_out.count, 0) + 1,
      r_out.raw_relation = coalesce(rel.props.raw_relation, rel.relation)
      
  RETURN count(*) AS rel_count
}

RETURN count(*) AS processed
"""

# Formula nodes deterministici
NEO4J_FORMULA_QUERY = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk_id})
MERGE (f:Formula {fid: r.fid})
SET f.latex = r.latex, 
    f.latex_raw = r.latex_raw,  // Salvataggio del dato puro
    f.plain = r.plain, 
    f.meaning_it = r.meaning_it, 
    f.keywords = r.keywords,
    f.page = r.page_no, 
    f.source = r.filename
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

def canonicalize_edges(edges: list[dict]) -> list[dict]:
    """
    Canonicalizza relation types senza whitelist linguistica:
    - se esiste sia BASE che BASE_SUFFIX nello stesso batch di edges,
      collassa BASE_SUFFIX -> BASE e salva suffix in props["qualifier"].
    """
    if not edges:
        return edges

    # set dei relation type presenti (già puliti da _clean_rel)
    rel_types = set((e.get("relation") or "").strip().upper() for e in edges if e.get("relation"))

    out = []
    for e in edges:
        rel = (e.get("relation") or "").strip().upper()
        if not rel or "_" not in rel:
            out.append(e)
            continue

        base, suffix = rel.rsplit("_", 1)

        # guardrail: base non troppo corto e suffix ragionevole
        if base in rel_types and len(base) >= 8 and 1 <= len(suffix) <= 12:
            ee = dict(e)
            ee["relation"] = base
            props = dict(ee.get("props") or {})
            props.setdefault("raw_relation", rel)
            props.setdefault("qualifier", suffix)
            ee["props"] = props
            out.append(ee)
        else:
            out.append(e)

    return out

def _normalize_graph_schema(js):
    if js is None:
        return None
    if isinstance(js, list):
        return {"nodes": js, "edges": []}
    if not isinstance(js, dict):
        return None

    g = dict(js)

    # top-level aliases
    if "nodes" not in g:
        for k in ("entities", "entity", "concepts", "items"):
            if k in g and isinstance(g[k], list):
                g["nodes"] = g[k]
                break
    if "edges" not in g:
        for k in ("relationships", "relations", "links"):
            if k in g and isinstance(g[k], list):
                g["edges"] = g[k]
                break
        g.setdefault("edges", [])

    # node aliases
    nnodes = []
    for n in (g.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        x = dict(n)
        if "label" not in x and "name" in x:
            x["label"] = x["name"]
        if "type" not in x and "category" in x:
            x["type"] = x["category"]
        if "id" not in x:
            t = (x.get("type") or "Entity").strip()
            l = (x.get("label") or "").strip()
            x["id"] = f"{t}:{l}" if l else f"{t}:{uuid.uuid4().hex[:8]}"
        if "properties" not in x or not isinstance(x.get("properties"), dict):
            props = {}
            for kk, vv in x.items():
                if kk not in ("id", "label", "type", "properties"):
                    props[kk] = vv
            x["properties"] = props
        nnodes.append(x)
    g["nodes"] = nnodes

    # edge aliases
    eedges = []
    for e in (g.get("edges") or []):
        if not isinstance(e, dict):
            continue
        x = dict(e)
        if "source" not in x:
            x["source"] = x.get("from") or x.get("src")
        if "target" not in x:
            x["target"] = x.get("to") or x.get("dst")
        if "relation" not in x:
            x["relation"] = x.get("type") or x.get("predicate") or x.get("rel") or "RELATED_TO"
        if "properties" not in x or not isinstance(x.get("properties"), dict):
            props = {}
            for kk, vv in x.items():
                if kk not in ("source", "target", "relation", "properties"):
                    props[kk] = vv
            x["properties"] = props
        if x.get("source") and x.get("target"):
            eedges.append(x)
    g["edges"] = eedges

    return g


from typing import Any, Dict, List, Tuple

def _sanitize_graph(graph: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(graph, dict):
        return [], []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()

    # --- NODES ---
    raw_nodes = graph.get("nodes") or graph.get("entities") or graph.get("concepts") or []
    for n in raw_nodes or []:
        if not isinstance(n, dict):
            continue

        nid = n.get("id") or n.get("name") or n.get("label") or n.get("key")
        label = n.get("label") or n.get("name") or nid
        ntype = n.get("type") or n.get("kind") or n.get("category") or "Entity"
        props = n.get("properties") or n.get("metadata") or n.get("attributes") or {}

        if not nid:
            continue

        nid = str(nid)
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)

        if not isinstance(props, dict):
            props = {}

        nodes.append({
            "id": nid,
            "label": str(label) if label else nid,
            "type": str(ntype),
            "properties": props
        })

    # --- EDGES ---
    raw_edges = graph.get("edges") or graph.get("relationships") or graph.get("relations") or graph.get("links") or []
    for e in raw_edges or []:
        if not isinstance(e, dict):
            continue

        src = e.get("source") or e.get("from") or e.get("src")
        tgt = e.get("target") or e.get("to") or e.get("dst")
        rel = e.get("relation") or e.get("type") or e.get("predicate") or e.get("rel") or "RELATED_TO"
        props = e.get("properties") or e.get("metadata") or e.get("attributes") or {}

        if not src or not tgt:
            continue

        if not isinstance(props, dict):
            props = {}

        edges.append({
            "source": str(src),
            "target": str(tgt),
            "relation": str(rel),
            "properties": props
        })

    # hard cap SOLO alla fine
    return nodes[:20], edges[:30]



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
            '''resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens
            )'''
            resp : ChatResponse = chat(
                model=model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}]
            )
            #return resp.choices[0].message.content or ""
            return resp['message']['content'] or ""
        except Exception:
            time.sleep(0.35)
    return ""

def llm_chat_multimodal(prompt: str, image_bytes: bytes, model: str, max_tokens: int = 900) -> str:
    if not image_bytes:
        return ""
    for _ in range(LLM_RETRIES + 1):
        try:
            resp = chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }],
                format="json",  # 🔒 forza JSON lato Ollama
                options={
                    "temperature": 0.0,  # 🔒 deterministico
                    "num_predict": int(max_tokens),
                },
            )
            return resp.get("message", {}).get("content", "") or ""
        except Exception as e:
            print(f"   ⚠️ Errore Vision (Ollama): {e}")
            time.sleep(0.5)
    return ""


def _downscale_and_compress_for_vision(
    img_bytes: bytes,
    max_side: int = 1400,     # ⬅️ prima era più basso
    jpeg_quality: int = 90   # ⬅️ prima troppo aggressivo
) -> bytes:
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return img_bytes


def ocr_extract_text(img_bytes: bytes) -> str:
    """
    Deterministic OCR used ONLY to assist Vision on titles/sources.
    No hallucinations possible here.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang="ita+eng")
        return normalize_ws(text)
    except Exception:
        return ""



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


def render_full_page_png(page_obj, dpi: int = 200) -> bytes:
    """
    Render pagina PDF in PNG (migliore per formule e testo fine).
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page_obj.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def render_full_page_png(page, dpi: int = 220, clip=None) -> bytes:
    """
    Render lossless PNG della pagina (migliore per OCR formule/simboli).
    - page: fitz.Page
    - dpi: 180-240 tipico per formule; 220 è un buon compromesso
    - clip: fitz.Rect opzionale per render parziale
    """
    if dpi <= 0:
        dpi = 220

    zoom = dpi / 72.0  # PyMuPDF usa 72 DPI come base
    mat = fitz.Matrix(zoom, zoom)

    # alpha=False => evita canale trasparenza (più compatto e stabile per OCR)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

    # In alcuni casi pix può risultare in colorspace non RGB.
    # get_pixmap(alpha=False) di solito è già ok; questo è un extra safety:
    try:
        if pix.n not in (3, 4):  # 3=RGB, 4=CMYK/altro
            pix = fitz.Pixmap(fitz.csRGB, pix)
    except Exception:
        pass

    return pix.tobytes("png")




def extract_formulas_vision(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Estrazione specifica per formule matematiche/finanziarie complesse.
    Converte l'immagine della pagina in LaTeX strutturato.
    """
    if not img_bytes:
        return None

    # Downscaling leggermente meno aggressivo per formule piccole (DPI virtuale più alto)
    vbytes = _downscale_and_compress_for_vision(img_bytes, max_side=1600, jpeg_quality=95)
    
    key = sha256_hex(vbytes) + "::formula_latex_v2"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    try:
        # Usa il nuovo prompt FORMULA_VISION_PROMPT definito sopra
        raw = llm_chat_multimodal(FORMULA_VISION_PROMPT, vbytes, VISION_MODEL_NAME, max_tokens=1200)
        js = safe_json_extract(raw)
        
        if not js or "formulas" not in js:
            return None
            
        _vision_cache_put(key, js)
        return js
    except Exception as e:
        print(f"   ⚠️ Formula Vision Error: {e}")
        return None

def extract_chart_via_vision(img_bytes: bytes, context_hint: str = "") -> Optional[Dict[str, Any]]:
    """
    Grounded Vision Extraction (anti-hallucination):
    - OCR deterministic injected as optional evidence
    - JSON forced (Ollama format=json)
    - temperature 0
    - cache enabled
    """
    if not img_bytes:
        return None

    vbytes = _downscale_and_compress_for_vision(img_bytes)
    if not vbytes:
        return None

    key = sha256_hex(vbytes) + "::chart_grounded_v2"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    # OCR (deterministic). Used only as auxiliary evidence.
    ocr_text = ocr_extract_text(vbytes)
    ocr_text = (ocr_text or "").strip()

    grounded_prompt = CHART_VISION_PROMPT
    if context_hint:
        grounded_prompt += f"\n\nCONTEXT HINT (may help disambiguate):\n{context_hint[:600]}\n"
    if ocr_text:
        grounded_prompt += f"""
        VERIFIED OCR TEXT (may be partial/noisy):
        \"\"\"{ocr_text[:1200]}\"\"\"
        RULES ABOUT OCR:
        - You may use OCR text ONLY if it matches what is visually visible.
        - If OCR conflicts with visual content, IGNORE OCR.
        """



    raw = llm_chat_multimodal(grounded_prompt, vbytes, VISION_MODEL_NAME, max_tokens=1100)
    js = safe_json_extract(raw)
    
    # If timeframe looks like years, keep only 4-digit years found
    tf = str(js.get("timeframe", "") or "")
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", tf)
    if years:
        js["timeframe"] = " vs ".join(sorted(set(years)))

    if not isinstance(js, dict):
        return None

    legend = js.get("legend_it", {}) or {}
    legend_ok = isinstance(legend, dict) and legend.get("is_readable") is True and legend.get("mapping")

    if not legend_ok:
        # remove claims about what colors represent
        obs = js.get("observations_it", []) or []
        js["observations_it"] = [o for o in obs if "color" not in o.lower() and "colore" not in o.lower()]


    # normalize required fields (schema-hardening)
    js.setdefault("kind", "other")
    js.setdefault("title", "NOT READABLE")
    js.setdefault("source", "NOT READABLE")
    js.setdefault("timeframe", "NOT READABLE")
    js.setdefault("what_is_visible_it", "")
    js.setdefault("observations_it", [])
    js.setdefault("visual_trends_it", [])
    js.setdefault("numbers", [])
    js.setdefault("unreadable_parts", [])
    js.setdefault("confidence", 0.0)

    # hardening: types
    if not isinstance(js.get("observations_it"), list):
        js["observations_it"] = []
    if not isinstance(js.get("visual_trends_it"), list):
        js["visual_trends_it"] = []
    if not isinstance(js.get("unreadable_parts"), list):
        js["unreadable_parts"] = []
    if not isinstance(js.get("numbers"), list):
        js["numbers"] = []

    # clamp confidence
    try:
        conf = float(js.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    js["confidence"] = conf

    # if low readability => drop numbers (avoid “guessed” values)
    if conf < 0.45:
        js["numbers"] = []

    # trim lists
    js["observations_it"] = [str(x)[:260] for x in js["observations_it"][:12]]
    js["visual_trends_it"] = [str(x)[:260] for x in js["visual_trends_it"][:12]]
    js["unreadable_parts"] = [str(x)[:160] for x in js["unreadable_parts"][:12]]

    cleaned_nums = []
    for n in js["numbers"][:12]:
        if not isinstance(n, dict):
            continue
        cleaned_nums.append({
            "label": str(n.get("label", ""))[:120],
            "value": str(n.get("value", ""))[:80],
            "unit": str(n.get("unit", ""))[:40],
            "period": str(n.get("period", ""))[:60],
        })
    js["numbers"] = cleaned_nums


    CHART_MIN_CONF = float(os.getenv("CHART_MIN_CONF", "0.55"))
    if js.get("kind") == "other" or float(js.get("confidence") or 0.0) < CHART_MIN_CONF:
        return None

    _vision_cache_put(key, js)
    return js


def build_chart_semantic_chunk(page_no: int, chart_json: Dict[str, Any], prefix: str = "VISUAL") -> str:
    kind = chart_json.get("kind", "immagine")
    title = chart_json.get("title", "")
    source = chart_json.get("source", "")
   
    vis = chart_json.get("what_is_visible_it", "") or ""
    obs = chart_json.get("observations_it", []) or []
    nums = chart_json.get("numbers", []) or []
    unread = chart_json.get("unreadable_parts", []) or []
    conf = chart_json.get("confidence", 0.0)

    subtitle = chart_json.get("subtitle", "")
    legend = chart_json.get("legend_it", {}) or {}

    analysis_it = chart_json.get("analysis_it", "") or ""

    raw_timeframe = chart_json.get("timeframe", "")
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(raw_timeframe))
    timeframe = " vs ".join(years) if years else "NOT READABLE"

    lines = [f"--- CONTENUTO VISUALE ({kind}) - Pagina {page_no} ---"]
    if title: lines.append(f"Titolo: {title}")
    if source: lines.append(f"Fonte: {source}")
    

    raw_timeframe = chart_json.get("timeframe", "")
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(raw_timeframe))
    timeframe = " vs ".join(years) if years else "NOT READABLE"

    if timeframe:
        lines.append(f"Periodo: {timeframe}")




    if subtitle and subtitle != "NOT READABLE":
        lines.append(f"Sottotitolo: {subtitle}")

    if isinstance(legend, dict) and legend.get("is_readable") and legend.get("mapping"):
        lines.append("Legenda (visibile):")
        for m in (legend.get("mapping") or [])[:8]:
            lines.append(f" - {m.get('label','')}: {m.get('color_or_style','')}")

    lines.append(f"Affidabilità (confidence): {conf}")

    if vis:
        lines.append("Descrizione fattuale (IT):")
        lines.append(vis)

    if obs:
        lines.append("Osservazioni visibili:")
        for o in obs[:10]:
            lines.append(f" - {o}")

    if nums:
        lines.append("Numeri leggibili (solo se visibili):")
        for n in nums[:10]:
            label = n.get("label", "")
            value = n.get("value", "")
            unit = n.get("unit", "")
            period = n.get("period", "")
            lines.append(f" - {label}: {value} {unit} {('('+period+')') if period else ''}".strip())

    if unread:
        lines.append("Parti non leggibili:")
        for u in unread[:10]:
            lines.append(f" - {u}")

    # ✅ NUOVO BLOCCO: analisi discorsiva grounded
    if analysis_it:
        lines.append("Analisi (IT) - interpretazione qualitativa:")
        lines.append(analysis_it)

    return normalize_ws("\n".join(lines))



# =========================
# Chunk builders
# =========================
def build_formula_semantic_chunk(page_no: int, formulas_json: Dict[str, Any]) -> str:
    """
    Crea un chunk ottimizzato per il RAG contenente spiegazioni e LaTeX puro.
    """
    formulas = (formulas_json or {}).get("formulas") or []
    summary = (formulas_json or {}).get("summary_it") or ""
    
    if not formulas and not summary:
        return ""

    lines = [f"--- FORMULE E MODELLI MATEMATICI - Pagina {page_no} ---"]
    if summary:
        lines.append(f"Contenuto: {summary}\n")

    for f in formulas:
        desc = f.get("meaning_it") or f.get("description_it") or "Formula"
        latex = f.get("latex", "")
        # Normalizza LaTeX per evitare errori di rendering
        latex = latex.replace("\\\\", "\\") 
        
        vars_list = []
        for v in f.get("variables", []):
            if isinstance(v, dict):
                vars_list.append(f"{v.get('name')}: {v.get('meaning')}")
        
        vars_str = "; ".join(vars_list)
        
        # Blocco semantico: Concetto + LaTeX + Variabili
        block = f"## {desc}\nModello (LaTeX): $${latex}$$\nVariabili: {vars_str}"
        lines.append(block)

    return "\n".join(lines).strip()

# =========================
# KG extraction (LLM) - SOLO dove serve
# =========================
def llm_extract_kg(filename: str, page_no, text: str, model_name: str):
    base = os.path.basename(str(filename))

    # anti-UnboundLocalError
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _safe_head(x, n: int = 900) -> str:
        try:
            s = str(x)
        except Exception:
            s = "<unprintable>"
        s = s.replace("\r", " ").replace("\n", " ")
        return (s[:n] + "...") if len(s) > n else s

    def _normalize_graph_schema(js: Any) -> Optional[Dict[str, Any]]:
        if js is None:
            return None
        if isinstance(js, list):
            return {"nodes": js, "edges": []}
        if not isinstance(js, dict):
            return None

        g = dict(js)

        if "nodes" not in g:
            for k in ("entities", "entity", "concepts", "items", "objects"):
                if k in g and isinstance(g[k], list):
                    g["nodes"] = g[k]
                    break

        if "edges" not in g:
            for k in ("relationships", "relations", "links", "rels"):
                if k in g and isinstance(g[k], list):
                    g["edges"] = g[k]
                    break
            g.setdefault("edges", [])

        return g

    def _extract_balanced_nodes_prefix(raw: str) -> Optional[str]:
            if not raw: return None
            s = raw.strip()
            
            # Se il JSON è chiaramente troncato alla fine (es: finisce con virgola, virgolette o lettera incompleta)
            if s.endswith(",") or s.endswith('"') or s.endswith("e") or not s.endswith("}"):
                # Cerca l'ultima chiusura valida di un oggetto nodo }
                last_obj = s.rfind("}")
                if last_obj != -1:
                    # Ricostruiamo la struttura minima per renderlo un JSON valido
                    # Chiudiamo l'array nodes e aggiungiamo una chiave edges vuota
                    return s[:last_obj+1] + "\n  ],\n  \"edges\": []\n}"
            return None

    def _parse_json_best_effort(raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None

        s = raw.strip()

        # strict
        try:
            return json.loads(s)
        except Exception:
            pass

        # extractor
        try:
            js = safe_json_extract(s)
            if js is not None:
                return js
        except Exception:
            pass

        # repair
        repaired = _extract_balanced_nodes_prefix(s)
        if repaired:
            try:
                return json.loads(repaired)
            except Exception:
                return None

        return None

    def _call_llm(payload_text: str, *, num_predict: int, num_ctx: int) -> str:
        """
        Wrapper unico per la chiamata a Ollama chat. Mantiene coerenza
        tra prima chiamata e retry, e riduce errori “funzioni non definite”.
        """
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": KG_PROMPT},
                {"role": "user", "content": payload_text or ""},
            ],
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": int(num_predict),
                "num_ctx": int(num_ctx),
            },
        )
        return resp.get("message", {}).get("content", "") or ""

    # ------------------------------------------------------------
    # page number safe
    # ------------------------------------------------------------
    try:
        page_no_int = int(page_no)
    except Exception:
        print(f"   ⚠️ [KG-BAD-PAGENO] {base} page_no={_safe_head(page_no,120)}")
        page_no_int = 1

    # ------------------------------------------------------------
    # model params
    # ------------------------------------------------------------
    tlen = len(text or "")
    num_predict = 1800 if tlen > 2200 else 1100
    num_ctx = int(globals().get("KG_NUM_CTX", 8192))

    # ------------------------------------------------------------
    # LLM call (1)
    # ------------------------------------------------------------
    try:
        raw = _call_llm(text or "", num_predict=num_predict, num_ctx=num_ctx)
    except Exception as e:
        print(f"   ❌ [KG-CHAT-ERROR] {base} p{page_no_int}: {e}")
        return [], []

    if not raw.strip():
        print(f"   ⚠️ [KG-EMPTY-RAW] {base} p{page_no_int}")
        return [], []

    # ------------------------------------------------------------
    # parse + repair (1)
    # ------------------------------------------------------------
    js = _parse_json_best_effort(raw)

    # ------------------------------------------------------------
    # retry parse se js None (con testo corto)
    # ------------------------------------------------------------
    if js is None:
        short_text = (text or "")[:1800]
        try:
            print(f"   🔁 [KG-RETRY-PARSE] {base} p{page_no_int}")
            raw2 = _call_llm(short_text, num_predict=1100, num_ctx=4096)
            js = _parse_json_best_effort(raw2)
        except Exception as e:
            print(f"   ❌ [KG-RETRY-ERROR] {base} p{page_no_int}: {e}")
            return [], []

        if js is None:
            print(f"   ❌ [KG-PARSE-FAIL] {base} p{page_no_int}")
            print(f"   [KG-RAW-HEAD]: {_safe_head(raw, 900)}")
            return [], []

    # ------------------------------------------------------------
    # normalize + sanitize (1)
    # ------------------------------------------------------------
    jsn = _normalize_graph_schema(js)
    if jsn is None:
        return [], []

    try:
        nodes, edges = _sanitize_graph(jsn)
    except Exception as e:
        print(f"   ❌ [KG-SANITIZE-ERROR] {base} p{page_no_int}: {e}")
        return [], []

    # ------------------------------------------------------------
    # retry sanitize se vuoto ma testo “ricco”
    # (qui NON usiamo llm_chat/prompt/KG_MODEL_NAME: solo ciò che esiste)
    # ------------------------------------------------------------
    if (not nodes and not edges) and len(text or "") > 800:
        try:
            print(f"   ♻️ [KG-RETRY-EMPTY] {base} p{page_no_int}: empty graph -> retry once")
            # retry con un payload ridotto ma non troppo aggressivo
            retry_text = (text or "")[:2400]
            raw3 = _call_llm(retry_text, num_predict=1100, num_ctx=min(num_ctx, 4096))
            js3 = _parse_json_best_effort(raw3)
            if js3 is not None:
                jsn3 = _normalize_graph_schema(js3)
                if jsn3 is not None:
                    n3, e3 = _sanitize_graph(jsn3)
                    if n3 or e3:
                        nodes, edges = n3, e3
        except Exception as e:
            print(f"   ❌ [KG-RETRY-EMPTY-ERROR] {base} p{page_no_int}: {e}")
            # non fallire l’intera pagina: mantieni output vuoto
            return [], []

    return nodes, edges





def extract_pdf_text_by_page_pdfminer(file_path: str) -> list[str]:
    """
    VERSIONE UPGRADE 3: Estrazione ottimizzata per velocità.
    Utilizza LAParams minimi per evitare l'analisi complessa del layout.
    """
    if extract_pages is None or LTTextContainer is None:
        return []

    # LAParams ottimizzati: disabilitiamo il rilevamento verticale e 
    # allarghiamo i margini per processare i blocchi di testo più velocemente.
    fast_params = LAParams(
        detect_vertical=False, 
        all_texts=True, 
        char_margin=2.0, 
        line_margin=0.5,
        word_margin=0.1
    )
    
    pages: list[str] = []
    try:
        # L'argomento 'caching=True' è fondamentale per non ri-analizzare 
        # le risorse comuni (font, immagini) ad ogni cambio pagina.
        for layout in extract_pages(file_path, laparams=fast_params, caching=True):
            # Usiamo una list comprehension per una raccolta dei chunk più rapida
            chunks = [element.get_text() for element in layout if isinstance(element, LTTextContainer)]
            pages.append("".join(chunks))
    except Exception as e:
        print(f"   ⚠️ PDFMiner Extraction Error: {e}")
        return []

    return pages


# ==============================================================================
# HELPER: CHUNKING RICORSIVO (Text Splitter)
# ==============================================================================
def recurse_text_chunking(text: str, base_meta: Dict[str, Any], max_chars: int = 1000) -> List[Dict[str, Any]]:
    """
    Divide il testo ricorsivamente cercando di non spezzare frasi o paragrafi.
    Usa una gerarchia di separatori: \n\n -> \n -> . -> spazio.
    Restituisce una lista di dizionari pronti per l'ingestion.
    """
    text = text.strip()
    if not text:
        return []

    # CASO BASE: Il testo rientra nella dimensione massima
    if len(text) <= max_chars:
        # Costruiamo il chunk semantico standard
        source = base_meta.get("source", "unknown")
        p_no = base_meta.get("page", 0)
        
        # Header semantico per aiutare il RAG a capire il contesto
        sem_header = f"Doc: {source} | Pag: {p_no}"
        
        # Recupera eventuali override di metadati (es. formule JSON)
        meta_final = base_meta.copy()
        if "metadata_override" in base_meta:
            del meta_final["metadata_override"]
            meta_final.update(base_meta["metadata_override"])

        return [{
            "text_raw": text,
            "text_sem": f"{sem_header}\n{text}",
            "page_no": p_no,
            "toon_type": base_meta.get("type", "text"),
            "section_hint": "content",
            "image_id": base_meta.get("original_image_id"),
            "metadata_override": meta_final
        }]

    # LOGICA RICORSIVA: Il testo è troppo lungo
    chunks = []
    
    # 1. Gerarchia separatori (dal più forte al più debole)
    separators = ["\n\n", "\n", ". ", " "]
    split_char = ""
    
    for sep in separators:
        if sep in text:
            # Verifica euristica: se splittiamo qui, otteniamo pezzi validi?
            temp_parts = text.split(sep)
            if len(temp_parts) > 1:
                split_char = sep
                break
    
    # 2. Se nessun separatore funziona (taglio brutale)
    if not split_char:
        mid = len(text) // 2
        return (
            recurse_text_chunking(text[:mid], base_meta, max_chars) + 
            recurse_text_chunking(text[mid:], base_meta, max_chars)
        )

    # 3. Accumulatore (Greedy Bin Packing)
    raw_parts = text.split(split_char)
    current_chunk_str = ""
    
    for part in raw_parts:
        candidate = part if not current_chunk_str else (current_chunk_str + split_char + part)
        
        if len(candidate) <= max_chars:
            current_chunk_str = candidate
        else:
            if current_chunk_str:
                chunks.extend(recurse_text_chunking(current_chunk_str, base_meta, max_chars))
            current_chunk_str = part

    # 4. Aggiungi l'ultimo pezzo rimasto
    if current_chunk_str:
        chunks.extend(recurse_text_chunking(current_chunk_str, base_meta, max_chars))

    return chunks


def render_full_page_png(page, dpi: int = 220, clip=None) -> bytes:
    """
    Render lossless PNG della pagina (migliore per OCR formule/simboli).
    - page: fitz.Page
    - dpi: 180-240 tipico per formule; 220 è un buon compromesso
    - clip: fitz.Rect opzionale per render parziale
    """
    if dpi <= 0:
        dpi = 220

    zoom = dpi / 72.0  # PyMuPDF usa 72 DPI come base
    mat = fitz.Matrix(zoom, zoom)

    # alpha=False => evita canale trasparenza (più compatto e stabile per OCR)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

    # In alcuni casi pix può risultare in colorspace non RGB.
    # get_pixmap(alpha=False) di solito è già ok; questo è un extra safety:
    try:
        if pix.n not in (3, 4):  # 3=RGB, 4=CMYK/altro
            pix = fitz.Pixmap(fitz.csRGB, pix)
    except Exception:
        pass
    return pix.tobytes("png")

# ==============================================================================
# MOTORE VISIONE QWEN (PURE VISION - NO HYBRID)
# ==============================================================================
def extract_pdf_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Versione RADICALE: Ignora completamente il layer testo del PDF.
    Tratta ogni pagina come una foto (screenshot) e usa Qwen per trascriverla.
    """
    out_chunks: List[Dict[str, Any]] = []
    filename = os.path.basename(file_path)

    # Reset stats
    if "VISION_STATS" in globals() and "_VISION_STATS_LOCK" in globals():
        with _VISION_STATS_LOCK:
            VISION_STATS["pages_total"] = 0

    try:
        doc0 = fitz.open(file_path)
        total_pages = len(doc0)
        doc0.close()
    except Exception as e:
        print(f"   ❌ Errore critico apertura file {filename}: {e}")
        return []

    print(f"   🚀 Ingestion Start: {total_pages} pagine | Mode: Qwen-Vision-ONLY (No Text Fallback)")

    # --- WORKER INTERNO ---
    def process_page_worker(p_idx: int) -> List[Dict[str, Any]]:
        p_no = p_idx + 1
        local_results: List[Dict[str, Any]] = []
        
        try:
            doc_worker = fitz.open(file_path)
            p_obj = doc_worker.load_page(p_idx)
            
            # 1. RENDERIZZA IMMAGINE (L'unica fonte di verità)
            # 200 DPI è il bilanciamento perfetto per Qwen 8B

            
            full_png = render_full_page_png(p_obj, dpi=220)   # lossless, meglio per simboli
            vision_bytes = _downscale_and_compress_for_vision(full_png, max_side=2000, prefer_png=True)

            full_jpg = render_full_page_jpeg(p_obj, dpi=160)

            # 2. SALVA DB (Opzionale, utile per debug)
            img_id = None
            t_conn = pg_get_conn()
            if t_conn:
                try:
                    with t_conn.cursor() as t_cur:
                        img_id = pg_save_image(log_id, full_png, "image/png", f"Page_{p_no}", cur=t_cur)
                    t_conn.commit()
                except: t_conn.rollback()
                finally: pg_put_conn(t_conn)

            # 3. CHIAMA QWEN (Trascrizione JSON strutturata)
            vision_bytes = _downscale_and_compress_for_vision(full_jpg, max_side=1600)
            
            # Usiamo un prompt specifico per la trascrizione completa
            # Nota: Definisci questo prompt o usa quello sotto
            # 0) testo veloce solo per routing
            page_text_fast = ""
            try:
                page_text_fast = p_obj.get_text("text") or ""
            except:
                page_text_fast = ""

            is_formula = is_formula_candidate_page(page_text_fast)
            is_chart   = is_chart_candidate_page(page_text_fast)

            if is_formula:
                prompt = FORMULA_VISION_PROMPT
                max_tok = 1400
            elif is_chart:
                prompt = CHART_VISION_PROMPT
                max_tok = 1100
            else:
                # prompt “plain OCR text” (molto più leggero)
                prompt = "Extract readable text only. NO hallucinations. Return JSON: {\"text\":\"...\",\"confidence\":0.0}"
                max_tok = 800

            ai_json_str = llm_chat_multimodal(
                prompt=prompt,
                image_bytes=vision_bytes,
                model=VISION_MODEL_NAME,
                max_tokens=max_tok
            )

            
            # 4. Parsing JSON
            f_js = safe_json_extract(ai_json_str)

            # second pass solo se serve (conf bassa o latex “rotto”)
            f_js = reconcile_formulas_with_strict_rules(
                f_js=f_js,
                page_image_bytes=vision_bytes,   # stessa immagine inviata alla prima pass
                model=VISION_MODEL_NAME,
                max_tokens=650
            )


            if isinstance(f_js, dict):
                conf = float(f_js.get("confidence") or 0.0)
                if conf < 0.70 and isinstance(f_js.get("formulas"), list) and f_js["formulas"]:
                    f_js = reconcile_formulas_with_strict_rules(f_js, vision_bytes)


            # Costruiamo il testo del chunk dai dati Vision (NO TESTO PDF)
            chunk_text = ""

            def _latex_clean(s: str) -> str:
                s = (s or "").strip()
                # rimuove delimitatori $...$ se presenti
                s = s.replace("$$", "").replace("$", "")
                # normalizza backslash doppi
                s = s.replace("\\\\", "\\")
                return s.strip()

                        
            def _latex_clean(s: str) -> str:
                s = (s or "").strip()
                s = s.replace("$$", "").replace("$", "")
                s = s.replace("\\\\", "\\")
                return s.strip()

            _LATEX_SPACE_FIX = re.compile(r"\s+")
            _LATEX_EMPTY_PARENS = re.compile(r"\(\s*\)")
            _LATEX_WEIRD_DASH = re.compile(r"[–—−]")

            def reconcile_formulas_with_strict_rules(latex: str) -> str:
                """
                Post-clean deterministico:
                - NON inventa niente
                - normalizza spazi e simboli
                - rimuove pattern OCR tipici che rompono la coerenza
                """
                s = (latex or "").strip()
                if not s:
                    return ""

                # normalizza trattini
                s = _LATEX_WEIRD_DASH.sub("-", s)

                # rimuove parentesi vuote "( )"
                s = _LATEX_EMPTY_PARENS.sub("", s)

                # normalizza spazi
                s = _LATEX_SPACE_FIX.sub(" ", s).strip()

                # fix comuni OCR: "1+ i ( )" => "1+i"
                s = s.replace("1+ ", "1+").replace("+ ", "+").replace(" +", "+")
                s = s.replace(" - ", "-").replace(" = ", "=")

                # elimina duplicazioni immediate di simboli causate da OCR (senza aggressività)
                s = s.replace("× ×", "×").replace("= =", "=")

                return s.strip()


            def _norm_table(s: str) -> str:
                # normalizza il "middot" OCR (0·990) in punto (0.990)
                return (s or "").replace("·", ".").strip()

            chunk_text = f"--- CONTENUTO PAGINA {p_no} (Vision OCR) ---\n"

            formulas_out = []
            tables_out = []
            summary_it = ""
            confidence = 0.0

            if isinstance(f_js, dict):
                summary_it = (f_js.get("summary_it") or "").strip()
                try:
                    confidence = float(f_js.get("confidence") or 0.0)
                except Exception:
                    confidence = 0.0

                # ✅ 1) Schema “ufficiale” del FORMULA_VISION_PROMPT
                if isinstance(f_js.get("formulas"), list) or isinstance(f_js.get("tables_md"), list):
                    formulas_out = f_js.get("formulas") or []
                    tables_out   = f_js.get("tables_md") or []

                # ✅ 2) Compat: schema alternativo “content[]” (se presente)
                elif isinstance(f_js.get("content"), list):
                    content = f_js.get("content") or []
                    formulas_out = [x for x in content if isinstance(x, dict) and x.get("type") == "formula"]
                    tables_out   = [x for x in content if isinstance(x, dict) and x.get("type") == "table"]

            has_any = bool(formulas_out or tables_out or summary_it)

            if has_any:
                if summary_it:
                    chunk_text += f"Sintesi: {summary_it}\n"
                chunk_text += f"Affidabilità: {confidence:.2f}\n\n"

                if formulas_out:
                    chunk_text += f"--- FORMULE - Pagina {p_no} ---\n"
                    for i, form in enumerate(formulas_out[:VISION_MAX_FORMULAS_PER_PAGE], start=1):
                        latex = _latex_clean(form.get("latex", ""))
                        meaning = (form.get("meaning_it") or form.get("description") or "").strip()
                        if latex:
                            chunk_text += f"{i}) LaTeX: $${latex}$$\n"
                        if meaning:
                            chunk_text += f"   Significato: {meaning}\n"
                        # variabili (se presenti nello schema prompt)
                        vars_ = form.get("variables") if isinstance(form, dict) else None
                        if isinstance(vars_, list) and vars_:
                            chunk_text += "   Variabili: " + ", ".join(
                                f"{(v.get('name') or '').strip()}={(v.get('meaning') or '').strip()}"
                                for v in vars_ if isinstance(v, dict)
                            ) + "\n"
                        chunk_text += "\n"

                if tables_out:
                    chunk_text += f"--- TABELLE - Pagina {p_no} ---\n"
                    for t in tables_out[:5]:
                        cap = (t.get("caption") or "").strip()
                        md  = _norm_table(t.get("markdown") or t.get("tables_md") or "")
                        if cap:
                            chunk_text += f"Caption: {cap}\n"
                        if md:
                            chunk_text += md + "\n"
                        chunk_text += "\n"

            else:
                chunk_text += "(Nessun contenuto strutturato rilevato)\n"
                chunk_text += str(ai_json_str)[:800]

            
                       
            # 5. CREAZIONE CHUNK
            # Solo se abbiamo contenuto valido
            if len(chunk_text) > 50:
                meta = {
                    "source": filename, 
                    "page": p_no, 
                    "type": "vision_ocr", 
                    "original_image_id": img_id,
                    "metadata_override": {
                        "has_formulas": True if f_js.get("formulas") else False,
                        "formulas_data": f_js.get("formulas"),
                        # NO skip_kg -> Vogliamo i nodi!
                    }
                }
                
                chunks = recurse_text_chunking(chunk_text, meta, max_chars=CHUNK_MAX_CHARS)
                local_results.extend(chunks)

        except Exception as e:
            print(f"   ⚠️ Vision Error p.{p_no}: {e}")
        finally:
            try: doc_worker.close()
            except: pass

        return local_results

    # --- ESECUZIONE PARALLELA (Qwen è veloce, usiamo 3-4 worker) ---
    with ThreadPoolExecutor(max_workers=VISION_PARALLEL_WORKERS) as executor:
        future_to_page = {executor.submit(process_page_worker, i): i for i in range(total_pages)}
        
        completed = 0
        for future in as_completed(future_to_page):
            res = future.result()
            if res: out_chunks.extend(res)
            completed += 1
            print(f"   🔄 Processing: {completed}/{total_pages} pages...", end="\r")
            
    print("")
    out_chunks.sort(key=lambda x: x["page_no"])
    return add_context_windows(out_chunks)

# ==============================================================================
# DISPATCHER FILE
# ==============================================================================
def extract_file_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_chunks(file_path, log_id)
    elif ext == ".md":
        return extract_markdown_chunks(file_path, log_id)
    return []


def extract_pdf_as_markdown_assets(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Trasforma PDF in MD virtuale e parcheggia immagini in RAM.
    Usa MIN_ASSET_SIZE per decidere cosa processare.
    """
    doc = fitz.open(file_path)
    try:
        filename = os.path.basename(file_path)
        asset_park = {}
        virtual_md_content = []
        total_pages = len(doc)

        print(f"   🔄 Asset Pipeline: Analisi {total_pages} pagine | Soglia Asset: {MIN_ASSET_SIZE} bytes")

        for p_idx in range(total_pages):
            page = doc[p_idx]
            p_no = p_idx + 1

            p_text = page.get_text("text")
            if is_structural_page(p_text): 
                continue

            p_text = normalize_ws(p_text)
            virtual_md_content.append(f"\n# PAGE {p_no}\n{p_text}")

            # --- FIX 1: Rilevamento Ibrido (Immagini + Diagrammi Vettoriali) ---
            image_list = page.get_images(full=True)
            
            # Verifichiamo se la pagina contiene disegni complessi (linee/archi) tipici dei diagrammi
            # e se il testo cita esplicitamente una "Figura" o "Fig."
            has_vector_content = len(page.get_drawings()) > 10 
            has_figure_label = bool(re.search(r"\b(Fig|Figure|Figura)\b", p_text))

            # Se get_images è vuoto ma ci sono disegni e riferimenti, catturiamo l'intera pagina come immagine
            if not image_list and has_vector_content and has_figure_label:
                pix = page.get_pixmap(dpi=VISION_DPI)
                img_bytes = pix.tobytes("jpg")
                asset_id = f"fig_vector_p{p_no}.jpg"
                asset_park[asset_id] = img_bytes
                virtual_md_content.append(f"\n![{asset_id}]({asset_id})\n")

            # Processiamo comunque le immagini raster standard se presenti
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                if len(img_bytes) < MIN_ASSET_SIZE:
                    continue

                asset_id = f"img_p{p_no}_{img_idx}.jpg"
                asset_park[asset_id] = img_bytes
                virtual_md_content.append(f"\n![{asset_id}]({asset_id})\n")

            if p_no % 10 == 0 or p_no == total_pages:
                print(f"   📄 Pagina {p_no}/{total_pages} processata...")

        full_virtual_md = "\n".join(virtual_md_content)
        return process_virtual_md_chunks(full_virtual_md, asset_park, filename, log_id)

    finally:
        # ✅ CRITICAL on Windows: release file handle before shutil.move
        try:
            doc.close()
        except Exception:
            pass



def process_virtual_md_chunks(content: str, asset_park: dict, filename: str, log_id: int) -> List[Dict[str, Any]]:
    """Versione 2.9.1: Include Debug Saving, Context Hint e Normalizzazione Universale."""
    out_chunks = []
    content = clean_markdown_structure(content)
    raw_paras = re.split(r'\n(?=# PAGE)|\n\n', content)

    for para in raw_paras:
        para_strip = para.strip()
        if not para_strip: continue

        # Identificazione Pagina
        current_page = 1
        page_match = re.search(r'# PAGE (\d+)', para_strip)
        if page_match: current_page = int(page_match.group(1))

        # Scomposizione granulare dei chunk troppo lunghi (>1200 char)
        sub_segments = [para_strip] if len(para_strip) <= 1200 else split_text_with_overlap(para_strip, 1000, 200)

        for sub_p in sub_segments:
            # Fix Caratteri Corrotti universale
            clean_sub_p = safe_normalize_text(sub_p)
            
            chunk_data = {
                "text_raw": clean_sub_p,
                "text_sem": f"Doc: {filename} | Content: {clean_sub_p[:80]}...\n{clean_sub_p}",
                "page_no": current_page, 
                "toon_type": "text",
                "section_hint": find_section_hint(clean_sub_p)
            }

            # --- LOGICA ASSET VISUALI (Surgical Vision AI) ---
            img_match = re.search(r'!\[.*?\]\(((?:img_|fig_).*?\.jpg)\)', clean_sub_p)
            if img_match and PDF_VISION_ENABLED:
                asset_id = img_match.group(1)
                img_bytes = asset_park.get(asset_id)

                if img_bytes and len(img_bytes) >= MIN_ASSET_SIZE and ai_vision_gatekeeper(img_bytes):
                    chunk_data["toon_type"] = "immagine"

                    hint = f"{filename} | page {current_page}"
                    
                    c_js = extract_chart_via_vision(img_bytes, context_hint=hint) or {}

                    # 🔒 GEO SANITIZATION (STRICT)
                    bad_geo_tokens = [" sud", "south"]
                    cats = c_js.get("categories_it", [])
                    if isinstance(cats, list):
                        c_js["categories_it"] = [
                            c for c in cats
                            if not any(tok in c.lower() for tok in bad_geo_tokens)
                        ]

                    # --- GEO SANITIZE ---
                    cats = c_js.get("categories_it", [])
                    if isinstance(cats, list):
                        cleaned = []
                        for c in cats:
                            low = str(c).lower()
                            if " sud" in low or "south" in low:
                                continue
                            cleaned.append(c)
                        c_js["categories_it"] = cleaned


                    # --- FIX DEFINITIVO CATEGORIE GEO ---
                    cats = c_js.get("categories_it", [])
                    if isinstance(cats, list):
                        cleaned = []
                        for c in cats:
                            low = str(c).lower()
                            if " sud" in low or "south" in low:
                                continue
                            cleaned.append(c)
                        c_js["categories_it"] = cleaned

                    # Se timeframe non è affidabile, NON passarla all’analyst
                    if c_js.get("timeframe") == "NOT READABLE":
                        c_js.pop("timeframe", None)

                    # 🔒 BLOCK AMBIGUOUS TIMEFRAME FROM ANALYST
                    tf = c_js.get("timeframe")
                    years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(tf)) if tf else []

                    # Se non ho ALMENO 2 anni puliti, l’analyst NON deve vederlo
                    if len(years) < 2:
                        c_js.pop("timeframe", None)

                    analysis_it = generate_chart_analysis_it(c_js, page_text="")
                    if analysis_it:
                        c_js["analysis_it"] = analysis_it


                    semantic = build_chart_semantic_chunk(current_page, c_js)
                    if c_js:
                        # If the paragraph is basically only the image markdown, replace semantic text fully
                        md_only = clean_sub_p.strip()
                        is_image_only = (md_only == f"![{asset_id}]({asset_id})") or (len(md_only) < 80 and "img_" in md_only)

                        semantic = build_chart_semantic_chunk(current_page, c_js)
                        chunk_data["text_sem"] = semantic
                        if is_image_only:
                            chunk_data["text_raw"] = semantic  # so embeddings + preview are meaningful
                    else:
                        chunk_data["text_sem"] = normalize_ws(
                            f"--- CONTENUTO VISIVO - Pagina {current_page} ---\n"
                            f"Asset: {asset_id}\n"
                            f"Descrizione: immagine estratta dal PDF.\n"
                            f"Stato: non interpretata automaticamente."
                        )
                        chunk_data["text_raw"] = chunk_data["text_sem"]

                    # save image to PG
                    conn = pg_get_conn()
                    try:
                        with conn.cursor() as cur:
                            chunk_data["image_id"] = pg_save_image(
                                log_id, img_bytes, "image/jpeg", f"RAM_{asset_id}", cur
                            )
                        conn.commit()
                    finally:
                        pg_put_conn(conn)


            out_chunks.append(chunk_data)
    return out_chunks

# =========================
# 3. DISPATCHER E KG (FIX VALUE HUNTER)
# =========================

def extract_file_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # ✅ FORZIAMO IL MOTORE "VISION SUPREMACY" (Qwen legge i pixel, ignora '2C0D')
        return extract_pdf_chunks(file_path, log_id)
    elif ext == ".md":
        return extract_markdown_chunks(file_path, log_id)
    return []

# =========================
# FILE DISPATCH (PDF only here)
# =========================
def process_single_file(file_path: str, source_type: str, doc_meta: dict):
    """
    Definitive, coherent ingestion pipeline (v2.4 fix):
    - Uses extract_file_chunks() (PDF -> virtual MD + asset park, MD -> chunker)
    - Deterministic IDs (doc_id + page + chunk_index + toon_type + text hash + image_id)
    - Batch flush to Postgres / Qdrant / Neo4j
    - KG per-page with consistent constants: KG_TEXT_MAX_CHARS, KG_TIMEOUT, KG_MAX_TRIPLES
    """
    t0 = time.time()
    filename = os.path.basename(file_path)

    # Ensure dirs
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    # Resolve meta
    tier = (doc_meta or {}).get("tier", DEFAULT_TIER_META.get("tier", "B"))
    ontology = (doc_meta or {}).get("ontology", DEFAULT_ONTOLOGY)
    effective_date = (doc_meta or {}).get("effective_date") or ""
    topics = (doc_meta or {}).get("topics") or []
    content_type = (doc_meta or {}).get("content_type") or ""
    source_kind = (doc_meta or {}).get("source_kind") or ""

    print(f"   ⚙️ Engine Start: {filename} | tier={tier} | ontology={ontology}")

    log_id = pg_start_log(filename, source_type)
    doc_id = sha256_file(file_path)[:32]

    # Lazy init clients (globals)
    global embedder, qdrant_client
    embedder = get_embedder()
    qdrant_client = get_qdrant_client()
    ensure_qdrant_collection()

    # Extract chunks (multiformat dispatcher)
    chunks = extract_file_chunks(file_path, log_id)
    if not chunks:
        pg_close_log(log_id, "FAILED", 0, _ms(t0), "No chunks extracted")
        shutil.move(file_path, os.path.join(FAILED_DIR, filename))
        print(f"   ❌ No chunks extracted: {filename}")
        return

    # Ensure chunk_index exists and attach doc-level metadata into per-chunk metadata
    for idx, ch in enumerate(chunks):
        ch.setdefault("chunk_index", idx)
        ch.setdefault("toon_type", "text")
        ch.setdefault("page_no", 1)

        meta = {}
        if isinstance(ch.get("metadata_override"), dict):
            meta.update(ch["metadata_override"])
        if isinstance(ch.get("metadata"), dict):
            meta.update(ch["metadata"])

        # doc-level meta
        meta.update({
            "doc_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "tier": tier,
            "ontology": ontology,
            "effective_date": effective_date,
            "topics": topics,
            "content_type": content_type,
            "source_kind": source_kind,
            "section_hint": ch.get("section_hint", ""),
            "image_id": ch.get("image_id"),
        })
        ch["metadata"] = meta

    # -------------------------
    # helpers: Qdrant flush
    # -------------------------
    def flush_qdrant_points_batch(points: list):
        if not points:
            return
        try:
            pts = [
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in points
            ]
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=pts)
        except Exception as e:
            print(f"   ⚠️ Qdrant Batch Error: {e}")

# -------------------------
    # 1. Buffers e Contatori (Sostituzione Integrale)
    # -------------------------
    qdrant_points: list = []
    pg_rows: list = []
    neo4j_rows: list = []
    total_chunks = 0  # Manteniamo il nome originale per il log finale
    num_chunks_totali = len(chunks)

    print(f"   🚀 Inizio elaborazione: {num_chunks_totali} chunks totali (Batch Size: {EMBED_BATCH_SIZE})")

    # -------------------------
    # 2. Ciclo di Batching con Progress Tracker
    # -------------------------
    for i in range(0, num_chunks_totali, EMBED_BATCH_SIZE):
        batch_t0 = time.time()
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        texts = [c.get("text_sem", "") for c in batch]

        # Feedback visivo del progresso
        percentuale = min(100, int((i + len(batch)) / num_chunks_totali * 100))
        print(f"   🔄 [{percentuale}%] Analisi chunk {i + len(batch)}/{num_chunks_totali}...", end="\r")

        # Generazione Embeddings
        vecs = embedder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Estrazione KG (Knowledge Graph)
        batch_kg_results: dict[int, tuple[list, list]] = {}
        if KG_ENABLED:
            pages_map: dict[int, list[tuple[int, dict]]] = {}
            for local_j, ch in enumerate(batch):
                global_idx = i + local_j
                p = int(ch.get("page_no") or 1)
                pages_map.setdefault(p, []).append((global_idx, ch))

            futures_kg: dict[int, cf.Future] = {}
            for p_no, indexed_chunks in pages_map.items():
                # Controllo Flag 'skip_kg' (impostato in passo 4 per le formule)
                should_skip_kg = False
                if indexed_chunks:
                    # Controlliamo il primo chunk della pagina
                    _, first_ch = indexed_chunks[0]
                    if first_ch.get("metadata", {}).get("skip_kg") is True:
                        should_skip_kg = True

                if should_skip_kg:
                    # FIX VELOCITÀ: Salta esecuzione KG inutile su pagine formule
                    continue

                combined_text = "\n".join([ch.get("text_sem", "") for _, ch in indexed_chunks])
                text_for_ai = re.sub(r'Doc:.*?\| Content:.*?\n|# PAGE \d+', '', combined_text).strip()
                text_for_ai = safe_normalize_text(text_for_ai)[:KG_TEXT_MAX_CHARS]

                # Filtro Value Hunter (Standard)
                if is_keyword_candidate_hybrid(text_for_ai):
                    futures_kg[p_no] = kg_executor.submit(
                        llm_extract_kg, filename, p_no, text_for_ai, LLM_MODEL_NAME
                    )

            # Raccolta risultati KG
            for p_no, fut in futures_kg.items():
                current_page_chunks = pages_map.get(p_no, [])
                try:
                    nodes, edges = fut.result(timeout=KG_TIMEOUT)
                    if nodes or edges:
                        # Canonicalizzazione relazioni
                        edges = canonicalize_edges_to_verb_object(
                            canonicalize_edges_by_base_presence(edges)
                        )
                        for global_idx, _ in current_page_chunks:
                            batch_kg_results[global_idx] = (nodes, edges)
                except Exception as ex:
                    print(f"\n   ⚠️ KG ERROR p.{p_no}: {ex}")

        # Costruzione dei dati per i Database
        for j, ch in enumerate(batch):
            global_idx = i + j
            vector = vecs[j]
            
            chunk_id = deterministic_chunk_id(
                doc_id=doc_id,
                page_no=int(ch.get("page_no") or 1),
                chunk_index=int(ch.get("chunk_index") or global_idx),
                toon_type=str(ch.get("toon_type") or "text"),
                text_sem=str(ch.get("text_sem") or ""),
                image_id=ch.get("image_id"),
            )

            # Payload Qdrant
            payload = dict(ch.get("metadata") or {})
            payload.update({
                "text_sem": str(ch.get("text_sem") or ""),
                "log_id": log_id,
                "page_no": int(ch.get("page_no") or 1),
                "chunk_index": int(ch.get("chunk_index") or global_idx),
                "toon_type": ch.get("toon_type"),
            })
            qdrant_points.append({"id": chunk_id, "vector": vector.tolist(), "payload": payload})

            # Record Postgres
            pg_rows.append((
                log_id, int(ch.get("chunk_index") or global_idx),
                ch.get("toon_type"), ch.get("text_raw"), ch.get("text_sem"),
                json.dumps(ch.get("metadata") or {}, ensure_ascii=False), chunk_id
            ))

            # Dati Neo4j
            kg_nodes, kg_edges = batch_kg_results.get(global_idx, ([], []))
            neo4j_rows.append({
                "doc_id": doc_id, "filename": filename, "doc_type": source_type,
                "log_id": log_id, "page_no": int(ch.get("page_no") or 1),
                "chunk_id": chunk_id, "chunk_index": int(ch.get("chunk_index") or global_idx),
                "toon_type": ch.get("toon_type"), "text_sem": ch.get("text_sem") or "",
                "section_hint": ch.get("section_hint") or "", "ontology": ontology,
                "nodes": kg_nodes, "edges": kg_edges
            })
            total_chunks += 1 # Incremento fondamentale per pg_close_log

        # Flush dei dati nei DB per ogni batch
        flush_postgres_chunks_batch(pg_rows)
        flush_qdrant_points_batch(qdrant_points)
        flush_neo4j_rows_batch(neo4j_rows)
        
        # Pulizia buffer per il prossimo ciclo
        pg_rows.clear()
        qdrant_points.clear()
        neo4j_rows.clear()

        batch_ms = int((time.time() - batch_t0) * 1000)
        print(f"   📦 Batch {int(i/EMBED_BATCH_SIZE)+1} completato in {batch_ms}ms")

    # Final flush
    flush_postgres_chunks_batch(pg_rows)
    flush_qdrant_points_batch(qdrant_points)
    flush_neo4j_rows_batch(neo4j_rows)

    # Close log
    pg_close_log(log_id, "DONE", total_chunks, _ms(t0))

    # Archive file
    try:
        shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
    except Exception:
        pass

    print(f"   ✅ Completed: {filename} | chunks={total_chunks} | ms={_ms(t0)}")




def main():
    """
    Punto di ingresso principale dell'Ingestion Engine.
    Configura l'ambiente, ottimizza Ollama e processa i file supportati.
    """
    # 1. Preparazione delle cartelle di lavoro
    # Crea la struttura dei Tier (A, B, C) se non presente
    os.makedirs(INBOX_DIR, exist_ok=True)
    ensure_inbox_structure(INBOX_DIR)
  
    # 2. Reset Totale OLLAMA (Turbo Mode per P5000)
    # Riavvia il server con NUM_PARALLEL=1 per garantire massima stabilità 
    # evitando conflitti di VRAM durante l'analisi Vision e KG.
    if not force_restart_ollama(num_parallel="1"):
        print("   ❌ Errore: Impossibile avviare Ollama in modalità ottimizzata.")
        print("   ⚠️ L'ingestion potrebbe fallire o risultare estremamente lenta.")
    
    print("\n" + "="*60)
    print("=== Ingestion Engine v2.3 (FAST + Markdown Support + Value Hunter) ===")
    print("="*60 + "\n")

    # 3. Definizione estensioni supportate (Upgrade Markdown)
    supported = {".pdf", ".md"}

    # 4. Scansione ricorsiva della cartella INBOX
    input_files = []
    for root, _, files in os.walk(INBOX_DIR):
        for fname in files:
            # Salta i file di metadati sidecar
            if fname.lower().endswith(".meta.json"):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext in supported:
                # Memorizza la radice per il dispatching corretto del Tier
                input_files.append((root, os.path.join(root, fname)))

    # 5. Verifica se ci sono file da processare
    if not input_files:
        print("   ✅ INBOX vuota: nessuna operazione necessaria.")
        return

    print(f"   📂 Trovati {len(input_files)} file da processare. Inizio sequenza...")

    # 6. Ciclo di elaborazione principale
    for root_folder, file_path in input_files:
        try:
            # Determina Tier e Ontology in base alla cartella di origine
            doc_meta = dispatch_document(file_path, root_folder)
            
            # Avvia l'ingestion multimodale (Text + Vision + KG)
            process_single_file(file_path, "document", doc_meta)
            
        except Exception as e:
            print(f"   ❌ Errore critico durante il processing di {os.path.basename(file_path)}: {e}")
            # Sposta il file in FAILED se non gestito da process_single_file
            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(FAILED_DIR, os.path.basename(file_path)))

    print("\n" + "="*60)
    print("   ✨ Ingestion completata con successo.")
    print("="*60)


if __name__ == "__main__":
    main()
