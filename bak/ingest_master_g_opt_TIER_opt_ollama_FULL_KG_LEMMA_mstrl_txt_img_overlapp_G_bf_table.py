
"""
set EMBED_BATCH_SIZE=16
set DB_FLUSH_SIZE=96
set VISION_PARALLEL_WORKERS=1
s1300
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

import sys

# --- FIX ANTI-BLOCCO ---
# Impostiamo queste variabili PRIMA di importare altre librerie pesanti.
# Questo risolve il freeze quando si calcolano gli embeddings mentre Ollama è attivo.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# valori "safe" (evitano oversubscription e spesso evitano freeze)
CPU_THREADS = os.environ.get("EMBED_CPU_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = CPU_THREADS
os.environ["MKL_NUM_THREADS"] = CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = CPU_THREADS

import re
import json
import time
import uuid
import shutil
import hashlib
import base64
from typing import List, Dict, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures as cf
import subprocess
import requests
from threading import Lock
import gc

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

from sentence_transformers import util
import torch



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

KG_TEXT_MAX_CHARS = int(os.getenv("KG_TEXT_MAX_CHARS", "2000"))   # chars sent to KG model per page
KG_MAX_TRIPLES = int(os.getenv("KG_MAX_TRIPLES", "50"))           # 10 soft cap (sanitize already caps)
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
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))    # se la tua VRAM regge, aumenta velocità

# Vision switches
PDF_VISION_ENABLED = os.getenv("PDF_VISION_ENABLED", "1") == "1"
PDF_VISION_ONLY_IF_TEXT_SCARSO = False #= os.getenv("PDF_VISION_ONLY_IF_TEXT_SCARSO", "0") == "1"
PDF_MIN_TEXT_LEN_FOR_NO_VISION = 0 #= int(os.getenv("PDF_MIN_TEXT_LEN_FOR_NO_VISION", "450"))

VISION_DPI = int(os.getenv("VISION_DPI", "130"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "2000000"))

VISION_MAX_FORMULAS_PER_PAGE = int(os.getenv("VISION_MAX_FORMULAS_PER_PAGE", "10"))

# --- SOGLIE ASSET VISUALI ---
# Immagini più piccole di questo valore (in byte) verranno ignorate.
# 7000 = icone/loghi | 2000 = molto permissivo | 15000 = molto sever
PDF_EXTRACT_EMBEDDED_IMAGES = True #= os.getenv("PDF_EXTRACT_EMBEDDED_IMAGES", "1") == "1"
PDF_VISION_ON_EMBEDDED_IMAGES = True # = os.getenv("PDF_VISION_ON_EMBEDDED_IMAGES", "1") == "1"
PDF_MAX_IMAGES_PER_PAGE = int(os.getenv("PDF_MAX_IMAGES_PER_PAGE", "8"))
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", "1"))
MIN_ASSET_SIZE = int(os.getenv("MIN_ASSET_SIZE", "2000"))


# Speed: Vision parallel + cache
VISION_PARALLEL_WORKERS = 1 #int(os.getenv("VISION_PARALLEL_WORKERS", "4"))  # 4-6 di solito ok
OLLAMA_NUM_PARALLEL=1
VISION_CACHE_MAX = int(os.getenv("VISION_CACHE_MAX", "5000"))             # entries in-memory

# Commit policy
PG_COMMIT_EVERY_N_PAGES = int(os.getenv("PG_COMMIT_EVERY_N_PAGES", "25"))

# KG extraction (solo dove serve)
KG_ENABLED = os.getenv("KG_ENABLED", "1") == "1"
KG_MIN_LEN = int(os.getenv("KG_MIN_LEN", "300")) #
MAX_KG_CHUNKS_PER_DOC = int(os.getenv("MAX_KG_CHUNKS_PER_DOC", "50")) #30



PDF_TEXT_EXTRACTOR = "fitz"
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "pdfminer").lower()
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "fitz").lower() #<--- OPZIONALE PIù VELOCE
#PDF_TEXT_EXTRACTOR = os.getenv("PDF_TEXT_EXTRACTOR", "fitz").lower()  # fitz | pdfminer

# --- Nella sezione A. GESTIONE FORMULE ---
FULLPAGE_DPI = 110 
CROP_DPI = 160 
KG_WORKERS = 2  # Forza l'elaborazione seriale per non saturare la VRAM
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

    # --- AGGIUNTA PER TRANSCRIPT E GESTIONE ---
    "benchmark", "drawdown", "asset allocation", "sottoperformance", "outperformance",
    "overweight", "underweight", "gestore", "fund", "fondo", "allocazione", "esposizione",
    "volatilità", "volatility", "sharpe", "sortino", "beta", "alpha", "track record",


    # --- AGGIUNTA: Strumenti Derivati & Tecnici ---
    "opzione", "option", "call", "put", "future", "futures", "forward", "swap",
    "certificato", "certificate", "warrant", "covered warrant",
    "premio", "premium", "strike", "strike price", "sottostante", "underlying",
    "scadenza", "maturity", "esercizio", "exercise", "leva", "leverage",
    "hedging", "copertura", "speculazione", "arbitraggio", "volatilità implicita",
    "benchmark", "etf", "nav", "greeks", "delta", "gamma", "theta", "vega"


    # --- Analisi Dati & Visual (già nel tuo script) ---
    "grafico", "graph", "tabella", "trend", "asse", "legenda", "chart", "table", "axis", "legend",
    "regression", "regressione", "model", "modello", "algorithm", "algoritmo", "correlation", "correlazione",
    "inference", "inferenza", "variance", "varianza","slope", "mean", "average", "moda", "mode", "modale", "modal"
]


# =========================
# SEMANTIC GATEKEEPER CONFIG
# =========================
# Definiamo i "Centroidi" tematici. Se un chunk è simile a questi, passa.
GATEKEEPER_CONCEPTS = [
    "Financial markets analysis and trading strategies",
    "Risk management, volatility, and performance metrics",
    "Derivatives, options, futures, and financial instruments definition", # <--- Questo cattura il tuo chunk sulle opzioni!
    "Corporate strategy, mergers, acquisitions, and dividends",
    "Macroeconomic indicators, inflation, and central bank policies",
    "ESG sustainability, regulations, and compliance (SFDR, Taxonomy)"
]

# Cache per gli embedding delle ancore (calcolati una volta sola all'avvio)
_GK_ANCHOR_EMBEDDINGS = None



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

# ---- FAST KG GATEKEEPER (NO LLM) ----
_KG_GK_CACHE = {}
_KG_GK_CACHE_MAX = 50000
_PROPER_NOUN_FAST = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")


# def con pytorch
def ai_gatekeeper_decision(text: str) -> bool:
    """
    Gatekeeper Semantico (V2):
    Invece di contare keyword, calcola quanto il testo è 'vicino' ai concetti finanziari.
    """
    if not text or len(text) < KG_MIN_LEN: # Filtro lunghezza minima (es. 250-300 char)
        return False

    global _GK_ANCHOR_EMBEDDINGS
    embedder = get_embedder() # Recupera il modello bge-m3 già caricato

    # 1. Inizializzazione Lazy delle ancore (fatta solo alla prima chiamata)
    if _GK_ANCHOR_EMBEDDINGS is None:
        # print("   🧠 Inizializzazione Semantic Gatekeeper...")
        _GK_ANCHOR_EMBEDDINGS = embedder.encode(GATEKEEPER_CONCEPTS, convert_to_tensor=True)

    # 2. Embedding del Chunk corrente
    # Nota: bge-m3 è molto veloce, su CPU impiega pochi millisecondi per 1000 char
    chunk_embedding = embedder.encode(text, convert_to_tensor=True)

    # 3. Calcolo Similarità (Coseno)
    # Confronta il chunk con TUTTI i concetti e prende il punteggio massimo
    scores = util.cos_sim(chunk_embedding, _GK_ANCHOR_EMBEDDINGS)
    max_score = float(torch.max(scores))

    # 4. Soglia di Decisione
    # 0.35 è solitamente una buona soglia per "vagamente correlato".
    # 0.50 è "molto correlato".
    # Se il testo parla di "cucinare la pasta", lo score sarà < 0.20.
    THRESHOLD = 0.38 

    # Debug (opzionale: scommenta per calibrare la soglia)
    if max_score > 0.3:
       print(f"   [GK] Score: {max_score:.3f} | Text: {text[:50]}...")

    return max_score >= THRESHOLD


def ai_gatekeeper_decision_from_vec(vec, threshold: float = 0.38) -> bool:
    """
    Gatekeeper Semantico (V2) ma usando un embedding già calcolato (NO re-encode).
    vec: np.ndarray | list[float] | torch.Tensor
    """
    global _GK_ANCHOR_EMBEDDINGS

    if vec is None:
        return False

    embedder = get_embedder()

    # Lazy init ancore (una sola volta)
    if _GK_ANCHOR_EMBEDDINGS is None:
        _GK_ANCHOR_EMBEDDINGS = embedder.encode(GATEKEEPER_CONCEPTS, convert_to_tensor=True)

    # vec -> torch tensor (1, dim)
    if not isinstance(vec, torch.Tensor):
        vec_t = torch.tensor(vec, dtype=torch.float32)
    else:
        vec_t = vec.float()

    if vec_t.dim() == 1:
        vec_t = vec_t.unsqueeze(0)

    scores = util.cos_sim(vec_t, _GK_ANCHOR_EMBEDDINGS)
    max_score = float(torch.max(scores))

    if max_score > 0.3:
        print(f"   [GK] Score(vec): {max_score:.3f}")

    return max_score >= threshold



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


#qwen3-8b-gguf:latest
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-vl-8b-instruct")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma2:9b") 

#VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "ministral-3:8b")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b") 
#VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "minicpm-v")
#VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "llama3.2-vision:11b")

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "ministral-3:8b")



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

# ==============================================================================
# PROMPT VISIONE UNIVERSALE (Vision Supremacy)
# ==============================================================================
FORMULA_VISION_PROMPT = """
You are a Scientific OCR Engine.
Your task is to transcribe mathematical content from the image into structured JSON with LaTeX.

### INPUT SOURCE HANDLING:
- **Text Layer**: If you see garbled text, reconstruction it into valid math.
- **Vector/Image**: Transcribe graphical formulas exactly as they appear.

### RULES:
1. **LATEX MANDATORY**: Use standard LaTeX for all math (e.g., `\\frac{a}{b}`, `\\int`, `\\sum`, `\\sigma`).
2. **FIDELITY**: Transcribe EXACTLY symbols found in the image. Do not hallucinate formulas not present.
3. **NO ASCII MATH**: Do not use "x^2". Use "$x^2$".

### OUTPUT FORMAT (JSON ONLY):
{
  "summary_it": "Breve descrizione del contenuto matematico (es. 'Equazione differenziale', 'Modello statistico').",
  "formulas": [
    {
      "description_it": "Nome o etichetta visibile (es. 'Eq. 1.2' o 'Definizione')",
      "latex": " ... write here the LaTeX code inside dollars, e.g. $a + b = c$ ... ", 
      "variables": [
        {"name": "symbol", "meaning": "variable meaning (if context allows) or 'unknown'"}
      ]
    }
  ]
}
"""

CHART_VISION_PROMPT = """
You are a SENIOR FINANCIAL ANALYST AI.
Your goal: Extract precise financial data and strategic insights from charts in financial reports.

CONTEXT: The user is a financial professional looking for trends, volatility, margins, and key performance indicators (KPIs).

ABSOLUTE PROHIBITIONS:
- Do NOT invent numbers. If a bar is between 10 and 20, do NOT guess "15.3" unless explicitly labeled.
- Do NOT confuse "Estimates" (e) with "Actuals" (a).
- Do NOT ignore negative signs (e.g., values in parentheses "(100)" mean "-100").

Return ONLY valid JSON (no markdown), EXACT schema:

{
  "kind": "line_chart|bar_chart|waterfall|pie|table|candlestick|heatmap|other",
  "title": "Exact chart title (e.g., 'EBITDA Evolution 2020-2024')",
  "subtitle": "Subtitle including currency/scale (e.g., 'In € millions')",
  "source": "Source if visible (e.g., 'Bloomberg', 'Company Data')",
  "timeframe": "Explicit period (e.g., 'Q1 2023 vs Q1 2024') or 'NOT READABLE'",

  "what_is_visible_it": "Description in ITALIAN of the visual structure (e.g., 'Grafico a cascata che mostra il ponte tra Ricavi 2022 e 2023').",
  
  "analysis_it": "A professional 3-sentence financial summary in ITALIAN. Focus on: Growth rates (CAGR/YoY), Margin expansion/contraction, and Volatility. Use financial terminology (bullish, bearish, flat, spike).",
  
  "data_table_md": "| Period | Value (Unit) |\n|---|---|\n| FY23 | € 14.5M |\n| FY24(e) | € 16.2M |",

  "observations_it": [
    "Fact 1: Identify the peak and trough values.",
    "Fact 2: Note any 'CAGR' or '%' labels visible.",
    "Fact 3: Mention if data is 'Pro-forma' or 'GAAP' if labeled."
  ],
  
  "visual_trends_it": ["Describe the slope (steep increase, plateau, decline). Mention specific colors if they indicate risk (red) or profit (green/black)."],

  "legend_it": {
    "is_readable": true,
    "mapping": [{"label": "Net Profit", "color_or_style": "Blue Bar"}, {"label": "Margin %", "color_or_style": "Orange Line"}]
  },
  
  "numbers": [
    {
      "label": "Entity/Category (e.g., 'Revenue', 'Q3')",
      "value": "Exact value read (e.g., '1,234')",
      "unit": "Currency/Scale (e.g., 'EUR Million', '%', 'bps')",
      "period": "Time ref (e.g., '2024E')"
    }
  ],
  "confidence": 0.0
}
"""

# ==============================================================================
# PROMPT: VISION-FIRST (PAGE-TO-MARKDOWN)
# ==============================================================================
# Questo prompt istruisce Ministral a trascrivere tutto in un unico flusso Markdown.
# È cruciale per catturare grafici e tabelle nel contesto del testo.
# ==============================================================================
# PROMPT VISIONE (Vision-First v2 - CHART HUNTER)
# ==============================================================================
VISION_FIRST_PROMPT = r"""
You are a Financial Document Analyst with Computer Vision capabilities.
Your goal is to transcribe text AND deeply analyze any visual chart.

PAGE ANALYSIS PROTOCOL:

1. **SCAN FOR CHARTS FIRST**: Look immediately for Trading Charts, Time Series, MACD/RSI indicators, or Candlesticks.
   - If found, you MUST insert a detailed description block using this EXACT format:
   
   > **### 🖼️ VISUAL ANALYSIS: [Title/Type of Chart]**
   > *Visual Elements:* Describe the lines (colors, trends), bars, or markers.
   > *Data Insights:* Describe the X/Y axes values, peaks, bottoms, and crossovers (e.g., "Price crosses moving average").
   > *Context:* Relate the chart to the surrounding text labels (e.g., "Fig 2.1").

2. **TEXT TRANSCRIPTION**: After looking for visuals, transcribe all text headers and paragraphs exactly.
3. **TABLES**: Transcribe tables using Markdown pipes (|).
4. **FORMULAS**: Use Unicode symbols.

STRICT RULES:
- **DO NOT IGNORE IMAGES**: Even if they contain text labels, treat them as visual data to describe.
- **Merge** the visual description naturally into the reading order where the image appears.
- Output ONLY Markdown.
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
You are a High-Fidelity Knowledge Engineer.
Extract a DETAILED Knowledge Graph.

RULES:
- Extract AS MANY entities as possible (Concepts, Metrics, Instruments, Regulations).
- Extract ALL relationships defined in the text.
- DO NOT summarize. Be exhaustive.
- If a sentence contains a causal link, extract it.

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


CHART_DATA_PROMPT = """
You are a Lead Data Scientist specializing in Financial Chart Reconstruction.
Your goal is to extract the EXACT underlying data table from the chart image.

### PHASE 1: STRUCTURAL ANCHORING (CRITICAL)
1. **Identify the MAIN CATEGORIES (X-Axis)**:
   - Look at the labels *under* the groups of bars.
   - Examples: "China", "USA", "EU", "Product A". 
   - *Constraint*: These are the Row Headers.
2. **Identify the SERIES LEGEND (Sub-groups)**:
   - Look for the text that distinguishes the bars *within* a single category.
   - *Visual Cue*: Are there dates (e.g., "2020", "2024") written below/inside the bars?
   - *Visual Cue*: Is there a color legend (e.g., "Blue=Revenue, Red=Cost")?
   - *Constraint*: These are the Column Headers.
3. **Determine Cluster Size (N)**:
   - Count how many bars exist for the first category (e.g., China).
   - If China has 2 bars, N=2. You MUST extract exactly 2 values for every other category.

### PHASE 2: PRECISION EXTRACTION
For EACH Main Category found:
1. **Locate**: Focus on the cluster of bars for that category.
2. **Measure**: Trace the top of each bar to the Y-Axis value. 
   - *Interpolate*: If a bar is between 2.0 and 4.0, it is likely 3.0.
   - *Ordering*: Extract values in the logical order of the Series (e.g., 2020 value then 2024 value).
3. **Values**: Return specific numbers. DO NOT return arrays of random numbers.

### PHASE 3: OUTPUT JSON
Return VALID JSON:
{
  "title": "Chart Title",
  "chart_type": "Clustered Bar / Stacked Bar / Line",
  "series_discriminators": "The exact labels for the series (e.g., '2020, 2024'). NOT the country names.",
  "data_points": [
    {
      "category": "Main Axis Label (e.g. China)",
      "visual_check": "Short description (e.g. 'Red bar (low), Red bar (high)')", 
      "value": "val1, val2" 
    }
  ]
}
"""

# ==============================================================================
# NUOVO PROMPT: FULL PAGE TO MARKDOWN (LaTeX Native)
# ==============================================================================
MARKER_VISION_PROMPT = """
You are an advanced AI conversion engine (OCR + Layout Analysis).
Your task: Convert this document image into clean, structured MARKDOWN.

RULES FOR MATHEMATICS (CRITICAL):
1. Identify ALL mathematical formulas, equations, and symbols.
2. Transcribe them EXACTLY into LaTeX format enclosed in single dollars ($...$) for inline or double dollars ($$...$$) for block equations.
3. Example: Convert "WACC = Ve/V * ke" into "$$WACC = \\frac{V_e}{V} k_e$$".
4. Do NOT output ascii math (like x^2). ALWAYS use LaTeX ($x^2$).

RULES FOR STRUCTURE:
1. Preserve headers (###), lists, and tables (using Markdown | col | col |).
2. Ignore page footers, page numbers, and copyright disclaimers.
3. If the text is garbled in the image, infer the correct words based on context.

OUTPUT ONLY THE MARKDOWN. NO CONVERSATIONAL FILLER.
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
    if not image_bytes or len(image_bytes) < MIN_ASSET_SIZE:
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


def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set)):
        parts = [to_text(i) for i in x]
        parts = [p.strip() for p in parts if p and str(p).strip()]
        return "; ".join(parts)
    if isinstance(x, dict):
        return str(x)
    return str(x)



def set_toon_type(chunk: dict, *, is_image: bool) -> dict:
    chunk["toon_type"] = "imagine" if is_image else "testo"
    return chunk


def fast_chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """
    Divide il testo in chunk basati su una stima dei token (1 tok ~= 4 char).
    - Veloce (nessun tokenizer pesante).
    - Rispetta i confini delle parole/frasi.
    - Include un overlap automatico per continuità semantica.
    """
    if not text:
        return []
    
    # Stima caratteri: OpenAI usa circa 4 char per token in media
    chunk_size_char = max_tokens * 4
    overlap_char = int(chunk_size_char * 0.1)  # 10% overlap
    
    text_len = len(text)
    if text_len <= chunk_size_char:
        return [text]
        
    chunks = []
    start = 0
    
    while start < text_len:
        # Definiamo il punto di fine teorico
        end = min(start + chunk_size_char, text_len)
        
        # Se non siamo alla fine assoluta del testo, cerchiamo un punto di taglio "morbido"
        if end < text_len:
            # Cerchiamo l'ultimo 'a capo' o 'spazio' prima del limite
            # Questo evita di troncare parole a metà es: "ingegner" | "ia"
            search_window = text[start:end]
            
            # Preferenza 1: Tagliare su un doppio a capo (fine paragrafo)
            last_break = search_window.rfind('\n\n')
            
            # Preferenza 2: Tagliare su un singolo a capo
            if last_break == -1:
                last_break = search_window.rfind('\n')
                
            # Preferenza 3: Tagliare su uno spazio (ultima risorsa)
            if last_break == -1:
                last_break = search_window.rfind(' ')
            
            # Se abbiamo trovato un punto valido nella seconda metà del chunk, usiamolo
            # (Se è troppo presto, meglio tagliare brutale che fare un chunk minuscolo)
            if last_break != -1 and last_break > (chunk_size_char * 0.5):
                end = start + last_break + 1  # +1 per includere il carattere di stop
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Calcolo del prossimo start con overlap
        # Se siamo alla fine, usciamo
        if end >= text_len:
            break
            
        # Torniamo indietro di 'overlap' caratteri, ma senza superare l'inizio attuale
        start = max(start + 1, end - overlap_char)

    return chunks


def prep_text_for_embedding(s: str, max_chars: int = 2200) -> str:
    if not s:
        return ""
    # rimuove prefix tipo "Doc: ... | Sezione: ..."
    s = re.sub(r"^Doc:\s.*?\n", "", s.strip(), flags=re.DOTALL)
    # normalizza whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # tronca (tokenizer cost cresce con la lunghezza)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def page_has_vector_graphics(page: fitz.Page) -> bool:
    """
    Heuristica: se la pagina contiene molti 'drawings' (linee/rect/path),
    è molto probabile che ci sia un grafico vettoriale.
    """
    try:
        drawings = page.get_drawings()
        if not drawings:
            return False
        # Conteggio totale items (path ops)
        ops = 0
        for d in drawings:
            items = d.get("items") or []
            ops += len(items)
        # Soglia: alza/abbassa se necessario (20-60 tipico)
        return ops >= 20
    except Exception:
        return False



def extract_markdown_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Estrattore ad alte prestazioni per file Markdown.
    Gestisce l'auto-cleaning e la Vision AI chirurgica.
    FIX: Include sub-chunking per paragrafi giganti (evita errori 0 nodi).
    """
    out_chunks = []
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # 1. AUTO-CLEANING
        content = clean_markdown_structure(raw_content)

        # 2. CHUNKING STRUTTURATO (Macro-divisione)
        raw_paras = re.split(r'\n(?=# )|\n\n', content)
        
        for idx, macro_para in enumerate(raw_paras):
            macro_para = macro_para.strip()
            if len(macro_para) < MIN_CHUNK_LEN:
                continue
            
            # ### FIX 1: Gestione Vision AI (spostata prima del sub-chunking)
            # Analizziamo l'immagine una volta sola per il "macro blocco"
            vision_metadata = {}
            img_matches = re.findall(r'!\[.*?\]\((.*?)\)', macro_para)
            
            if img_matches and PDF_VISION_ENABLED:
                # ... (TUA LOGICA VISION INVARIATA) ...
                try:
                    img_path = img_matches[0]
                    full_img_path = os.path.join(os.path.dirname(file_path), img_path)
                    
                    if os.path.exists(full_img_path) and os.path.getsize(full_img_path) > 5000:
                        with open(full_img_path, "rb") as img_f:
                            img_bytes = img_f.read()
                        
                        CHART_MIN_CONF = float(os.getenv("CHART_MIN_CONF", "0.55"))
                        c_js = extract_chart_via_vision(img_bytes)
                        
                        conf = float((c_js or {}).get("confidence") or 0.0)

                        if c_js and c_js.get("kind") != "other" and conf >= CHART_MIN_CONF:
                            # Salviamo i dati per iniettarli nel primo sub-chunk
                            vision_metadata = {
                                "chart_semantic": build_chart_semantic_chunk(1, c_js),
                                "metadata_override": c_js
                            }
                            print(f"   📝 Analisi Semantica Chart (conf={conf:.2f})")
                        else:
                            # Salvataggio Postgres (Asset Management)
                            conn = pg_get_conn()
                            try:
                                with conn.cursor() as cur:
                                    pg_save_image(log_id, img_bytes, "image/jpeg", f"MD_{filename}_{idx}", cur)
                                conn.commit()
                            finally:
                                pg_put_conn(conn)
                except Exception as e_img:
                    print(f"   ⚠️ Vision Error: {e_img}")


            # ### FIX 2: SUB-CHUNKING DI SICUREZZA
            # Se il paragrafo è > 1024 token, lo spezziamo ancora, altrimenti l'LLM fallisce.
            # Se è piccolo, fast_chunk_text ritorna una lista con 1 solo elemento (invariato).
            sub_chunks_text = fast_chunk_text(macro_para, max_tokens=1024)

            for sub_i, txt in enumerate(sub_chunks_text):
                
                # Se c'era un grafico, lo alleghiamo SOLO al primo pezzo del paragrafo
                # per evitare di duplicare l'informazione vision in N chunk.
                current_text_sem = f"Doc: {filename} | Sezione: {txt[:60]}...\n{txt}"
                current_meta = {}
                
                if sub_i == 0 and vision_metadata:
                    # Iniettiamo la descrizione del grafico all'inizio del testo semantico
                    current_text_sem = vision_metadata["chart_semantic"] + "\n\n" + current_text_sem
                    current_meta = vision_metadata.get("metadata_override", {})

                chunk_data = {
                    "text_raw": txt,
                    "text_sem": current_text_sem,
                    "page_no": 1,
                    "toon_type": "text" if not (sub_i == 0 and vision_metadata) else "chart_analysis",
                    "section_hint": txt[:80] if txt.startswith("#") else f"part_{sub_i+1}",
                    "metadata": current_meta # Importante per passare info Chart
                }
                
                # Se c'era un metadata override (es. grafico), assicurati che sia nel payload
                if "metadata_override" in vision_metadata and sub_i == 0:
                     chunk_data["metadata_override"] = vision_metadata["metadata_override"]

                out_chunks.append(chunk_data)
            
        print(f"   📄 Markdown Ingested: {len(out_chunks)} chunk validi.")
        return out_chunks

    except Exception as e:
        print(f"   ❌ Errore durante l'estrazione Markdown: {e}")
        import traceback
        traceback.print_exc()
        return []

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def force_unload_ollama(model_name: str):
    """
    Forza lo scaricamento e attende che la VRAM si stabilizzi.
    Versione Aggressiva per P5000.
    """
    if not model_name:
        return
    try:
        # print(f"   🧹 Unloading {model_name}...", end="\r")
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": "",
            "keep_alive": 0 
        }
        requests.post(url, json=payload, timeout=2)
        
        # --- MODIFICA FONDAMENTALE ---
        # La P5000 ha bisogno di tempo per de-allocare la memoria CUDA
        # 0.5s non bastano. Facciamo 3 secondi. È lento? Sì. Si blocca? No.
        time.sleep(3.0) 
        
    except Exception:
        pass


def force_restart_ollama(num_parallel: str = "1") -> bool:
    """
    Riavvio Ottimizzato per GPU 16GB (P5000).
    Forza OLLAMA_NUM_PARALLEL=1 per bilanciare Vision e Chat senza OOM.
    """
    print(f"🔄 Resetting Ollama Server (Target Parallelism={num_parallel})...")

    # 1) Kill processi esistenti (Pulizia VRAM)
    try:
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/f", "/im", "ollama_app.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3) # Un secondo in più per essere sicuri che la VRAM sia libera
    except Exception:
        pass

    # 2) Configura l'ambiente P5000 Friendly
    env = os.environ.copy()
    
    # Su 16GB, UNO alla volta è meglio. Massimizza la VRAM per il contesto lungo.
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    
    # Teniamo in memoria max 2 modelli (Vision e Brain) per evitare continui reload
    env["OLLAMA_MAX_LOADED_MODELS"] = "2"
    
    # Opzionale: Flash Attention se supportato (spesso aiuta su Pascal/Volta/Ampere)
    env["OLLAMA_FLASH_ATTENTION"] = "1" 

    # 3) Percorso Ollama
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        ollama_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")

    print(f"   🚀 Starting Ollama from: {ollama_path} with P5000 optimizations...")

    # 4) Avvio server
    try:
        subprocess.Popen(
            [ollama_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            shell=False
        )
    except Exception as e:
        print(f"   ❌ Errore critico avvio Ollama: {e}")
        return False

    # 5) Healthcheck
    for i in range(20): # Aumentato timeout a 20 per dare tempo al caricamento VRAM
        try:
            res = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            if res.status_code == 200:
                print(f"   ✨ Ollama is READY (Parallel={num_parallel})")
                return True
        except:
            time.sleep(1)
            
    print("   ⚠️ Ollama non ha risposto entro il timeout, ma potrebbe essere attivo.")
    return True


    # 2. Configura l'ambiente
    env = os.environ.copy()
    num_parallel=4
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
# 5. Verifica disponibilità
    for i in range(15):
        try:
            # FIX: Usiamo /api/tags che accetta GET e risponde 200 OK
            res = requests.get("http://localhost:11434/api/tags", timeout=1)
            if res.status_code == 200:
                print(f"   ✨ Ollama is READY (Parallel={num_parallel})")
                return True
        except Exception:
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
from typing import Any, Dict, Optional, Tuple, Union


def safe_json_extract(raw: str):
    """
    Estrae in modo robusto il primo JSON valido (dict/list) da una risposta LLM.
    Gestisce:
      - code fences ```json ... ```
      - testo prima/dopo il JSON
      - caratteri di controllo / null bytes
      - smart quotes
      - trailing commas
    Ritorna: dict | list | None
    """
    import json, re

    if raw is None:
        return None

    s = str(raw)

    # 1) strip code fences
    #    prende il contenuto interno se la risposta è tipo ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if fence:
        s = fence.group(1)

    # 2) rimuove caratteri di controllo (salva \t \n \r)
    s = "".join(ch for ch in s if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 32)

    # 3) normalizza smart quotes
    s = (s.replace("“", '"').replace("”", '"')
           .replace("‘", "'").replace("’", "'"))

    # helper: trova primo JSON bilanciato {..} o [..]
    def _first_balanced_json(text: str):
        starts = []
        for i, ch in enumerate(text):
            if ch in "{[":
                starts.append(i)

        for start in starts:
            open_ch = text[start]
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            in_str = False
            esc = False

            for j in range(start, len(text)):
                c = text[j]

                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                else:
                    if c == '"':
                        in_str = True
                        continue
                    if c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            return text[start:j+1]
        return None

    cand = _first_balanced_json(s)
    if not cand:
        return None

    # 4) prova parse diretto
    try:
        return json.loads(cand)
    except Exception:
        pass

    # 5) micro-repair: trailing commas + spazi strani
    repaired = cand
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
    repaired = repaired.strip()

    # 6) riprova
    try:
        return json.loads(repaired)
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
            # AGGIUNTO format='json' per evitare testo extra
            format='json', 
            options={"temperature": 0.1, "num_predict": 600, "num_ctx": 4096},
        )
        return (resp.get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        print(f"   ⚠️ Chart Analysis Error: {e}")
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
    Trigger bilanciato: abbassiamo la soglia di attivazione per 
    distribuire i nodi su più chunk.
    """
    if not text or len(text) < 300: # Soglia minima più bassa
        return False
    
    clean_text = safe_normalize_text(text)
    
    # Cerchiamo almeno 1 nome proprio E 1 keyword finanziaria 
    # oppure almeno 3 keyword finanziarie/tecniche totali
    proper_nouns = set(_ENTITY_PROPER_NOUNS.findall(clean_text))
    finance_keywords = set(_KG_PAT.findall(clean_text))

    if (len(proper_nouns) >= 1 and len(finance_keywords) >= 1) or len(finance_keywords) >= 3:
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
    return nodes[80], edges[:100]
    #return nodes[:20], edges[:30]


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


# ==============================================================================
# API CALLER (OLLAMA NATIVE - MINISTRAL OPTIMIZED)
# ==============================================================================
# Assicurati di avere in cima al file: from ollama import chat, ChatResponse

# ==============================================================================
# API CALLER (OLLAMA NATIVE LIBRARY)
# ==============================================================================
# Assicurati di avere l'import in alto: from ollama import chat, ChatResponse

# ---- Vision call lock: serializza SOLO la chiamata LLM (evita deadlock su Ollama Vision) ----
# ---- Vision call lock: evita deadlock Ollama Vision ----
_VISION_CALL_LOCK = Lock()

OLLAMA_API_GENERATE = os.getenv(
    "OLLAMA_API_GENERATE",
    "http://127.0.0.1:11434/api/generate"
)

OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "600")) #180
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))


def llm_chat_multimodal(
    prompt: str,
    image_bytes: bytes,
    model: str,
    max_tokens: int = 4000,
    num_ctx: int = 4096,
    response_format_json: bool = False,  # ✅ usa format="json"
    force_json: Optional[bool] = None,   # ✅ ALIAS retro-compat (fix definitivo)
) -> str:
    """
    Vision via Ollama /api/generate (robusto).
    - max_tokens -> options.num_predict
    - timeout + retry
    - lock per evitare deadlock con Vision in parallelo
    - response_format_json: usa format='json' per forzare JSON valido
    - force_json: alias compatibile con chiamate già presenti nel codice
    """
    if not image_bytes:
        return ""

    # ✅ alias: se qualcuno chiama force_json=True, lo mappiamo su response_format_json
    if force_json is not None:
        response_format_json = bool(force_json)

    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f"   ❌ Base64 encode error: {e}")
        return ""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "options": {
            "temperature": 0.0,
            "num_ctx": int(num_ctx),
            "num_predict": int(max_tokens) if max_tokens is not None else 4000,
        },
        "stream": False,
    }

    # ✅ forza JSON lato Ollama (riduce drasticamente parse-fail)
    if response_format_json:
        payload["format"] = "json"

    with _VISION_CALL_LOCK:
        last_err = None
        for _ in range(OLLAMA_RETRIES + 1):
            try:
                r = requests.post(
                    OLLAMA_API_GENERATE,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT_S,
                )
                r.raise_for_status()
                data = r.json() or {}
                return data.get("response", "") or ""
            except Exception as e:
                last_err = e
                time.sleep(0.5)

        print(f"   ⚠️ Vision generate failed (model={model}): {last_err}")
        return ""


def _downscale_and_compress_for_vision(
    img_bytes: bytes,
    max_side: int = 1800,
    jpeg_quality: int = 92,
    output_format: str = "PNG",   # ✅ CHART: PNG = testo più nitido
) -> bytes:
    """
    Prepara immagini per Vision mantenendo il testo leggibile.
    - PNG consigliato per grafici (etichette piccole).
    - JPEG ok per foto, ma può "impastare" le scritte.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        buf = io.BytesIO()
        fmt = (output_format or "PNG").upper().strip()

        if fmt == "JPEG" or fmt == "JPG":
            img.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        else:
            # ✅ PNG: preserva edge e testo
            img.save(buf, format="PNG", optimize=True)

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

def render_full_page_png(page_obj, dpi: int = 200) -> bytes:
    """
    Render pagina PDF in PNG (migliore per formule e testo fine).
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page_obj.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


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


def extract_formulas_vision(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Estrazione Formule V3: Supersampling + PNG per pedici nitidi.
    """
    if not img_bytes:
        return None

    # 1. UPSCALING AGGRESSIVO (Fondamentale per i vettoriali renderizzati)
    # 2400px garantisce che un pedice 'i' sia leggibile.
    # PNG è obbligatorio: il JPEG 'sbaverebbe' le linee sottili delle frazioni.
    vbytes = _downscale_and_compress_for_vision(
        img_bytes, 
        max_side=2400,    # <--- AUMENTATO DA 1600
        output_format="PNG", 
        jpeg_quality=100
    )
    
    key = sha256_hex(vbytes) + "::formula_latex_v3_highres"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    try:
        # Usa il nuovo prompt LaTeX-centrico
        raw = llm_chat_multimodal(
            FORMULA_VISION_PROMPT, 
            vbytes, 
            VISION_MODEL_NAME, 
            max_tokens=1500, # Più token per formule lunghe
            response_format_json=True
        )
        
        js = safe_json_extract(raw)
        if not js or ("formulas" not in js and "summary_it" not in js):
            return None
            
        _vision_cache_put(key, js)
        return js
    except Exception as e:
        print(f"   ⚠️ Formula Vision Error: {e}")
        return None


def normalize_chart_json_for_semantics(js: Dict[str, Any], page_no: int, context_hint: str = "") -> Dict[str, Any]:
    """
    Normalizza output Vision eterogeneo (V8 - Robustezza Totale).
    Gestisce varianti di schema (value/values, category/labels) tipiche di Ministral/Qwen.
    """
    if not isinstance(js, dict):
        return {}

    out = dict(js)  # shallow copy

    # ----------------------------
    # Page Alignment
    # ----------------------------
    out["page_no"] = page_no

    # ----------------------------
    # Kind / Toon Type Inference
    # ----------------------------
    if not out.get("kind"):
        # Se ci sono dati strutturati e assi, è probabilmente un grafico a barre
        if out.get("data_points") and (out.get("x-axis_labels") or out.get("x_axis_labels")):
            out["kind"] = "bar_chart"
        else:
            out["kind"] = out.get("toon_type") or "immagine"

    # Uniformiamo toon_type per il downstream (Neo4j/Postgres)
    out["toon_type"] = "immagine"

    # ----------------------------
    # Timeframe Auto-Detection
    # ----------------------------
    tf = out.get("timeframe")
    if not tf or str(tf).strip() in ("", "NOT READABLE", "None"):
        # Se timeframe è vuoto, cerca anni nelle etichette dell'asse X
        xlab = out.get("x-axis_labels") or out.get("x_axis_labels") or out.get("xAxis") or []
        if isinstance(xlab, (list, tuple)):
            years = []
            for v in xlab:
                years += re.findall(r"\b(19\d{2}|20\d{2})\b", str(v))
            years = list(dict.fromkeys(years))  # unique preserving order
            if len(years) >= 2:
                out["timeframe"] = f"{years[0]} vs {years[-1]}"
            elif len(years) == 1:
                out["timeframe"] = years[0]

    # ----------------------------
    # Confidence Score Calculation
    # ----------------------------
    conf = out.get("confidence", None)
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None

    if conf is None:
        # Euristica: calcola un punteggio basato sulla completezza dei dati
        score = 0.25
        if out.get("title"): score += 0.15
        if out.get("source"): score += 0.10
        if out.get("unit_of_measure"): score += 0.05

        dps = out.get("data_points") or out.get("numbers") or []
        if isinstance(dps, list):
            if len(dps) >= 2: score += 0.25
            if len(dps) >= 4: score += 0.10

        tf2 = out.get("timeframe") or ""
        if re.findall(r"\b(19\d{2}|20\d{2})\b", str(tf2)):
            score += 0.10

        # Clamp tra 0.0 e 0.95
        conf = max(0.0, min(0.95, score))

    out["confidence"] = conf

    # ----------------------------
    # Text Fields Defaults
    # ----------------------------
    if not out.get("what_is_visible_it"):
        title = str(out.get("title") or "").strip()
        uom = str(out.get("unit_of_measure") or "").strip()
        out["what_is_visible_it"] = (
            f"Grafico/immagine informativa. Titolo: {title}."
            + (f" Unità di misura: {uom}." if uom else "")
        ).strip()

    if not out.get("observations_it"):
        # Se c'è una descrizione in inglese, la usiamo come osservazione
        vtd = out.get("visual_trend_description")
        if vtd:
            out["observations_it"] = [str(vtd)]
        else:
            out["observations_it"] = []

    if not out.get("analysis_it"):
        out["analysis_it"] = ""

    # ----------------------------
    # Numbers Normalization (Legacy Sync + Robustness)
    # ----------------------------
    # Questa sezione popola 'numbers' (usato da parti legacy) usando 'data_points'
    if not out.get("numbers"):
        numbers = []
        dps = out.get("data_points") or []
        if isinstance(dps, list):
            for dp in dps[:12]: # Limitiamo a 12 per evitare payload enormi
                if isinstance(dp, dict):
                    
                    # --- FIX ROBUSTEZZA (Value/Values + Label/Category) ---
                    # 1. Cattura valori singolari o plurali
                    v_raw = dp.get("value") or dp.get("values")
                    
                    # 2. Cattura etichette in qualsiasi formato allucinato dal modello
                    l_raw = (dp.get("category") or 
                             dp.get("categories") or 
                             dp.get("label") or 
                             dp.get("labels") or 
                             "")
                    # ------------------------------------------------------

                    numbers.append({
                        "label": l_raw,
                        "value": v_raw,
                        "unit": out.get("unit_of_measure") or "",
                        "period": out.get("timeframe") or ""
                    })
        out["numbers"] = numbers

    return out


def extract_chart_via_vision(img_bytes: bytes, context_hint: str = "") -> Optional[Dict[str, Any]]:
    """
    FIX 3 (definitivo):
    - prepara immagine in PNG (testo più nitido)
    - OCR su immagine migliore
    - 2-pass retry se data_points sono pochi / NOT_READABLE
    - sceglie automaticamente l'output "migliore" (più categorie, meno NOT_READABLE)
    """
    if not img_bytes:
        return None

    def _score_chart(js: Dict[str, Any]) -> int:
        dps = js.get("data_points", [])
        if not isinstance(dps, list):
            return -9999
        n = 0
        bad = 0
        for dp in dps:
            if not isinstance(dp, dict):
                continue
            cat = str(dp.get("category", "") or "")
            val = str(dp.get("value", "") or "")
            if cat.strip():
                n += 1
            if "NOT_READABLE" in cat.upper() or "NOT_READABLE" in val.upper():
                bad += 1
        # più categorie = meglio; meno NOT_READABLE = meglio
        return (n * 10) - (bad * 25)

    # PAGE: prova a leggerla da context_hint tipo "... Page 2 ..."
    page_no = 0
    if context_hint:
        m = re.search(r"\bpage\s+(\d+)\b", context_hint, re.I)
        if m:
            try:
                page_no = int(m.group(1))
            except Exception:
                page_no = 0

    # PASS 1 (PNG nitido)
    vbytes1 = _downscale_and_compress_for_vision(img_bytes, max_side=1900, output_format="PNG")
    key1 = sha256_hex(vbytes1) + "::chart_grounded_fix3_p1"
    cached = _vision_cache_get(key1)
    if cached:
        return cached

    ocr1 = ocr_extract_text(vbytes1)

    prompt1 = CHART_DATA_PROMPT
    if context_hint:
        prompt1 += f"\n\nCONTEXT HINT: {context_hint[:600]}\n"
    if ocr1:
        prompt1 += f"\nVERIFIED OCR TEXT (Use as evidence): \"\"\"{ocr1[:1600]}\"\"\""

    raw1 = llm_chat_multimodal(
        prompt1, vbytes1, VISION_MODEL_NAME, max_tokens=2000, response_format_json=True
    )
    js1 = safe_json_extract(raw1)
    if not isinstance(js1, dict):
        js1 = {}

    # Condizione retry: poche categorie o NOT_READABLE presenti
    need_retry = True
    if isinstance(js1.get("data_points", None), list):
        dps1 = js1.get("data_points", [])
        n1 = sum(1 for x in dps1 if isinstance(x, dict) and str(x.get("category", "")).strip())
        has_bad = any(
            "NOT_READABLE" in str(x.get("category", "")).upper() or "NOT_READABLE" in str(x.get("value", "")).upper()
            for x in dps1 if isinstance(x, dict)
        )
        # se ho almeno 4 categorie leggibili e niente NOT_READABLE, ok
        need_retry = (n1 < 4) or has_bad

    # PASS 2 (più grande, istruzioni più “hard”)
    js2 = {}
    if need_retry:
        vbytes2 = _downscale_and_compress_for_vision(img_bytes, max_side=2600, output_format="PNG")
        ocr2 = ocr_extract_text(vbytes2)

        prompt2 = CHART_DATA_PROMPT + """
            SECOND PASS (IMPORTANT):
            - You MUST list ALL categories visible (legend + bars/lines labels), even if some values are only estimates.
            - If a category label is partially readable, return the best possible label instead of NOT_READABLE.
            - Prefer OCR evidence for labels (countries/regions) and units.
            """
        if context_hint:
            prompt2 += f"\n\nCONTEXT HINT: {context_hint[:600]}\n"
        if ocr2:
            prompt2 += f"\nVERIFIED OCR TEXT (Use as evidence): \"\"\"{ocr2[:2200]}\"\"\""

        raw2 = llm_chat_multimodal(
            prompt2, vbytes2, VISION_MODEL_NAME, max_tokens=2400, response_format_json=True
        )
        js2 = safe_json_extract(raw2)
        if not isinstance(js2, dict):
            js2 = {}

    # scegli output migliore
    best = js1 if _score_chart(js1) >= _score_chart(js2) else js2
    if not isinstance(best, dict) or not best:
        return None

    # --- CLEANING (anni / geo) ---
    tf = str(best.get("timeframe", "") or "")
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", tf)
    if years:
        best["timeframe"] = " vs ".join(sorted(set(years)))

    bad_geo = [" sud", "south", " nord", "north"]
    for dp in best.get("data_points", []) if isinstance(best.get("data_points", []), list) else []:
        if not isinstance(dp, dict):
            continue
        cat = str(dp.get("category", "") or "")
        for token in bad_geo:
            if token in cat.lower():
                dp["category"] = cat.lower().replace(token, "").capitalize().strip()

    js_norm = normalize_chart_json_for_semantics(best, page_no=page_no, context_hint=context_hint)
    js_norm["semantic_description"] = build_chart_semantic_chunk(page_no, js_norm)
    js_norm["toon_type"] = "immagine"

    _vision_cache_put(key1, js_norm)
    return js_norm




def to_text(x) -> str:
    """Converte in stringa sicura (None, list annidate, dict)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set)):
        parts = [to_text(i) for i in x]
        parts = [p.strip() for p in parts if p and str(p).strip()]
        return "; ".join(parts)
    if isinstance(x, dict):
        # qui NON forziamo json.dumps per evitare output enorme, ma è ok usare str
        return str(x)
    return str(x)


def build_chart_semantic_chunk(page_no: int, chart_json: Dict[str, Any], prefix: str = "VISUAL") -> str:
    """
    Versione V8: Ministral-Ready.
    Accetta 'value' (singolare) e 'values' (plurale).
    """
    page_human = (page_no + 1) if isinstance(page_no, int) else page_no

    if not isinstance(chart_json, dict):
        return normalize_ws(f"--- ANALISI VISUALE - Pagina {page_human} ---\n[Dati non validi]")

    title = to_text(chart_json.get("title", ""))
    ctype = to_text(chart_json.get("chart_type", "Grafico"))
    discriminators = to_text(chart_json.get("series_discriminators") or chart_json.get("series_legend", ""))
    
    lines = [f"--- ANALISI VISUALE ({ctype}) - Pagina {page_human} ---"]
    if title: lines.append(f"Titolo: {title}")
    if discriminators: lines.append(f"Legenda Serie: {discriminators}")

    datap = chart_json.get("data_points") or []
    if datap:
        lines.append("\nDati Estratti:")
        if not isinstance(datap, list): datap = [datap]
        
        for d in datap[:40]:
            if isinstance(d, dict):
                cat = to_text(d.get("category", ""))
                
                # --- FIX CRITICO PER MINISTRAL (Value vs Values) ---
                # Cerca prima 'value', se vuoto cerca 'values', se vuoto stringa vuota
                raw_val = d.get("value") or d.get("values") or ""
                
                if isinstance(raw_val, list):
                    # Unisce lista [0.8, 5.5] -> "0.8, 5.5"
                    val = ", ".join([str(v) for v in raw_val if v is not None])
                else:
                    # Pulisce stringa
                    val = to_text(raw_val).replace("[", "").replace("]", "").replace("'", "")
                # ---------------------------------------------------

                vis_check = to_text(d.get("visual_check", ""))
                check_str = f" ({vis_check})" if (vis_check and len(vis_check) < 80) else ""

                if cat or val:
                    lines.append(f" - {cat}: {val}{check_str}")

    return normalize_ws("\n".join(lines))


# =========================
# Chunk builders
# =========================
def build_formula_semantic_chunk(page_no: int, formulas_json: Dict[str, Any]) -> str:
    """
    Crea un chunk ottimizzato per il RAG contenente spiegazioni e LaTeX puro.
    FIX ROBUSTEZZA: Gestisce casi in cui 'latex' è una lista o dict invece di str.
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
        
        # --- FIX QUI: Normalizzazione forzata a stringa ---
        raw_latex = f.get("latex", "")
        if isinstance(raw_latex, list):
            # Se è una lista, uniamo gli elementi
            latex_str = " ".join([str(x) for x in raw_latex])
        elif isinstance(raw_latex, dict):
            latex_str = str(raw_latex)
        else:
            latex_str = str(raw_latex)
            
        # Ora possiamo fare replace sicuro
        latex = latex_str.replace("\\\\", "\\") 
        # --------------------------------------------------
        
        vars_list = []
        # Fix anche per variables se il modello impazzisce
        raw_vars = f.get("variables", [])
        if isinstance(raw_vars, list):
            for v in raw_vars:
                if isinstance(v, dict):
                    vars_list.append(f"{v.get('name')}: {v.get('meaning')}")
                elif isinstance(v, str):
                    vars_list.append(v)
        
        vars_str = "; ".join(vars_list)
        
        # Blocco semantico: Concetto + LaTeX + Variabili
        block = f"## {desc}\nModello (LaTeX): $${latex}$$\nVariabili: {vars_str}"
        lines.append(block)

    return "\n".join(lines).strip()


# =========================
# KG extraction (LLM) - ROBUST DEBUG & RETRY
# =========================
def llm_extract_kg(filename: str, page_no, text: str, model_name: str):
    """
    Estrazione KG Robusta (V3.2 - Anti-Hang & Type Safe):
    - Tentativo 1: JSON Mode Strict.
    - Tentativo 2: Fallback su RAW Mode con contesto ridotto (evita freeze su bibliografie).
    - Gestisce risposte JSON che sono liste invece di dict.
    """
    base = os.path.basename(str(filename))
    
    # Se il testo è troppo breve, inutile chiamare l'LLM
    if not text or len(text) < 50: 
        return [], []

    # ------------------------------------------------------------
    # TENTATIVO 1: JSON MODE (Strict & High Quality)
    # ------------------------------------------------------------
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": KG_PROMPT},
                {"role": "user", "content": f"Extract entities and relations from this text:\n\n{text}"}
            ],
            format="json", 
            options={
                "temperature": 0.0, 
                "num_predict": 3000, # Abbondante per evitare troncamenti
                "num_ctx": 4096      # Contesto pieno
            }
        )
        
        raw_content = resp.get("message", {}).get("content", "").strip()
        if not raw_content:
            return [], []

        # Parsing JSON
        data = json.loads(raw_content)
        
        # --- FIX ROBUSTEZZA TIPO (List vs Dict) ---
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # Caso: il modello ha ritornato [{"nodes": ...}]
                data = data[0]
            else:
                # Caso: lista piatta o incomprensibile, invalida per noi
                data = {} 
        # ------------------------------------------

        nodes = data.get("nodes", data.get("entities", []))
        edges = data.get("edges", data.get("relationships", []))
        
        # Se abbiamo trovato qualcosa, ritorniamo subito
        if nodes or edges:
            return nodes, edges

    except json.JSONDecodeError:
        print(f"   ⚠️ [KG-RETRY] {base} p{page_no}: JSON incompleto/troncato. Riprovo in RAW mode...")
        # Non ritorniamo, lasciamo scorrere verso il Tentativo 2
    except Exception as e:
        print(f"   ⚠️ [KG-ERR] {base} p{page_no}: {e}")
        return [], []
    
    # ------------------------------------------------------------
    # TENTATIVO 2: RAW MODE (Fallback / Anti-Hang)
    # ------------------------------------------------------------
    # Usato quando il JSON mode fallisce o si tronca.
    # PROMPT SEMPLIFICATO e CONTESTO RIDOTTO per sbloccare i thread "Zombie".
    
    SIMPLE_KG_PROMPT = """You are a Knowledge Graph Expert.
Extract entities and relationships from the text.
Return ONLY valid JSON.
Schema: {"nodes": [{"id": "X", "label": "Y"}], "edges": [{"source": "A", "target": "B", "relation": "C"}]}
"""
    try:
        # Tagliamo il testo per il retry per evitare che un chunk enorme
        # (es. bibliografia piena di link) mandi in loop il modello.
        retry_text = text[:2500] 

        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SIMPLE_KG_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{retry_text}"}
            ],
            # NO format="json" qui (lasciamo libertà al modello per correggere errori)
            options={
                "temperature": 0.1, 
                "num_predict": 2048, # Meno token output
                "num_ctx": 2048      # <--- ANTI-HANG: Riduciamo la RAM richiesta per il retry
            }
        )
        raw_content = resp.get("message", {}).get("content", "")
        
        # Usa la funzione helper (che deve essere presente nello script)
        js = safe_json_extract(raw_content)
        
        if js:
            # --- FIX ROBUSTEZZA TIPO ANCHE QUI ---
            if isinstance(js, list):
                if len(js) > 0 and isinstance(js[0], dict):
                    js = js[0]
                else:
                    js = {}
            # -------------------------------------

            nodes = js.get("nodes", js.get("entities", []))
            edges = js.get("edges", js.get("relationships", []))
            return nodes, edges
        else:
            # Se fallisce anche il retry, amen. Non blocchiamo tutto.
            return [], []

    except Exception as e:
        # Silenziamo l'errore critico nel retry per non sporcare il log
        return [], []
    
    # ------------------------------------------------------------
    # TENTATIVO 2: RAW MODE (Fallback)
    # ------------------------------------------------------------
    # (Inserisci qui il resto della funzione come nel codice precedente...)
    # Assicurati di applicare lo stesso controllo "isinstance(js, list)" anche qui sotto
    
    SIMPLE_KG_PROMPT = """You are a Knowledge Graph Expert.
    Extract entities and relationships from the text.
    Return ONLY valid JSON.
    Schema: {"nodes": [], "edges": []}
    """
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SIMPLE_KG_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text[:3000]}"}
            ],
            options={"temperature": 0.1, "num_predict": 3000, "num_ctx": 4096}
        )
        raw_content = resp.get("message", {}).get("content", "")
        js = safe_json_extract(raw_content)
        
        if js:
            # --- FIX LISTA ANCHE QUI ---
            if isinstance(js, list):
                if len(js) > 0 and isinstance(js[0], dict):
                    js = js[0]
                else:
                    js = {}
            # ---------------------------

            nodes = js.get("nodes", js.get("entities", []))
            edges = js.get("edges", js.get("relationships", []))
            return nodes, edges
        else:
            # print(f"   ❌ [KG-FAIL] {base} p{page_no}: Recupero fallito.")
            return [], []

    except Exception as e:
        # print(f"   ❌ [KG-CRITICAL] {base} p{page_no}: {e}")
        return [], []


    
    # ------------------------------------------------------------
    # Chiamata LLM
    # ------------------------------------------------------------
    
    # SYSTEM PROMPT SEMPLIFICATO PER EVITARE CONFUSIONE
    SIMPLE_KG_PROMPT = """You are a Knowledge Graph Expert.
Extract entities (concepts, persons, metrics) and relationships from the text.
Return ONLY valid JSON with this schema:
{
  "nodes": [{"id": "ConceptName", "label": "CATEGORY"}],
  "edges": [{"source": "ConceptName", "target": "OtherConcept", "relation": "VERB"}]
}
"""

    # TENTATIVO 1: JSON MODE (Strict)
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SIMPLE_KG_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text[:3500]}"}
            ],
            format="json",
            options={
                            "temperature": 0.0, 
                            "num_predict": 2500, # Aumentato da 1200 a 2500
                            "num_ctx": 8192      # Aumentato per gestire meglio la memoria di lavoro di Gemma 2
                        }
                    )
        content = resp.get("message", {}).get("content", "")
        
        # Validazione Immediata
        if content:
            js = json.loads(content)
            nodes = js.get("nodes", js.get("entities", []))
            edges = js.get("edges", js.get("relationships", []))
            return nodes, edges
            
    except Exception as e:
        # Se fallisce, non stampiamo errore, andiamo diretti al retry
        pass

    # TENTATIVO 2: RAW MODE (Fallback) + DEBUG
    print(f"   ⚠️ [KG-RETRY] {base} p{page_no}: JSON mode failed. Retrying raw...")
    
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SIMPLE_KG_PROMPT + "\nIMPORTANT: Do not use Markdown blocks. Just raw JSON."},
                {"role": "user", "content": f"Analyze this text:\n\n{text[:3500]}"}
            ],
            # NO format="json" qui
            options={"temperature": 0.1, "num_ctx": 3584}
        )
        raw_content = resp.get("message", {}).get("content", "")
        
        # Debug: Vediamo cosa risponde se fallisce ancora
        # print(f"   🐛 DEBUG RAW RESPONSE: {raw_content[:100]}...") 

        js = _extract_json_substring(raw_content)
        if js:
            nodes = js.get("nodes", js.get("entities", []))
            edges = js.get("edges", js.get("relationships", []))
            return nodes, edges
        else:
            print(f"   ❌ [KG-FAIL] {base} p{page_no}: No valid JSON found in response.")
            return [], []

    except Exception as e:
        print(f"   ❌ [KG-CRITICAL] {base} p{page_no}: {e}")
        return [], []



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
def recurse_text_chunking(text: str, base_meta: Dict[str, Any], max_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    Versione Ottimizzata: Aumentata la soglia max_chars a 1200 (da 1000)
    per ridurre il numero totale di nodi e chunk.
    """
    text = text.strip()
    if not text:
        return []

    # CASO BASE: Aumentiamo la tolleranza per non spezzare troppo
    if len(text) <= max_chars + 200: 
        source = base_meta.get("source", "unknown")
        p_no = base_meta.get("page", 0)
        
        # Ridotto l'header: rimosso il nome file che è già nei metadati 
        # per risparmiare token e ridurre la ripetitività nel testo semantico.
        sem_header = f"PAG. {p_no}" 
        
        meta_final = base_meta.copy()
        if "metadata_override" in base_meta:
            del meta_final["metadata_override"]
            meta_final.update(base_meta["metadata_override"])

        return [{
            "text_raw": text,
            "text_sem": f"{sem_header}: {text}",
            "page_no": p_no,
            "toon_type": base_meta.get("type", "text"),
            "section_hint": "content",
            "image_id": base_meta.get("original_image_id"),
            "metadata_override": meta_final
        }]
    
    # ... resto della logica ricorsiva ...
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


def prep_text_for_embedding(s: str, max_chars: int = 1800) -> str:
    """
    PDF-optimized: riduce rumore e costo tokenizer.
    - rimuove header "Doc: ...\n" se presente
    - normalizza whitespace
    - tronca a max_chars (importantissimo per i PDF)
    """
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^Doc:\s.*?\n", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


# ==============================================================================
# MOTORE VISIONE QWEN (Thread-Safe + Vision Supremacy)
# ==============================================================================
# ==============================================================================
def extract_pdf_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Strategia 'Vision-First' con Classificazione Granulare (Per-Chunk).
    """
    out_chunks = []
    filename = os.path.basename(file_path)

    # Reset stats
    if "VISION_STATS" in globals() and "_VISION_STATS_LOCK" in globals():
        with _VISION_STATS_LOCK: VISION_STATS["pages_total"] = 0

    try:
        doc0 = fitz.open(file_path)
        total_pages = len(doc0)
        doc0.close()
    except Exception as e:
        print(f"   ❌ Errore critico file {filename}: {e}")
        return []

    print(f"   🚀 Ingestion: {total_pages} pagine | Vision: {VISION_MODEL_NAME} | Brain: {LLM_MODEL_NAME}")

    # --- WORKER INTERNO ---
    def process_page_worker(p_idx: int):
        p_no = p_idx + 1
        local_res = []
        doc_worker = None
        try:
            doc_worker = fitz.open(file_path)
            page = doc_worker.load_page(p_idx)
            
            # 1. Rendering
            zoom = VISION_DPI / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            # ✅ riduce tempo + VRAM + rischio blocchi Vision
            img_bytes = _downscale_and_compress_for_vision(img_bytes, max_side=1400, jpeg_quality=90)
            # 2. Salva img nel DB
            img_id = None
            t_conn = pg_get_conn()
            if t_conn:
                try:
                    with t_conn.cursor() as t_cur:
                        # Salva lo screenshot della pagina intera
                        img_id = pg_save_image(log_id, img_bytes, "image/png", f"Page_{p_no}", cur=t_cur)
                    t_conn.commit()
                except Exception as e:
                    print(f"   ⚠️ Errore salvataggio immagine PG (Pagina {p_no}): {e}")
                    t_conn.rollback()
                finally:
                    pg_put_conn(t_conn)

            # 3. VISIONE TOTALE
            chunk_text = llm_chat_multimodal(
                prompt=VISION_FIRST_PROMPT, 
                image_bytes=img_bytes, 
                model=VISION_MODEL_NAME 
            )

            # 4. Creazione Chunk
            if chunk_text and "NO_CONTENT" not in chunk_text and len(chunk_text) > 10:
                
                # Partiamo assumendo che sia tutto TESTO.
                # La classificazione avverrà pezzo per pezzo DOPO lo split.
                meta = {
                    "source": filename, 
                    "page": p_no, 
                    "type": "text", # Default
                    "original_image_id": img_id
                }
                
                # Generiamo i chunk grezzi
                # 1. Generazione chunk con soglia più alta per evitare frammentazione eccessiva
                # Usiamo CHUNK_MAX_CHARS (consigliato 1200-1500)
                raw_chunks = recurse_text_chunking(chunk_text, meta, max_chars=CHUNK_MAX_CHARS)
                
                # 2. Set per la deduplicazione semantica locale (per pagina)
                seen_hashes = set()
                
                # 3. POST-PROCESSING: Classificazione e Pulizia
                for ch in raw_chunks:
                    txt_content = ch.get("text_raw", "").strip()
                    
                    # --- FILTRO RIDONDANZA ---
                    # Evitiamo chunk troppo corti o duplicati esatti generati dalla ricorsione
                    content_hash = hashlib.md5(txt_content.encode()).hexdigest()[:12]
                    if len(txt_content) < 100 or content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)

                    # --- CLASSIFICAZIONE GRANULARE AVANZATA ---
                    is_visual = False
                    
                    # A. Marcatori espliciti del Prompt Vision
                    visual_markers = [
                        "### 🖼️ VISUAL ANALYSIS", 
                        "*Visual Elements:*", 
                        "*Data Insights:*",
                        "ANALISI VISIVA"
                    ]
                    
                    # B. Analisi euristica del contenuto (Keywords visive)
                    # Cerca termini come "asse", "legenda", "andamento" nel chunk
                    visual_keywords = r"\b(asse|assi|axis|axes|legenda|legend|grafico|chart|plot|pendenza|slope|barre|bars)\b"
                    has_visual_terms = len(re.findall(visual_keywords, txt_content, re.IGNORECASE)) >= 2
                    
                    if any(m in txt_content for m in visual_markers) or has_visual_terms:
                        is_visual = True

                    # C. Protezione Formule: Se è presente LaTeX ($$), non classificarlo come immagine
                    # a meno che non ci siano riferimenti espliciti a grafici.
                    if "$$" in txt_content and not has_visual_terms:
                        is_visual = False

                    # 4. Assegnazione finale e salvataggio
                    ch["toon_type"] = "imagine" if is_visual else "testo"
                    local_res.append(ch)

        except Exception as e:
            print(f"   ⚠️ Error p.{p_no}: {e}")
        finally:
            if doc_worker: doc_worker.close()
            
        return local_res

    # --- ESECUZIONE PARALLELA ---
    workers = int(os.environ.get("VISION_PARALLEL_WORKERS", "1")) #3
    per_page_timeout = int(os.getenv("PDF_PAGE_TIMEOUT_S", "240"))  # timeout per pagina

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(process_page_worker, i): i for i in range(total_pages)}

        done_count = 0
        for f in as_completed(future_map):
            p_idx = future_map[f]
            try:
                res = f.result(timeout=per_page_timeout)
                if res:
                    out_chunks.extend(res)
            except Exception as e:
                # Non bloccare ingestion: logga e vai avanti
                print(f"\n   ⚠️ Pagina {p_idx+1}/{total_pages} fallita o in timeout: {e}")

            done_count += 1
            print(f"   🔄 Processed {done_count}/{total_pages}...", end="\r")

    print("")
    out_chunks.sort(key=lambda x: x["page_no"])
    return add_context_windows(out_chunks)


def is_page_math_heavy_or_broken(text: str) -> bool:
    """
    Rileva se la pagina contiene formule complesse che PyMuPDF ha rotto
    o se è densa di matematica che richiede LaTeX.
    """
    if not text: return False
    
    # 1. Rileva artefatti di codifica PDF (il problema che hai avuto col WACC)
    if "(cid:" in text or "2C0D" in text: 
        return True
        
    # 2. Rileva simboli di corruzione comuni nelle formule
    # Spesso le linee di frazione diventano caratteri strani
    if text.count("•") > 5 or text.count("—") > 5 or text.count("˜") > 3:
        return True

    # 3. Rileva keyword matematiche forti
    # Se c'è "WACC =" o "Equation", vogliamo vederla bene in LaTeX
    math_triggers = ["WACC", "CAPM", "Black-Scholes", "Equation", "Formula", "Theorem", "Teorema"]
    # Controlliamo se c'è una trigger word E se il testo non è lunghissimo (evitiamo libri interi)
    if any(t in text for t in math_triggers) and len(text) < 3000:
        return True
        
    return False

# ==============================================================================
# PATTERN GENERALISTA PER LA MATEMATICA
# ==============================================================================
# Rileva simboli matematici universali, operatori logici, lettere greche e keyword standard.
MATH_BROAD_PAT = re.compile(
    r"(?i)("
    r"formula|equation|equazione|modello|model|theorem|teorema|lemma|"  # Keyword generiche
    r"[∑∏∫√∂∇∆∀∃∄∈∉⊆⊂∪∩≠≈≡≤≥±∓×÷∝∞]|"                             # Simboli matematici avanzati
    r"\^\{|\_\{|"                                                 # Sintassi stile LaTeX residua
    r"[a-z]_[a-z0-9]|"                                            # Pedici (es. x_i, t_0)
    r"\b[A-Za-z]=\d+"                                             # Assegnazioni (es. x=5)
    r")"
)

def extract_file_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Estrazione Universale (PDF) v2.5:
    1. Testo Nativo con OVERLAP (Rolling Buffer).
    2. Immagini Embedded (grafici raster).
    3. Matematica/Schemi (Visione su render Hi-Res).
    """
    
    # --- CONFIGURAZIONE OVERLAP ---
    OVERLAP_SIZE = 250  # Caratteri (~40 parole)
    prev_page_tail = "" # Buffer per la coda della pagina precedente
    # ------------------------------

    # 1. Estrazione del testo nativo
    native_text_pages = extract_pdf_text_by_page_pdfminer(file_path)

    filename = os.path.basename(file_path)
    final_chunks: List[Dict[str, Any]] = []
    doc = None 

    # Helper per contare elementi vettoriali
    def _count_vectors(p: fitz.Page) -> int:
        try:
            ops = 0
            drawings = p.get_drawings() or []
            for d in drawings:
                ops += len(d.get("items") or [])
            return ops
        except Exception:
            return 0

    try:
        doc = fitz.open(file_path)
        # Calcolo ID univoco documento (hash)
        doc_id = sha256_file(file_path)[:32]

        for i, page_text in enumerate(native_text_pages):
            page_no = i + 1
            
            # Pulizia base per evitare spazi bianchi eccessivi
            clean_text = page_text.strip()
            
            # =========================================================
            # FIX VISION-NATIVE: Rilevamento Testo Corrotto / Matematica
            # =========================================================
            use_vision_replacement = False
            if PDF_VISION_ENABLED and is_page_math_heavy_or_broken(page_text):
                print(f"   👁️‍🗨️ Pagina {page_no}: Rilevata matematica/corruzione. Attivo Vision-to-Markdown...")
                use_vision_replacement = True

            # Se serve la Visione, sostituiamo 'clean_text' con l'output dell'LLM
            if use_vision_replacement:
                try:
                    # Carichiamo la pagina come immagine ad alta risoluzione (DPI 250 cruciale per pedici)
                    page_obj = doc[i]
                    hq_bytes = render_full_page_png(page_obj, dpi=180) 
                    
                    # Svuotiamo la VRAM prima del task pesante
                    #if VISION_MODEL_NAME: force_unload_ollama(VISION_MODEL_NAME)

                    # ✅ NON fare unload qui: costa moltissimo e spesso peggiora la latenza.
                    # Manteniamo il modello caldo durante l'intero documento.
                    vision_md = llm_chat_multimodal(
                        prompt=MARKER_VISION_PROMPT,
                        image_bytes=hq_bytes,
                        model=VISION_MODEL_NAME,
                        max_tokens=3500
                    )

                    
                    if len(vision_md) > 50:
                        clean_text = vision_md 
                        # Aggiungiamo un marker invisibile per debug futuro
                        clean_text = f"\n{clean_text}"
                    
                except Exception as e:
                    print(f"   ⚠️ Vision fallback failed: {e}")
                    # Se fallisce, teniamo il clean_text originale
            # =========================================================

            # ---------------------------------------------------------
            # A) CHUNK TESTO BASE (CON OVERLAP)
            # ---------------------------------------------------------
            if len(clean_text) > MIN_CHUNK_LEN:
                
                # --- LOGICA OVERLAP ---
                
                # --- LOGICA OVERLAP ---
                # Costruiamo il testo semantico (quello che verrà vettorizzato)
                # incollando la fine della pagina precedente all'inizio di questa.
                if prev_page_tail:
                    text_semantic_content = f"... {prev_page_tail}\n{clean_text}"
                else:
                    text_semantic_content = clean_text
                
                # Aggiorniamo il buffer per il prossimo giro
                if len(clean_text) > OVERLAP_SIZE:
                    prev_page_tail = clean_text[-OVERLAP_SIZE:]
                else:
                    prev_page_tail = clean_text # Pagina corta, prendiamo tutto
                # ----------------------

                final_chunks.append({
                    "text_raw": clean_text,          # Il testo originale resta pulito
                    "text_sem": text_semantic_content, # Il testo per il RAG ha l'overlap
                    "toon_type": "testo",
                    "page_no": page_no,
                    "metadata": {"source": filename, "doc_id": doc_id}
                })
            else:
                # Se la pagina è vuota/sporca, resettiamo il buffer per non incollare
                # roba vecchia su una pagina che magari è un nuovo capitolo dopo pagina bianca
                prev_page_tail = ""

            # Se PyMuPDF riesce ad aprire la pagina corrispondente
            if i < len(doc):
                page = doc[i]

                # ---------------------------------------------------------
                # B) IMMAGINI EMBEDDED (Grafici Raster, Foto)
                # ---------------------------------------------------------
                if PDF_VISION_ENABLED:
                    page_objs = page.get_images(full=True) or []
                    
                    t_conn = pg_get_conn()
                    try:
                        with t_conn.cursor() as t_cur:
                            for img_idx, img in enumerate(page_objs):
                                xref = img[0]
                                try:
                                    base_image = doc.extract_image(xref)
                                    img_bytes = base_image.get("image", b"")
                                    
                                    if not img_bytes or len(img_bytes) < MIN_ASSET_SIZE:
                                        continue

                                    img_name = f"PDF_{doc_id}_P{page_no}_IMG{img_idx}"
                                    image_id = pg_save_image(log_id, img_bytes, "image/jpeg", img_name, cur=t_cur)

                                    analysis = None
                                    if VISION_MODEL_NAME:
                                        try:
                                            analysis = extract_chart_via_vision(
                                                img_bytes,
                                                context_hint=f"Page {page_no} of {filename}"
                                            )
                                        except Exception: 
                                            analysis = None
                                    
                                    if analysis:
                                        sem_text = build_chart_semantic_chunk(page_no, analysis)
                                        meta = analysis
                                    else:
                                        sem_text = normalize_ws(f"--- ASSET VISIVO - P{page_no} ---\nNome: {img_name}")
                                        meta = {"asset_name": img_name}

                                    final_chunks.append({
                                        "text_raw": json.dumps(meta) if analysis else sem_text,
                                        "text_sem": sem_text,
                                        "toon_type": "imagine",
                                        "page_no": page_no,
                                        "image_id": image_id,
                                        "metadata": {**meta, "source": filename}
                                    })

                                except Exception:
                                    continue
                        t_conn.commit()
                    finally:
                        pg_put_conn(t_conn)

                # ---------------------------------------------------------
                # C) MATEMATICA & VETTORIALI (Render & Transcribe)
                # ---------------------------------------------------------
                has_math_text = bool(MATH_BROAD_PAT.search(page_text))
                vector_count = _count_vectors(page)
                has_vectors = vector_count > 10 

                if PDF_VISION_ENABLED and (has_math_text or has_vectors):
                    try:
                        pix = page.get_pixmap(dpi=300, alpha=False)
                        hq_bytes = pix.tobytes("png")

                        math_json = extract_formulas_vision(hq_bytes)

                        if math_json and math_json.get("formulas"):
                            sem_math = build_formula_semantic_chunk(page_no, math_json)
                            
                            final_chunks.append({
                                "text_raw": json.dumps(math_json, ensure_ascii=False),
                                "text_sem": sem_math,
                                "toon_type": "formula",
                                "page_no": page_no,
                                "metadata": {
                                    "source": filename,
                                    "type": "mathematical_content",
                                    "doc_id": doc_id,
                                    "formulas_found": len(math_json.get("formulas", []))
                                }
                            })
                            print(f"   Σ  Matematica rilevata a pag {page_no}: {len(math_json['formulas'])} formule.")

                    except Exception as e_math:
                        print(f"   ⚠️ Errore estrazione matematica pag {page_no}: {e_math}")

    except Exception as e:
        print(f"   ❌ Errore critico file {filename}: {e}")
    finally:
        if doc: doc.close()

        # ✅ Unload una sola volta a fine documento (opzionale)
        if PDF_VISION_ENABLED and VISION_MODEL_NAME:
            force_unload_ollama(VISION_MODEL_NAME)

    return final_chunks



def extract_pdf_as_markdown_assets(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Estrae testo e asset da PDF.
    FIX:
    - Crea SEMPRE un chunk 'immagine' per ogni asset embedded (anche se Vision fallisce o ritorna kind=other)
    - Evita che tutto finisca come 'testo' quando il PDF contiene grafici.
    """
    chunks_payload: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"   ❌ Errore apertura PDF {file_path}: {e}")
        return []

    filename = os.path.basename(file_path)
    total_pages = len(doc)
    doc_id = sha256_file(file_path)[:32]

    print(f"   🚀 Ingestion: {total_pages} pagine | Vision: {VISION_MODEL_NAME} | Brain: {LLM_MODEL_NAME}")

    # Pulizia VRAM preventiva
    if PDF_VISION_ENABLED and VISION_MODEL_NAME:
        force_unload_ollama(VISION_MODEL_NAME)

    for i, page in enumerate(doc):
        page_no = i + 1
        try:
            # ------------------------------------------------------------
            # 1) Chunk TESTO pagina
            # ------------------------------------------------------------
            page_text = page.get_text()
            text_sem = safe_normalize_text(page_text) or ""

            # se vuoi, puoi mantenere lo skip condizionale; qui lo lasciamo conservativo
            if len(text_sem) < 50 and not PDF_VISION_ENABLED:
                continue

            page_chunk = {
                "text_raw": text_sem,
                "text_sem": f"Page {page_no} content: {text_sem[:250]}...\n{text_sem}",
                "page_no": page_no,
                "toon_type": "testo",
                "metadata": {
                    "source": filename,
                    "page": page_no,
                    "doc_id": doc_id,
                }
            }
            chunks_payload.append(page_chunk)

            # ------------------------------------------------------------
            # 2) Chunk IMMAGINI embedded (grafici inclusi)
            # ------------------------------------------------------------
            if PDF_VISION_ENABLED:
                images = page.get_images(full=True) or []
                if images:
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        try:
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image.get("image", b"")

                            # filtro dimensione (usa la tua soglia configurabile)
                            if not image_bytes or len(image_bytes) < MIN_ASSET_SIZE:
                                continue

                            # salva asset in Postgres
                            img_name = f"PDF_{doc_id}_P{page_no}_IMG{img_index}"
                            conn = pg_get_conn()
                            try:
                                with conn.cursor() as cur:
                                    image_id = pg_save_image(log_id, image_bytes, "image/jpeg", img_name, cur)
                                conn.commit()
                            finally:
                                pg_put_conn(conn)

                            # prova Vision
                            c_js = None
                            if VISION_MODEL_NAME:
                                try:
                                    c_js = extract_chart_via_vision(
                                        image_bytes,
                                        context_hint=f"{filename} | page {page_no}"
                                    )
                                except Exception as _:
                                    c_js = None

                            # costruisci semantica (anche fallback)
                            conf = float((c_js or {}).get("confidence") or 0.0)
                            kind = (c_js or {}).get("kind")

                            if c_js and isinstance(c_js, dict):
                                # FIX: uniforma tipo (evita 'imagine' typo)
                                c_js["toon_type"] = "immagine"

                            CHART_MIN_CONF = float(os.getenv("CHART_MIN_CONF", "0.55"))

                            if c_js and kind != "other" and conf >= CHART_MIN_CONF:
                                sem = build_chart_semantic_chunk(page_no, c_js)
                                meta = c_js
                            else:
                                sem = normalize_ws(
                                    f"--- CONTENUTO VISIVO (immagine) - Pagina {page_no} ---\n"
                                    f"Asset: {img_name}\n"
                                    f"Nota: immagine estratta dal PDF (grafico/tabella possibile).\n"
                                    f"Stato: non interpretata automaticamente oppure conf bassa.\n"
                                    f"Hint: prova ad aumentare VISION_DPI o abbassare CHART_MIN_CONF.\n"
                                )
                                meta = {
                                    "asset_name": img_name,
                                    "confidence": conf,
                                    "kind": kind or "unknown",
                                    "toon_type": "immagine"
                                }

                            # crea chunk IMMAGE dedicato (questo è il FIX chiave)
                            img_chunk = {
                                "text_raw": sem,               # utile anche per embedding
                                "text_sem": sem,               # retrieval “forte”
                                "page_no": page_no,
                                "toon_type": "immagine",
                                "image_id": image_id,
                                "metadata": {
                                    "source": filename,
                                    "page": page_no,
                                    "doc_id": doc_id,
                                    **(meta if isinstance(meta, dict) else {})
                                }
                            }
                            chunks_payload.append(img_chunk)

                        except Exception as e_img:
                            print(f"   ⚠️ Err. Img {img_index} pg {page_no}: {e_img}")
                            continue


            # ✅ VRAM safety: no unload per pagina (troppo costoso).
            # Se serve, lasciamo solo una GC leggera.
            if PDF_VISION_ENABLED:
                gc.collect()


        except Exception as e_page:
            print(f"   ⚠️ Err. Pagina {page_no}: {e_page}")
            continue

    try:
        doc.close()
    except Exception:
        pass

    return chunks_payload



def normalize_toon_type(ch: dict) -> str:
    """
    Rileva se un chunk è 'immagine' basandosi sui marcatori del prompt di visione
    o sulla presenza di un asset ID salvato.
    """
    # 1. Controllo se esiste un riferimento a un'immagine salvata nel DB
    if ch.get("image_id") is not None:
        return "imagine"
    
    # 2. Scansione dei marcatori testuali generati dalla Vision AI
    # Cerca i tag impostati in VISION_FIRST_PROMPT
    content_raw = ch.get("text_raw", "")
    content_sem = ch.get("text_sem", "")
    
    markers = [
        "### 🖼️ VISUAL ANALYSIS", 
        "VISUAL ANALYSIS:", 
        "*Visual Elements:*", 
        "--- CONTENUTO VISUALE"
    ]
    
    if any(m in content_raw for m in markers) or any(m in content_sem for m in markers):
        return "imagine"
    
    # 3. Fallback sul tipo assegnato durante l'estrazione
    if ch.get("toon_type") == "immagine":
        return "imagine"

    return "testo"


def process_virtual_md_chunks(content: str, asset_park: dict, filename: str, log_id: int) -> List[Dict[str, Any]]:
    """
    Versione 2.9.2 (Optimized):
    - Include Debug Saving e Context Hint.
    - Prompt Merging: Rimossa la seconda chiamata LLM per l'analisi (ora fatta dalla Vision).
    """
    out_chunks = []

    # ✅ Cap Vision per documento
    vision_done = 0
    VISION_MAX_ASSETS_PER_DOC = int(os.getenv("VISION_MAX_ASSETS_PER_DOC", "25"))

    content = clean_markdown_structure(content)
    raw_paras = re.split(r'\n(?=# PAGE)|\n\n', content)

    for para in raw_paras:
        para_strip = para.strip()
        if not para_strip:
            continue

        # Identificazione Pagina
        current_page = 1
        page_match = re.search(r'# PAGE (\d+)', para_strip)
        if page_match: current_page = int(page_match.group(1))

        # Scomposizione granulare
        sub_segments = [para_strip] if len(para_strip) <= 1200 else split_text_with_overlap(para_strip, 1000, 200)

        for sub_p in sub_segments:
            clean_sub_p = safe_normalize_text(sub_p)
            
            chunk_data = {
                "text_raw": clean_sub_p,
                "text_sem": f"Doc: {filename} | Content: {clean_sub_p[:80]}...\n{clean_sub_p}",
                "page_no": current_page, 
                "toon_type": "text",
                "section_hint": find_section_hint(clean_sub_p)
            }

            # --- LOGICA ASSET VISUALI ---
            img_match = re.search(r'!\[.*?\]\(((?:img_|fig_).*?\.jpg)\)', clean_sub_p)
            if img_match and PDF_VISION_ENABLED:
                asset_id = img_match.group(1)
                img_bytes = asset_park.get(asset_id)

                if img_bytes and len(img_bytes) >= MIN_ASSET_SIZE and ai_vision_gatekeeper(img_bytes):
                    if vision_done >= VISION_MAX_ASSETS_PER_DOC:
                        out_chunks.append(chunk_data)
                        continue

                    chunk_data["toon_type"] = "immagine"
                    hint = f"{filename} | page {current_page}"

                    # Estrazione (ora include già 'analysis_it' grazie al Prompt Merging)
                    c_js = extract_chart_via_vision(img_bytes, context_hint=hint) or {}
                    vision_done += 1
                    
                    if c_js:
                        # 🔒 GEO SANITIZATION (STRICT)
                        bad_geo_tokens = [" sud", "south"]
                        cats = c_js.get("categories_it", [])
                        if isinstance(cats, list):
                            c_js["categories_it"] = [
                                c for c in cats
                                if not any(tok in c.lower() for tok in bad_geo_tokens)
                            ]

                        # 🔒 BLOCK AMBIGUOUS TIMEFRAME
                        tf = c_js.get("timeframe")
                        years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(tf)) if tf else []
                        if len(years) < 2:
                            c_js.pop("timeframe", None)

                        # --- OPTIMIZATION START: Prompt Merging ---
                        # Abbiamo rimosso la chiamata a generate_chart_analysis_it()
                        # perché il dato è già presente in c_js['analysis_it'] dal modello Vision.
                        # --- OPTIMIZATION END ---

                        c_js = normalize_chart_json_for_semantics(c_js, page_no=current_page, context_hint=hint)
                        semantic = build_chart_semantic_chunk(current_page, c_js)
                        
                        # Replace semantico
                        # is_image_only: chunk che contiene solo immagine markdown (es. ![...](...)) oppure chunk di tipo image
                        md_only = (chunk_data.get("text_raw") or "").strip()

                        toon_type = (chunk_data.get("toon_type") or "").strip().lower()
                        # accetta anche il vecchio typo "imagine"
                        if toon_type == "imagine":
                            toon_type = "image"
                        chunk_data["toon_type"] = toon_type  # normalizza per downstream

                        is_image_only = (toon_type == "image") or (
                            md_only.startswith("![") and "](" in md_only and len(md_only) < 120
                        )

                        chunk_data["text_sem"] = semantic
                        if is_image_only:
                            chunk_data["text_raw"] = semantic 
                    else:
                        # Fallback se la vision fallisce o restituisce None
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

                                # 🔥 FIX: rendi l'immagine "risolvibile" dal solo record document_chunks
                                chunk_data.setdefault("metadata", {})
                                chunk_data["metadata"]["image_id"] = chunk_data["image_id"]
                                chunk_data["metadata"]["asset_name"] = asset_id
                                chunk_data["metadata"]["mime_type"] = "image/jpeg"

                        conn.commit()
                    finally:
                        pg_put_conn(conn)

            out_chunks.append(chunk_data)
    return out_chunks


# =========================
# FILE DISPATCH (PDF only here)
# =========================
def process_single_file(file_path: str, source_type: str, doc_meta: dict):
    """
    Pipeline v2.5 - Optimized for Gemma 2:9b & P5000 (Final Clean)
    Rimuove log doppi e ottimizza la visualizzazione della console.
    """
    t0 = time.time()
    filename = os.path.basename(file_path)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    tier = (doc_meta or {}).get("tier", "B")
    ontology = (doc_meta or {}).get("ontology", DEFAULT_ONTOLOGY)
    
    print(f"   ⚙️ Engine Start: {filename} | tier={tier} | Brain={LLM_MODEL_NAME}")

    log_id = pg_start_log(filename, source_type)
    doc_id = sha256_file(file_path)[:32]

    global embedder, qdrant_client
    
    # Init Lazy dei client
    embedder = get_embedder()
    qdrant_client = get_qdrant_client()
    ensure_qdrant_collection()

    # 🔥 Warmup embeddings (singolo, per evitare il doppio log)
    try:
        # Check se è già caldo (opzionale, ma male non fa)
        pass 
    except Exception:
        pass

    # 1. Estrazione Chunks (Vision AI avviene qui per i PDF)
    # Nota: Qui potrebbe esserci una pausa lunga mentre Ministral legge le pagine
    if file_path.lower().endswith(".md"):
            print(f"   📝 Detected Markdown: {filename}")
            chunks = extract_markdown_chunks(file_path, log_id)
    else:
        # Default assume PDF
        chunks = extract_file_chunks(file_path, log_id)
    
    if not chunks:
        pg_close_log(log_id, "FAILED", 0, _ms(t0), "No chunks extracted")
        try:
            shutil.move(file_path, os.path.join(FAILED_DIR, filename))
        except Exception:
            pass
        return

    
    # 2. Iniezione metadati e normalizzazione tipo
    for idx, ch in enumerate(chunks):
        ch.setdefault("chunk_index", idx)
        
        # Applica la nuova classificazione invece di forzare 'text'
        ch["toon_type"] = normalize_toon_type(ch)
        
        meta = ch.get("metadata", {})
        meta.update({
            "doc_id": doc_id, 
            "filename": filename, 
            "tier": tier, 
            "ontology": ontology
        })
        ch["metadata"] = meta

    # Buffers
    qdrant_points, pg_rows, neo4j_rows = [], [], []
    total_chunks = 0
    num_chunks_totali = len(chunks)

    # PDF: batch più piccolo per stabilità VRAM su P5000
    if file_path.lower().endswith(".pdf"):
        pdf_batch = int(os.environ.get("PDF_EMBED_BATCH_SIZE", "8"))
        if pdf_batch > 0:
            global EMBED_BATCH_SIZE
            EMBED_BATCH_SIZE = pdf_batch

    # LOG SINGOLO (Prima ne avevi due)
    print(f"   🚀 Inizio elaborazione: {num_chunks_totali} chunks (Batch: {EMBED_BATCH_SIZE})")


    # 3. Ciclo Batches
    for i in range(0, num_chunks_totali, EMBED_BATCH_SIZE):
        batch_t0 = time.time()
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        
        # Prep text
        PDF_EMBED_MAX_CHARS = int(os.environ.get("PDF_EMBED_MAX_CHARS", "1800"))
        texts = [prep_text_for_embedding(c.get("text_sem", ""), max_chars=PDF_EMBED_MAX_CHARS) for c in batch]

        # 3a. Embeddings (Con barra di progresso reale)
        print(f"   [DEBUG] Calcolo embeddings batch {i//EMBED_BATCH_SIZE + 1}...")
        t_emb0 = time.time()
        try:
            vecs = embedder.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True)
        except Exception as e:
            print(f"   ⚠️ Errore Embeddings: {e}")
            break
        t_emb1 = time.time()

        # 3b. Knowledge Graph (Parallel)
        batch_kg_results = {}
        if KG_ENABLED:
            pages_map = {}
            for local_j, ch in enumerate(batch):
                p = int(ch.get("page_no") or 1)
                # salviamo anche l'indice locale nel batch, così riusiamo vecs[local_j]
                pages_map.setdefault(p, []).append((i + local_j, local_j, ch))

            # Convertiamo i vecs del batch una volta sola in torch (utile per mean/max)
            try:
                vecs_t = torch.tensor(vecs, dtype=torch.float32)
            except Exception:
                vecs_t = None

            futures_kg = {}
            for p_no, indexed_chunks in pages_map.items():
                combined_text = "\n".join([ch.get("text_sem", "") for _, _, ch in indexed_chunks])
                text_clean = safe_normalize_text(combined_text)[:KG_TEXT_MAX_CHARS]

                # ✅ Gatekeeper su embedding già calcolato (media dei chunk della pagina)
                if vecs_t is not None:
                    local_idxs = [local_j for _, local_j, _ in indexed_chunks]
                    page_vec = vecs_t[local_idxs].mean(dim=0)
                    ok_kg = ai_gatekeeper_decision_from_vec(page_vec)
                else:
                    # fallback (non dovrebbe quasi mai servire)
                    ok_kg = ai_gatekeeper_decision(text_clean)

                if ok_kg:
                    futures_kg[p_no] = kg_executor.submit(
                        llm_extract_kg, filename, p_no, text_clean, LLM_MODEL_NAME
                    )

            # Raccolta risultati KG
            for p_no, fut in futures_kg.items():
                try:
                    res = fut.result(timeout=120) 
                    if res:
                        nodes, edges = res
                        edges = canonicalize_edges_to_verb_object(edges)
                        for g_idx, _ in pages_map[p_no]:
                            batch_kg_results[g_idx] = (nodes, edges)
                except Exception:
                    pass

        # 3c. Costruzione Record DB
        for j, ch in enumerate(batch):
            g_idx = i + j

            # ✅ NORMALIZZA toon_type (solo "testo" / "imagine")
            ch["toon_type"] = normalize_toon_type(ch)

            vector = vecs[j]
            chunk_id = deterministic_chunk_id(
                doc_id,
                ch.get("page_no", 1),
                g_idx,
                ch.get("toon_type"),
                ch.get("text_sem")
            )

            # Qdrant Payload
            payload = ch["metadata"].copy()
            payload.update({"text_sem": ch.get("text_sem", ""), "page_no": ch.get("page_no", 1)})
            qdrant_points.append({"id": chunk_id, "vector": vector.tolist(), "payload": payload})

            # Postgres Rows
            pg_rows.append((log_id, g_idx, ch.get("toon_type"), ch.get("text_raw"), ch.get("text_sem"), json.dumps(ch["metadata"]), chunk_id))

            # Neo4j Rows (Con tutte le proprietà corrette)
            k_nodes, k_edges = batch_kg_results.get(g_idx, ([], []))
            neo4j_rows.append({
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": source_type,
                "log_id": log_id,
                "chunk_id": chunk_id,
                "chunk_index": g_idx,
                "toon_type": ch.get("toon_type"),
                "page_no": ch.get("page_no", 1),
                "nodes": k_nodes,
                "edges": k_edges,
                "ontology": ontology,
                "text_sem": ch.get("text_sem", ""),
                "section_hint": ch.get("section_hint", "")
            })
            total_chunks += 1

        # 4. Flush "intelligente" (meno roundtrip, stessa modalità di scrittura)
        must_flush = (len(pg_rows) >= DB_FLUSH_SIZE) or (i + len(batch) >= num_chunks_totali)

        if must_flush:
            flush_postgres_chunks_batch(pg_rows)

            try:
                pts = [models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in qdrant_points]
                qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=pts)
            except Exception as e:
                print(f"   ⚠️ Qdrant Error: {e}")

            flush_neo4j_rows_batch(neo4j_rows)

            # Reset buffers (solo dopo flush)
            pg_rows.clear()
            qdrant_points.clear()
            neo4j_rows.clear()

        
        # Log Avanzamento
        percentuale = min(100, int((i + len(batch)) / num_chunks_totali * 100))
        print(f"   📦 Batch {int(i/EMBED_BATCH_SIZE)+1} | {percentuale}% completato | Tempo Batch: {_ms(batch_t0)}ms")

    # 5. Chiusura Finale
    total_ms = _ms(t0)
    pg_close_log(log_id, "DONE", total_chunks, total_ms)

    if NEO4J_ENABLED:
        try:
            with neo4j_driver.session() as session:
                session.run("MATCH (d:Document {doc_id: $did}) SET d.processing_time_ms = $ms", did=doc_id, ms=total_ms)
        except Exception: pass

    # Sposta in PROCESSED
    try:
        shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
    except Exception as e:
        print(f"   ⚠️ Errore spostamento file: {e}")

    print(f"   ✅ Completed: {filename} | chunks={total_chunks} | time={total_ms/1000:.2f}s")

def main():
    """
    Punto di ingresso principale dell'Ingestion Engine.
    Configura l'ambiente, ottimizza Ollama e processa i file supportati.
    """
    # 1. Preparazione delle cartelle di lavoro
    os.makedirs(INBOX_DIR, exist_ok=True)
    ensure_inbox_structure(INBOX_DIR)
  
    # 2. Reset Totale OLLAMA (Turbo Mode per P5000)
    # Riavvia il server con NUM_PARALLEL=1 per evitare OOM su GPU 16GB
    # (Rimuovi "2" e metti "1" o lascia vuoto per usare il default)
    if not force_restart_ollama(num_parallel="1"): 
        print("   ❌ Errore: Impossibile avviare Ollama in modalità ottimizzata.")
        print("   ⚠️ L'ingestion potrebbe fallire o risultare estremamente lenta.")
    
    print("\n" + "="*60)
    print("=== Ingestion Engine v2.4 (FAST + Markdown Support + Value Hunter) ===")
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
