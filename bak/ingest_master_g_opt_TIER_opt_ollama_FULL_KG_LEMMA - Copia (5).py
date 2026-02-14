
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
import subprocess
import requests
from threading import Lock

import shutil # Aggiungi questo import in cima al file

import fitz  # PyMuPDF
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import SimpleConnectionPool

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

# =========================
# CONFIG
# =========================
BASE_DATA_DIR = "./data_ingestion"
INBOX_DIR = os.path.join(BASE_DATA_DIR, "INBOX")
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "PROCESSED")
FAILED_DIR = os.path.join(BASE_DATA_DIR, "FAILED")

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "40"))

CONTEXT_WINDOW_CHARS = int(os.getenv("CONTEXT_WINDOW_CHARS", "260"))
INCLUDE_CONTEXT_IN_KG = os.getenv("INCLUDE_CONTEXT_IN_KG", "1") == "1"

DB_FLUSH_SIZE = int(os.getenv("DB_FLUSH_SIZE", "120"))          # un po' più alto per meno flush - 
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "24"))    # se la tua VRAM regge, aumenta velocità

# Vision switches
PDF_VISION_ENABLED = os.getenv("PDF_VISION_ENABLED", "1") == "1"
PDF_VISION_ONLY_IF_TEXT_SCARSO = False #= os.getenv("PDF_VISION_ONLY_IF_TEXT_SCARSO", "0") == "1"
PDF_MIN_TEXT_LEN_FOR_NO_VISION = 0 #= int(os.getenv("PDF_MIN_TEXT_LEN_FOR_NO_VISION", "450"))

VISION_DPI = int(os.getenv("VISION_DPI", "160"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "2000000"))

VISION_MAX_FORMULAS_PER_PAGE = int(os.getenv("VISION_MAX_FORMULAS_PER_PAGE", "10"))

PDF_EXTRACT_EMBEDDED_IMAGES = True #= os.getenv("PDF_EXTRACT_EMBEDDED_IMAGES", "1") == "1"
PDF_VISION_ON_EMBEDDED_IMAGES = True # = os.getenv("PDF_VISION_ON_EMBEDDED_IMAGES", "1") == "1"
PDF_MAX_IMAGES_PER_PAGE = int(os.getenv("PDF_MAX_IMAGES_PER_PAGE", "8"))
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", "2000"))

# Speed: Vision parallel + cache
VISION_PARALLEL_WORKERS = 2 #int(os.getenv("VISION_PARALLEL_WORKERS", "4"))  # 4-6 di solito ok
OLLAMA_NUM_PARALLEL=1
VISION_CACHE_MAX = int(os.getenv("VISION_CACHE_MAX", "5000"))             # entries in-memory

# Commit policy
PG_COMMIT_EVERY_N_PAGES = int(os.getenv("PG_COMMIT_EVERY_N_PAGES", "50"))

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
KG_WORKERS = 1  # Forza l'elaborazione seriale per non saturare la VRAM
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

def is_structural_page(text: str) -> bool:
    """Rileva se la pagina è un indice o una sezione di servizio."""
    if not text: return False
    # Controlliamo l'inizio del testo per capire se è un indice
    return bool(STRUCTURAL_PAT.search(text[:400]))

def count_unique_keywords(text: str) -> int:
    """Conta quanti concetti finanziari diversi sono presenti nel chunk."""
    found = set(_KG_PAT.findall(text))
    return len(found)

def ai_gatekeeper_decision(text: str) -> bool:
    """
    Usa Gemma 3 12B come Gatekeeper ultra-veloce.
    Configurato con num_predict:2 per rispondere istantaneamente SI/NO.
    """
    try:
        resp = chat(
            model=LLM_MODEL_NAME, 
            messages=[{
                "role": "user", 
                "content": f"Is this financial text worth a Knowledge Graph? Respond ONLY YES or NO.\n\nText: {text[:500]}"
            }],
            options={
                "temperature": 0.0,
                "num_predict": 2,    # Limita la generazione a una sola parola
                "num_ctx": 1024      # Riduce il pre-caricamento per velocità
            }
        )
        return "YES" in resp['message']['content'].upper()
    except Exception as e:
        print(f"   ⚠️ Gatekeeper Error: {e}")
        return True # In caso di errore, meglio analizzare che perdere dati
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
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b-finstudio")
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


# =========================
# PROMPTS
# =========================
FORMULA_VISION_PROMPT = f"""
You are a careful scientific parser. The image contains mathematical or financial formulas.
Extract ONLY formulas that are clearly visible. 

Return ONLY valid JSON (no markdown):

{{
  "has_formulas": true/false,
  "formulas": [
    {{
      "latex": "LaTeX code without surrounding $",
      "plain": "normalized plain text version",
      "meaning_it": "Spiegazione approfondita in italiano del significato matematico e finanziario della formula. Descrivi a cosa serve (es. calcolo medie mobili, volatilità, segnali trading) e come deve essere interpretata.",
      "keywords": ["finance_term1", "math_term2"],
      "variables": [
        {{"name": "alpha", "meaning": "parametro di smoothing"}},
        {{"name": "P_t", "meaning": "prezzo al tempo t"}}
      ]
    }}
  ],
  "notes": ["uncertainties"]
}}

Rules:
- meaning_it: MUST be in Italian and detailed enough to allow a user to find the formula by its function.
- Max {VISION_MAX_FORMULAS_PER_PAGE} formulas.
"""

CHART_VISION_PROMPT = """
You are a strict financial and quantitative data extraction engine.
The image is a chart, table, diagram, or photo extracted from a technical PDF.

MANDATORY: 
- Your goal is to provide a "semantic bridge" for RAG systems.
- You MUST provide a rich, discursive analysis in the "summary_it" field.

Return ONLY valid JSON (no markdown):

{
  "kind": "table|chart|diagram|photo|other",
  "title": "visible title or empty",
  "source": "visible source label or empty",
  "timeframe": "visible timeframe or empty",
  "summary_it": "Descrizione analitica e discorsiva in italiano (minimo 5 righe). Spiega cosa rappresenta l'immagine, i trend principali, i valori di picco, le conclusioni logiche visibili e il contesto finanziario/tecnico.",
  "axes": { "x": "label or empty", "y": "label or empty" },
  "legend": ["series names as visible"],
  "data_points": [
    { "series": "series", "x": "x category/date", "y": "y value", "unit": "unit" }
  ],
  "key_points": ["fact 1", "fact 2"],
  "numbers": [
    { "label": "as visible", "value": "value", "unit": "unit", "period": "period" }
  ],
  "entities": [ { "type": "Company|Index|Metric|Instrument", "label": "label" } ],
  "unreadable_parts": []
}

Rules:
- summary_it: MUST be in Italian and highly descriptive for semantic search.
- Never invent numbers. If kind="other", summary_it should explain why it was ignored.
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

'''
KG_PROMPT = """
You are a High-Fidelity Knowledge Engineer. Extract a HYPER-DETAILED Knowledge Graph.

Extraction Rules:
1. NODES: Extract EVERY person, organization, date, specific monetary value, country, and financial concept. 
   - Be exhaustive. Do not summarize.
   - If the text mentions '1971' or 'Richard Nixon', these MUST be separate nodes.
2. EDGES: Extract every single action, causal link, or historical fact.
   - Use specific relations like 'ABANDONED', 'VENDS_TREASURIES', 'ESTABLISHED_SYSTEM'.

Return ONLY valid JSON. Focus on atomic granularity.
"""


KG_PROMPT = """
You are a High-Fidelity Financial Knowledge Engineer. Your task is to extract a hyper-detailed and atomic Knowledge Graph from the provided text.

Extraction Rules:
1. NODES: Extract every specific Person, Organization, Date, Monetary Value, Country, and Financial Concept. 
   - Every distinct entity mentioned must be its own node. 
   - Use clean, concise labels (e.g., "China" instead of "The People's Republic of China").
2. EDGES: Map every individual action, causal link, or historical fact connecting two nodes.
   - Use uppercase English infinitive verbs (e.g., ABANDON, SELL_TREASURY, IMPACT, ESTABLISH, FORECAST).
   - Be granular: if the text mentions multiple actions, create a separate edge for each.
3. CONSTRAINTS: Do not summarize or skip entities. Do not provide any conversational text or explanations.

Return ONLY a valid JSON object containing "nodes" and "edges".
"""
'''
KG_PROMPT = """
You are a High-Fidelity Financial Knowledge Engineer. Your task is to extract a hyper-detailed and atomic Knowledge Graph from the provided text.

Extraction Rules:
1. NODES: Extract every specific Person, Organization, Date, Monetary Value, Country, and Financial Concept. 
   - Every distinct entity mentioned must be its own node. 
   - For each entity, populate the 'properties' field with relevant attributes (sector, ticker, value, unit, timeframe, impact_level).
   - Use clean, concise labels (e.g., "China" instead of "The People's Republic of China").
2. EDGES: Map every individual action, causal link, or historical fact connecting two nodes.
   - Use uppercase English infinitive verbs (e.g., ABANDON, SELL_TREASURY, IMPACT, ESTABLISH).
3. CONSTRAINTS: Return ONLY a valid JSON object containing "nodes" and "edges". Do not provide explanations.
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
    env["OLLAMA_NUM_PARALLEL"] = num_parallel
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    
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
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1" # Ottimizza la VRAM
    
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

def safe_normalize_text(text: str) -> str:
    """
    Pulisce il testo per l'estrazione KG senza distruggere la struttura matematica.
    """
    if not text:
        return ""
    
    # Se il chunk contiene già simboli matematici chiari, evitiamo sostituzioni rischiose
    is_math = bool(MATH_CANDIDATE_PAT.search(text))
    
    # Legature tipografiche sicure (comuni nei testi ma quasi mai in LaTeX)
    safe_replacements = {
        'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
    }
    
    # Legature rischiose (es. Ʃ può essere Sigma o 'tt')
    # Le attiviamo solo se NON è un chunk matematico
    risky_replacements = {
        'Ʃ': 'tt', 'ƫ': 'tt', 'Ʃo': 'tto', 'Ʃi': 'tti', 'Ʃa': 'tta', 'Ʃe': 'tte'
    }
    
    clean_text = text
    for old, new in safe_replacements.items():
        clean_text = clean_text.replace(old, new)
        
    if not is_math:
        for old, new in risky_replacements.items():
            clean_text = clean_text.replace(old, new)
            
    return clean_text

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
      r_out.count = coalesce(r_out.count, 0) + 1,
      r_out.raw_relations =
          apoc.coll.toSet(
              coalesce(r_out.raw_relations, []) +
              [coalesce(rel.props.raw_relation, rel.relation)]
          ),
      r_out.canon_objects =
          apoc.coll.toSet(
              coalesce(r_out.canon_objects, []) +
              CASE
                WHEN rel.props.canon_object IS NULL OR rel.props.canon_object = ""
                THEN []
                ELSE [rel.props.canon_object]
              END
          )

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
            # FORMATO CORRETTO: l'immagine deve stare nel dizionario del messaggio
            resp = chat(
                model=model,
                messages=[{
                    'role': 'user', 
                    'content': prompt,
                    'images': [image_bytes] # <--- Spostato qui
                }],
                options={"temperature": 0.1, "num_predict": max_tokens}
            )
            return resp.get('message', {}).get('content', "")
        except Exception as e:
            print(f"   ⚠️ Errore Vision (Ollama): {e}")
            time.sleep(0.5)
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
    """
    Estrae info da grafici/tabelle/diagrammi con Vision (Ollama multimodal).
    - Usa downscale/compress
    - Cache in-memory per evitare chiamate ripetute
    - Validazione minima ma robusta
    """
    vbytes = _downscale_and_compress_for_vision(img_bytes)
    if not vbytes:
        return None

    key = sha256_hex(vbytes) + "::chart"
    cached = _vision_cache_get(key)
    if cached:
        return cached

    raw = llm_chat_multimodal(CHART_VISION_PROMPT, vbytes, VISION_MODEL_NAME, max_tokens=900)
    js = safe_json_extract(raw)

    # Validazione minima: basta avere kind o summary_it
    if not isinstance(js, dict):
        return None

    # Normalizza campi essenziali
    if not (js.get("kind") or js.get("summary_it")):
        return None

    js.setdefault("kind", "other")
    js.setdefault("title", "")
    js.setdefault("source", "")
    js.setdefault("timeframe", "")
    js.setdefault("summary_it", js.get("summary_it", "") or "")
    js.setdefault("axes", {"x": "", "y": ""})
    js.setdefault("legend", [])
    js.setdefault("data_points", [])
    js.setdefault("key_points", [])
    js.setdefault("numbers", [])
    js.setdefault("entities", [])
    js.setdefault("unreadable_parts", [])

    _vision_cache_put(key, js)
    return js


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
    # Usiamo summary_it come contenuto principale per il RAG
    summary = chart_json.get("summary_it") or chart_json.get("detailed_description") or ""
    kind = chart_json.get("kind", "immagine")
    title = chart_json.get("title", "")
    
    lines = [f"--- CONTENUTO VISUALE ({kind}) - Pagina {page_no} ---"]
    if title: lines.append(f"Titolo: {title}")
    if summary: lines.append(f"Analisi Semantica: {summary}")
    
    # Aggiunta dati numerici per precisione RAG
    nums = chart_json.get("numbers", [])
    if nums:
        lines.append("Dati chiave:")
        for n in nums[:8]:
            lines.append(f" - {n.get('label')}: {n.get('value')} {n.get('unit','')}")
            
    return normalize_ws("\n".join(lines))

def build_formula_semantic_chunk(page_no: int, formulas_json: Dict[str, Any]) -> str:
    lines = [f"--- EQUAZIONE MATEMATICA - Pagina {page_no} ---"]
    for f in (formulas_json.get("formulas") or []):
        latex = f.get("latex", "")
        meaning = f.get("meaning_it", "") # <--- Chiave aggiornata
        if latex: lines.append(f"Formula LaTeX: ${latex}$")
        if meaning: lines.append(f"Spiegazione: {meaning}")
        
        vars = f.get("variables", [])
        if vars:
            v_list = [f"{v.get('name')}={v.get('meaning')}" for v in vars]
            lines.append(f"Parametri: {', '.join(v_list)}")
            
    return normalize_ws("\n".join(lines))

# =========================
# KG extraction (LLM) - SOLO dove serve
# =========================
def llm_extract_kg(
    filename: str,
    page_no: int,
    text: str,
    model_name: str = LLM_MODEL_NAME
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Estrae il Knowledge Graph con opzioni ottimizzate per GPU 16GB (Pascal P5000).
    - num_predict dinamico (riduce latenza sui testi brevi)
    - num_ctx controllato
    - input tagliato a 6000 chars (coerente con prompt)
    """
    t0 = time.perf_counter()

    clean_text = (text or "").strip()
    if not clean_text:
        return [], []

    nchars = len(clean_text)
    dynamic_num_predict = 1200 if nchars < 2500 else 2000  # <-- ora viene usato davvero

    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": KG_PROMPT},
                {"role": "user", "content": clean_text[:6000]},
            ],
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": int(dynamic_num_predict),
                "num_ctx": 3072,
                "num_thread": 8,
            },
        )
        raw = resp["message"]["content"]
    except Exception as e:
        print(f"   ⚠️ Ollama Error: {e}")
        return [], []

    dt = time.perf_counter() - t0
    print(f"   ⏱️ {filename} | KG ({model_name}) p{page_no}: {dt:.2f}s | predict={dynamic_num_predict}")

    js = safe_json_extract(raw)
    return _sanitize_graph(js) if js else ([], [])


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

# =========================
# PDF extraction (FAST)
# =========================
def extract_pdf_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """
    VERSIONE FINALE INGESTION:
    - Independent blocks per Formule e Immagini
    - NoneType protection per Vision AI
    - Symbolic Garbage Filter per fitz
    - LaTeX/Semantic splitting
    - Fix 3: fallback full-page solo quando serve davvero
    """
    out_chunks: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)
    filename = os.path.basename(file_path)

    try:
        total_pages = len(doc)
        print(f"   🔍 Analisi PDF: {total_pages} pagine | Vision DPI: {VISION_DPI}")

        def process_page_worker(p_idx):
            p_no = p_idx + 1
            p_obj = doc[p_idx]
            
            # AGGIORNA STATS SUBITO (per ogni pagina)
            with _VISION_STATS_LOCK:
                VISION_STATS["pages_total"] += 1
            
            local_results = []

            # Testo pagina
            p_text = normalize_ws(p_obj.get_text("text"))
            s_hint = find_section_hint(p_text)

            # Immagini embedded
            image_info = p_obj.get_image_info(hashes=True)
            has_imgs = len(image_info) > 0

            # Trigger logic
            # --- TRIGGER LOGIC AGGIORNATA ---
            is_struct = is_structural_page(p_text)
            
            is_math_page = bool(MATH_CANDIDATE_PAT.search(p_text))
            is_visual_page = bool(CHART_CANDIDATE_PAT.search(p_text))
            is_low_text_page = len(p_text.strip()) < 250 

            # ATTIVAZIONE VISION: solo se NON è una pagina di indice/servizio
            do_f = PDF_VISION_ENABLED and (is_math_page or is_low_text_page) and not is_struct
            do_c = PDF_VISION_ENABLED and (is_visual_page or has_imgs or is_low_text_page) and not is_struct
            
            # 1) Chunking testuale (con filtro garbage)
            if p_text.strip():
                for ch in split_paragraphs(p_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
                    if is_garbage_text(ch):
                        continue

                    local_results.append({
                        "text_raw": ch,
                        "text_sem": f"Documento: {filename} | Sezione: {s_hint}\n{ch}",
                        "page_no": p_no,
                        "toon_type": "text",
                        "section_hint": s_hint
                    })

            # 2) Vision & Storage
            if do_f or do_c:
                try:
                    t_conn = pg_get_conn()
                    try:
                        with t_conn.cursor() as t_cur:

                            # A) FORMULE
                            if do_f:
                                full_jpg = render_full_page_jpeg(p_obj, dpi=FULLPAGE_DPI)  # 120 DPI
                                f_js_raw = extract_formulas_via_vision(full_jpg)
                                f_js = f_js_raw or {"has_formulas": False, "formulas": []}

                                if f_js.get("has_formulas") or is_low_text_page:
                                    img_id = pg_save_image(
                                        log_id, full_jpg, "image/jpeg",
                                        f"Formula p.{p_no}", cur=t_cur
                                    )
                                    f_list = f_js.get("formulas", [])
                                    raw_latex = "\n".join([f.get("latex", "") for f in f_list if f.get("latex")])
                                    sem_desc = (
                                        build_formula_semantic_chunk(p_no, f_js)
                                        if f_js_raw else
                                        f"Formula a pagina {p_no}"
                                    )

                                    local_results.append({
                                        "text_raw": raw_latex if raw_latex else f"Formule pagina {p_no}",
                                        "text_sem": sem_desc,
                                        "page_no": p_no,
                                        "toon_type": "formula",
                                        "section_hint": s_hint,
                                        "image_id": img_id,
                                        "metadata_override": f_js
                                    })

                            # B) IMMAGINI / GRAFICI
                            if do_c:
                                found_valid_crop = False

                                # 1) Analisi crop embedded (se presenti)
                                if has_imgs:
                                    with _VISION_STATS_LOCK:
                                        VISION_STATS["pages_total"] += 1
                                        if has_imgs:
                                            VISION_STATS["pages_with_imgs"] += 1                                   
                                    for img_meta in image_info[:PDF_MAX_IMAGES_PER_PAGE]:
                                        bbox = img_meta["bbox"]
                                        crop_pix = p_obj.get_pixmap(clip=bbox, dpi=CROP_DPI)  # 160 DPI
                                        crop_jpg = crop_pix.tobytes("jpg")

                                        if len(crop_jpg) < MIN_IMAGE_BYTES:
                                            continue

                                        c_js_raw = extract_chart_via_vision(crop_jpg)
                                        c_js = c_js_raw or {}

                                        img_id = pg_save_image(
                                            log_id, crop_jpg, "image/jpeg",
                                            f"Crop p.{p_no}", cur=t_cur
                                        )

                                        raw_info = f"Immagine: {c_js.get('kind', 'visual')} - {c_js.get('title', '')}"
                                        sem_desc = build_chart_semantic_chunk(p_no, c_js, prefix="IMAGE")

                                        local_results.append({
                                            "text_raw": raw_info,
                                            "text_sem": sem_desc,
                                            "page_no": p_no,
                                            "toon_type": "immagine",
                                            "image_id": img_id,
                                            "metadata_override": c_js
                                        })

                                        # Se il crop è almeno “qualificato”, consideralo valido
                                        if c_js.get("kind") and c_js.get("kind") != "other":
                                            found_valid_crop = True
                                            with _VISION_STATS_LOCK:
                                                VISION_STATS["pages_crop_only"] += 1                                            
                                        else:
                                            # anche se kind non è perfetto, se c’è summary, spesso basta
                                            if c_js.get("summary_it"):
                                                found_valid_crop = True

                                # 2) FIX 3 — Fallback Full Page SOLO se serve davvero
                                # - se non ho crop validi
                                # - oppure pagina low-text (scansione/figura piena)
                                # - oppure pagina visual ma SENZA embedded images (grafici vettoriali)
                                need_fullpage = (
                                    (not found_valid_crop) or
                                    is_low_text_page or
                                    (is_visual_page and not has_imgs)
                                )

                                if need_fullpage:
                                    if need_fullpage:
                                        with _VISION_STATS_LOCK:
                                            VISION_STATS["pages_fullpage"] += 1
                                    full_jpg = render_full_page_jpeg(p_obj, dpi=VISION_DPI)  # es. 150/160
                                    c_js_raw = extract_chart_via_vision(full_jpg)
                                    c_js = c_js_raw or {}

                                    # Salva solo se non è "other" (evita rumore)
                                    if c_js.get("kind") != "other":
                                        img_id = pg_save_image(
                                            log_id, full_jpg, "image/jpeg",
                                            f"Visual Full p.{p_no}", cur=t_cur
                                        )

                                        raw_info = f"Grafico/Pagina: {c_js.get('title', 'Senza titolo')} ({c_js.get('kind', 'visual')})"
                                        sem_desc = build_chart_semantic_chunk(p_no, c_js, prefix="VISUAL")

                                        local_results.append({
                                            "text_raw": raw_info,
                                            "text_sem": sem_desc,
                                            "page_no": p_no,
                                            "toon_type": "immagine",
                                            "image_id": img_id,
                                            "metadata_override": c_js
                                        })

                        t_conn.commit()
                    finally:
                        pg_put_conn(t_conn)

                except Exception as e:
                    print(f"   ⚠️ Vision DB Error p.{p_no}: {e}")
                    
            return local_results

        # Parallelizza pagine
        with ThreadPoolExecutor(max_workers=VISION_PARALLEL_WORKERS) as p_executor:
            futures = [p_executor.submit(process_page_worker, i) for i in range(total_pages)]
            
            for fut in as_completed(futures):
                try:
                    # Recupera i risultati della singola pagina
                    res = fut.result()
                    if res:
                        out_chunks.extend(res)
                    
                    # LOG DI PROGRESSO IMMEDIATO
                    # VISION_STATS['pages_total'] viene incrementato dentro process_page_worker
                    done = VISION_STATS['pages_total']
                    print(f"   📄 Pagina {done}/{total_pages} completata (Vision & Text)...")
                
                except Exception as e:
                    print(f"   ⚠️ Errore durante l'elaborazione di una pagina: {e}")

        # --- REPORT FINALE VISION ---
        # Viene stampato una sola volta quando tutti i thread hanno terminato
        print(
            f"\n   📊 Vision Report Finale | Pagine Totali: {VISION_STATS['pages_total']} | "
            f"Con Immagini: {VISION_STATS['pages_with_imgs']} | "
            f"Solo Crop: {VISION_STATS['pages_crop_only']} | "
            f"Full-page: {VISION_STATS['pages_fullpage']}"
        )

        # Ordinamento e aggiunta contesto
        out_chunks.sort(key=lambda x: (x["page_no"]))
        return add_context_windows(out_chunks)

    finally:
        doc.close()


def extract_file_chunks(file_path: str, log_id: int) -> List[Dict[str, Any]]:
    """Funzione dispatcher che smista il file all'estrattore corretto."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_chunks(file_path, log_id)
    return []


# =========================
# FILE DISPATCH (PDF only here)
# =========================
def process_single_file(file_path: str, doc_type: str, doc_meta: dict):
    """
    Versione 2.5 - FULL FIX:
    - Sincronizzazione totale toon_type (immagine/formula)
    - Fix errore chiavi duplicate in Postgres
    - Bypass filtro lunghezza per oggetti visuali
    - Iniezione metadati deterministici in Neo4j
    """
    _ = get_qdrant_client()
    _ = get_embedder()
    ensure_qdrant_collection()

    filename = os.path.basename(file_path)
    doc_meta = doc_meta or {}

    # Metadati deterministici dal dispatcher
    tier = doc_meta.get("tier", "B")
    ontology = (doc_meta.get("ontology") or DEFAULT_ONTOLOGY).strip().lower()
    effective_date = doc_meta.get("effective_date")
    model_for_kg = LLM_MODEL_NAME 

    start_time = time.time()
    print(f"   ⚙️ Engine Start: {filename}")

    # Generazione ID documento per l'idempotenza
    doc_sha = ""
    try: doc_sha = sha256_file(file_path)
    except Exception: pass
    doc_id = f"{filename}::{doc_sha[:16] if doc_sha else 'nohash'}"
    
    # Inizio log in Postgres
    log_id = pg_start_log(filename, doc_type)

    # 1. Estrazione Chunk (Multimodale)
    t_extract = time.time()
    chunks = extract_file_chunks(file_path, log_id) 
    log_phase(filename, "extract_file_chunks", _ms(t_extract))

    try:
        # FIX: Non scartare mai immagini o formule anche se la descrizione è breve
        valid_chunks = [
            c for c in chunks 
            if c.get("toon_type") in ("immagine", "formula") or len(c.get("text_sem", "")) >= MIN_CHUNK_LEN
        ]
        
        if not valid_chunks:
            pg_close_log(log_id, "SKIPPED_EMPTY", 0, 0)
            print(f"   ⚠️ {filename} saltato: nessun contenuto valido trovato.")
            return

        print(f"   🚀 Chunks validi: {len(valid_chunks)}. Elaborazione batch...")

        qdrant_points, pg_rows, neo4j_rows = [], [], []
        flush_bg_executor = ThreadPoolExecutor(max_workers=1)

        # 2. Elaborazione Batch (Embedding + KG)
        for i in range(0, len(valid_chunks), EMBED_BATCH_SIZE):
            batch = valid_chunks[i:i + EMBED_BATCH_SIZE]
            texts = [b["text_sem"][:QDRANT_TEXT_MAX_CHARS] for b in batch]

            # A. Generazione Vettori (Eseguita su CPU per risparmiare VRAM)
            vecs = get_embedder().encode(texts, normalize_embeddings=True)


            # B. Knowledge Graph Batching (Ollama) - LOGICA "CACCIATORE DI VALORE"
            batch_kg_results = {}
            if KG_ENABLED:
                t_kg_start = time.time()
                pages_map = {}
                for idx, ch in enumerate(batch):
                    p = ch.get("page_no", 1)
                    pages_map.setdefault(p, []).append((idx, ch))

# B. Knowledge Graph Batching (Ollama) - LOGICA SELETTIVA "GEMMA-3 GATEKEEPER"
            batch_kg_results = {}
            if KG_ENABLED:
                t_kg_start = time.time()
                pages_map = {}
                for idx, ch in enumerate(batch):
                    p = ch.get("page_no", 1)
                    pages_map.setdefault(p, []).append((idx, ch))

                futures_kg = {}
                for p_no, indexed_chunks in pages_map.items():
                    combined_text = "\n ".join([c[1].get("text_sem", "") for c in indexed_chunks])
                    
                    # 1. Filtro Simbolico (Regex)
                    unique_keys = count_unique_keywords(combined_text)
                    is_struct = is_structural_page(combined_text)

                    # 2. Regole di Selettività Gerarchica
                    if is_struct:
                        should_extract_kg = False # Mai estrarre grafo da indici
                    elif tier == "A":
                        # Per la Metodologia siamo inclusivi: basta 1 concetto e testo minimo
                        should_extract_kg = len(combined_text) > 400 and unique_keys >= 1
                    else:
                        # Per News e Reference (Tier B/C) siamo severi
                        should_extract_kg = len(combined_text) > 600 and unique_keys >= 3

                    # 3. Decisione finale tramite AI Gatekeeper (Solo per News/Ref)
                    if should_extract_kg and tier != "A":
                        # Chiediamo a Gemma 3 se questa pagina merita lo sforzo del KG
                        should_extract_kg = ai_gatekeeper_decision(combined_text)

                    if should_extract_kg:
                        # Lancio tramite l'executor seriale (1 worker)
                        fut = kg_executor.submit(llm_extract_kg, filename, p_no, combined_text, model_for_kg)
                        futures_kg[fut] = indexed_chunks
                
                # Raccolta Risultati (Codice esistente per Neo4j)
                if futures_kg:
                    for fut in as_completed(futures_kg):
                        current_page_chunks = futures_kg[fut]
                        try:
                            n, e = fut.result()
                            if n or e:
                                e = canonicalize_edges_to_verb_object(canonicalize_edges_by_base_presence(e))
                                for original_idx, _ in current_page_chunks:
                                    batch_kg_results[original_idx] = (n, e)
                        except Exception as ex:
                            print(f"   ⚠️ KG Batch Task failed: {ex}")
                
                log_phase(filename, f"kg_page_batching_{len(pages_map)}", _ms(t_kg_start))

            # 3. Costruzione Record (Sincronizzazione DB)
            for j, ch in enumerate(batch):
                c_idx, p_no = i + j, ch.get("page_no", 1)
                
                # Recupero sicuro metadati per la GUI
                t_type = ch.get("toon_type", "text")
                img_id = ch.get("image_id")
                text_s = ch["text_sem"][:QDRANT_TEXT_MAX_CHARS]

                # ID unico deterministico (include t_type per separare testo da immagini)
                c_uuid = deterministic_chunk_id(doc_id, p_no, c_idx, t_type, text_s)

                # Payload completo per Qdrant e colonna JSON di Postgres
                payload = {
                    "tier": tier, 
                    "ontology": ontology, 
                    "filename": filename, 
                    "page": p_no, 
                    "toon_type": t_type,      
                    "image_id": img_id,        
                    "text_sem": text_s, 
                    "effective_date": effective_date
                }
                
                # --- AGGIUNTA SINGOLA AI BUFFER (Fix Duplicati) ---
                qdrant_points.append(models.PointStruct(id=c_uuid, vector=vecs[j].tolist(), payload=payload))
                pg_rows.append((log_id, c_idx, t_type, ch["text_raw"], ch["text_sem"], Json(payload), c_uuid))  
                                         
                if NEO4J_ENABLED:
                    kg_nodes, kg_edges = batch_kg_results.get(j, ([], []))
                    
                    # Iniezione metadati deterministici nei nodi Entity
                    for node in kg_nodes:
                        node["props"].update({
                            "source_tier": tier, 
                            "ontology": ontology,
                            "document_date": effective_date, 
                            "source_file": filename
                        })

                    neo4j_rows.append({
                        "doc_id": doc_id, 
                        "filename": filename, 
                        "doc_type": doc_type, 
                        "log_id": log_id,
                        "chunk_id": c_uuid, 
                        "chunk_index": c_idx, 
                        "toon_type": t_type, 
                        "page_no": p_no,
                        "section_hint": ch.get("section_hint", ""), 
                        "text_sem": ch["text_sem"], 
                        "ontology": ontology,
                        "nodes": kg_nodes, 
                        "edges": kg_edges,
                    })

            # 4. Flush Asincrono dei Dati
            if len(qdrant_points) >= DB_FLUSH_SIZE:
                q_copy, p_copy, n_copy = list(qdrant_points), list(pg_rows), list(neo4j_rows)
                def run_flush(q, p, n):
                    if q: get_qdrant_client().upsert(QDRANT_COLLECTION, points=q)
                    if p: flush_postgres_chunks_batch(p)
                    if n: flush_neo4j_rows_batch(n)
                flush_bg_executor.submit(run_flush, q_copy, p_copy, n_copy)
                qdrant_points.clear(); pg_rows.clear(); neo4j_rows.clear()

        # Attendere completamento flush asincroni
        flush_bg_executor.shutdown(wait=True)
        
        # Flush finale rimasugli
        if qdrant_points: get_qdrant_client().upsert(QDRANT_COLLECTION, points=qdrant_points)
        if pg_rows: flush_postgres_chunks_batch(pg_rows)
        if neo4j_rows: flush_neo4j_rows_batch(neo4j_rows)

        # Chiusura log e spostamento file
        elapsed = int((time.time() - start_time) * 1000)
        pg_close_log(log_id, "COMPLETED", len(valid_chunks), elapsed)
        print(f"✅ Successo: {filename} processato in {elapsed}ms")
        shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))

    except Exception as e:
        print(f"❌ Errore critico durante l'ingestion: {e}")
        pg_close_log(log_id, "FAILED", 0, 0, str(e))
        shutil.move(file_path, os.path.join(FAILED_DIR, filename))



def main():
    # 1. Preparazione cartelle
    os.makedirs(INBOX_DIR, exist_ok=True)
    ensure_inbox_structure(INBOX_DIR)
  
    # 2. Reset Totale OLLAMA (Turbo Mode)
    # Questa funzione è l'unica necessaria: chiude le app tray/fantasmi 
    # e applica NUM_PARALLEL=2 per saturare i 16GB di VRAM senza errori.
    if not force_restart_ollama(num_parallel="1"):
        print("❌ Impossibile avviare Ollama in modalità Turbo. L'ingestion sarà lenta.")
    #ensure_ollama_parallel(num_parallel="2")
       
    print("=== Ingestion Engine v2.3 (FAST + Formulas deterministic + Charts Vision) ===")

    supported = {".pdf"}

    # 3. Raccogli i file supportati
    pdf_files = []
    for root, _, files in os.walk(INBOX_DIR):
        for fname in files:
            if fname.lower().endswith(".meta.json"):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue

            pdf_files.append((root, os.path.join(root, fname)))

    # 4. Early-exit
    if not pdf_files:
        print("✅ INBOX vuota: nessuna ingestion eseguita.")
        return

    # 5. Processa i file
    for root, file_path in pdf_files:
        doc_meta = dispatch_document(file_path, root)
        process_single_file(file_path, "pdf", doc_meta)


if __name__ == "__main__":
    main()
