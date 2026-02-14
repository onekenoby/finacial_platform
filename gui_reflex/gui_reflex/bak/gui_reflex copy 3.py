import reflex as rx
import torch
import uuid
import os
import time
import re
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
# RAG TIER POLICY
# =========================
RAG_DEFAULT_TIERS = os.getenv("RAG_DEFAULT_TIERS", "A,B")  # default prudente
RAG_NEWS_KEYWORDS = os.getenv(
    "RAG_NEWS_KEYWORDS",
    "news,oggi,ieri,ultima,ultime,rumor,breaking,aggiornamenti,recente,recent"
)


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

# =========================
# 🧠 CARICAMENTO RISORSE
# =========================
print("⏳ Init Backend...")

device_embed = "cuda" if torch.cuda.is_available() else "cpu"
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

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceItem] = []


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
    if any(k in q for k in ["formula", "equazione", "latex", "lift", "support", "confidence", "probabilità", "probability"]):
        return "formula"
    # chart intent
    if any(k in q for k in ["grafico", "chart", "figura", "tabella", "table", "asse", "legend", "legenda", "trend", "candela", "ohlc", "volumi", "heatmap"]):
        return "chart"
    return "text"


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

def allow_tier_c_in_graph_expansion(query_text: str) -> bool:
    # per ora: Tier C nei neighbor SOLO se query news
    return is_news_query(query_text)



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


# =========================
# 🔍 RAG v2 Retrieval
# =========================
def retrieve_v2(query_text: str) -> List[SourceItem]:
    if not embedder or not qdrant_client_inst:
        return [SourceItem(id="error", content="Backend non disponibile", filename="System")]

    intent = detect_intent(query_text)

    # 1) Dense retrieval candidates
    query_vector = embedder.encode(query_text, normalize_embeddings=True).tolist()
    try:
        hits = qdrant_client_inst.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=QDRANT_CANDIDATES,
            with_payload=True,
            query_filter=tier_qdrant_filter(query_text),
        )

    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
        return []

    if not hits:
        return []

    # 2) Build candidate list + intent-aware boost to improve recall for formula/chart
    candidates = []
    cross_inputs = []

    for hit in hits:
        payload = hit.payload or {}
        content = safe_payload_text(payload)
        tier = get_payload_tier(payload)  #helper che hai già aggiunto
        
        if not content:
            continue

        f = str(payload.get("filename", "Unknown"))
        page = get_payload_page(payload)
        ttype = get_payload_type(payload)
        section_hint = get_payload_section(payload)
        image_id = get_payload_image_id(payload)
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
        if intent == "chart" and any(k in content.lower() for k in ["axes:", "legend:", "numbers:", "kind=chart", "kind=table"]):
            boost += 0.06

        # preserve vector score but slightly steer
        tier_delta = tier_score_delta(tier, query_text)
        final_score = base_score + boost + tier_delta

        cand = {
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
        candidates.append(cand)

    if not candidates:
        return []

    # sort by boosted score and keep only rerank candidates
    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:RERANK_CANDIDATES]

    # 3) Cross-encoder reranking (CPU)
    if reranker:
        try:
            # Usiamo solo i primi 500 caratteri per il reranking rapido
            cross_inputs = [[query_text, c["content"][:500]] for c in candidates] 
            rerank_scores = reranker.predict(cross_inputs)
            for i, sc in enumerate(rerank_scores):
                # mix: rerank dominates, keep tiny influence of boosted score
                candidates[i]["rerank_score"] = float(sc)
                candidates[i]["final_score"] = float(sc) + 0.02 * float(candidates[i]["final_score"])
        except Exception as e:
            print(f"⚠️ Reranker error, fallback: {e}")

    # 4) Diversify (avoid 10 chunks from same page)
    top = diversify(candidates, max_per_page=MAX_PER_PAGE, max_per_doc=MAX_PER_DOC, final_k=FINAL_SOURCES)
    
    # If user asks about formulas, ensure at least one formula chunk is present when available
    if intent == "formula":
        formula_candidates = [c for c in candidates if c["type"] in ("formula_page", "page_vision_math")]
        if formula_candidates:
            if not any(t["type"] in ("formula_page", "page_vision_math") for t in top):
                # replace the last one to guarantee a clean formula source
                top = top[:-1] + [formula_candidates[0]]

    

    # 5) Neo4j enrichment: entities + formulas + neighbor chunks (graph-augmented RAG)
    chunk_ids = [t["id"] for t in top]
    entities_map = get_graph_entities(chunk_ids)

    # Attach entities
    sources: List[SourceItem] = []
    for t in top:
        ents = entities_map.get(t["id"], [])
        # dedup
        uniq = []
        seen = set()
        for e in ents:
            if e.name not in seen:
                uniq.append(e)
                seen.add(e.name)

        sources.append(
            SourceItem(
                id=t["id"],
                content=t["content"],
                filename=t["filename"],
                page=t["page"],
                type=t["type"],
                score=float(t.get("final_score", t.get("score", 0.0))),
                graph_context=uniq,
                section_hint=t.get("section_hint", ""),
                image_id=t.get("image_id"),
                tier=t.get("tier", ""),   # ✅ NEW
            )
        )

    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        # add short formula block (for prompt)
        formulas = get_formulas_for_chunks(chunk_ids, limit=GRAPH_MAX_FORMULAS)

        # neighbor chunks (fetch from qdrant by id)
        neighbor_ids = get_neighbor_chunk_ids(chunk_ids, limit=GRAPH_MAX_NEIGHBOR_CHUNKS)
        neighbor_sources = fetch_chunks_from_qdrant_by_ids(neighbor_ids)

        # ✅ Tier-aware filter: niente Tier C nei neighbor se non è news query
        if not allow_tier_c_in_graph_expansion(query_text):
            neighbor_sources = [
                s for s in neighbor_sources
                if (s.tier or "").upper() != "C"
                and (s.tier or "").upper() != "TIER_C_NEWS"
                and not (s.tier or "").upper().endswith("_C_NEWS")
            ]

        # (opzionale ma consigliato) evita di duplicare chunk già selezionati
        already_ids = {s.id for s in sources}
        neighbor_sources = [s for s in neighbor_sources if s.id not in already_ids]

        # tagga come neighbor (se stai usando type per distinguerli)
        for s in neighbor_sources:
            # Se vuoi mantenere il tipo originale puoi omettere
            s.type = "graph_neighbor"

        # aggiungi in coda (non davanti) per evitare che “sovrastino” i top source
        sources.extend(neighbor_sources)


        # Attach as extra sources (but mark type)
        # We keep them small to avoid prompt bloat
        for ns in neighbor_sources:
            ns.type = "graph_neighbor"
            ns.score = 0.0
            sources.append(ns)

        # store formulas as an extra "virtual source" for the model
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

    # final trim (keep prompt bounded)
    # prefer primary sources over neighbors if too many
    primary = [s for s in sources if s.type not in ("graph_neighbor",)]
    neighbors = [s for s in sources if s.type == "graph_neighbor"]
    formulas_src = [s for s in sources if s.type == "graph_formulas"]

    final_sources = primary[:FINAL_SOURCES] + neighbors[:max(0, 3)] + formulas_src[:1]
    return final_sources


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
    """Finance-grade instructions, but fluent and without leaking technical IDs."""
    base = (
        "Sei un analista finanziario e quantitativo. Rispondi in modo naturale e scorrevole, come un umano.\n"
        "Regole:\n"
        "1) Usa SOLO le informazioni presenti nel contesto.\n"
        "2) Quando citi una fonte, usa esclusivamente il formato [numero] (es: [1]).\n"
        "3) NON scrivere mai codici tecnici, id, UUID, né stringhe tipo 'id=...'.\n"
        "4) Se il contesto non basta, dillo chiaramente e spiega cosa manca.\n"
        "5) Per formule: riporta la formula in LaTeX tra $...$ e spiega le variabili solo se nel contesto.\n"
        "6) Se nel testo compaiono simboli strani/illeggibili (□ o simili), ignorali e preferisci i blocchi con LaTeX.\n"
        "\n"
        "Formato:\n"
        "- Risposta (discorsiva)\n"
        "- Formule (se presenti)\n"
        "- Fonti: elenco finale tipo '[1] Doc pag X; [2] ...'\n"
    )
    if intent == "formula":
        base += "\nNota: la domanda riguarda formule. Dai priorità ai blocchi con LaTeX ($...$) e/o 'FORMULE'.\n"
    elif intent == "chart":
        base += "\nNota: la domanda riguarda grafici/tabelle. Descrivi assi, legenda, trend e numeri visibili.\n"
    return base

def tier_guardrail_instructions(query_text: str) -> str:
    news = is_news_query(query_text)
    return (
        "GUARDRAILS (TIER-FIRST, FINANCE-GRADE):\n"
        "1) Use Tier A sources as the primary authority for definitions, mechanisms, and explanations.\n"
        "2) Use Tier B sources for examples, applications, and case studies.\n"
        "3) Use Tier C sources ONLY as time-sensitive context (news/rumors/headlines). "
        "If you cite Tier C, ALWAYS include the source date (effective_date or ingestion timestamp).\n"
        "4) NEVER make claims that are not supported by the provided sources. "
        "If the sources do not contain the answer, say you do not have enough information.\n"
        f"5) {'This is a news/recency query: you may rely on Tier C for what happened, but still ground explanations in Tier A/B.' if news else 'This is NOT a news query: do not rely on Tier C for core claims; Tier A/B must dominate.'}\n"
        "6) If sources conflict, explicitly state the conflict and prefer Tier A over B over C.\n"
    )

def has_sufficient_ab_sources(sources: list) -> bool:
    # considera solo fonti reali (escludi graph_formulas se vuoi)
    tiers = [(getattr(s, "tier", "") or "").upper() for s in sources]
    ab = sum(1 for t in tiers if (t == "A" or t.endswith("_A_METHODOLOGY") or t == "TIER_A_METHODOLOGY"
                                 or t == "B" or t.endswith("_B_REFERENCE") or t == "TIER_B_REFERENCE"))
    # soglia minima: almeno 1 fonte A/B per query non-news
    return ab >= 1


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
    rx.text(short_text(s.content), size="1", color="gray", truncate=True),
    return s[:n] + ("..." if len(s) > n else "")


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

    # Sidebar info
    vram_info: str = "N/A"
    vram_free: str = "N/A"
    backend_status: str = "OK"

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

        try:
            self.refresh_gpu()
            self.refresh_backend_status()

            self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", content=user_query))
            yield rx.scroll_to("chat_bottom")

            intent = detect_intent(user_query)
            sources = retrieve_v2(user_query)

            context_str = build_context_block(sources, max_chars=MAX_CONTEXT_CHARS)
            system_instructions = build_system_instructions(intent)

            final_prompt_content = (
                f"### ISTRUZIONI ###\n{system_instructions}\n\n"
                f"### CONTESTO DOCUMENTALE ###\n{context_str}\n\n"
                f"### DOMANDA UTENTE ###\n{user_query}"
            )

            messages_payload = build_alternating_history(self.messages, MEMORY_LIMIT)
            if messages_payload and messages_payload[-1]["role"] == "user":
                messages_payload.pop()
            messages_payload.append({"role": "user", "content": final_prompt_content})

            self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="assistant", content="", sources=sources))
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
                    is_bot & (msg.sources.length() > 0),
                    rx.badge(f"{msg.sources.length()} Fonti", color_scheme="green", variant="surface"),
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

            rx.cond(
                is_bot & (msg.sources.length() > 0),
                rx.accordion.root(
                    rx.accordion.item(
                        header=rx.hstack(
                            rx.icon("book-open", size=14),
                            rx.text("Fonti Analizzate", size="1"),
                            align_items="center",
                            spacing="2",
                        ),
                        content=rx.vstack(
                            rx.foreach(
                                msg.sources,
                                lambda s: rx.card(
                                    rx.vstack(
                                        rx.hstack(
                                            rx.icon("file-text", size=14),
                                            rx.text(s.filename, weight="bold", size="1"),
                                            rx.spacer(),
                                            source_badge(f"Pag {s.page}", "gray", "hash"),
                                            source_badge(s.type, "blue", "tag"),
                                            rx.cond(
                                                    (s.tier != "") & (s.type != "graph_formulas"),
                                                    source_badge(s.tier, "orange", "layers"),
                                             ),
                                            width="100%",
                                            align_items="center",
                                        ),
                                        # ✅ NO short_text(...) su Var: usa rx.cond + slicing
                                        rx.text(
                                            rx.cond(
                                                s.content.length() > 320,
                                                s.content[:320] + "...",
                                                s.content,
                                            ),
                                            size="1",
                                            color="gray",
                                            truncate=True,
                                        ),
                                        rx.cond(
                                            s.graph_context.length() > 0,
                                            rx.flex(
                                                rx.text("Graph: ", size="1", weight="bold", color="gray"),
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
                                        ),
                                        align_items="start",
                                        spacing="1",
                                    ),
                                    size="1",
                                    width="100%",
                                ),
                            ),
                            width="100%",
                            spacing="2",
                        ),
                    ),
                    collapsible=True,
                    type="single",
                    width="100%",
                    margin_top="1em",
                ),
            ),
            align_items="start",
            spacing="3",
            width="100%",
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
