import reflex as rx
import torch
import uuid
import os
import re
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from openai import OpenAI

# =========================
# ⚙️ CONFIGURAZIONE
# =========================
PAGE_TITLE = "Financial AI Analyst 📊"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "financial_docs")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password_sicura"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-3-12b")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================
# 📦 DATA MODELS
# =========================
class GraphEntity(BaseModel):
    name: str
    type: str

class SourceItem(BaseModel):
    id: str
    content: str
    filename: str
    page: int = 0
    type: str = "text"
    score: float = 0.0

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceItem] = []

# =========================
# 🧠 BACKEND INIT
# =========================
embedder = None
reranker = None
llm_client = None
qdrant_client_inst = None
neo4j_driver = None

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    qdrant_client_inst = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
except Exception as e:
    print(f"⚠️ Init Error: {e}")

# =========================
# 🔄 STATE MANAGEMENT
# =========================
class State(rx.State):
    messages: List[ChatMessage] = [ChatMessage(id="i", role="assistant", content="Pronto per l'analisi.")]
    input_text: str = ""
    is_processing: bool = False
    vram_free: str = "N/A"

    def set_input_text(self, text: str):
        self.input_text = text

    def dummy_action(self):
        """Azione vuota per sostituire no_op se non disponibile."""
        pass

    def on_load(self):
        if torch.cuda.is_available():
            f, _ = torch.cuda.mem_get_info()
            self.vram_free = f"{f / 1024**3:.1f} GB"
        else:
            self.vram_free = "CPU"

    def clear_chat(self):
        self.messages = [self.messages[0]]

    async def handle_submit(self):
        if not self.input_text.strip() or self.is_processing: return
        user_q = self.input_text
        self.input_text = ""
        self.is_processing = True
        
        self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", content=user_q))
        yield rx.scroll_to("chat_bottom")

        # Retrieval logica (semplificata per stabilità)
        query_vec = embedder.encode(user_q, normalize_embeddings=True).tolist()
        hits = qdrant_client_inst.search(COLLECTION_NAME, query_vec, limit=5, with_payload=True)
        sources = [SourceItem(id=str(h.id), content=h.payload.get("text_sem", ""), filename=h.payload.get("filename", "N/A"), page=h.payload.get("page", 0)) for h in hits]
        
        ctx = "\n\n".join([s.content for s in sources])
        self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="assistant", content="", sources=sources))
        
        try:
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "system", "content": "Sei un analista finanziario."}, {"role": "user", "content": f"Contesto: {ctx}\nDomanda: {user_q}"}],
                stream=True
            )
            full_resp = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_resp += token
                    self.messages[-1].content = full_resp
                    yield rx.scroll_to("chat_bottom")
        except Exception as e:
            self.messages[-1].content = f"Errore: {e}"
        
        self.is_processing = False
        yield rx.scroll_to("chat_bottom")

# =========================
# 🎨 UI COMPONENTS
# =========================
def message_ui(msg: ChatMessage):
    is_bot = (msg.role == "assistant")
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.avatar(fallback=rx.cond(is_bot, "🤖", "👤"), size="1"),
                rx.text(rx.cond(is_bot, "AI Analyst", "Tu"), weight="bold", size="2"),
                spacing="2"
            ),
            # Markdown con colore forzato per evitare eredità strane
            rx.markdown(
                msg.content,
                color=rx.cond(is_bot, "#1a202c", "#ffffff") # Nero scuro (bot) / Bianco (user)
            ),
            rx.cond(
                msg.sources.length() > 0,
                rx.accordion.root(
                    rx.accordion.item(
                        header=rx.text("Fonti consultate", size="1"),
                        content=rx.vstack(
                            rx.foreach(msg.sources, lambda s: rx.card(
                                rx.vstack(
                                    rx.text(f"{s.filename} (Pag. {s.page})", size="1", weight="bold"),
                                    rx.text(s.content[:200] + "...", size="1", color="gray"),
                                    align_items="start"
                                ),
                                size="1", width="100%"
                            )),
                            width="100%", spacing="2"
                        )
                    ),
                    collapsible=True, type="single", width="100%"
                )
            ),
            align_items="start", width="100%"
        ),
        # Colori Sfondo Espliciti: Grigio chiarissimo (Bot) / Indaco scuro (User)
        bg=rx.cond(is_bot, "#f3f4f6", "#4338ca"), 
        # Colore Testo Principale
        color=rx.cond(is_bot, "#1a202c", "#ffffff"),
        
        padding="1em", 
        border_radius="10px", 
        margin_y="0.5em", 
        width="100%", 
        max_width="850px",
        align_self=rx.cond(is_bot, "start", "end"),
        box_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)" # Leggera ombra per stacco
    )

# =========================
# 🏗️ LAYOUT
# =========================
def index():
    return rx.flex(
        # Sidebar
        rx.vstack(
            rx.heading("System", size="3"),
            rx.text(f"VRAM: {State.vram_free}", size="1"),
            rx.divider(),
            rx.heading("Clusters", size="2"),
            rx.badge("Risk Management", color_scheme="red"),
            rx.spacer(),
            rx.button("Clear Chat", on_click=State.clear_chat, color_scheme="red", variant="soft", width="100%"),
            width="260px", height="100vh", padding="1.5em", bg=rx.color("gray", 2)
        ),
        # Chat
        rx.vstack(
            rx.heading(PAGE_TITLE, size="5", padding="1em"),
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(State.messages, message_ui),
                    rx.box(id="chat_bottom", height="50px"),
                    width="100%", max_width="800px", margin="0 auto"
                ),
                height="calc(100vh - 160px)", width="100%"
            ),
            rx.box(
                rx.hstack(
                    rx.input(
                        value=State.input_text, 
                        on_change=State.set_input_text,
                        on_key_down=lambda k: rx.cond(k == "Enter", State.handle_submit(), State.dummy_action()),
                        placeholder="Chiedi...", flex="1"
                    ),
                    rx.button(rx.icon("send"), on_click=State.handle_submit, loading=State.is_processing),
                    width="100%", max_width="800px", padding="1em"
                ),
                width="100%", border_top="1px solid #eee", display="flex", justify_content="center"
            ),
            height="100vh", width="100%", spacing="0", overflow="hidden"
        ),
        width="100%", height="100vh"
    )

app = rx.App()
app.add_page(index, on_load=State.on_load)