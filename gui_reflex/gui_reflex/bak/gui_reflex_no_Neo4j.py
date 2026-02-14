import reflex as rx
import torch
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel  # <--- NUOVO IMPORT STANDARD
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from openai import OpenAI

# --- CONFIGURAZIONE ---
PAGE_TITLE = "Financial RAG 🧠 (Pro)"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
MODEL_RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL_ID = "gemma-3-12b" 
MEMORY_LIMIT = 4 

# --- CARICAMENTO RISORSE ---
print("⏳ Caricamento modelli...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device=device)
    reranker = CrossEncoder(MODEL_RERANKER_NAME, device=device)
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)
    print("✅ Modelli caricati.")
except Exception as e:
    print(f"❌ Errore modelli: {e}")
    embedder = None
    reranker = None
    llm_client = None

# --- MODELLI DI DATI (Aggiornati a Pydantic) ---
# Sostituito rx.Base con BaseModel per eliminare il DeprecationWarning
class SourceItem(BaseModel):
    content: str = ""
    filename: str = "Unknown"
    type: str = "text"
    rerank_score: float = 0.0

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceItem] = []

# --- LOGICA RETRIEVAL ---
def retrieve_optimized_logic(query_text: str) -> List[Dict]:
    if not embedder: return []
    
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        search_result = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=35)
    except:
        try:
            search_result = client.search_points(collection_name=COLLECTION_NAME, vector=query_vector, limit=35, with_payload=True)
        except:
            return []

    if not search_result: return []

    hits_data = []
    cross_inp = []
    
    for hit in search_result:
        payload = hit.payload if hasattr(hit, 'payload') else hit.get('payload', {})
        content = payload.get('content_semantic', payload.get('text_preview', ''))
        filename = payload.get('filename', 'Sconosciuto')
        toon_type = payload.get('toon_type', 'text')
        
        hits_data.append({"content": content, "filename": filename, "type": toon_type})
        cross_inp.append([query_text, content])

    if not cross_inp: return []

    cross_scores = reranker.predict(cross_inp)
    for idx in range(len(cross_scores)):
        hits_data[idx]['rerank_score'] = float(cross_scores[idx])

    hits_sorted = sorted(hits_data, key=lambda x: x['rerank_score'], reverse=True)
    return hits_sorted[:10]

# --- STATE ---
class State(rx.State):
    messages: List[ChatMessage] = []
    memory_buffer: List[Dict[str, str]] = []
    input_text: str = ""
    is_processing: bool = False
    
    # Monitoraggio VRAM
    vram_free_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_percent: float = 0.0
    using_cpu: bool = False

    def on_load(self):
        if torch.cuda.is_available():
            try:
                info = torch.cuda.mem_get_info()
                props = torch.cuda.get_device_properties(0)
                self.vram_free_gb = info[0] / 1024**3
                self.vram_total_gb = props.total_memory / 1024**3
                if self.vram_total_gb > 0:
                    self.vram_percent = 1 - (self.vram_free_gb / self.vram_total_gb)
            except:
                self.using_cpu = True
        else:
            self.using_cpu = True

    def clear_history(self):
        self.messages = []
        self.memory_buffer = []

    def set_input_text(self, text: str):
        self.input_text = text

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    async def handle_submit(self):
        if not self.input_text:
            return

        user_query = self.input_text
        self.input_text = ""
        self.is_processing = True
        
        # User Message
        self.messages.append(
            ChatMessage(id=self._new_id(), role="user", content=user_query, sources=[])
        )
        yield rx.scroll_to("chat_bottom")

        # Retrieval
        raw_sources = retrieve_optimized_logic(user_query)
        structured_sources = [SourceItem(**s) for s in raw_sources]
        
        context_str = "\n\n".join([
            f"--- {s.filename} ---\n{s.content}" for s in structured_sources
        ])

        #system_prompt = "Sei un assistente finanziario esperto. Usa il contesto per rispondere."
        
        system_prompt = (
            "Sei un analista finanziario senior esperto in AI e trading quantitativo. "
            "Il tuo compito è rispondere alle domande basandoti ESCLUSIVAMENTE sui documenti forniti. "
            "\n\nGUIDA ALL'ANALISI:"
            "\n1. DATI VISIVI: I documenti contengono descrizioni di grafici marcate come '--- DESCRIZIONE IMMAGINE ---'. "
            "   Queste sono fonti primarie. Se l'utente chiede dati da un grafico (es. nomi assi, aziende, trend), DEVI usare queste descrizioni."
            "\n2. CODICE: Se cerchi librerie o algoritmi, guarda attentamente i blocchi di testo che contengono codice Python."
            "\n3. SINTESI: Se un concetto è frammentato tra testo e immagini, unisci le informazioni in una risposta coerente."
            "\n4. CITAZIONI: Alla fine di ogni affermazione chiave, cita la fonte tra parentesi quadre, es:."
            "\n5. ONESTÀ: Se l'informazione non c'è, dillo chiaramente, ma controlla bene tutto il contesto prima di arrenderti."
        )
        
        
        
        
        current_msg_payload = f"### CONTESTO ###\n{context_str}\n\n### DOMANDA ###\n{user_query}"

        full_payload = [{"role": "system", "content": system_prompt}] + self.memory_buffer
        full_payload.append({"role": "user", "content": current_msg_payload})

        # Assistant Placeholder
        self.messages.append(
            ChatMessage(id=self._new_id(), role="assistant", content="", sources=structured_sources)
        )
        yield rx.scroll_to("chat_bottom")

        # Streaming
        full_response = ""
        try:
            stream = llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL_ID, messages=full_payload, temperature=0.2, stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    self.messages[-1].content = full_response
                    yield
            
            yield rx.scroll_to("chat_bottom")

        except Exception as e:
            self.messages[-1].content = f"Errore: {str(e)}"
            yield

        # Update Memory
        self.memory_buffer.append({"role": "user", "content": user_query})
        self.memory_buffer.append({"role": "assistant", "content": full_response})
        if len(self.memory_buffer) > MEMORY_LIMIT * 2:
            self.memory_buffer = self.memory_buffer[-(MEMORY_LIMIT * 2):]

        self.is_processing = False

# --- UI ---

def sidebar_content():
    return rx.box(
        rx.vstack(
            rx.heading("⚙️ Monitoraggio", size="4"),
            rx.cond(
                State.using_cpu,
                rx.callout("⚠️ CPU Mode", icon="triangle_alert", color_scheme="yellow"),
                rx.vstack(
                    rx.text(f"VRAM Libera: {State.vram_free_gb.to(float):.2f} GB"),
                    rx.progress(value=(State.vram_percent * 100).to(int), width="100%"),
                    rx.text(f"Totale: {State.vram_total_gb.to(float):.1f} GB", size="1", color="gray"),
                    width="100%"
                )
            ),
            rx.divider(),
            rx.button("🗑️ Reset", on_click=State.clear_history, color_scheme="red", variant="outline", width="100%"),
            spacing="4",
        ),
        padding="2em", width="300px", height="100vh", bg=rx.color("gray", 2), border_right="1px solid #e0e0e0"
    )

def message_box(msg: ChatMessage):
    return rx.box(
        rx.vstack(
            rx.text(
                rx.cond(msg.role == "user", "👤 Tu", "🤖 Assistant"),
                weight="bold",
                color=rx.cond(msg.role == "user", "blue", "green")
            ),
            rx.markdown(msg.content),
            rx.cond(
                (msg.role == "assistant") & (msg.sources.length() > 0),
                rx.accordion.root(
                    rx.accordion.item(
                        header=rx.text("📚 Fonti", size="2"),
                        content=rx.vstack(
                            rx.foreach(
                                msg.sources,
                                lambda s: rx.box(
                                    rx.text(s.filename, weight="bold", size="1"),
                                    rx.text("Score: " + s.rerank_score.to_string(), size="1", color="gray"),
                                    rx.text(s.content, size="1", truncate=True),
                                    padding_bottom="0.5em", border_bottom="1px dashed gray", width="100%"
                                )
                            ),
                            width="100%"
                        )
                    ),
                    width="100%", type="multiple"
                )
            ),
            align_items="start", spacing="2", width="100%"
        ),
        bg=rx.cond(msg.role == "user", rx.color("blue", 3), rx.color("green", 3)),
        padding="1em", border_radius="md", width="100%", margin_bottom="1em",
        key=msg.id 
    )

def main_content():
    return rx.vstack(
        rx.heading(PAGE_TITLE, size="8"),
        rx.scroll_area(
            rx.vstack(
                rx.foreach(State.messages, message_box),
                rx.box(id="chat_bottom"), 
                width="100%",
            ),
            height="70vh", width="100%", padding="1em", border="1px solid #eaeaea", border_radius="lg",
        ),
        rx.hstack(
            rx.input(
                placeholder="Chiedi pure...",
                value=State.input_text,
                on_change=State.set_input_text,
                width="100%",
                on_key_down=lambda key: rx.cond(key == "Enter", State.handle_submit(), None)
            ),
            rx.button("Invia", on_click=State.handle_submit, loading=State.is_processing, color_scheme="blue"),
            width="100%", padding_top="1em"
        ),
        width="100%", padding="2em", height="100vh"
    )

def index():
    return rx.hstack(sidebar_content(), main_content(), spacing="0", align_items="start", width="100%")

app = rx.App(theme=rx.theme(appearance="light", accent_color="blue", radius="large"))
app.add_page(index, on_load=State.on_load)