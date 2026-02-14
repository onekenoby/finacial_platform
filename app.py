import streamlit as st
import torch
import time
from collections import deque
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
# Assicurati che questo ID sia corretto!
LM_STUDIO_MODEL_ID = "gemma-3-12b" 
MEMORY_LIMIT = 4  # Scambi ricordati

# --- SETUP PAGINA ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title("📊 Financial AI Assistant (Re-Rank + Memory)")

# --- CARICAMENTO RISORSE (Cachate) ---
@st.cache_resource
def load_resources():
    status = st.empty()
    status.info("⏳ Caricamento modelli (Embedding + Re-Ranker)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Embedder
    embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device=device)
    
    # 2. Re-Ranker (Leggero, veloce)
    reranker = CrossEncoder(MODEL_RERANKER_NAME, device=device)
    
    # 3. Client LLM
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)
    
    status.empty()
    return embedder, reranker, llm_client

embedder, reranker, llm_client = load_resources()

# --- FUNZIONE DI RICERCA AVANZATA (Copia logica di rag_chat_final.py) ---
def retrieve_optimized(query_text):
    # 1. Retrieval Ampio (35 chunks)
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    limit_broad = 35
    
    try:
        search_result = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=limit_broad)
    except:
        # Fallback Robusto
        try:
             search_result = client.search_points(collection_name=COLLECTION_NAME, vector=query_vector, limit=limit_broad, with_payload=True)
        except:
             search_result = client.http.points_api.search_points(
                collection_name=COLLECTION_NAME,
                search_points_body={"vector": query_vector.tolist(), "limit": limit_broad, "with_payload": True}
            ).result

    if not search_result: return []

    # 2. Re-Ranking
    hits_data = []
    cross_inp = []
    
    for hit in search_result:
        if isinstance(hit, dict): payload = hit.get('payload', {})
        else: payload = hit.payload
        content = payload.get('content_semantic', payload.get('text_preview', ''))
        filename = payload.get('filename', 'Sconosciuto')
        toon_type = payload.get('toon_type', 'text')
        
        hits_data.append({"content": content, "filename": filename, "type": toon_type})
        cross_inp.append([query_text, content])

    # Calcolo Score di Rilevanza
    cross_scores = reranker.predict(cross_inp)
    for idx in range(len(cross_scores)):
        hits_data[idx]['rerank_score'] = cross_scores[idx]

    # Ordinamento e taglio (Top 10)
    hits_sorted = sorted(hits_data, key=lambda x: x['rerank_score'], reverse=True)
    return hits_sorted[:10]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Monitoraggio")
    if torch.cuda.is_available():
        vram_free = torch.cuda.mem_get_info()[0] / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.metric("VRAM Libera", f"{vram_free:.2f} GB", f"Totale: {vram_total:.1f} GB")
        st.progress(1 - (vram_free / vram_total))
    else:
        st.warning("⚠️ CPU Mode")
    
    st.divider()
    if st.button("🗑️ Reset Memoria Chat"):
        st.session_state.messages = []
        st.session_state.memory_buffer = deque(maxlen=MEMORY_LIMIT*2)
        st.rerun()

# --- GESTIONE STATO ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Per la visualizzazione UI
if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = deque(maxlen=MEMORY_LIMIT*2) # Per il prompt LLM (solo testo)

# --- VISUALIZZAZIONE CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Fonti (Re-Ranked)"):
                for s in msg["sources"]:
                    icon = "🖼️" if s['type'] == 'image_description' else "📄"
                    st.markdown(f"**{icon} {s['filename']}** (Score: {s['rerank_score']:.3f})")
                    st.caption(s['content'][:300] + "...")

# --- INPUT UTENTE ---
if prompt := st.chat_input("Chiedi pure..."):
    # 1. Aggiungi user message alla UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generazione Risposta
    with st.chat_message("assistant"):
        with st.spinner("🧠 Re-Ranking e Ragionamento..."):
            
            # A. Recupero Contesto Ottimizzato
            sources = retrieve_optimized(prompt)
            
            context_str = "\n\n".join([
                f"--- {'DESCRIZIONE IMMAGINE' if s['type']=='image_description' else 'TESTO'} ({s['filename']}) ---\n{s['content']}"
                for s in sources
            ])
            
            # B. Costruzione Messaggi (con Memoria)
            system_prompt = (
                "Sei un assistente finanziario esperto. "
                "Usa il contesto fornito per rispondere. Cita sempre le fonti. "
                "Se la risposta richiede dati da un grafico, usa le descrizioni fornite."
            )
            
            # Messaggio corrente con documenti
            current_msg_payload = f"### DOCUMENTI ###\n{context_str}\n\n### DOMANDA ###\n{prompt}"
            
            # Costruzione payload completo: System -> Memoria -> Corrente
            full_payload = [{"role": "system", "content": system_prompt}]
            full_payload.extend(list(st.session_state.memory_buffer)) # Aggiunge storico
            full_payload.append({"role": "user", "content": current_msg_payload})

            # C. Chiamata LLM
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                stream = llm_client.chat.completions.create(
                    model=LM_STUDIO_MODEL_ID,
                    messages=full_payload,
                    temperature=0.2,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # D. Aggiornamento Stati
                # UI History (con fonti per display)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
                
                # LLM Memory Buffer (Solo testo pulito per risparmiare token)
                st.session_state.memory_buffer.append({"role": "user", "content": prompt})
                st.session_state.memory_buffer.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Errore: {e}")