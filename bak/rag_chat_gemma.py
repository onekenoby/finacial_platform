import sys
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- CONFIGURAZIONE DATABASE ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"

# --- CONFIGURAZIONE LLM (LM STUDIO) ---
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# 👇 INSERISCI QUI IL NOME ESATTO DEL TUO MODELLO 12B
# Esempio: "gemma-3-12b-it-Q4_K_M" (o come appare in LM Studio)
LM_STUDIO_MODEL_ID = "gemma-3-12b" 

print(f"⏳ Caricamento modello Embeddings ({MODEL_EMBEDDING_NAME})...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device=device)

# Verifica VRAM iniziale (Solo informativo)
if device == 'cuda':
    vram_free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"📊 VRAM Libera rilevata da PyTorch: {vram_free:.2f} GB")
    if vram_free < 2.0:
        print("⚠️  ATTENZIONE: Poca VRAM libera per gli embeddings. Potresti rallentare.")

llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def check_active_model():
    """Controlla cosa sta girando su LM Studio."""
    try:
        models = llm_client.models.list()
        active = models.data[0].id
        print(f"🔌 Modello agganciato su LM Studio: {active}")
        return active
    except Exception as e:
        print(f"❌ Errore connessione LM Studio: {e}")
        sys.exit(1)

def retrieve_context(query_text, limit=5):
    """Recupera il contesto ibrido (Testo + Descrizioni Immagini)."""
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    context_parts = []
    print(f"\n🔍 Fonti recuperate ({len(search_result)}):")
    
    for hit in search_result:
        payload = hit.payload
        content = payload.get('content_semantic', payload.get('text_preview', ''))
        filename = payload.get('filename', 'Sconosciuto')
        toon_type = payload.get('toon_type', 'text')
        
        # Gestione etichette per il modello 12B
        if toon_type == 'image_description':
            header = f"--- DESCRIZIONE IMMAGINE (File: {filename}) ---"
        else:
            header = f"--- TESTO ESTRATTO (File: {filename}) ---"
            
        print(f"   - {filename} [{toon_type}] (Score: {hit.score:.3f})")
        context_parts.append(f"{header}\n{content}")
    
    return "\n\n".join(context_parts)

def start_chat():
    active_model = check_active_model()
    
    print(f"\n💬 Chat pronta con modello 12B: {active_model}")
    print("Consiglio: Sii specifico nelle domande sui grafici.")
    print("Scrivi 'exit' per uscire.\n")
    
    while True:
        user_query = input("\nTu: ")
        if user_query.lower() in ['exit', 'quit']: break
            
        context = retrieve_context(user_query)
        
        if not context:
            print("🤖 AI: Nessun dato rilevante trovato nei documenti.")
            continue

        # System Prompt ottimizzato per modelli medi (12B)
        # I modelli 12B gestiscono istruzioni complesse meglio dei 7B-9B
        system_prompt = (
            "Sei un assistente finanziario avanzato. "
            "Hai accesso a frammenti di documenti che includono testo e descrizioni dettagliate di grafici/immagini. "
            "1. Analizza il contesto fornito. "
            "2. Se la risposta richiede dati da un grafico, usa la descrizione fornita tra i tag 'DESCRIZIONE IMMAGINE'. "
            "3. Se i dati sono in conflitto, dai priorità ai dati più recenti (controlla le date se presenti). "
            "4. Cita sempre la fonte (nome file)."
        )
        
        full_user_message = f"""
        ### CONTESTO DOCUMENTALE ###
        {context}
        
        ### DOMANDA ###
        {user_query}
        """

        print("🤖 AI sta elaborando...", end="\r")
        try:
            stream = llm_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_message}
                ],
                temperature=0.2, # Bassa temperatura per precisione
                max_tokens=1500,
                stream=True
            )
            
            print("🤖 AI: ", end="")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"\n❌ Errore generazione: {e}")

if __name__ == "__main__":
    start_chat()