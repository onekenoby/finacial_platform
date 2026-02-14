import sys
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- CONFIGURAZIONE ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
# Assicurati che questo combaci con LM Studio!
LM_STUDIO_MODEL_ID = "gemma-3-12b" 

print(f"⏳ Caricamento modello Embeddings ({MODEL_EMBEDDING_NAME})...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device=device)

llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def check_active_model():
    try:
        models = llm_client.models.list()
        if models.data:
            active = models.data[0].id
            print(f"🔌 Modello agganciato su LM Studio: {active}")
            return active
        return "Modello Sconosciuto"
    except Exception as e:
        print(f"❌ Errore connessione LM Studio: {e}")
        return "Errore"

def retrieve_context(query_text, limit=5):
    """Recupera il contesto usando una strategia a prova di bomba."""
    
    # 1. Encoding domanda
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    
    # 2. Ricerca Vettoriale
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        # Strategia A: Metodo standard .search()
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
    except (AttributeError, TypeError):
        try:
            # Strategia B: Metodo Legacy .search_points()
            search_result = client.search_points(
                collection_name=COLLECTION_NAME,
                vector=query_vector,
                limit=limit,
                with_payload=True
            )
        except Exception:
            # Strategia C: Accesso DIRETTO alle API (Indistruttibile)
            # Questo bypassa qualsiasi errore della libreria Python e parla direttamente al server
            print("⚠️ Uso fallback API HTTP diretto...")
            search_result = client.http.points_api.search_points(
                collection_name=COLLECTION_NAME,
                search_points_body={
                    "vector": query_vector.tolist(),
                    "limit": limit,
                    "with_payload": True
                }
            ).result

    # 3. Costruzione Contesto
    context_parts = []
    print(f"\n🔍 Fonti recuperate ({len(search_result)}):")
    
    for hit in search_result:
        # Normalizzazione dell'oggetto hit (può variare tra versioni)
        if isinstance(hit, dict):
            payload = hit.get('payload', {})
            score = hit.get('score', 0)
        else:
            payload = hit.payload
            score = hit.score

        content = payload.get('content_semantic', payload.get('text_preview', ''))
        filename = payload.get('filename', 'Sconosciuto')
        toon_type = payload.get('toon_type', 'text')
        
        if toon_type == 'image_description':
            header = f"--- DESCRIZIONE IMMAGINE (File: {filename}) ---"
        else:
            header = f"--- TESTO ESTRATTO (File: {filename}) ---"
            
        print(f"   - {filename} [{toon_type}] (Score: {score:.3f})")
        context_parts.append(f"{header}\n{content}")
    
    return "\n\n".join(context_parts)

def start_chat():
    active_model = check_active_model()
    print(f"\n💬 Chat pronta. (Versione Robustezza Massima)")
    
    while True:
        user_query = input("\nTu: ")
        if user_query.lower() in ['exit', 'quit']: break
            
        context = retrieve_context(user_query)
        if not context: 
            print("Nessun dato trovato.")
            continue

        system_prompt = (
            "Sei un assistente finanziario esperto. "
            "Usa il contesto fornito per rispondere. "
            "Se ci sono descrizioni di grafici, usale per arricchire la risposta. "
            "Cita sempre il nome del file."
        )
        
        full_user_message = f"### CONTESTO ###\n{context}\n\n### DOMANDA ###\n{user_query}"

        print("🤖 AI sta elaborando...", end="\r")
        try:
            stream = llm_client.chat.completions.create(
                model=active_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user_message}],
                temperature=0.2, max_tokens=1500, stream=True
            )
            print("🤖 AI: ", end="")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ Errore: {e}")

if __name__ == "__main__":
    start_chat()