import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- CONFIGURAZIONE ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"

# --- CONFIGURAZIONE LLM (LM Studio) ---
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
# 👇 Assicurati che sia lo stesso modello attivo su LM Studio
LM_STUDIO_MODEL_ID = "qwen3-vl-8b" 

print("⏳ Caricamento modello Embeddings (BGE-M3)...")
# Usiamo la GPU per convertire la tua domanda in numeri velocemente
embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device='cuda')

# Client per generare le risposte
llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def retrieve_context(query_text, limit=5):
    """Cerca i 5 pezzi di informazione più rilevanti in Qdrant."""
    
    # 1. Converti domanda in vettore
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    
    # 2. Cerca nel DB
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    # 3. Costruisci il testo del contesto
    context_parts = []
    print(f"\n🔍 Fonti trovate ({len(search_result)}):")
    
    for hit in search_result:
        payload = hit.payload
        content = payload.get('text_preview', '') # O il testo completo se l'hai salvato nel payload
        filename = payload.get('filename', 'Sconosciuto')
        toon_type = payload.get('toon_type', 'text')
        score = hit.score
        
        # Etichetta speciale per le descrizioni delle immagini
        prefix = "[IMMAGINE/GRAFICO]" if toon_type == 'image_description' else "[TESTO]"
        
        print(f"   - {prefix} {filename} (Score: {score:.3f})")
        context_parts.append(f"FONTE ({filename} - {toon_type}):\n{content}")
    
    return "\n\n".join(context_parts)

def chat_with_data():
    print(f"\n💬 Chat avviata con modello: {LM_STUDIO_MODEL_ID}")
    print("Scrivi 'exit' per uscire.\n")
    
    while True:
        user_query = input("Tu: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        # 1. RETRIEVAL (Recupero informazioni)
        context_text = retrieve_context(user_query)
        
        if not context_text:
            print("🤖 AI: Non ho trovato documenti pertinenti nel database.")
            continue

        # 2. AUGMENTATION (Creazione Prompt)
        system_prompt = (
            "Sei un assistente finanziario preciso e professionale. "
            "Usa SOLO le informazioni fornite nel CONTESTO sottostante per rispondere alla domanda. "
            "Se il contesto contiene descrizioni di grafici o immagini, usale per estrapolare i trend numerici. "
            "Se non sai la risposta basandoti sul contesto, dillo chiaramente."
        )
        
        full_prompt = f"""
        CONTESTO RECUPERATO DAI DOCUMENTI:
        ----------------------------------
        {context_text}
        ----------------------------------
        
        DOMANDA DELL'UTENTE:
        {user_query}
        """

        # 3. GENERATION (Risposta AI)
        print("🤖 AI sta scrivendo...", end="\r")
        try:
            response = llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3, # Bassa temperatura per risposte fattuali
                stream=True 
            )
            
            print("🤖 AI: ", end="")
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk
            print("\n")
            
        except Exception as e:
            print(f"\n❌ Errore LLM: {e}")
            print("Suggerimento: Controlla che il Server LM Studio sia avviato e l'ID del modello sia corretto.")

if __name__ == "__main__":
    chat_with_data()