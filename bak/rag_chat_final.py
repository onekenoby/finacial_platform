import sys
import torch
from collections import deque
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# --- CONFIGURAZIONE ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
MODEL_RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL_ID = "gemma-3-12b" 

# --- MEMORIA (Novità) ---
# Conserviamo solo gli ultimi 4 scambi (User + AI) per non saturare il contesto
MEMORY_LIMIT = 4 

print(f"⏳ Caricamento Embeddings e Re-Ranker...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(MODEL_EMBEDDING_NAME, device=device)
reranker = CrossEncoder(MODEL_RERANKER_NAME, device=device)
llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def retrieve_optimized(query_text):
    """Retrieval + Re-Ranking (Il tuo motore di ricerca avanzato)"""
    query_vector = embedder.encode(query_text, normalize_embeddings=True)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    limit_broad = 35 
    
    # 1. Recupero Ampio
    try:
        search_result = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=limit_broad)
    except:
        try:
            search_result = client.search_points(collection_name=COLLECTION_NAME, vector=query_vector, limit=limit_broad, with_payload=True)
        except:
             search_result = client.http.points_api.search_points(
                collection_name=COLLECTION_NAME,
                search_points_body={"vector": query_vector.tolist(), "limit": limit_broad, "with_payload": True}
            ).result

    if not search_result: return []

    # 2. Preparazione Re-Ranking
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

    # 3. Applicazione Re-Ranker
    cross_scores = reranker.predict(cross_inp)
    for idx in range(len(cross_scores)):
        hits_data[idx]['rerank_score'] = cross_scores[idx]

    hits_sorted = sorted(hits_data, key=lambda x: x['rerank_score'], reverse=True)
    final_context = hits_sorted[:10] # Top 10 chunk per Gemma

    # 4. Formattazione
    context_parts = []
    print(f"\n🔍 FONTI USATE (Top 3 su {len(hits_sorted)}):")
    for hit in final_context[:3]:
        print(f"   [{hit['type']}] {hit['filename']} (Re-Rank Score: {hit['rerank_score']:.4f})")

    for hit in final_context:
        header = f"--- {'DESCRIZIONE IMMAGINE' if hit['type']=='image_description' else 'TESTO'} ({hit['filename']}) ---"
        context_parts.append(f"{header}\n{hit['content']}")

    return "\n\n".join(context_parts)

def start_chat():
    print(f"\n💬 Chat Finanziaria (Memoria Attiva: ultimi {MEMORY_LIMIT} scambi)")
    
    # --- INIZIALIZZAZIONE MEMORIA ---
    # Usiamo 'deque' per una lista che si svuota automaticamente se supera il limite
    chat_history = deque(maxlen=MEMORY_LIMIT * 2) 
    
    # System Prompt fisso
    system_instruction = (
        "Sei un assistente finanziario intelligente. "
        "Hai accesso a una memoria della conversazione precedente e a nuovi documenti. "
        "Usa il contesto fornito per rispondere. Cita sempre le fonti."
    )

    while True:
        user_query = input("\nTu: ")
        if user_query.lower() in ['exit', 'quit']: break
            
        # 1. Recuperiamo il contesto SOLO per la domanda attuale
        context = retrieve_optimized(user_query)
        
        if not context:
            print("Nessun dato trovato nei documenti.")
            continue

        # 2. Costruiamo il messaggio attuale con il contesto fresco
        current_message_content = f"""
        ### DOCUMENTI RECUPERATI PER QUESTA DOMANDA ###
        {context}
        
        ### DOMANDA DELL'UTENTE ###
        {user_query}
        """

        # 3. Costruiamo la lista completa dei messaggi per l'LLM
        # Ordine: System Prompt -> Storia Vecchia -> Messaggio Attuale
        messages_payload = [{"role": "system", "content": system_instruction}]
        
        # Aggiungiamo lo storico (senza i vecchi documenti enormi, solo testo puro)
        messages_payload.extend(list(chat_history))
        
        # Aggiungiamo la domanda corrente con i documenti
        messages_payload.append({"role": "user", "content": current_message_content})

        print("🤖 AI sta pensando...", end="\r")
        
        full_response = ""
        try:
            stream = llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL_ID,
                messages=messages_payload,
                temperature=0.2,
                stream=True
            )
            
            print("🤖 AI: ", end="")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print("\n")
            
            # 4. AGGIORNAMENTO MEMORIA
            # Salviamo la domanda pulita (senza i documenti allegati) per risparmiare spazio
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"\n❌ Errore: {e}")

if __name__ == "__main__":
    start_chat()