import os
import time
import psycopg2
import fitz  # PyMuPDF
from datetime import datetime
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURAZIONE ---
PG_PARAMS = {
    "host": "localhost", "port": "5432",
    "database": "ai_ingestion", "user": "admin", "password": "admin_password"
}
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"

# Modello Embedding (Dimensione 1024)
MODEL_NAME = "BAAI/bge-m3" 

print("⏳ Caricamento modello AI (sarà lento solo la prima volta)...")
# device='cuda' usa la GPU NVIDIA. Se da errore, metti 'cpu'
embedder = SentenceTransformer(MODEL_NAME, device='cuda') 

def get_db_connection():
    return psycopg2.connect(**PG_PARAMS)

def process_pdf(file_path):
    filename = os.path.basename(file_path)
    print(f"\n📄 Avvio ingestion: {filename}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. CREA LOG (Stato: PROCESSING)
    cur.execute("""
        INSERT INTO ingestion_logs (source_name, source_type, status, ingestion_ts)
        VALUES (%s, 'pdf', 'processing', NOW())
        RETURNING log_id;
    """, (filename,))
    log_id = cur.fetchone()[0]
    conn.commit()
    
    try:
        # 2. ESTRAZIONE TESTO (PyMuPDF)
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # 3. CHUNKING (Tagliamo il testo in pezzi da ~1000 caratteri con sovrapposizione)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        print(f"   ✂️  Testo diviso in {len(chunks)} chunks.")

        # 4. GENERAZIONE EMBEDDINGS (AI)
        print("   🧠 Generazione Vettori...")
        embeddings = embedder.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        
        # 5. SALVATAGGIO SUI DATABASE
        q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        points_buffer = []
        
        print("   💾 Scrittura su DB...")
        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            # A. Salva su Postgres (Testo + Metadati)
            cur.execute("""
                INSERT INTO document_chunks (log_id, chunk_index, toon_type, content_raw, content_semantic)
                VALUES (%s, %s, 'text', %s, %s)
                RETURNING chunk_id;
            """, (log_id, i, chunk_text, chunk_text)) # content_semantic per ora è uguale al raw
            
            chunk_db_id = cur.fetchone()[0]
            
            # B. Prepara punto per Qdrant
            # Usiamo l'ID di Postgres come Payload per collegarli
            points_buffer.append(models.PointStruct(
                id=chunk_db_id,  # Usa lo stesso ID del DB SQL per coerenza
                vector=vector.tolist(),
                payload={
                    "log_id": log_id,
                    "filename": filename,
                    "text_preview": chunk_text[:100] # Anteprima veloce
                }
            ))

        # Push su Qdrant in batch
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_buffer
        )

        # 6. CHIUSURA LOG (Stato: SUCCESS)
        cur.execute("""
            UPDATE ingestion_logs 
            SET status = 'success', total_chunks = %s, processing_time_ms = 0
            WHERE log_id = %s
        """, (len(chunks), log_id))
        conn.commit()
        print("✅ Ingestion completata con successo!")

    except Exception as e:
        print(f"❌ ERRORE: {e}")
        cur.execute("UPDATE ingestion_logs SET status = 'failed', error_message = %s WHERE log_id = %s", (str(e), log_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # --- PROVA CON UN TUO PDF ---
    # Sostituisci con un percorso vero a un PDF sul tuo PC
    # Esempio: "D:\\Documenti\\Bilancio_2023.pdf"
    
    # Se non hai un PDF pronto, crea un file dummy.pdf con del testo dentro per provare
    TARGET_FILE = "E:\\Dev\\FinancialAI\\test_pdf.pdf" 
    
    if os.path.exists(TARGET_FILE):
        process_pdf(TARGET_FILE)
    else:
        print(f"⚠️ File non trovato: {TARGET_FILE}")
        print("Modifica la variabile TARGET_FILE nello script con un percorso valido.")