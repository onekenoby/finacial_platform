import os
import shutil  # Libreria per spostare i file
import base64
import hashlib
import json
import psycopg2
import fitz  # PyMuPDF
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURAZIONE CARTELLE ---
INPUT_FOLDER = r"E:\Dev\FinancialAI\input_docs"
PROCESSED_FOLDER = r"E:\Dev\FinancialAI\input_doc_processed"  # Nuova cartella archivio

# --- CONFIGURAZIONE DATABASE ---
PG_PARAMS = {
    "host": "localhost", "port": "5432",
    "database": "ai_ingestion", "user": "admin", "password": "admin_password"
}
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"
MODEL_NAME = "BAAI/bge-m3"

# --- CONFIGURAZIONE LM STUDIO (VISION) ---
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
# 👇 Assicurati che questo ID corrisponda a quello caricato in LM Studio
LM_STUDIO_MODEL_ID = "qwen3-vl-8b" 

print("⏳ Caricamento modello Embeddings (BGE-M3)...")
embedder = SentenceTransformer(MODEL_NAME, device='cuda')

# Client per Qwen
llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def get_db_connection():
    return psycopg2.connect(**PG_PARAMS)

def calculate_hash(data_bytes):
    return hashlib.md5(data_bytes).hexdigest()

def move_to_processed(filepath):
    """Sposta il file processato e lo rinomina con timestamp."""
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    # Genera Timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"
    
    dest_path = os.path.join(PROCESSED_FOLDER, new_filename)
    
    shutil.move(filepath, dest_path)
    print(f"📦 File archiviato: {new_filename}")

def analyze_image_with_qwen(image_bytes, image_id):
    """Invia l'immagine a Qwen VL per l'analisi."""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        system_prompt = "Sei un analista finanziario esperto. Analizza questa immagine estratta da un documento."
        user_prompt = (
            "Descrivi dettagliatamente questa immagine. "
            "Se è un grafico, estrai i numeri, le etichette degli assi e il trend. "
            "Se è una tabella, trascrivi i dati chiave. "
        )

        response = llm_client.chat.completions.create(
            model=LM_STUDIO_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=500
        )
        description = response.choices[0].message.content
        print(f"      👁️ Vision AI: {description[:50]}...")
        return f"[[DESCRIZIONE_AI (Img_ID: {image_id}): {description}]]"

    except Exception as e:
        print(f"      ⚠️ Errore Vision AI: {e}")
        return f"[[ERRORE ANALISI AI per Img_ID: {image_id}]]"

def save_image_to_db(cur, log_id, image_bytes, mime_type="image/png"):
    img_hash = calculate_hash(image_bytes)
    # Deduplica
    cur.execute("SELECT image_id FROM ingestion_images WHERE image_hash = %s", (img_hash,))
    res = cur.fetchone()
    if res: return res[0], False

    cur.execute("""
        INSERT INTO ingestion_images (log_id, image_data, image_hash, mime_type, ingestion_ts)
        VALUES (%s, %s, %s, %s, NOW())
        RETURNING image_id;
    """, (log_id, psycopg2.Binary(image_bytes), img_hash, mime_type))
    return cur.fetchone()[0], True

def process_multimodal_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\n📸 Processing: {filename}")
    
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Start Log
    cur.execute("""
        INSERT INTO ingestion_logs (source_name, source_type, status, ingestion_ts)
        VALUES (%s, 'pdf_multimodal_qwen', 'processing', NOW())
        RETURNING log_id;
    """, (filename,))
    log_id = cur.fetchone()[0]
    conn.commit()

    try:
        doc = fitz.open(filepath)
        full_semantic_stream = "" 
        
        print(f"   🔍 Analisi {len(doc)} pagine...")

        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict["blocks"]

            for block in blocks:
                # Gestione Testo
                if block["type"] == 0: 
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    if block_text.strip():
                        full_semantic_stream += block_text.strip() + "\n\n"
                
                # Gestione Immagini
                elif block["type"] == 1: 
                    image_bytes = block["image"]
                    ext = block["ext"]
                    if len(image_bytes) < 3000: continue # Salta icone piccole

                    img_id, is_new = save_image_to_db(cur, log_id, image_bytes, f"image/{ext}")
                    
                    # Analisi Qwen
                    ai_description = analyze_image_with_qwen(image_bytes, img_id)
                    
                    # Aggiorna DB e Flusso
                    cur.execute("UPDATE ingestion_images SET description_ai = %s WHERE image_id = %s", (ai_description, img_id))
                    full_semantic_stream += f"\n\n{ai_description}\n\n"

        # 2. Chunking
        print("   ✂️  Chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(full_semantic_stream)

        if not chunks: raise ValueError("PDF vuoto o illeggibile.")

        # 3. Embeddings & Save
        embeddings = embedder.encode(chunks, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        points_buffer = []

        print(f"   💾 Scrittura {len(chunks)} TOONs...")
        for i, (text, vector) in enumerate(zip(chunks, embeddings)):
            chunk_type = "image_description" if "[[DESCRIZIONE_AI" in text else "text_block"
            
            cur.execute("""
                INSERT INTO document_chunks 
                (log_id, chunk_index, toon_type, content_raw, content_semantic, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING chunk_id;
            """, (log_id, i, chunk_type, text, text, json.dumps({"filename": filename})))
            chunk_db_id = cur.fetchone()[0]

            points_buffer.append(models.PointStruct(
                id=chunk_db_id,
                vector=vector.tolist(),
                payload={
                    "log_id": log_id,
                    "toon_type": chunk_type,
                    "text_preview": text[:100],
                    "filename": filename
                }
            ))

        q_client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)

        # 4. Success Commit
        cur.execute("UPDATE ingestion_logs SET status = 'success', total_chunks = %s WHERE log_id = %s", (len(chunks), log_id))
        conn.commit()
        
        # 5. CHIUSURA CONNESSIONI PRIMA DI SPOSTARE IL FILE
        # Importante: chiudiamo il file handle di PyMuPDF
        doc.close() 
        cur.close()
        conn.close()

        print("✅ Ingestion completata!")
        
        # 6. SPOSTAMENTO FILE (Solo se successo)
        move_to_processed(filepath)

    except Exception as e:
        print(f"❌ ERRORE: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
            cur = conn.cursor()
            cur.execute("UPDATE ingestion_logs SET status = 'failed', error_message = %s WHERE log_id = %s", (str(e), log_id))
            conn.commit()
            cur.close()
            conn.close()

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    
    if files:
        print(f"🚀 Trovati {len(files)} documenti. Inizio pipeline...")
        for f in files:
            process_multimodal_file(os.path.join(INPUT_FOLDER, f))
    else:
        print(f"Nessun PDF trovato in {INPUT_FOLDER}")