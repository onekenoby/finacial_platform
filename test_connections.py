import psycopg2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import datetime

# --- CONFIGURAZIONE ---
# Nota: Usiamo 'localhost' perché Docker espone le porte su Windows
PG_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "database": "ai_ingestion",
    "user": "admin",
    "password": "admin_password"
}

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"

def test_system():
    print("🚀 Inizio Test Idraulica...")
    log_id = None

    # --- 1. TEST POSTGRESQL (Relazionale + Timescale) ---
    try:
        print("   Testing Postgres (Logs & Chunks)...", end=" ")
        
        # Connessione
        conn = psycopg2.connect(**PG_PARAMS)
        cur = conn.cursor()
        
        # A. Creiamo un Log finto (Anagrafica)
        cur.execute("""
            INSERT INTO ingestion_logs (source_name, source_type, status, ingestion_ts)
            VALUES (%s, %s, %s, NOW())
            RETURNING log_id;
        """, ("test_connection_file.pdf", "test_probe", "processing"))
        
        log_id = cur.fetchone()[0]
        
        # B. Creiamo un Chunk finto (Hypertable)
        # Nota: Qui testiamo se la FK funziona e se Timescale accetta il dato
        cur.execute("""
            INSERT INTO document_chunks (log_id, chunk_index, toon_type, content_raw)
            VALUES (%s, %s, %s, %s)
        """, (log_id, 0, "text", "Questo è un test di scrittura dal Python su Windows verso Docker."))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ OK! (Log ID creato: {log_id})")

    except Exception as e:
        print(f"\n❌ Errore Postgres: {e}")
        return

    # --- 2. TEST QDRANT (Vettoriale) ---
    try:
        print("   Testing Qdrant (Vectors)...", end=" ")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Generiamo un vettore casuale (dimensione 1024 come BGE-M3)
        fake_vector = np.random.rand(1024).tolist()
        
        # Inseriamo il punto collegandolo al Log ID appena creato
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,  # ID del punto (può essere un UUID o int)
                    vector=fake_vector,
                    payload={
                        "log_id": log_id,
                        "text": "Payload di test per verificare la persistenza."
                    }
                )
            ]
        )
        
        # Verifica di lettura immediata
        assert client.count(COLLECTION_NAME).count > 0
        
        print("✅ OK! (Vettore inserito)")

    except Exception as e:
        print(f"\n❌ Errore Qdrant: {e}")
        return

    print("\n✨ SISTEMA COMPLETAMENTE OPERATIVO! ✨")
    print("Postgres e Qdrant sono raggiungibili e scrivibili.")

if __name__ == "__main__":
    test_system()