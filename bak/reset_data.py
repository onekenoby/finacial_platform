import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURAZIONE ---
PG_PARAMS = {
    "host": "localhost", "port": "5432",
    "database": "ai_ingestion", "user": "admin", "password": "admin_password"
}
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_docs"

def reset_system():
    print("🧹 Inizio pulizia sistema...")

    # 1. PULIZIA POSTGRES (Truncate a cascata)
    try:
        print("   🛁 Svuotamento Postgres...", end=" ")
        conn = psycopg2.connect(**PG_PARAMS)
        cur = conn.cursor()
        
        # TRUNCATE cancella i dati ma mantiene la struttura delle tabelle.
        # RESTART IDENTITY resetta i contatori degli ID a 1.
        # CASCADE pulisce anche le tabelle collegate (chunks e images).
        cur.execute("TRUNCATE TABLE ingestion_logs RESTART IDENTITY CASCADE;")
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Fatto!")
    except Exception as e:
        print(f"\n❌ Errore Postgres: {e}")

    # 2. PULIZIA QDRANT (Ricreazione Collection)
    try:
        print("   🗑️  Svuotamento Qdrant...", end=" ")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Il modo più veloce per svuotare è cancellare e ricreare la collection
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        
        # La ricreiamo subito vuota
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024, 
                distance=models.Distance.COSINE
            )
        )
        print("✅ Fatto!")
    except Exception as e:
        print(f"\n❌ Errore Qdrant: {e}")

    print("\n✨ SISTEMA PULITO E PRONTO PER DATI REALI ✨")

if __name__ == "__main__":
    confirm = input("⚠️  SEI SICURO di voler cancellare TUTTI i dati? (s/n): ")
    if confirm.lower() == 's':
        reset_system()
    else:
        print("Operazione annullata.")