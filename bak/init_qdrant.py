from qdrant_client import QdrantClient
from qdrant_client.http import models

# Connessione a Qdrant (localhost, porta 6333)
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "financial_docs"

# Verifica esistenza
if not client.collection_exists(COLLECTION_NAME):
    print(f"🔧 Creazione collection '{COLLECTION_NAME}'...")
    
    # Creiamo la struttura per ospitare vettori da 1024 dimensioni (BGE-M3)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024, 
            distance=models.Distance.COSINE
        )
    )
    print("✅ Collection creata con successo!")
else:
    print(f"⚠️ La collection '{COLLECTION_NAME}' esiste già.")