import psycopg2
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
import os

# --- CONFIGURAZIONE (Allineata ai tuoi script) ---
PG_DSN = "dbname=ai_ingestion user=admin password=admin_password host=localhost"
QDRANT_HOST = "localhost"
QDRANT_COLLECTION = "financial_docs"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password_sicura")

def clean_postgres_targeted(filename):
    print(f"🐘 Postgres: Rimozione record per {filename}...", end=" ")
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        
        # 1. Recuperiamo i log_id associati al file
        cur.execute("SELECT log_id FROM ingestion_logs WHERE source_name = %s", (filename,))
        log_ids = [row[0] for row in cur.fetchall()]
        
        if log_ids:
            # 2. Eliminiamo i dati collegati
            # Usiamo CASCADE se definito, o eliminiamo manualmente dalle tabelle figlie
            cur.execute("DELETE FROM ingestion_images WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM document_chunks WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM ingestion_logs WHERE log_id = ANY(%s)", (log_ids,))
            conn.commit()
            print(f"✅ Rimossi {len(log_ids)} log e relativi chunk/immagini.")
        else:
            print("⚠️ Nessun dato trovato.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Errore: {e}")
		
def clean_neo4j_targeted(filename):
    print(f"🕸️  Neo4j: Rimozione sotto-grafo per {filename}...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            # Elimina Documento -> Pagine -> Chunk e le loro relazioni
            # Le Entity (nodi verdi) rimangono ma perdono il legame con questo file
            query = """
            MATCH (d:Document {filename: $fname})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p)-[:HAS_CHUNK]->(c)
            DETACH DELETE d, p, c
            """
            result = session.run(query, fname=filename)
            summary = result.consume()
            print(f"✅ Eliminati {summary.counters.nodes_deleted} nodi.")
        driver.close()
    except Exception as e:
        print(f"❌ Errore: {e}")
		

def clean_qdrant_targeted(filename):
    print(f"💠 Qdrant: Rimozione vettori per {filename}...", end=" ")
    try:
        from qdrant_client import models
        client = QdrantClient(host=QDRANT_HOST, port=6333)
        
        # Eliminiamo solo i punti dove il payload "filename" coincide
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    ),
                ]
            ),
        )
        print("✅ Punti eliminati.")
    except Exception as e:
        print(f"❌ Errore: {e}")
		


if __name__ == "__main__":
    target_file = input("Inserisci il nome del file da pulire (es. Intelligenza_Artificiale_Finanza.pdf) o 'ALL' per reset totale: ")
    
    if target_file.upper() == 'ALL':
        # Esegui le funzioni originali di reset totale
        pass
    else:
        clean_postgres_targeted(target_file)
        clean_neo4j_targeted(target_file)
        clean_qdrant_targeted(target_file)
        print(f"\n✨ Database pronti per la ri-ingestion di: {target_file}")
