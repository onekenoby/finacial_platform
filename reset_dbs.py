import psycopg2
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# --- CONFIG ---
# Postgres
PG_DSN = "dbname=ai_ingestion user=admin password=admin_password host=localhost"
# Qdrant
QDRANT_HOST = "localhost"
QDRANT_COLLECTION = "financial_docs"
# Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password_sicura")

def clean_postgres():
    print("🐘 Cleaning Postgres...", end=" ")
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        cur.execute("""
            TRUNCATE TABLE 
                ingestion_consistency_checks,
                doc_graph_index,
                chunk_vector_index,
                ingestion_run_logs,
                ingestion_documents,
                ingestion_images,
                document_chunks,
                ingestion_logs,
                ingestion_runs
            RESTART IDENTITY CASCADE;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Fatto.")
    except Exception as e:
        print(f"❌ Errore: {e}")

def clean_neo4j():
    print("🕸️  Cleaning Neo4j...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("✅ Fatto.")
    except Exception as e:
        print(f"❌ Errore: {e}")

def clean_qdrant():
    print("💠 Cleaning Qdrant...", end=" ")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=6333)
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)
            print("✅ Collection eliminata.", end=" ")
        else:
            print("⚠️ Collection non trovata.", end=" ")
        print("")
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    print("--- 🗑️  RESET TOTALE DATABASE 🗑️  ---")
    confirm = input("Sei sicuro di voler cancellare TUTTI i dati? (s/N): ")
    if confirm.lower() == 's':
        clean_postgres()
        clean_neo4j()
        clean_qdrant()
        print("\n✨ Ambiente pulito e pronto per l'ingestion.")
    else:
        print("Operazione annullata.")