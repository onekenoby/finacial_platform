import psycopg2
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
import os
import datetime
import re
import shutil

# --- CONFIGURAZIONE ---
PG_DSN = "dbname=ai_ingestion user=admin password=admin_password host=localhost"
QDRANT_HOST = "localhost"
QDRANT_COLLECTION = "financial_docs"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password_sicura")

# Percorso dei file processati del TIER C
PROCESSED_DIR = r"E:\Dev\FinancialAI\data_ingestion\PROCESSED\TIER_C_NEWS\MARKET_NEWS"
ARCHIVE_DIR = r"E:\Dev\FinancialAI\data_ingestion\ARCHIVE\TIER_C_NEWS" # Opzionale per non perdere i file fisici

def clean_postgres_targeted(filename):
    print(f"  🐘 Postgres: Rimozione record per {filename}...", end=" ")
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        
        cur.execute("SELECT log_id FROM ingestion_logs WHERE source_name = %s", (filename,))
        log_ids = [row[0] for row in cur.fetchall()]
        
        if log_ids:
            cur.execute("DELETE FROM ingestion_images WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM document_chunks WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM ingestion_logs WHERE log_id = ANY(%s)", (log_ids,))
            conn.commit()
            print(f"✅ Rimossi {len(log_ids)} log.")
        else:
            print("⚠️ Nessun dato.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Errore: {e}")
		
def clean_neo4j_targeted(filename):
    print(f"  🕸️  Neo4j: Rimozione sotto-grafo per {filename}...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
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
    print(f"  💠 Qdrant: Rimozione vettori per {filename}...", end=" ")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=6333)
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

def clean_neo4j_orphan_relations(days_old):
    """Pulizia globale degli archi invecchiati su Neo4j (Trend storici superati)"""
    print(f"\n🕸️  Neo4j: Pulizia relazioni orfane/scadute (> {days_old} giorni)...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            query = f"""
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.last_seen < datetime() - duration('P{days_old}D')
            DELETE r
            """
            result = session.run(query)
            summary = result.consume()
            print(f"✅ Eliminate {summary.counters.relationships_deleted} relazioni obsolete.")
        driver.close()
    except Exception as e:
        print(f"❌ Errore: {e}")

def run_retention_loop(days_old=90, clean_pg=False):
    """
    Scansiona la cartella PROCESSED, estrae la data dal nome file e
    cancella i dati associati se il file è più vecchio di 'days_old'.
    """
    if not os.path.exists(PROCESSED_DIR):
        print(f"⚠️ Cartella non trovata: {PROCESSED_DIR}")
        return

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    print(f"\n⏳ Avvio Retention Loop (Soglia: {days_old} giorni -> file precedenti al {cutoff_date.strftime('%Y-%m-%d')})")
    
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    files_processed = 0

    for filename in os.listdir(PROCESSED_DIR):
        if not filename.endswith(".md"):
            continue

        # Estrai la data dal formato NEWS_TICKER_YYYYMMDD_HHMMSS_HASH.md
        # Cerca un blocco di 8 cifre preceduto e seguito da underscore
        match = re.search(r'_(\d{8})_', filename)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue # Formato data non valido, salta
            
            # Se il file è scaduto, attiva le funzioni mirate
            if file_date < cutoff_date:
                print(f"\n🗑️  Trovato file scaduto: {filename} (Data: {date_str})")
                
                clean_qdrant_targeted(filename)
                clean_neo4j_targeted(filename)
                
                if clean_pg:
                    clean_postgres_targeted(filename)
                else:
                    print("  🐘 Postgres: Ignorato (Mantenimento Storico).")
                
                # Rimuovi il file dal disco (o spostalo in un archivio per sicurezza)
                filepath = os.path.join(PROCESSED_DIR, filename)
                archive_path = os.path.join(ARCHIVE_DIR, filename)
                shutil.move(filepath, archive_path)
                print(f"  📁 File spostato in ARCHIVE: {filename}")
                
                files_processed += 1

    # Pulizia degli archi temporali di Neo4j slegati dai singoli file
    clean_neo4j_orphan_relations(days_old=180) # Storico trend a 6 mesi

    print(f"\n✨ Loop completato. Rimossi {files_processed} documenti obsoleti.")


if __name__ == "__main__":
    print("=== Fin-Studio Data Retention & Reset Tool ===")
    print("1) Inserisci il nome di un file specifico (es. NEWS_AAPL...md)")
    print("2) 'ALL' per reset totale (attualmente disabilitato in questo stub)")
    print("3) 'RETENTION' per avviare la pulizia automatica dei file scaduti (TIER C)")
    
    target = input("\nScelta: ").strip()
    
    if target.upper() == 'RETENTION':
        days = input("Quanti giorni di storico vuoi mantenere in Qdrant? (Default: 90): ").strip()
        days_old = int(days) if days.isdigit() else 90
        
        clean_postgres_answer = input("Vuoi cancellare i log anche da Postgres? (s/N - Consigliato N per audit): ").strip().upper()
        clean_pg = clean_postgres_answer == 'S'
        
        run_retention_loop(days_old=days_old, clean_pg=clean_pg)
        
    elif target.upper() == 'ALL':
        print("Funzione ALL disabilitata in questo script di sicurezza.")
    else:
        # Pulisce un singolo file inserito a mano
        clean_qdrant_targeted(target)
        clean_neo4j_targeted(target)
        clean_postgres_targeted(target)
        print(f"\n✨ Database pronti per la ri-ingestion di: {target}")