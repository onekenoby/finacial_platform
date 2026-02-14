import json
import re
from neo4j import GraphDatabase
from openai import OpenAI

# --- CONFIGURAZIONE ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password_sicura"

LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
MODEL_ID = "gemma-3-12b"

# --- CONNESSIONE ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY)

def clean_json_output(text):
    """
    Pulisce l'output dell'LLM rimuovendo i backticks di Markdown
    se il modello decide di formattare il codice.
    """
    # Rimuove ```json all'inizio e ``` alla fine
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # Rimuove eventuali ``` generici
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
    return text.strip()

def extract_entities_relationships(text_chunk):
    """
    Usa l'LLM per estrarre nodi e relazioni.
    Rimossa la restrizione response_format per compatibilità con LM Studio.
    """
    system_prompt = """
    Sei un esperto analista dati. Il tuo compito è estrarre entità e relazioni dal testo fornito.
    Restituisci SOLO un JSON valido (nessun testo introduttivo, nessun commento) con questa struttura:
    {
      "nodes": [{"id": "NomeEntità", "type": "Tipo (es. Azienda, Persona, Metrica)"}],
      "edges": [{"source": "NomeEntità1", "target": "NomeEntità2", "relation": "RELAZIONE_IN_MAIUSCOLO"}]
    }
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Testo da analizzare:\n{text_chunk}"}
            ],
            temperature=0,
            # RIMOSSO response_format={"type": "json_object"} per evitare l'errore 400
        )
        
        raw_content = response.choices[0].message.content
        print(f"DEBUG - Raw LLM Output:\n{raw_content[:100]}...") # Vediamo cosa risponde
        
        cleaned_content = clean_json_output(raw_content)
        return json.loads(cleaned_content)
        
    except json.JSONDecodeError:
        print("❌ Errore: L'LLM non ha prodotto un JSON valido.")
        print(f"Output ricevuto: {raw_content}")
        return None
    except Exception as e:
        print(f"❌ Errore Generico: {e}")
        return None

def ingest_into_neo4j(data):
    """
    Scrive i dati nel Grafo usando Cypher.
    """
    if not data: return

    # Query ottimizzata con APOC (se disponibile) o standard Cypher
    with driver.session() as session:
        # 1. Crea Nodi
        print(f"   ↳ Creazione {len(data.get('nodes', []))} nodi...")
        for node in data.get("nodes", []):
            session.run(
                "MERGE (n:Entity {name: $name}) SET n.type = $type",
                name=node['id'], type=node['type']
            )
        
        # 2. Crea Relazioni
        print(f"   ↳ Creazione {len(data.get('edges', []))} relazioni...")
        for edge in data.get("edges", []):
            # Normalizziamo la relazione per evitare caratteri strani in Cypher
            rel_type = edge['relation'].upper().replace(" ", "_").replace("-", "_")
            if not rel_type: rel_type = "RELATED_TO"
            
            cypher = f"""
            MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
            MERGE (a)-[:{rel_type}]->(b)
            """
            try:
                session.run(cypher, source=edge['source'], target=edge['target'])
            except Exception as e:
                print(f"   ⚠️ Errore relazione {edge['source']}->{edge['target']}: {e}")

# --- TEST ---
if __name__ == "__main__":
    # Testo di prova
    testo_finanziario = """
    Elon Musk è il CEO di Tesla Inc.
    Tesla ha riportato un fatturato record nel Q4 2023.
    NVIDIA fornisce i chip AI per i server di Tesla.
    Apple sta competendo con Microsoft nel settore dell'Intelligenza Artificiale.
    """
    
    print("🧠 Estrazione dati con LLM (Gemma-3)...")
    graph_data = extract_entities_relationships(testo_finanziario)
    
    if graph_data:
        print(f"📊 Dati estratti correttamente!")
        print("💾 Scrittura su Neo4j...")
        ingest_into_neo4j(graph_data)
        print("✅ Fatto! Controlla http://localhost:7474")
        
        # Eseguiamo una query di prova per confermare
        with driver.session() as session:
            res = session.run("MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 5")
            print("\n🔍 Verifica dati nel DB:")
            for record in res:
                print(f"   {record['n.name']} --[{record['type(r)']}]--> {record['m.name']}")
    else:
        print("❌ Nessun dato estratto.")

    driver.close()