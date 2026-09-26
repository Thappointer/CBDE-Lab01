import chromadb
import time
import numpy as np
import os

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME_L2 = "corpus_collection" # Posa el nom que vas usar a C0.py
COLLECTION_NAME_COSINE = "corpus_collection_cosine"

TEST_IDS = [str(i) for i in range(1, 11)]

def print_stats(metric_name, times):
    times_array = np.array(times)
    print(f"\n--- Estadístiques de temps per a la mètrica: {metric_name.upper()} ---")
    print(f"Mínim:             {np.min(times_array):.6f} segons")
    print(f"Màxim:             {np.max(times_array):.6f} segons")
    print(f"Mitjana:           {np.mean(times_array):.6f} segons")
    print(f"Desviació Estàndard: {np.std(times_array):.6f} segons")
    print("-" * 65)

def run_queries(collection, metric_name, test_data):
    times = []
    
    print(f"\nIniciant cerca amb mètrica {metric_name}...")
    
    for i in range(len(test_data['ids'])):
        q_id = test_data['ids'][i]
        q_doc = test_data['documents'][i]
        q_emb = test_data['embeddings'][i]
        
        start_time = time.time()
        
        # n_results=3 perquè el primer resultat serà la pròpia frase que estem buscant
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"\n[Consulta {i+1}] ID original: {q_id} | Text: '{q_doc[:50]}...'")
        
        # Filtrem el propi document de la cerca per obtenir només els top-2 "altres" documents
        match_count = 0
        for res_id, res_doc, res_dist in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
            if str(res_id) != str(q_id) and match_count < 2:
                print(f"   -> Match {match_count+1} (ID: {res_id}, Distància: {res_dist:.4f}): '{res_doc}'")
                match_count += 1

    print_stats(metric_name, times)

def main():
    print("Connectant a ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 1. Obtenim la col·lecció existent (per defecte L2 / Euclidiana)
    collection_l2 = client.get_collection(name=COLLECTION_NAME_L2)
    
    # Obtenim les 10 frases i els seus embeddings ja generats
    test_data = collection_l2.get(ids=TEST_IDS, include=['embeddings', 'documents'])
    
    if len(test_data['ids']) != len(TEST_IDS):
        print(f"ALERTA: Només s'han trobat {len(test_data['ids'])} dels {len(TEST_IDS)} IDs a ChromaDB.")
    
    # EXECUCIÓ 1: Mètrica Euclidiana (L2)
    run_queries(collection_l2, "Euclidiana (L2)", test_data)
    
    # 2. Preparació per a la mètrica Cosinus
    # Com Chroma no permet canviar la distància al vol, 
    # necessitem una col·lecció nova configurada amb espai "cosine".
    try:
        collection_cosine = client.get_collection(name=COLLECTION_NAME_COSINE)
        print(f"\nCol·lecció {COLLECTION_NAME_COSINE} ja existeix. No cal bolcar les dades.")
    except Exception:
        print(f"\nCreant col·lecció nova per al Cosinus ({COLLECTION_NAME_COSINE})...")
        collection_cosine = client.create_collection(
            name=COLLECTION_NAME_COSINE,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Bolquem les 10.000 dades de L2 a la nova de Cosinus
        all_data = collection_l2.get(include=['embeddings', 'documents', 'metadatas'])
        batch_size = 5000
        total = len(all_data['ids'])
        
        for i in range(0, total, batch_size):
            collection_cosine.add(
                ids=all_data['ids'][i:i+batch_size],
                embeddings=all_data['embeddings'][i:i+batch_size],
                documents=all_data['documents'][i:i+batch_size],
                metadatas=all_data['metadatas'][i:i+batch_size] if all_data['metadatas'] else None
            )
        print(f"Copiats {total} registres a la col·lecció Cosinus.")
        
    # EXECUCIÓ 2: Mètrica Cosinus
    run_queries(collection_cosine, "Cosinus (Cosine)", test_data)

if __name__ == "__main__":
    main()