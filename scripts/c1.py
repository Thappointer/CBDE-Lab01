import chromadb
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import os

# 1. Configurar la connexió a la base de dades
db_path = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=db_path)

collection_name = "corpus_collection"
try:
    collection = client.get_collection(name=collection_name)
    print(f"Connectat correctament a la col·lecció: {collection_name}")
except Exception as e:
    print(f"Error en connectar a la col·lecció. Assegura't d'haver executat C0.py primer.\nDetalls: {e}")
    exit()

# 2. Obtenir totes les dades inserides a C0
print("Recuperant els documents emmagatzemats a Chroma...")
data = collection.get(include=["documents"])

ids = data['ids']
sentences = data['documents']
num_records = len(ids)

print(f"S'han recuperat {num_records} registres.")

if num_records == 0:
    print("La col·lecció està buida. Revisa l'script C0.py.")
    exit()

# 3. Carregar el model Transformer
print("Carregant el model 'all-MiniLM-L6-v2' (això pot trigar uns segons)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Generació d'embeddings i mesura del temps d'emmagatzematge
update_times = []

print("Generant embeddings i actualitzant Chroma registre a registre...")
for i in range(num_records):
    doc_id = ids[i]
    sentence = sentences[i]

    # GENERACIÓ (Fora del mesurament de temps)
    # .tolist() converteix l'array de numpy en una llista de floats nadius de Python
    # que és el format que Chroma necessita per als embeddings.
    embedding = model.encode(sentence).tolist()

    # EMMAGATZEMATGE
    start_time = time.perf_counter()
    
    collection.update(
        ids=[doc_id],
        embeddings=[embedding]
    )
    
    end_time = time.perf_counter()
    update_times.append(end_time - start_time)

    if (i + 1) % 1000 == 0:
        print(f"  -> {i + 1}/{num_records} embeddings actualitzats...")

# 5. Càlcul d'estadístiques
min_time = np.min(update_times)
max_time = np.max(update_times)
avg_time = np.mean(update_times)
std_time = np.std(update_times)

print("\n" + "="*50)
print(" RESULTATS: TEMPS D'ACTUALITZACIÓ D'EMBEDDINGS (CHROMA)")
print("="*50)
print(f"Total de registres actualitzats: {len(update_times)}")
print(f"Temps mínim:         {min_time:.6f} segons")
print(f"Temps màxim:         {max_time:.6f} segons")
print(f"Temps mitjà (AVG):   {avg_time:.6f} segons")
print(f"Desviació Est. (STD): {std_time:.6f} segons")
print("="*50)