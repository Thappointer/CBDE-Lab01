import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import psycopg2
import time
import statistics

DB_PARAMS = {
    "dbname": "lab1_db",
    "user": "postgres",
    "password": "admin123",
    "host": "localhost",
    "port": "5432"
}

# Passem uns embeddings falsos (tot zeros) de la mateixa dimensió (384) que el model all-MiniLM-L6-v2.
class DummyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [[0.0] * 384 for _ in input]

def main():
    print("1. Obtenint les dades idèntiques de PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        # Obtenim els mateixos ID i textos que tenim a PostgreSQL
        cursor.execute("SELECT id, sentence FROM corpus_data ORDER BY id ASC;")
        records = cursor.fetchall()
        conn.close()
        print(f"   -> S'han llegit {len(records)} frases de PostgreSQL.")
    except Exception as e:
        print(f"Error connectant a PostgreSQL: {e}")
        return

    print("\n2. Inicialitzant ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Creem la col·lecció aplicant la nostra funció dummy
    dummy_ef = DummyEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="corpus_collection",
        embedding_function=dummy_ef
    )

    print("\n3. Inserint text a ChromaDB frase a frase i mesurant temps...")
    insertion_times = []

    for record in records:
        doc_id = str(record[0])
        sentence = record[1]

        start_time = time.time()
        # Com en P0, fem insercions d'una en una per comparar sota les mateixes condicions
        collection.add(
            documents=[sentence],
            ids=[doc_id]
        )
        end_time = time.time()

        insertion_times.append(end_time - start_time)

        if int(doc_id) % 1000 == 0:
            print(f"   -> Inserides {doc_id} frases...")

    print("\n4. Resultats dels temps d'inserció de text a Chroma (C0):")
    min_time = min(insertion_times)
    max_time = max(insertion_times)
    avg_time = statistics.mean(insertion_times)
    std_dev = statistics.stdev(insertion_times)

    print(f"   Temps Mínim:           {min_time:.6f} segons")
    print(f"   Temps Màxim:           {max_time:.6f} segons")
    print(f"   Temps Mitjà:           {avg_time:.6f} segons")
    print(f"   Desviació Estàndard:   {std_dev:.6f} segons")
    print(f"   Temps TOTAL d'inserció:{sum(insertion_times):.4f} segons")
    
    print("\n[Procés C0 finalitzat. La base de dades local s'ha guardat a './chroma_db']")

if __name__ == "__main__":
    main()