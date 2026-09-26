import psycopg2
import time
import numpy as np
from sentence_transformers import SentenceTransformer


DB_PARAMS = {
    "dbname": "lab1_db",
    "user": "postgres",
    "password": "admin123",
    "host": "localhost",
    "port": "5432"
}


def main():
    print("Connectant a PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connectant a la base de dades: {e}")
        return


    # Afegim una columna per guardar l'embedding.
    # El tipus de paràmetre és REAL[], el qual és un array de floats de 32 bits. Així l’espai s’ajusta al tamany de les dades produïdes pel model.
    print("Preparant la taula 'corpus_data'...")
    cur.execute("ALTER TABLE corpus_data ADD COLUMN IF NOT EXISTS embedding REAL[];")
    conn.commit()


    # Recuperem totes les frases que encara no tenen embedding
    print("Recuperant frases de la base de dades...")
    cur.execute("SELECT id, sentence FROM corpus_data WHERE embedding IS NULL;")
    rows = cur.fetchall()
   
    if not rows:
        print("No hi ha frases per processar. Totes tenen ja el seu embedding.")
        return


    total_rows = len(rows)
    print(f"S'han recuperat {total_rows} frases.")


    print("Carregant el model all-MiniLM-L6-v2 (això pot trigar una mica la primera vegada)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')


    # Temps d'inserció de cada embedding
    storage_times = []


    print("Generant i emmagatzemant embeddings...")
    for idx, (row_id, sentence) in enumerate(rows):
        # Converteix el resultat (numpy array) a una llista de Python  perquè la llibreria psycopg2 la pugui inserir a PostgreSQL
        vector = model.encode(sentence).tolist()


        # Temporitzador que marca la transacció de la Base de Dades
        start_time = time.time()
       
        cur.execute("UPDATE corpus_data SET embedding = %s WHERE id = %s;", (vector, row_id))
        conn.commit() # Fem commit a cada fila per simular transaccions individuals
       
        end_time = time.time()
       
        storage_times.append(end_time - start_time)


        if (idx + 1) % 1000 == 0:
            print(f"Processades {idx + 1}/{total_rows} frases...")


    print("\n--- RESULTATS DEL RENDIMENT D'EMMAGATZEMATGE D'EMBEDDINGS ---")
    print(f"Total d'embeddings emmagatzemats: {len(storage_times)}")
    print(f"Temps Mínim:   {np.min(storage_times):.6f} segons")
    print(f"Temps Màxim:   {np.max(storage_times):.6f} segons")
    print(f"Temps Mitjà:   {np.mean(storage_times):.6f} segons")
    print(f"Desviació Std: {np.std(storage_times):.6f} segons")
    print("-------------------------------------------------------------")


    cur.close()
    conn.close()
    print("Procés finalitzat. Connexió tancada.")


if __name__ == "__main__":
    main()
