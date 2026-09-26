import time
import statistics
import psycopg2
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize


# Mòdul necessari per separar frases la primera vegada
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


DB_PARAMS = {
    "dbname": "lab1_db",
    "user": "postgres",
    "password": "admin123",
    "host": "localhost",
    "port": "5432"
}


NUM_SENTENCES_REQUIRED = 10000


# Hem creat un mètode apart per fer el codi més llegible
def get_sentences_from_corpus(required_sentences):
    print("Iniciant la descàrrega en streaming del BookCorpus...")
    # Descàrrega parcial del dataset
    dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)
   
    sentences_list = []
   
    for row in dataset:
        text = row["text"]
        sentences = sent_tokenize(text)
       
        for sentence in sentences:
            # Filtra les frases buides o massa curtes per homogenitzar el dataset
            if len(sentence.strip()) > 10:
                sentences_list.append(sentence.strip())
               
            if len(sentences_list) >= required_sentences:
                return sentences_list


    return sentences_list


def main():
    # 1. Extracció de dades
    sentences = get_sentences_from_corpus(NUM_SENTENCES_REQUIRED)
    print(f"S'han obtingut {len(sentences)} frases amb èxit.")


    # 2. Connexió a PostgreSQL
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connectant a PostgreSQL: {e}")
        return


    # 3. Creació de la taula de frases
    print("Creant la taula 'corpus_data' a PostgreSQL...")
    cur.execute("""
        DROP TABLE IF EXISTS corpus_data;
        CREATE TABLE corpus_data (
            id SERIAL PRIMARY KEY,
            sentence TEXT NOT NULL
        );
    """)
    conn.commit()


    # 4. Inserció de dades i mesura de temps
    insertion_times = []
    print("Iniciant la inserció de dades a PostgreSQL...")


    for sentence in sentences:
        start_time = time.time()
       
        cur.execute("INSERT INTO corpus_data (sentence) VALUES (%s)", (sentence,))
        conn.commit() # Fem el commit a cada iteració per simular transaccions individuals
       
        end_time = time.time()
        insertion_times.append(end_time - start_time)


        if len(insertion_times) % 1000 == 0:
            print(f"Inserides {len(insertion_times)} frases...")


    cur.close()
    conn.close()


    min_time = min(insertion_times)
    max_time = max(insertion_times)
    avg_time = statistics.mean(insertion_times)
    std_time = statistics.stdev(insertion_times)


    print("\n" + "="*50)
    print("RESULTATS DE RENDIMENT (Inserció de Text a PostgreSQL)")
    print("="*50)
    print(f"Temps Mínim:            {min_time:.6f} segons")
    print(f"Temps Màxim:            {max_time:.6f} segons")
    print(f"Temps Mitjà:            {avg_time:.6f} segons")
    print(f"Desviació Estàndard:    {std_time:.6f} segons")
    print(f"Temps Total Inserció:   {sum(insertion_times):.4f} segons")
    print("="*50)


if __name__ == "__main__":
    main()
