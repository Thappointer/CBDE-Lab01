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

def create_distance_functions(cursor):
    """
    Crea funcions natives a PostgreSQL (PL/pgSQL) per calcular distàncies.
    Forcem el càlcul a FLOAT8 per evitar 'underflow' amb els números tan petits dels embeddings.
    """
    print("Creant/Actualitzant funcions de distància a PostgreSQL...")
   
    # Mètrica 1: Distància Euclidiana
    cursor.execute("""
        CREATE OR REPLACE FUNCTION euclidean_distance(a REAL[], b REAL[]) RETURNS FLOAT8 AS $$
        DECLARE
            s FLOAT8 := 0;
        BEGIN
            FOR i IN 1 .. array_length(a, 1) LOOP
                -- Fem un cast a FLOAT8 per evitar l'error de desbordament per sota (underflow)
                s := s + (a[i]::FLOAT8 - b[i]::FLOAT8) * (a[i]::FLOAT8 - b[i]::FLOAT8);
            END LOOP;
            RETURN sqrt(s);
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
   
    # Mètrica 2: Distància del Cosinus
    cursor.execute("""
        CREATE OR REPLACE FUNCTION cosine_distance(a REAL[], b REAL[]) RETURNS FLOAT8 AS $$
        DECLARE
            dot_product FLOAT8 := 0;
            norm_a FLOAT8 := 0;
            norm_b FLOAT8 := 0;
        BEGIN
            FOR i IN 1 .. array_length(a, 1) LOOP
                -- Cast a FLOAT8 a totes les operacions matemàtiques intenses
                dot_product := dot_product + (a[i]::FLOAT8 * b[i]::FLOAT8);
                norm_a := norm_a + (a[i]::FLOAT8 * a[i]::FLOAT8);
                norm_b := norm_b + (b[i]::FLOAT8 * b[i]::FLOAT8);
            END LOOP;
            IF norm_a = 0 OR norm_b = 0 THEN RETURN 2; END IF;
            RETURN 1 - (dot_product / (sqrt(norm_a) * sqrt(norm_b)));
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    print("Funcions creades amb èxit.")


def main():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()


        create_distance_functions(cur)


        print("\nSeleccionant 10 frases per fer els tests de cerca...")
        cur.execute("SELECT id, sentence, embedding FROM corpus_data ORDER BY id ASC LIMIT 10")
        test_samples = cur.fetchall()
       
        euclidean_times = []
        cosine_times = []


        print("\n=== INICIANT CERCA DE SIMILITUDS (Això pot trigar bastant per culpa de l'Impedance Mismatch!) ===")
       
        for sample in test_samples:
            target_id = sample[0]
            target_sentence = sample[1]
            target_embedding = sample[2]
           
            print(f"\n--- Cercant similituds per la frase ID {target_id} ---")
            print(f"Frase original: '{target_sentence[:60]}...'")


            # Calcula el temps de les distàncies euclidianes
            start_time = time.time()
            cur.execute("""
                SELECT id, sentence, euclidean_distance(embedding, %s::real[]) as dist
                FROM corpus_data
                WHERE id != %s
                ORDER BY dist ASC
                LIMIT 2
            """, (target_embedding, target_id))
            euclidean_results = cur.fetchall()
            end_time = time.time()
           
            e_time = end_time - start_time
            euclidean_times.append(e_time)
           
            print(f" [Euclidiana] Temps: {e_time:.4f} segons")
            for res in euclidean_results:
                print(f"    - ID: {res[0]} | Dist: {res[2]:.4f} | Frase: '{res[1][:60]}...'")


            # Calcula el temps de la distància del cosinus
            start_time = time.time()
            cur.execute("""
                SELECT id, sentence, cosine_distance(embedding, %s::real[]) as dist
                FROM corpus_data
                WHERE id != %s
                ORDER BY dist ASC
                LIMIT 2
            """, (target_embedding, target_id))
            cosine_results = cur.fetchall()
            end_time = time.time()
           
            c_time = end_time - start_time
            cosine_times.append(c_time)
           
            print(f" [Cosinus]    Temps: {c_time:.4f} segons")
            for res in cosine_results:
                print(f"    - ID: {res[0]} | Dist: {res[2]:.4f} | Frase: '{res[1][:60]}...'")


        # Resultats
        print("\n" + "="*50)
        print("=== ESTADÍSTIQUES DE TEMPS DE CONSULTA ===")
        print("="*50)
       
        def print_stats(name, times):
            print(f"Mètrica: {name}")
            print(f" - Mínim:   {min(times):.4f} segons")
            print(f" - Màxim:   {max(times):.4f} segons")
            print(f" - Mitjana: {statistics.mean(times):.4f} segons")
            if len(times) > 1:
                print(f" - Std Dev: {statistics.stdev(times):.4f} segons")
            print("-" * 30)


        print_stats("Distància Euclidiana", euclidean_times)
        print_stats("Distància del Cosinus", cosine_times)
       
        cur.close()
        conn.close()


    except Exception as e:
        print(f"Hi ha hagut un error: {e}")


if __name__ == "__main__":
    main()
