from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time

print("1. Descarregant un fragment del BookCorpus...")
# Hem afegit bookcorpus/bookcorpus i trust_remote_code=True
dataset = load_dataset("bookcorpus/bookcorpus", split="train[:1000]", trust_remote_code=True)

# Agafem només 5 frases per veure com són
frases_exemple = dataset['text'][:5]
print("\nFrases d'exemple:")
for i, frase in enumerate(frases_exemple):
    print(f"  {i+1}: {frase}")

print("\n2. Carregant el model Transformer (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n3. Generant els embeddings...")
inici = time.time()
embeddings = model.encode(frases_exemple)
fi = time.time()

print(f"\nS'han generat {len(embeddings)} embeddings en {fi - inici:.4f} segons.")
print(f"Dimensions de cada vector: {len(embeddings[0])}.")