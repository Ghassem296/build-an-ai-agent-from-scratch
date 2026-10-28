
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "How can I change my password?",
    "How do I reset my account password?",
    "How do I cook pasta?"
]

vectors = model.encode(texts, normalize_embeddings=True)

print("Number of texts:", len(texts))
print("Vector dimensions:", vectors.shape)

similarities = vectors @ vectors[0]

for text, score in zip(texts, similarities):
    print(f"{score:.4f} - {text}")
