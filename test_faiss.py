from summarizer.embed_faiss import FaissEngine

chunks = [
    "Deep learning requires large datasets.",
    "Our method reduces training time and cost.",
    "Experiments show high accuracy.",
    "Future work includes scaling the model."
]

faiss_engine = FaissEngine()

emb = faiss_engine.embed(chunks)
faiss_engine.build_index(emb)

query = "How does the method improve efficiency?"

results = faiss_engine.search(query, top_k=2)

print("\nTop results:")
for r in results:
    print("-", r)
