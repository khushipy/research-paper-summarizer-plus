# summarizer/embed_faiss.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def embed(self, chunks):
        """Convert chunks to embeddings using MiniLM."""
        self.text_chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")

    def build_index(self, embeddings):
        """Create FAISS index."""
        dimension = embeddings.shape[1]  # 384 for MiniLM
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        """Retrieve top-K similar chunks."""
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results
