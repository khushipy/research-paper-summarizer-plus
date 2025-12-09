from summarizer.embed_faiss import FaissEngine
from summarizer.abstractive_local import abstractive_summary

class ResearchChatbot:
    def __init__(self, faiss_engine: FaissEngine):
        self.faiss_engine = faiss_engine

    def answer(self, question, top_k=3):
        """
        Retrieve relevant chunks using FAISS and generate an answer 
        using T5-small abstractive summarizer.
        """

        # 1. Retrieve relevant text chunks
        retrieved_chunks = self.faiss_engine.search(question, top_k=top_k)

        # 2. Combine chunks into one context
        context = " ".join(retrieved_chunks)

        # 3. Create a Q&A formatted text
        input_text = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        # 4. Generate abstractive answer
        answer = abstractive_summary(input_text, max_length=120)

        return answer
