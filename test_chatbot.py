from summarizer.embed_faiss import FaissEngine
from summarizer.chatbot import ResearchChatbot

# Sample chunks from a research paper
chunks = [
    "Deep learning requires large datasets and computational resources.",
    "Our method reduces training time by 40% and improves efficiency.",
    "Experimental results show 92% accuracy on benchmark datasets.",
    "The model is lightweight and suitable for low-resource devices."
]

# Build FAISS engine
faiss_engine = FaissEngine()
emb = faiss_engine.embed(chunks)
faiss_engine.build_index(emb)

# Chatbot instance
bot = ResearchChatbot(faiss_engine)

# Ask a question
question = "How does the method improve efficiency?"

print("Question:", question)
answer = bot.answer(question)
print("\nAnswer:", answer)
