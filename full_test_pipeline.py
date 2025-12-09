# full_test_pipeline.py

print("\n==============================")
print("      LOADING MODULES")
print("==============================\n")

from summarizer.parser import parse_pdf_local
from summarizer.clean import clean_text
from summarizer.chunker import sentence_chunk
from summarizer.extractive import extractive_summary
from summarizer.abstractive_local import abstractive_summary
from summarizer.hybrid import hybrid_summary
from summarizer.embed_faiss import FaissEngine
from summarizer.chatbot import ResearchChatbot


print("\n==============================")
print("      1. PARSING PDF")
print("==============================\n")

pdf_path = "data/sample.pdf"   # <-- change if file name differs
raw_text = parse_pdf_local(pdf_path)
print(raw_text[:300], "...\n")  # only first 300 chars


print("\n==============================")
print("      2. CLEANING TEXT")
print("==============================\n")

cleaned = clean_text(raw_text)
print(cleaned[:300], "...\n")


print("\n==============================")
print("      3. CHUNKING TEXT")
print("==============================\n")

chunks = sentence_chunk(cleaned)
for i, ch in enumerate(chunks[:5]):   # show first 5 chunks
    print(f"--- Chunk {i+1} ---\n{ch}\n")


print("\n==============================")
print("  4. EXTRACTIVE SUMMARY")
print("==============================\n")

extractive = extractive_summary(cleaned, num_sentences=3)
print(extractive, "\n")


print("\n==============================")
print("  5. ABSTRACTION (T5-small)")
print("==============================\n")

abstractive = abstractive_summary(cleaned)
print(abstractive, "\n")


print("\n==============================")
print("  6. HYBRID SUMMARY")
print("==============================\n")

hybrid = hybrid_summary(cleaned)
print(hybrid, "\n")


print("\n==============================")
print("  7. BUILDING FAISS INDEX")
print("==============================\n")

faiss_engine = FaissEngine()
emb = faiss_engine.embed(chunks)
faiss_engine.build_index(emb)
print("FAISS index built on", len(chunks), "chunks.\n")


print("\n==============================")
print("  8. FAISS RETRIEVAL TEST")
print("==============================\n")

query = "What is the main contribution of this research?"
results = faiss_engine.search(query, top_k=3)
print("Query:", query, "\n")
for r in results:
    print("-", r)


print("\n==============================")
print("  9. CHATBOT (RAG) TEST")
print("==============================\n")

bot = ResearchChatbot(faiss_engine)

question = "Explain the key idea of the research paper."
answer = bot.answer(question)

print("Question:", question)
print("Answer:", answer)


print("\n==============================")
print("    PIPELINE TEST COMPLETE")
print("==============================\n")
