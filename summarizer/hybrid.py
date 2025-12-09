# summarizer/hybrid.py

from sentence_transformers import SentenceTransformer, util
from summarizer.extractive import extractive_summary
from summarizer.abstractive_local import abstractive_summary

model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_summary(text):
    """
    Combines extractive + abstractive + embedding fusion.
    """

    # 1. Extractive component
    ext = extractive_summary(text, num_sentences=3)

    # 2. Abstractive component
    abs_sum = abstractive_summary(text, max_length=120)

    # 3. Embed both
    embeddings = model.encode([ext, abs_sum], convert_to_tensor=True)

    # 4. Blend using similarity score
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # 5. Fusion rule
    if similarity > 0.6:
        # Very similar → combine both
        return f"{ext} {abs_sum}"
    else:
        # Dissimilar → abstractive is more human-like
        return abs_sum
