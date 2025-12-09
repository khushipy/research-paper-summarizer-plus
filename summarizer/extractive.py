# summarizer/extractive.py

import nltk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extractive_summary(text, num_sentences=3):
    """
    Extract top N key sentences using TextRank approach.
    """

    # Step 1: Sentence tokenize
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text  # nothing to summarize

    # Step 2: TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Step 3: Sentence similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Step 4: Build graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Step 5: PageRank to get important sentences
    scores = nx.pagerank(graph)


    # Step 6: Rank sentences
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Step 7: Select top-k sentences
    summary_sentences = [sent for _, sent in ranked[:num_sentences]]

    # Step 8: Preserve original order
    final_summary = " ".join([s for s in sentences if s in summary_sentences])

    return final_summary
