import nltk

def sentence_chunk(text, max_sentences=5, overlap=1):
    """
    Break text into chunks where each chunk contains a fixed number of sentences.
    Overlap ensures smoother context transitions.
    """

    sentences = nltk.sent_tokenize(text)

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

        # move forward with overlap
        start += max_sentences - overlap

    return chunks
