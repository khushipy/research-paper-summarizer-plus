from summarizer.chunker import sentence_chunk

text = """
Machine learning models require large datasets. 
However, deep learning models also require extensive computational power. 
This research introduces an efficient approach. 
Our method reduces both cost and training time. 
Experimental results show improvements.
"""

chunks = sentence_chunk(text, max_sentences=2, overlap=1)

for idx, c in enumerate(chunks):
    print(f"\n--- Chunk {idx+1} ---")
    print(c)
