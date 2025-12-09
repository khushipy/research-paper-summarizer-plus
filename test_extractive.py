from summarizer.extractive import extractive_summary

text = """
Machine learning models require large datasets. 
Deep learning models also require extensive computational power. 
This research introduces an efficient approach. 
Our method reduces both cost and training time. 
Experimental results show improvements.
"""

summary = extractive_summary(text, num_sentences=2)
print(summary)
