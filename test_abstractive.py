from summarizer.abstractive_local import abstractive_summary

text = """
Deep learning models require very large datasets and computational power.
This research proposes a new lightweight framework for efficient learning.
"""

print(abstractive_summary(text))
