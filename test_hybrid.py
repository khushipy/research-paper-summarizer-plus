from summarizer.hybrid import hybrid_summary

text = """
Deep learning requires very large datasets and heavy computational power.
This research introduces a new lightweight method that reduces training cost and time.
Experiments show that the method achieves competitive accuracy.
"""

print(hybrid_summary(text))
