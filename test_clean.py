from summarizer.clean import clean_text

sample = """
This   is   a   test   text.  

Data- 
driven research is important.   

"""

print(clean_text(sample))
