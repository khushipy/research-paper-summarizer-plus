import sys
import os
sys.path.append(os.getcwd())

from summarizer.parser import parse_pdf_local

print(parse_pdf_local("data/sample.pdf")[:1000])
