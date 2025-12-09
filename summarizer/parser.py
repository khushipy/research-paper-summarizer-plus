import fitz  

def parse_pdf_local(pdf_path: str):
    """Extract raw text from PDF (fallback method)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text
