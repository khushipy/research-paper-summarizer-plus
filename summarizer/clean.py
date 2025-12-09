def clean_text(text: str) -> str:
    import re
    import unicodedata

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # FIX 1: remove hyphen-line-breaks (e.g., "Data-\ndriven" → "Data driven")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1 \2", text)

    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Fix spaced punctuation
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)

    return text.strip()
