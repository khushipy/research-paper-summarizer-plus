"""
Full scientific-paper summarization pipeline.
User uploads a PDF → returns structured research insights.
"""

import re

from summarizer.parser import parse_pdf_local
from summarizer.clean import clean_text
from summarizer.chunker import sentence_chunk
from summarizer.extractive import extractive_summary
from summarizer.abstractive_local import abstractive_summary
from summarizer.hybrid import hybrid_summary
from summarizer.embed_faiss import FaissEngine
from summarizer.chatbot import ResearchChatbot


# =====================================
#       GENERIC UTILS
# =====================================

def extract_title(text: str):
    lines = text.strip().split("\n")
    title = lines[0].strip() if len(lines[0]) < 150 else None
    return title or "Title not detected"


def extract_formulas(text: str):
    formulas = set()

    # 1. Inline LaTeX math: $ ... $
    inline = re.findall(r"\$(.+?)\$", text)
    formulas.update(inline)

    # 2. Display math: \[ ... \]
    display = re.findall(r"\\\[(.+?)\\\]", text, flags=re.DOTALL)
    formulas.update(display)

    # 3. Equation environment
    equation_env = re.findall(
        r"\\begin{equation\*?}(.+?)\\end{equation\*?}",
        text,
        flags=re.DOTALL
    )
    formulas.update(equation_env)

    # 4. Simple symbolic formulas (e.g., F = ma)
    symbolic = re.findall(
        r"[A-Za-z]\w*\s*(=|\+|\-|\*|/|\^)\s*[A-Za-z0-9\^\+\-\*/=\(\) ]+",
        text
    )
    formulas.update(symbolic)

    # Clean & return
    cleaned = [f.strip() for f in formulas if len(f.strip()) > 3]
    return cleaned[:10]


# =====================================
#     SECTION DETECTION HELPERS
# =====================================

def find_section(text: str, section_name: str):
    t = text.lower()
    s = section_name.lower()

    idx = t.find(s)
    if idx == -1:
        return ""

    return text[idx: idx + 1800]


def extract_introduction(text):
    return (
        find_section(text, "introduction")
        or find_section(text, "background")
        or ""
    )


def extract_method(text):
    return (
        find_section(text, "methodology")
        or find_section(text, "methods")
        or find_section(text, "approach")
        or find_section(text, "proposed method")
        or ""
    )


def extract_results(text):
    return (
        find_section(text, "results")
        or find_section(text, "experiments")
        or find_section(text, "evaluation")
        or ""
    )


def extract_conclusion(text):
    return (
        find_section(text, "conclusion")
        or find_section(text, "discussion")
        or find_section(text, "future work")
        or ""
    )


# =====================================
#       SECTION-WISE SUMMARIZATION
# =====================================

def abstractive_section_summary(text, label="section"):
    if not text or not text.strip():
        return f"{label.capitalize()} section not found."
    try:
        return abstractive_summary(text)
    except:
        return f"Could not summarize {label} section."


# =====================================
#        MAIN PIPELINE
# =====================================

def run_full_summarization(pdf_path: str, user_question: str = None):
    """
    Full pipeline:
    1. Parse → Clean
    2. Section extraction
    3. Extractive / Abstractive / Hybrid summaries
    4. FAISS retrieval
    5. RAG chatbot answer
    """

    # 1. Parse PDF
    raw_text = parse_pdf_local(pdf_path)

    # 2. Clean text
    cleaned = clean_text(raw_text)

    # 3. Metadata extraction
    title = extract_title(raw_text)
    formulas = extract_formulas(raw_text)

    # 4. Section extraction
    intro_text = extract_introduction(cleaned)
    method_text = extract_method(cleaned)
    results_text = extract_results(cleaned)
    conclusion_text = extract_conclusion(cleaned)

    # 5. Section-wise abstractive summaries
    intro_summary = abstractive_section_summary(intro_text, "introduction")
    method_summary = abstractive_section_summary(method_text, "methodology")
    results_summary = abstractive_section_summary(results_text, "results")
    conclusion_summary = abstractive_section_summary(conclusion_text, "conclusion")

    # 6. Chunking
    chunks = sentence_chunk(cleaned)

    # 7. Full-paper summaries
    extractive = extractive_summary(cleaned, num_sentences=3)
    abstractive = abstractive_summary(cleaned)
    hybrid = hybrid_summary(cleaned)

    # 8. FAISS Index
    faiss_engine = FaissEngine()
    embeddings = faiss_engine.embed(chunks)
    faiss_engine.build_index(embeddings)

    # 9. FAISS retrieval (default question)
    default_query = "What is the main contribution of this research?"
    retrieved_chunks = faiss_engine.search(default_query, top_k=3)

    # 10. Chatbot answer
    chatbot_answer = None
    if user_question:
        bot = ResearchChatbot(faiss_engine)
        chatbot_answer = bot.answer(user_question)

    # 11. Output structure
    return {
        "paper_title": title,

        "summary": {
            "extractive": extractive,
            "abstractive": abstractive,
            "hybrid": hybrid
        },

        "section_summaries": {
            "introduction": intro_summary,
            "methodology": method_summary,
            "results": results_summary,
            "conclusion": conclusion_summary
        },

        "metadata": {
            "formulas_detected": formulas,
        },

        "retrieval": retrieved_chunks,
        "chatbot_answer": chatbot_answer
    }

