from fastapi import FastAPI, UploadFile, File
import shutil
from summarizer.summarizer import run_full_summarization

app = FastAPI()

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run full summarization
    output = run_full_summarization(temp_path)

    return output
