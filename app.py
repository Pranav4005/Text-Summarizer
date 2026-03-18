from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Text Summarizer", description="A simple text summarization API using transformers")

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    try:
        summary = summarizer(request.text, max_length=request.max_length, min_length=request.min_length, do_sample=False)
        return SummarizeResponse(summary=summary[0]['summary_text'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Text Summarizer API", "status": "running"}