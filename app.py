from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import os

from textSummarizer import pipeline
from textSummarizer.pipeline.prediction_pipeline import PredictionPipeline


# ====================================
# FastAPI App
# ====================================

app = FastAPI(title="Text Summarizer")


# ====================================
# Static + Templates
# ====================================

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# ====================================

# ====================================
# Home Page
# ====================================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ====================================
# Prediction API
# ====================================

@app.post("/predict")
async def predict(text: str = Form(...)):

    try:

        print("Loading model...")

        prediction_pipeline = PredictionPipeline()

        print("Generating summary...")

        summary = prediction_pipeline.predict(text)

        print("Summary generated")

        return JSONResponse(
            content={
                "summary": summary
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# ====================================
# Run App
# ====================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
