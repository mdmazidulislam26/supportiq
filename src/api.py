# FastAPI service: POST /ask, GET /health, GET /stats. SUPPORTIQ_MODE=offline|real.
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field

from factory import build_stack, calibrate_threshold, load_threshold
from config import THRESHOLD_PATH
from rag import RAGService

app = FastAPI(title="SupportIQ API", version="1.0")

MODE = os.environ.get("SUPPORTIQ_MODE", "offline")          # offline (CI/dev) | real (HF LLM + MiniLM)
_retriever, _llm = build_stack(MODE)
if not THRESHOLD_PATH.exists():
    calibrate_threshold(_retriever)
SERVICE = RAGService(_retriever, _llm, threshold=load_threshold())


# Request body for POST /ask.
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "threshold": SERVICE.threshold}


@app.post("/ask")
def ask(req: AskRequest):
    return SERVICE.ask(req.question)


@app.get("/stats")
def stats():
    return SERVICE.stats.summary()
