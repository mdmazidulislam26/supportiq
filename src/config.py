# Central settings: paths, chunking, retrieval sizes, model names. Override the root with SUPPORTIQ_ROOT.
import os
from pathlib import Path

ROOT = Path(os.environ.get("SUPPORTIQ_ROOT", "/content/supportiq"))
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = str(ARTIFACT_DIR / "llm_cache.db")
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"

CHUNK_WORDS = 55          # target words per chunk (sentence-aware)
TOP_K = 4                 # chunks passed to the LLM
CANDIDATES = 10           # per-retriever candidates before fusion
RRF_K = 60
DEFAULT_THRESHOLD = 0.25  # fallback only; real value is calibrated on the dev split

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_PROVIDER = "featherless-ai"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Illustrative price per 1K tokens (0 = free/serverless tier). Set real numbers to see $ estimates.
PRICE_PER_1K_PROMPT = 0.0
PRICE_PER_1K_COMPLETION = 0.0
