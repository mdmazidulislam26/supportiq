# Builds the offline/real stack and calibrates the out-of-scope threshold on the dev split.
import json
import os
import numpy as np

from config import DEFAULT_THRESHOLD, EMBED_MODEL, THRESHOLD_PATH
from corpus import DOCS
from golden import GOLDEN
from guardrails import redact_pii
from llm import CachedLLM, FakeLLM, HFLLM, LocalLLM
from retrieval import HybridRetriever, MiniLMEmbedder, TfidfEmbedder, chunk_documents


# offline = TF-IDF + fake LLM (CI); real = MiniLM + cached HF/local LLM.
def build_stack(mode="offline", token=None, llm_backend=None):
    chunks = chunk_documents(DOCS)
    texts = [c.title + ". " + c.text for c in chunks]
    if mode == "offline":
        return HybridRetriever(chunks, TfidfEmbedder(texts)), FakeLLM()
    backend = llm_backend or os.environ.get("SUPPORTIQ_LLM", "hf")      # hf | local
    retriever = HybridRetriever(chunks, MiniLMEmbedder(EMBED_MODEL))
    if backend == "local":
        return retriever, CachedLLM(LocalLLM())
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set (needed for llm_backend='hf')")
    return retriever, CachedLLM(HFLLM(token=token))


# Pick the out-of-scope cut-off on the DEV split. recall_first keeps every dev answerable question.
def calibrate_threshold(retriever, golden=GOLDEN, split="dev", save=True, strategy="recall_first", margin=0.9):
    # Dense-score cut-off for the out-of-scope gate, chosen on the DEV split only.
    # Queries are redacted exactly like in RAGService.ask, so calibration sees what serving sees.
    pos = [retriever.dense_search(redact_pii(g["q"]), 1)[0][1] for g in golden if g["split"] == split and g["kind"] == "answerable"]
    neg = [retriever.dense_search(redact_pii(g["q"]), 1)[0][1] for g in golden if g["split"] == split and g["kind"] == "out_of_scope"]
    if strategy == "recall_first":
        # Cost-asymmetric: refusing a real customer is worse than letting a near-domain question reach the LLM
        # (which can still answer "I don't know"). So keep every dev answerable question and reject only clear junk.
        # min() instead of a percentile: with ~12 samples a percentile lands between two points and can reject a valid question.
        threshold = float(margin * min(pos))
    else:
        # Balanced accuracy: treats both errors equally; with very few negatives it over-rejects.
        best_bal, best_ts = -1, []
        for t in sorted(set(pos + neg)):
            bal = (np.mean([p >= t for p in pos]) + np.mean([n < t for n in neg])) / 2
            if bal > best_bal + 1e-9:
                best_bal, best_ts = bal, [t]
            elif abs(bal - best_bal) <= 1e-9:
                best_ts.append(t)
        threshold = float(np.mean([min(best_ts), max(best_ts)])) if best_ts else DEFAULT_THRESHOLD
    info = {"strategy": strategy, "threshold": round(threshold, 4),
            "answerable_recall_dev": round(float(np.mean([p >= threshold for p in pos])), 3),
            "oos_rejected_dev": round(float(np.mean([n < threshold for n in neg])), 3),
            "answerable_score_min": round(float(min(pos)), 3), "answerable_score_median": round(float(np.median(pos)), 3),
            "oos_score_max": round(float(max(neg)), 3), "oos_score_median": round(float(np.median(neg)), 3)}
    if save:
        THRESHOLD_PATH.write_text(json.dumps(info, indent=2))
    return info


# Read the saved threshold (falls back to a default).
def load_threshold():
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text())["threshold"]
    return DEFAULT_THRESHOLD
