# LLM clients (HF API, local, fake for CI), retry/backoff, disk cache, and runtime stats.
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from config import CACHE_DB, LLM_MODEL, LLM_PROVIDER


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False


# Provider failure. retryable=False means retrying is pointless (bad key, no credits, bad request).
class LLMError(RuntimeError):
    # retryable=False -> the caller must not retry (bad key, no credits, bad request)
    def __init__(self, msg, retryable=True):
        super().__init__(msg)
        self.retryable = retryable


# Extract the HTTP status code from an exception (attribute or message).
def _status_code(e):
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code is None:
        m = re.search(r"(?:Client|Server) error '(\d{3})", str(e))
        code = int(m.group(1)) if m else None
    return code


# Rough token estimate, used only when the API returns no usage numbers.
def _est_tokens(text):
    return max(1, int(len(text.split()) * 1.3))   # rough fallback when the API gives no usage


# Hugging Face Inference API client with retry + exponential backoff.
class HFLLM:
    # client is injectable so retry logic can be tested without network
    def __init__(self, token=None, model=LLM_MODEL, provider=LLM_PROVIDER, client=None,
                 retries=3, sleep_fn=time.sleep):
        if client is None:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=provider, api_key=token)
        self.client, self.model, self.retries, self.sleep_fn = client, model, retries, sleep_fn

    # Call the model; retry transient errors, fail fast on 4xx (e.g. 402 = credits exhausted).
    def generate(self, messages, max_tokens=300, temperature=0.0):
        last_err = None
        for attempt in range(self.retries):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature)
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                p = getattr(usage, "prompt_tokens", None) or _est_tokens(" ".join(m["content"] for m in messages))
                c = getattr(usage, "completion_tokens", None) or _est_tokens(text)
                return LLMResult(text.strip(), int(p), int(c), time.perf_counter() - t0)
            except Exception as e:                       # network / rate limit / provider error
                last_err = e
                code = _status_code(e)
                if code in (400, 401, 402, 403, 404):    # retrying can never succeed (402 = credits exhausted)
                    raise LLMError(f"HTTP {code}: {str(e)[:160]}", retryable=False)
                print(f"LLM attempt {attempt + 1}/{self.retries} failed: {type(e).__name__}")
                self.sleep_fn(2 ** attempt)              # exponential backoff 1s, 2s, 4s
        raise LLMError(f"LLM failed after {self.retries} attempts: {last_err}", retryable=True)


# Small open model running inside Colab (free; weaker and slower than the API model).
class LocalLLM:
    # Runs a small open model inside Colab (free, no API credits). Slower and weaker than a 7B API model.
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = model                                   # name (also used in the cache key)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")

    # Greedy decoding (deterministic), so results are reproducible and cacheable.
    def generate(self, messages, max_tokens=300, temperature=0.0):
        t0 = time.perf_counter()
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.lm.device)
        out = self.lm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)   # greedy = deterministic
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return LLMResult(text.strip(), int(inputs["input_ids"].shape[1]), int(len(gen)), time.perf_counter() - t0)


# Deterministic extractive 'LLM' for CI: returns the best-matching context sentence + citation.
class FakeLLM:
    # Deterministic extractive "LLM" for CI: answers with the best-matching context sentence + citation.
    STOP = set("the a an of to is are what how do does i my can in for and or on at it be when who which".split())

    # Count calls so tests can assert that the LLM was (not) used.
    def __init__(self):
        self.calls = 0

    # Pick the context sentence with the most word overlap with the question.
    def generate(self, messages, max_tokens=300, temperature=0.0):
        self.calls += 1
        user = messages[-1]["content"]
        question = user.split("Question:")[-1].strip().lower()
        q_words = {w for w in re.findall(r"[a-z0-9%]+", question) if w not in self.STOP}
        best, best_score, best_idx = None, -1, 1
        for m in re.finditer(r"\[(\d+)\] \(.*?\) (.*?)(?=\n\[\d+\] |\n\nQuestion:|\Z)", user, flags=re.S):
            idx, body = int(m.group(1)), m.group(2)
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                score = len(q_words & set(re.findall(r"[a-z0-9%]+", sent.lower())))
                if score > best_score:
                    best, best_score, best_idx = sent, score, idx
        text = f"{best} [{best_idx}]" if best else "I don't know."
        return LLMResult(text, _est_tokens(user), _est_tokens(text), 0.0)


# Disk cache keyed by (model, messages, temperature): repeat runs are free and reproducible.
class CachedLLM:
    # Disk cache keyed by (model, messages, temperature): re-running the eval costs nothing and is reproducible.
    def __init__(self, inner, db_path=CACHE_DB):
        self.inner, self.db_path = inner, db_path
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, text TEXT, p INT, c INT)")
        con.commit()
        con.close()

    # Stable hash of everything that affects the output.
    def _key(self, messages, temperature):
        raw = json.dumps({"m": getattr(self.inner, "model", "fake"), "msgs": messages, "t": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # Return the cached answer if present, otherwise call the model and store the result.
    def generate(self, messages, max_tokens=300, temperature=0.0):
        key = self._key(messages, temperature)
        con = sqlite3.connect(self.db_path)
        row = con.execute("SELECT text, p, c FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            con.close()
            return LLMResult(row[0], row[1], row[2], 0.0, cached=True)
        res = self.inner.generate(messages, max_tokens, temperature)
        con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (key, res.text, res.prompt_tokens, res.completion_tokens))
        con.commit()
        con.close()
        return res


class Stats:
    def __init__(self):
        self.rows = []

    def add(self, status, latency, ptok, ctok, cached):
        self.rows.append((status, latency, ptok, ctok, cached))

    def summary(self):
        if not self.rows:
            return {"requests": 0}
        lat = np.array([r[1] for r in self.rows])
        by_status = {}
        for r in self.rows:
            by_status[r[0]] = by_status.get(r[0], 0) + 1
        llm_calls = [r for r in self.rows if r[2] > 0]
        return {
            "requests": len(self.rows), "by_status": by_status,
            "latency_p50_s": round(float(np.percentile(lat, 50)), 4),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 4),
            "prompt_tokens": int(sum(r[2] for r in self.rows)),
            "completion_tokens": int(sum(r[3] for r in self.rows)),
            "llm_calls": len(llm_calls),
            "cache_hit_rate": round(sum(1 for r in llm_calls if r[4]) / len(llm_calls), 3) if llm_calls else 0.0,
        }
