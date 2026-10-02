# SupportIQ: an evaluated, guardrailed RAG service

A support assistant for an online learning platform. It answers questions about refunds, deadlines, exams, payments and accounts **only from policy documents**, cites its sources, refuses what it should not answer, and is built to be **measured**, not just demoed.

The interesting part is not the chatbot. It is the engineering around it: a golden-set evaluation harness, an out-of-scope gate calibrated on held-out data, guardrails, caching, retries and tests that run without any paid API.

> Corpus and questions are **synthetic** (10 policy documents). Numbers below show the system works as designed on this data. They are not claims about real-world accuracy.

## How a request flows

```
question
  -> injection check            (blocked: no LLM call)
  -> PII redaction              (email / phone / card never reach the LLM)
  -> hybrid retrieval           (BM25 + MiniLM embeddings, fused with RRF)
  -> out-of-scope gate          (low dense similarity: refused, no LLM call)
  -> LLM                        (disk-cached, retried, fails fast on HTTP 402)
  -> citation check             (no valid [n] citation: safe fallback)
  -> answer + sources + tokens + latency
```

Every outcome has an explicit status: `answered`, `refused_out_of_scope`, `refused_by_model`, `blocked_injection`, `uncited_fallback`, `llm_error`.

| Problem | Design choice |
|---|---|
| Is the bot correct? | Golden set (33 questions) + eval harness with retrieval, accuracy, refusal and guardrail metrics |
| Unrelated questions | Cosine-similarity gate; costs nothing when it refuses |
| Prompt injection | Pattern guardrail before any LLM call |
| PII | Redacted before retrieval and prompting |
| Ungrounded answers | A valid citation is required, otherwise a fallback is returned |
| Repeat runs cost money | Sqlite cache keyed by (model, messages, temperature) |
| Provider outage / no credits | Retry with backoff; 4xx (e.g. 402) is not retried and becomes `llm_error` |
| CI cannot call paid APIs | Offline stack (TF-IDF + deterministic fake LLM) for tests |

## Results

Real run: `all-MiniLM-L6-v2` embeddings + `Qwen2.5-7B-Instruct` via Hugging Face Inference Providers (featherless-ai). Splits: **dev** is used to calibrate the gate, **test** is for reporting.

| | Gate `balanced` (threshold 0.505) | Gate `recall_first` (threshold 0.19) |
|---|---|---|
| Answer accuracy (dev / test) | 0.667 / 0.250 | **0.917 / 0.833** |
| Valid questions wrongly refused by the gate (dev / test) | 3 of 12 / 8 of 12 | **0 / 0** |
| Out-of-scope refusal rate | 1.0 | 1.0 |
| Injection block rate | 1.0 | 1.0 |

Only the gate changed between the two columns. Retrieval, model and prompt were identical.

**Retrieval:** chunk-level hit@4 = 23/24 (the correct fact is inside one of the top-4 chunks). Document-level hit-rate looked better than this, which is why I measure at chunk level.

## Key engineering finding

My first gate wrongly refused **46% of valid questions** (11 of 24). Its threshold was chosen to maximise *balanced accuracy* on a dev split with only 3 out-of-scope questions. One of them (a near-domain question about job guarantees) scored high, so the threshold was pushed up to reject it and many valid questions fell below it.

Fix: change the objective, not the model.

- Refusing a real customer is worse than letting a near-domain question reach the LLM, which can still answer "I don't know".
- So the gate became a cheap first filter that rejects only clearly unrelated questions (`threshold = 0.9 x lowest dev answerable score`), and hard cases are left to the LLM and the citation check (defence in depth).
- Out-of-scope refusal stayed at 1.0 because the LLM refused the near-domain questions the gate let through.

The eval harness now separates **gate errors** from **generation errors** (`gate_false_refusal_rate`, `accuracy_when_reached_llm`), which is what made this visible.

## Limitations

- **Small golden set:** 12 answerable questions per split, so one question is about 8 percentage points. Treat small differences as noise.
- **Synthetic corpus and questions.**
- **Injection guardrail is circular:** the regexes and the test attacks were written together. 100% block rate shows the code works, not that it is robust to unseen attacks.
- **Test split was partly seen** when choosing the threshold (a score table printed both splits).
- **English-only embeddings.** Bangla or mixed-language questions need a multilingual embedding model and a re-run of the eval.
- **Latency:** most calls in the real runs were served from cache, so p95 mixes cached and cold calls. Use cold runs for latency claims.
- **Accuracy is a proxy:** it checks that required facts appear in the answer, not that the answer is faithful.

## Known failures (3 of 24 answerable)

1. A ranking miss: the chunk with the answer was retrieved at rank 7-8, outside the top 4.
2. A wrong "I don't know" even though the answer was in the retrieved context.
3. A correct answer without a citation, which the guardrail replaced with a fallback.

## Next steps

- Cross-encoder reranker (fixes failure 1).
- Citation example in the prompt (fixes failure 3); re-run to measure.
- Try `TOP_K = 3` (same chunk-level hit, about 25% fewer prompt tokens).
- Larger golden set with more near-domain "hard negative" questions and unseen attack phrasings.
- LLM-as-judge faithfulness, validated against a few human labels.
- Multilingual embeddings for Bangla.

## Project layout

```
src/
  config.py         settings (paths, chunking, model names)
  corpus.py         synthetic knowledge base
  golden.py         evaluation questions (answerable / out_of_scope / injection, dev / test)
  retrieval.py      chunking, BM25 + dense retrieval, RRF
  llm.py            HF / local / fake LLM, retry, cache, stats
  guardrails.py     injection, PII, citation checks
  rag.py            RAGService (request flow above)
  factory.py        build offline/real stack, threshold calibration
  eval.py           evaluation harness
  score_table.py    free diagnostic: scores and threshold sweep
  inspect_cache.py  free diagnostic: rejected LLM outputs
  api.py            FastAPI service
tests/              30 tests, no network or credits needed
```

## Run it

The whole project is a Colab notebook (`Milestone_6_week_clean.ipynb`): run it top to bottom. Sections 0-4 are free. The real-LLM evaluation is opt-in (`RUN_REAL_EVAL = True`) because it spends API credits for uncached prompts.

Commands (from `/content/supportiq`):

```bash
python -m pytest -q                          # 30 tests, offline
python src/eval.py --mode offline            # plumbing + guardrails, no token
python src/score_table.py --mode real        # free: score distribution + threshold sweep
python src/eval.py --mode real               # needs HF_TOKEN; --strategy balanced for the old gate
python src/eval.py --mode real --llm local   # small free local model (a different model; report separately)
```

Set `SUPPORTIQ_ROOT` to run outside Colab, and `HF_TOKEN` for real-LLM runs. Never commit the token.

### API

```bash
SUPPORTIQ_MODE=offline uvicorn api:app --app-dir src --port 8000
```

```
POST /ask    {"question": "How long is the password reset link valid?"}
GET  /health
GET  /stats  (requests, status counts, p50/p95 latency, tokens, cache hit rate)
```

Response fields: `answer`, `status`, `sources`, `retrieved_docs`, `latency_s`, `prompt_tokens`, `completion_tokens`, `cached`, `est_cost`.

`offline` mode uses TF-IDF and a fake LLM, so its answers are for plumbing and demos only. Use `SUPPORTIQ_MODE=real` (with `HF_TOKEN`) for real answers.

## Tech

Python, FastAPI, sentence-transformers (MiniLM), rank-bm25, scikit-learn, Hugging Face Inference Providers, sqlite3, pytest.
