# Failure-mode tests: HTTP 402 is not retried, LLM errors degrade gracefully, gate strategy.
from types import SimpleNamespace
from factory import calibrate_threshold
from llm import FakeLLM, HFLLM, LLMError
from rag import RAGService


# Fake client that raises HTTP 402 (Payment Required).
class Client402:
    # Mimics Hugging Face "402 Payment Required": retrying cannot help
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    # Always fail with a 402 error.
    def _create(self, **kw):
        self.calls += 1
        err = RuntimeError("Client error '402 Payment Required' for url x")
        err.response = SimpleNamespace(status_code=402)
        raise err


# A 402 must raise immediately; retrying cannot help.
def test_402_is_not_retried():
    client = Client402()
    llm = HFLLM(client=client, sleep_fn=lambda s: None)
    try:
        llm.generate([{"role": "user", "content": "hi"}])
        assert False, "should have raised"
    except LLMError as e:
        assert e.retryable is False
    assert client.calls == 1


# An LLM outage becomes status 'llm_error', not an exception.
def test_llm_failure_becomes_status_not_crash(stack):
    retriever, _ = stack

    class Broken(FakeLLM):
        def generate(self, messages, **kw):
            raise LLMError("no credits", retryable=False)

    r = RAGService(retriever, Broken(), threshold=0.0).ask("How long is the password reset link valid?")
    assert r["status"] == "llm_error" and "temporarily unavailable" in r["answer"]


# Returns fixed scores so calibration can be tested in isolation.
class StubRetriever:
    # Map each question to a fixed score.
    def __init__(self, scores):
        self.scores = scores

    # Return the preset score for the question.
    def dense_search(self, q, n=1):
        return [(0, self.scores[q])]


# Recall-first keeps the weakest answerable question; balanced sacrifices it.
def test_recall_first_lets_every_dev_answerable_pass():
    golden = [{"q": "a1", "kind": "answerable", "split": "dev"},
              {"q": "a2", "kind": "answerable", "split": "dev"},
              {"q": "o1", "kind": "out_of_scope", "split": "dev"}]
    ret = StubRetriever({"a1": 0.6, "a2": 0.2, "o1": 0.5})
    recall = calibrate_threshold(ret, golden, strategy="recall_first", save=False)
    balanced = calibrate_threshold(ret, golden, strategy="balanced", save=False)
    assert recall["threshold"] < 0.2          # a2 (the weakest answerable) still passes
    assert balanced["threshold"] > 0.2        # balanced sacrifices a2 to block the hard negative o1
