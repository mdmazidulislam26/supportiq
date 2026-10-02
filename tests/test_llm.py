# Tests for retry/backoff and the disk cache (no network).
from types import SimpleNamespace
import pytest
from llm import CachedLLM, FakeLLM, HFLLM

MSG = [{"role": "user", "content": "Context:\n[1] (T) Refunds take 14 days.\n\nQuestion: how long refund"}]


# Fake client that fails a few times, then succeeds (tests retry without a network).
class FlakyClient:
    # Fails twice, then succeeds -> proves retry + backoff without any network
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    # Fail until the configured number of failures is used up.
    def _create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("boom")
        msg = SimpleNamespace(content=" ok [1] ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


# Two failures then success; backoff waits 1s then 2s.
def test_retry_then_success():
    sleeps = []
    llm = HFLLM(client=FlakyClient(2), sleep_fn=sleeps.append)
    res = llm.generate(MSG)
    assert res.text == "ok [1]" and res.prompt_tokens == 10
    assert sleeps == [1, 2]


# After the retry budget the call fails loudly.
def test_gives_up_after_retries():
    llm = HFLLM(client=FlakyClient(5), retries=3, sleep_fn=lambda s: None)
    with pytest.raises(RuntimeError):
        llm.generate(MSG)


# The second identical call is served from the cache.
def test_cache_hits_second_time(tmp_path):
    inner = FakeLLM()
    cached = CachedLLM(inner, db_path=str(tmp_path / "c.db"))
    a = cached.generate(MSG)
    b = cached.generate(MSG)
    assert not a.cached and b.cached
    assert inner.calls == 1 and a.text == b.text
