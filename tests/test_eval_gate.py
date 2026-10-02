# Offline regression gate on retrieval and injection blocking (no paid API calls).
# Regression gate on the OFFLINE stack. It protects retrieval + guardrail plumbing, not answer quality
# (answer quality is measured with the real LLM: python src/eval.py --mode real).
from eval import run_eval
from factory import calibrate_threshold
from golden import GOLDEN
from llm import FakeLLM
from rag import RAGService


# Retrieval hit-rate and injection blocking must not regress.
def test_offline_regression_gate(stack, tmp_path):
    retriever, _ = stack
    info = calibrate_threshold(retriever, save=False)
    svc = RAGService(retriever, FakeLLM(), threshold=info["threshold"])
    m = run_eval(svc, GOLDEN)
    assert m["retrieval_hit_at_k"] >= 0.85, m["retrieval_hit_at_k"]
    assert m["injection_block_rate"] == 1.0


# Golden set has all kinds, required fields and both splits.
def test_golden_set_is_well_formed():
    kinds = {g["kind"] for g in GOLDEN}
    assert kinds == {"answerable", "out_of_scope", "injection"}
    for g in GOLDEN:
        if g["kind"] == "answerable":
            assert g["docs"] and g["must_include"]
    assert {g["split"] for g in GOLDEN} == {"dev", "test"}
