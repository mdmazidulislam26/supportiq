# Evaluation harness. Run: python src/eval.py --mode offline|real [--strategy recall_first|balanced]
import argparse
import json
import numpy as np

from config import ARTIFACT_DIR
from factory import build_stack, calibrate_threshold
from golden import GOLDEN
from rag import RAGService

REFUSED = {"refused_out_of_scope", "refused_by_model"}
REACHED_LLM = {"answered", "refused_by_model", "uncited_fallback"}


# Run the golden set and compute retrieval, answer, refusal and guardrail metrics.
def run_eval(service, golden, split=None):
    items = [g for g in golden if split in (None, g["split"])]
    ans, oos, inj = [], [], []
    n_err = 0
    for g in items:
        r = service.ask(g["q"])
        n_err += r["status"] == "llm_error"
        if g["kind"] == "answerable":
            docs = r["retrieved_docs"]
            rank = next((i for i, d in enumerate(docs, 1) if d in g["docs"]), None)
            correct = r["status"] == "answered" and all(m in r["answer"].lower() for m in g["must_include"])
            ans.append({"hit": rank is not None, "rr": 1 / rank if rank else 0.0, "correct": correct,
                        "refused": r["status"] != "answered", "q": g["q"], "status": r["status"], "answer": r["answer"]})
        elif g["kind"] == "out_of_scope":
            oos.append({"refused": r["status"] in REFUSED, "q": g["q"], "status": r["status"]})
        else:
            inj.append({"blocked": r["status"] == "blocked_injection", "q": g["q"]})
    reached = [a for a in ans if a["status"] in REACHED_LLM]
    m = {
        "llm_error_rate": round(n_err / max(1, len(items)), 3),
        "n": len(items), "n_answerable": len(ans), "n_oos": len(oos), "n_injection": len(inj),
        "retrieval_hit_at_k": round(float(np.mean([a["hit"] for a in ans])), 3) if ans else None,
        "mrr": round(float(np.mean([a["rr"] for a in ans])), 3) if ans else None,
        "answer_accuracy": round(float(np.mean([a["correct"] for a in ans])), 3) if ans else None,
        "false_refusal_rate": round(float(np.mean([a["refused"] for a in ans])), 3) if ans else None,
        # Split the blame: did the cheap gate wrongly refuse, or did the LLM step fail after the gate let it through?
        "gate_false_refusal_rate": round(float(np.mean([a["status"] == "refused_out_of_scope" for a in ans])), 3) if ans else None,
        "n_reached_llm": len(reached),
        "accuracy_when_reached_llm": round(float(np.mean([a["correct"] for a in reached])), 3) if reached else None,
        "oos_refusal_rate": round(float(np.mean([o["refused"] for o in oos])), 3) if oos else None,
        "injection_block_rate": round(float(np.mean([i["blocked"] for i in inj])), 3) if inj else None,
    }
    m["failures"] = [a for a in ans if not a["correct"]] + [o for o in oos if not o["refused"]]
    return m


# Command-line entry point.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="offline", choices=["offline", "real"])
    ap.add_argument("--llm", default=None, choices=["hf", "local"], help="LLM backend for --mode real")
    ap.add_argument("--strategy", default="recall_first", choices=["recall_first", "balanced"])
    args = ap.parse_args()

    retriever, llm = build_stack(args.mode, llm_backend=args.llm)
    info = calibrate_threshold(retriever, strategy=args.strategy)
    print("threshold calibration (dev split):", json.dumps(info))
    service = RAGService(retriever, llm, threshold=info["threshold"])

    report = {"mode": args.mode, "calibration": info}
    for split in ["dev", "test"]:
        report[split] = run_eval(service, GOLDEN, split)
        m = report[split]
        print(f"\n[{split}] n={m['n']}")
        if m["llm_error_rate"] > 0:
            print(f"  WARNING: llm_error_rate={m['llm_error_rate']} -> these numbers are INVALID, fix the LLM first")
        for k in ["retrieval_hit_at_k", "mrr", "answer_accuracy", "false_refusal_rate",
                  "gate_false_refusal_rate", "n_reached_llm", "accuracy_when_reached_llm",
                  "oos_refusal_rate", "injection_block_rate"]:
            print(f"  {k:22s} {m[k]}")
        for f in m["failures"]:
            print("  FAIL:", f["q"], "->", f.get("status"), "|", str(f.get("answer", ""))[:90])
    report["runtime"] = service.stats.summary()
    print("\nruntime stats:", json.dumps(report["runtime"]))
    (ARTIFACT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
