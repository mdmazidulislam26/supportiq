# FREE diagnostic (no LLM calls): top dense score per question + threshold sweep.
# Free diagnostic (no LLM calls): top dense score of every golden question + a threshold sweep.
import argparse
import numpy as np

from factory import build_stack
from golden import GOLDEN
from guardrails import redact_pii


# Command-line entry point.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="offline", choices=["offline", "real"])
    args = ap.parse_args()
    retriever, _ = build_stack(args.mode)

    rows = []
    for g in GOLDEN:
        if g["kind"] != "injection":
            rows.append((g["kind"], g["split"], retriever.dense_search(redact_pii(g["q"]), 1)[0][1], g["q"]))

    print("ANSWERABLE (weakest first - these are at risk of being wrongly refused)")
    for kind, split, score, q in sorted([r for r in rows if r[0] == "answerable"], key=lambda r: r[2])[:10]:
        print(f"  {score:.3f} [{split:4s}] {q[:70]}")
    print("OUT-OF-SCOPE (strongest first - these are at risk of slipping through)")
    for kind, split, score, q in sorted([r for r in rows if r[0] == "out_of_scope"], key=lambda r: -r[2]):
        print(f"  {score:.3f} [{split:4s}] {q[:70]}")

    print("\nTHRESHOLD SWEEP  (acc_ok = answerable accepted, oos_rej = out-of-scope rejected)")
    print("  thr   dev_acc_ok dev_oos_rej | test_acc_ok test_oos_rej")
    for t in np.arange(0.05, 0.61, 0.05):
        cells = []
        for split in ["dev", "test"]:
            pos = [r[2] for r in rows if r[0] == "answerable" and r[1] == split]
            neg = [r[2] for r in rows if r[0] == "out_of_scope" and r[1] == split]
            cells += [np.mean([p >= t for p in pos]), np.mean([n < t for n in neg])]
        print(f"  {t:.2f}  {cells[0]:9.2f} {cells[1]:11.2f} | {cells[2]:11.2f} {cells[3]:12.2f}")


if __name__ == "__main__":
    main()
