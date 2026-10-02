# FREE diagnostic: show cached LLM outputs that the guardrails would reject.
# Free diagnostic: read cached LLM outputs (no API call) and flag the ones the guardrails will reject.
import sqlite3
from config import CACHE_DB
from guardrails import extract_citations, model_refused


# Command-line entry point.
def main(limit=60):
    con = sqlite3.connect(CACHE_DB)
    rows = con.execute("SELECT text FROM cache").fetchall()
    con.close()
    print("cached outputs:", len(rows))
    flagged = 0
    for (text,) in rows:
        refused = model_refused(text)
        uncited = not extract_citations(text, 99)
        if refused or uncited:
            flagged += 1
            if flagged <= limit:
                tag = "REFUSAL-MATCH" if refused else "NO-CITATION"
                print(f"[{tag}] {text[:300]!r}")
    print("flagged:", flagged)


if __name__ == "__main__":
    main()
