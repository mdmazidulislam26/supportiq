# Input/output guardrails: prompt-injection detection, PII redaction, citation validation.
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|context)",
    r"disregard (all |any )?(the )?(previous|prior|above|context)",
    r"(reveal|print|show|repeat) (me )?(your|the) (system|hidden|secret) (prompt|rules|instructions)",
    r"you are now (in )?(developer|dan|jailbreak|god) ?mode",
    r"developer mode",
    r"\bjailbreak\b",
    r"admin password",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
BD_PHONE = re.compile(r"(?:\+?880|0)1[3-9]\d{8}")
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


# True if the text matches a known prompt-injection pattern.
def is_injection(text):
    return bool(_INJ.search(text))


# Replace card numbers, emails and BD phone numbers with placeholders.
def redact_pii(text):
    # Order matters: cards first so their digits are not eaten by the phone pattern
    text = CARD.sub("[CARD]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = BD_PHONE.sub("[PHONE]", text)
    return text


# Valid chunk numbers (1..n) cited in the answer; made-up numbers are ignored.
def extract_citations(answer, n_chunks):
    # Returns the valid chunk numbers (1..n) cited in the answer; invalid ones are ignored
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    return sorted({n for n in nums if 1 <= n <= n_chunks})


# Remove citations that point to chunks that do not exist.
def strip_invalid_citations(answer, n_chunks):
    def repl(m):
        return m.group(0) if 1 <= int(m.group(1)) <= n_chunks else ""
    return re.sub(r"\[(\d+)\]", repl, answer).strip()


REFUSAL_HINTS = ("i don't know", "i do not know", "cannot find", "not mentioned in the", "no information")


# True if the model said it does not know.
def model_refused(answer):
    low = answer.lower()
    return any(h in low for h in REFUSAL_HINTS)
