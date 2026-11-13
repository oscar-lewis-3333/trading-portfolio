
#structured reading of news articles by an LLM, and checks on the returns

import re
import unicodedata

CATEGORIES = ["earnings", "guidance", "m_and_a", "legal_regulatory", "analyst_rating", "macro", "product", "management", "capital_return", "other"]

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subjects", "category", "facts", "direction", "magnitude", "surprise", "novelty", "source_quality"],
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "is_primary_subject"],
                "properties": {"symbol": {"type": "string"}, "is_primary_subject": {"type": "boolean"}},
            },
        },
        "category": {"type": "string", "enum": CATEGORIES},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "quote"],
                "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}},
            },
        },
        "direction": {"type": "string", "enum": ["positive", "negative", "mixed", "neutral"]},
        "magnitude": {"type": "integer", "minimum": 1, "maximum": 5},
        "surprise": {"type": "string", "enum": ["above_expectations", "below_expectations", "in_line", "not_applicable", "unclear"]},
        "novelty": {"type": "string", "enum": ["new", "follow_up", "rehash"]},
        "source_quality": {"type": "string", "enum": ["primary", "major_outlet", "other", "rumour"]},
    },
}

EXTRACTION_TEMPLATE = """You are extracting structured facts from one financial news item for a research log.

Rules:
- Use only the text below. Do not use outside knowledge about prices or what happened next.
- Every fact must include a quote copied exactly, word for word, from the text (5 to 40 words).
- direction is the likely effect on the primary subject's share price given only this text.
- magnitude: 1 = routine noise, 3 = notable for this company, 5 = thesis-changing.
- surprise compares reported results with expectations only if the text states the expectation.
- source_quality: primary = company release or filing; rumour = unconfirmed or anonymous sourcing.
- If the item is not about any of these symbols, mark none as primary subject.

Symbols tagged on this item: {symbols}
Published: {published_at}

TEXT:
{text}
"""


def build_extraction_text(version, max_chars=12000):
    return EXTRACTION_TEMPLATE.format(symbols=", ".join(version["symbols"]), published_at=version["published_at"], text=version["text"][:max_chars])


def _normalise(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def grounding(extraction, text):
    #share of facts whose quote appears verbatim

    facts = extraction.get("facts") or []
    haystack = _normalise(text)
    failed = [f["quote"] for f in facts if not f.get("quote") or _normalise(f["quote"]) not in haystack]
    return {"n_facts": len(facts), "n_failed": len(failed), "failed_quotes": failed[:5], "passed": len(facts) > 0 and not failed}


def primary_subjects(extraction):
    return sorted({s["symbol"].upper() for s in extraction.get("subjects", []) if s.get("is_primary_subject")})


def symbol_mentioned(symbol, company_name, text):
    #stock must actually be named in the text, by ticker, or by the first word(s) of its name
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol.upper())}(?![A-Za-z0-9])", text or ""):
        return True
    
    haystack = _normalise(text)
    name = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|holdings|group|the)\b\.?", "", _normalise(company_name)).strip(" ,.")
    key = " ".join(name.split()[:2])
    return bool(key) and key in haystack


def compare(a, b):
    #field by field agreement between the two extractors

    if a is None or b is None:
        return {"both_present": False}
    return {"both_present": True,
        "direction": a["direction"] == b["direction"],
        "category": a["category"] == b["category"],
        "novelty": a["novelty"] == b["novelty"],
        "source_quality": a["source_quality"] == b["source_quality"],
        "magnitude_diff": abs(a["magnitude"] - b["magnitude"]),
        "primary_subjects": primary_subjects(a) == primary_subjects(b)}
