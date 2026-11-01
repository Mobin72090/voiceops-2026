"""Evaluation utilities for VoiceOps 2026.

Provides helpers for:
- Fact checking against practice/questions.csv and practice/conversations.csv:
  - must_include verification (semicolon clauses, slash alternatives)
  - must_not_include verification (forbidden claim detection)
  - honest uncertainty detection on unanswerable queries
  - citation extraction and grounding verification
  - order and proof parsing for conversation handling
"""

import re
from typing import Any

# Keywords and phrases expressing honest uncertainty or inability to confirm from documents
UNCERTAINTY_PHRASES = [
    # English phrases
    "don't have",
    "do not have",
    "cannot confirm",
    "can't confirm",
    "not sure",
    "no information",
    "cannot find",
    "can't find",
    "not able to find",
    "couldn't find",
    "not covered",
    "not available in the documents",
    "cannot verify",
    "can't verify",
    "unable to verify",
    "unable to confirm",
    "connect you to a colleague",
    "connect you with a colleague",
    "connect you to human",
    "human support",
    "human agent",
    "colleague who can help",
    # Hindi and Hinglish phrases
    "jaankari nahi",
    "pata nahi",
    "जानकारी नहीं",
    "पुष्टि नहीं",
    "उपलब्ध नहीं",
    "असमर्थ",
    "माफ़ कीजिए",
    "माफ कीजिए",
    "क्षमा",
    "सहयोगी",
]


def normalize_string(val: Any) -> str:
    """Safely convert any input value to a normalized lowercase string."""
    if val is None:
        return ""
    if isinstance(val, float) and (val != val):  # NaN check
        return ""
    return str(val).strip()


def check_must_include(text: str, must_include_raw: str | None) -> dict:
    """Verify that required facts are present in text.

    Dataset specification:
    - Clauses are separated by ' ; '.
    - Alternatives within a clause are separated by ' / ' (any one is sufficient).
    """
    raw = normalize_string(must_include_raw)
    if not raw or raw.lower() == "nan":
        return {
            "passed": True,
            "total_clauses": 0,
            "matched_clauses": 0,
            "missing_clauses": [],
            "coverage_rate": 1.0,
        }

    clauses = [c.strip() for c in raw.split(";") if c.strip()]
    if not clauses:
        return {
            "passed": True,
            "total_clauses": 0,
            "matched_clauses": 0,
            "missing_clauses": [],
            "coverage_rate": 1.0,
        }

    text_lower = text.lower()
    matched = 0
    missing = []

    for clause in clauses:
        alternatives = [alt.strip().lower() for alt in clause.split("/") if alt.strip()]
        clause_matched = False
        for alt in alternatives:
            # Check for literal match or word match
            if alt in text_lower:
                clause_matched = True
                break

        if clause_matched:
            matched += 1
        else:
            missing.append(clause)

    total = len(clauses)
    return {
        "passed": matched == total,
        "total_clauses": total,
        "matched_clauses": matched,
        "missing_clauses": missing,
        "coverage_rate": round(matched / total, 3) if total > 0 else 1.0,
    }


def check_must_not_include(text: str, must_not_include_raw: str | None) -> dict:
    """Verify that forbidden phrases/claims do NOT appear in text.

    Dataset specification:
    - Forbidden claims separated by ' ; ' or ' / '.
    - Presence of any forbidden claim indicates a violation/hallucination.
    """
    raw = normalize_string(must_not_include_raw)
    if not raw or raw.lower() == "nan":
        return {
            "passed": True,
            "violations": [],
        }

    # Split on semicolons first, then slashes if any
    parts = []
    for item in raw.split(";"):
        for sub in item.split("/"):
            cleaned = sub.strip()
            if cleaned and cleaned.lower() != "nan":
                parts.append(cleaned)

    if not parts:
        return {
            "passed": True,
            "violations": [],
        }

    text_lower = text.lower()
    violations = []
    for forbidden in parts:
        forbidden_clean = forbidden.lower().strip()
        if forbidden_clean in text_lower:
            violations.append(forbidden)

    return {
        "passed": len(violations) == 0,
        "violations": violations,
    }


def check_uncertainty_response(text: str) -> bool:
    """Check if the text expresses honest uncertainty or inability to confirm."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in UNCERTAINTY_PHRASES)


def extract_citations(text: str) -> list[str]:
    """Extract cited section IDs in square brackets (e.g., [RET-2], [DWM-3])."""
    matches = re.findall(r"\[([A-Za-z0-9_.-]+)\]", text)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def check_citation_grounding(text: str, gold_evidence_raw: str | None) -> dict:
    """Verify citations in text against gold evidence section IDs."""
    cited = extract_citations(text)
    raw = normalize_string(gold_evidence_raw)

    gold_sections = set()
    if raw and raw.lower() != "nan":
        for item in raw.split(";"):
            cleaned = item.strip()
            if cleaned:
                # Remove file extensions like products.csv:SKU
                sec_id = cleaned.split(":")[0].strip()
                if not sec_id.endswith(".csv"):
                    gold_sections.add(sec_id)

    cited_set = set(cited)
    has_gold_citation = bool(cited_set & gold_sections) if gold_sections else False

    return {
        "cited_sections": cited,
        "gold_sections": sorted(gold_sections),
        "has_citation": len(cited) > 0,
        "has_gold_citation": has_gold_citation,
    }


def extract_order_id(text: str) -> str | None:
    """Extract an 8-digit order number from customer message (e.g. 8978 2837 or 89782837)."""
    # Spaced pattern: e.g. 8978 2837
    spaced = re.search(r"\b(\d{4})\s+(\d{4})\b", text)
    if spaced:
        return f"{spaced.group(1)}{spaced.group(2)}"

    # Contiguous 8 digits: e.g. 89782837
    contiguous = re.search(r"\b(\d{8})\b", text)
    if contiguous:
        return contiguous.group(1)

    return None


def extract_verification_proof(text: str) -> str | None:
    """Extract 4-digit phone suffix or 6-digit delivery pincode from caller message."""
    # Look for 4 digits explicitly mentioned with digits, last four, digits, etc.
    pincode = re.search(r"\b([1-9]\d{5})\b", text)
    if pincode:
        return pincode.group(1)

    # 4 digits (e.g., "4533", "last four digits 4533")
    four_digits = re.search(r"\b(\d{4})\b", text)
    if four_digits:
        return four_digits.group(1)

    return None
