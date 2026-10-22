
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = (
    PROJECT_ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "kb"
)


def parse_front_matter(text):
    """Read document metadata from Markdown front matter."""
    metadata = {}

    match = re.match(
        r"\A---\s*\n(.*?)\n---\s*(?:\n|$)",
        text,
        re.DOTALL,
    )

    if not match:
        return metadata, text

    for line in match.group(1).splitlines():
        item = re.match(
            r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$",
            line,
        )
        if item:
            metadata[item.group(1)] = (
                item.group(2).strip().strip("\"'")
            )

    return metadata, text[match.end():]


def load_documents():
    """Load Markdown files and split them into sections."""
    chunks = []

    for file_path in sorted(KB_DIR.glob("*.md")):
        raw_text = file_path.read_text(encoding="utf-8-sig")
        metadata, body = parse_front_matter(raw_text)

        parts = re.split(
            r"(?m)(?=^#{1,6}\s+\[[A-Z0-9_.-]+\])",
            body,
        )

        for part in parts:
            part = part.strip()
            if not part:
                continue

            heading_match = re.search(
                r"(?m)^#{1,6}\s+(.+)$",
                part,
            )

            heading = (
                heading_match.group(1).strip()
                if heading_match
                else metadata.get("title", file_path.stem)
            )

            id_match = re.search(
                r"\[([A-Z0-9_.-]+)\]",
                heading,
            )

            section_id = (
                id_match.group(1)
                if id_match
                else file_path.stem
            )

            chunks.append({
                "source": file_path.name,
                "section_id": section_id,
                "heading": heading,
                "doc_id": metadata.get(
                    "doc_id", file_path.stem
                ),
                "doc_type": metadata.get(
                    "doc_type", "unspecified"
                ),
                "status": metadata.get(
                    "status", "unspecified"
                ),
                "effective_from": metadata.get(
                    "effective_from", "unspecified"
                ),
                "version": metadata.get(
                    "version", "unspecified"
                ),
                "text": part,
            })

    return chunks



STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it", "they",
    "that", "this", "these", "those", "there", "what", "which", "who", "whom",
    "how", "when", "where", "why", "if", "so", "as", "by", "from", "up", "out",
    "about", "into", "through", "after", "before", "just", "now", "still",
    "not", "m", "t", "s", "d", "ll", "ve", "re", "no", "nor",
    # Common Hindi / Hinglish stopwords
    "है", "हैं", "हो", "था", "थी", "थे", "का", "के", "की", "को", "में", "पे",
    "पर", "से", "ने", "भी", "तो", "ही", "क्या", "अब", "हुई", "हुए", "गया", "गई",
    "hai", "hain", "ho", "tha", "thi", "the", "ka", "ke", "ki", "ko", "mein", "pe",
    "par", "se", "ne", "bhi", "to", "hi", "kya", "ab", "wala", "wali", "wale",
    "liye", "hoga", "hogi", "hoge",
}


def stem(word):
    """Normalize common English inflectional suffixes."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("able") and len(word) > 5:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


# Reusable idioms and conversational phrases
GENERAL_PHRASES = (
    ("send back", "return"),
    ("send it back", "return"),
    ("money back", "refund"),
    ("cash on delivery", "cod cash delivery"),
    ("collection fee", "pickup fee"),
    ("collection", "pickup"),
    # Escalation trigger: activity customer did not make (ESC-1)
    ("never placed", "did not make"),
    ("didn't place", "did not make"),
    ("did not place", "did not make"),
    ("never made", "did not make"),
    # Hindi / Hinglish conversational phrases
    ("खोल लिया", "opened seal"),
    ("खोल", "opened"),
    ("कर सकती हूँ", "can"),
    ("कर सकती", "can"),
    ("कर सकते", "can"),
    ("हो सकता है", "can"),
    ("kitne din", "days window"),
    ("कितने दिन", "days window"),
    ("kab tak", "timeline when"),
)

# Individual word mappings for multilingual terms and catalog categories
GENERAL_WORDS = {
    # Hindi/Hinglish vocabulary to policy terminology
    "वापस": "return",
    "लौटाने": "return",
    "लौटा": "return",
    "वापसी": "return",
    "दिनों": "days",
    "दिन": "days",
    "din": "days",
    "dino": "days",
    "टूटा": "damaged",
    "टूट": "damaged",
    "खराब": "damaged",
    "गलत": "wrong",
    "शिकायत": "complaint",
    "रिप्लेसमेंट": "replacement",
    "रिफ़ंड": "refund",
    "कैंसिल": "cancel",
    "रद्द": "cancel",
    "शिप": "shipped",
    "सामान": "item",
    "मिला": "received",
    "parso": "delivered",
    # Physical condition synonyms
    "torn": "damaged",
    "crack": "damaged",
    "cracked": "damaged",
    "broken": "damaged",
    # E-commerce product categories (no policy decisions or durations injected)
    "swimsuit": "swimwear",
    "briefs": "innerwear",
    "kurta": "clothing",
    "jeans": "clothing",
    "lipstick": "beauty",
    "parfum": "beauty",
    "perfume": "beauty",
    "laptop": "electronics",
    "smartphone": "electronics",
    "earbuds": "electronics",
}


def tokenize(text, filter_stops=False):
    """Normalize English, Hindi, and Hinglish searchable terms."""
    text = text.lower().replace("-", " ")

    for source, target in GENERAL_PHRASES:
        text = text.replace(source, target)

    for source, target in GENERAL_WORDS.items():
        text = text.replace(source, target)

    tokens = re.findall(r"[a-z0-9]+", text)
    stemmed = []
    for token in tokens:
        st = stem(token)
        if filter_stops and st in STOP_WORDS:
            continue
        stemmed.append(st)

    return set(stemmed)



def search_documents(question, top_k=5):
    """Rank relevant sections using overlap and document authority."""
    query_terms = tokenize(question, filter_stops=True)

    if not query_terms:
        query_terms = tokenize(question, filter_stops=False)
        if not query_terms:
            return []

    question_lower = question.lower()

    historical_indicators = (
        "historical",
        "previous policy",
        "old policy",
        "version 3",
        "10-day",
        "10 day",
        "before 1 august 2026",
        "before august 2026",
        "delivered before",
        "older electronics",
    )

    asks_historical = any(
        indicator in question_lower
        for indicator in historical_indicators
    )

    authority_weights = {
        "policy": 1.5,
        "agent_guide": 1.2,
        "help": 1.0,
        "faq": 0.8,
        "meta": 0.8,
        "marketing": 0.4,
        "unspecified": 0.7,
    }

    scored_chunks = []

    for chunk in load_documents():
        status = chunk["status"].lower()

        if status == "superseded" and not asks_historical:
            continue

        document_terms = tokenize(
            chunk["heading"] + " " + chunk["text"],
            filter_stops=False,
        )

        overlap = query_terms & document_terms

        if not overlap:
            continue

        overlap_score = len(overlap) / len(query_terms)

        heading_terms = tokenize(chunk["heading"], filter_stops=False)
        heading_score = (
            len(query_terms & heading_terms) / len(query_terms)
        )

        authority = authority_weights.get(
            chunk["doc_type"].lower(), 0.7
        )

        score = overlap_score * authority + heading_score * 0.5

        if status == "superseded":
            score *= 0.7

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {**chunk, "score": round(score, 3)}
        for score, chunk in scored_chunks[:top_k]
    ]


if __name__ == "__main__":
    chunks = load_documents()
    results = search_documents(
        "What is the return policy and return window?"
    )

    print(f"Knowledge base: {KB_DIR}")
    print(f"Sections found: {len(chunks)}")
    print("Superseded sections excluded unless historical intent is detected.")
    print("\nTop search results:")

    for result in results:
        print(
            f"\n[{result['section_id']}] "
            f"{result['heading']} "
            f"(score={result['score']})"
        )
        print(f"Source: {result['source']}")
        print(f"Document type: {result['doc_type']}")
        print(f"Status: {result['status']}")
        print(f"Effective from: {result['effective_from']}")
        print(result["text"][:350])
