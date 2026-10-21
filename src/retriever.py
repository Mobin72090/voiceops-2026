
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


def tokenize(text):
    """Convert text into lowercase searchable terms."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_documents(question, top_k=5):
    """Rank relevant sections using overlap and document authority."""
    query_terms = tokenize(question)

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
            chunk["heading"] + " " + chunk["text"]
        )

        overlap = query_terms & document_terms

        if not overlap:
            continue

        overlap_score = len(overlap) / len(query_terms)

        heading_terms = tokenize(chunk["heading"])
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
