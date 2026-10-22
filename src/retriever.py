"""Knowledge-base retriever for ZentroMart VoiceOps 2026.

Supports hybrid semantic retrieval using Google Gen AI embeddings
(gemini-embedding-001) with local caching and an offline lexical fallback.
Preserves document authority weighting, current document precedence,
and superseded-policy filtering.
Integrates verified product catalog facts from data/products.csv for targeted
policy routing (such as high-value damage escalation under DWM-3 & ESC-1).
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

try:
    from catalog import (
        DEFAULT_PRODUCTS_CSV,
        HIGH_VALUE_THRESHOLD,
        get_product_policy_expansion,
        load_products,
        lookup_product,
    )
except ImportError:
    from src.catalog import (
        DEFAULT_PRODUCTS_CSV,
        HIGH_VALUE_THRESHOLD,
        get_product_policy_expansion,
        load_products,
        lookup_product,
    )

KB_DIR = (
    PROJECT_ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "kb"
)
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_CACHE_FILE = CACHE_DIR / "kb_embeddings.json"

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)

AUTHORITY_WEIGHTS = {
    "policy": 1.5,
    "agent_guide": 1.2,
    "help": 1.0,
    "faq": 0.8,
    "meta": 0.8,
    "marketing": 0.4,
    "unspecified": 0.7,
}

HISTORICAL_INDICATORS = (
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


def load_documents(kb_dir: Path | str = KB_DIR):
    """Load Markdown files and split them into sections with full metadata."""
    chunks = []
    kb_path = Path(kb_dir)

    for file_path in sorted(kb_path.glob("*.md")):
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


def compute_kb_hash(kb_dir: Path | str = KB_DIR) -> str:
    """Compute a SHA-256 fingerprint across all KB markdown files for cache invalidation."""
    kb_path = Path(kb_dir)
    hasher = hashlib.sha256()

    for file_path in sorted(kb_path.glob("*.md")):
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(file_path.read_bytes())

    return hasher.hexdigest()


def load_cached_embeddings(
    cache_path: Path | str = DEFAULT_CACHE_FILE,
    expected_model: str = DEFAULT_EMBEDDING_MODEL,
    kb_dir: Path | str = KB_DIR,
) -> list[dict] | None:
    """Load cached chunk embeddings if present, valid, and not stale."""
    cache_file = Path(cache_path)
    if not cache_file.is_file():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        if data.get("model") != expected_model:
            return None

        current_hash = compute_kb_hash(kb_dir)
        if data.get("kb_hash") != current_hash:
            return None

        chunks = data.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return None

        # Verify chunks have embeddings
        for chunk in chunks:
            if "embedding" not in chunk or not chunk["embedding"]:
                return None

        return chunks
    except Exception:
        return None


def save_cached_embeddings(
    chunks: list[dict],
    cache_path: Path | str = DEFAULT_CACHE_FILE,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    kb_dir: Path | str = KB_DIR,
) -> None:
    """Save chunk embeddings to disk atomically alongside the KB fingerprint."""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    kb_hash = compute_kb_hash(kb_dir)
    payload = {
        "version": 1,
        "model": model_name,
        "kb_hash": kb_hash,
        "chunks": chunks,
    }

    temp_file = cache_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    temp_file.replace(cache_file)


def get_embedding_client(api_key: str | None = None):
    """Instantiate a Google GenAI client if credentials are available."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def embed_texts(
    client,
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 50,
) -> list[list[float]] | None:
    """Batch embed a list of texts using the official Google Gen AI SDK."""
    if client is None or not texts:
        return None

    embeddings = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.models.embed_content(
                model=model,
                contents=batch,
            )
            if not response or not hasattr(response, "embeddings"):
                return None
            for item in response.embeddings:
                values = getattr(item, "values", None)
                if values is None and isinstance(item, dict):
                    values = item.get("values")
                embeddings.append(values)
        return embeddings
    except Exception:
        return None


def get_embedded_chunks(
    cache_path: Path | str = DEFAULT_CACHE_FILE,
    client=None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    kb_dir: Path | str = KB_DIR,
    force_refresh: bool = False,
) -> list[dict]:
    """Retrieve KB chunks with embeddings, loading from cache or embedding dynamically."""
    if not force_refresh:
        cached = load_cached_embeddings(
            cache_path=cache_path,
            expected_model=model,
            kb_dir=kb_dir,
        )
        if cached:
            return cached

    chunks = load_documents(kb_dir)
    if not chunks:
        return []

    client = client or get_embedding_client()
    if client is None:
        return chunks

    texts = [f"{c['heading']}\n\n{c['text']}" for c in chunks]
    vectors = embed_texts(client, texts, model=model)
    if vectors and len(vectors) == len(chunks):
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        try:
            save_cached_embeddings(
                chunks,
                cache_path=cache_path,
                model_name=model,
                kb_dir=kb_dir,
            )
        except Exception:
            pass

    return chunks


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vector embeddings."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return dot / (mag1 * mag2)


def search_documents_lexical(question: str, top_k: int = 5, chunks: list[dict] | None = None) -> list[dict]:
    """Rank relevant sections using keyword overlap and document authority (offline fallback)."""
    if not question or not question.strip():
        return []

    expansion = get_product_policy_expansion(question)
    search_query = f"{question}{expansion}" if expansion else question

    query_terms = tokenize(search_query, filter_stops=True)
    if not query_terms:
        query_terms = tokenize(search_query, filter_stops=False)
        if not query_terms:
            return []

    question_lower = search_query.lower()
    asks_historical = any(
        indicator in question_lower
        for indicator in HISTORICAL_INDICATORS
    )

    chunks_to_search = chunks if chunks is not None else load_documents()
    scored_chunks = []

    for chunk in chunks_to_search:
        status = chunk.get("status", "unspecified").lower()

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

        authority = AUTHORITY_WEIGHTS.get(
            chunk.get("doc_type", "unspecified").lower(), 0.7
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
        {
            key: value
            for key, value in chunk.items()
            if key != "embedding"
        } | {"score": round(score, 3)}
        for score, chunk in scored_chunks[:top_k]
    ]


def search_documents_semantic(
    question: str,
    top_k: int = 5,
    chunks: list[dict] | None = None,
    client=None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: Path | str = DEFAULT_CACHE_FILE,
) -> list[dict] | None:
    """Rank sections purely using semantic embedding similarity and document authority."""
    if not question or not question.strip():
        return []

    client = client or get_embedding_client()
    if client is None:
        return None

    if chunks is None:
        chunks = get_embedded_chunks(cache_path=cache_path, client=client, model=model)

    has_embeddings = any(c.get("embedding") for c in chunks)
    if not has_embeddings:
        return None

    expansion = get_product_policy_expansion(question)
    search_query = f"{question}{expansion}" if expansion else question

    q_vectors = embed_texts(client, [search_query], model=model)
    if not q_vectors or not q_vectors[0]:
        return None
    q_emb = q_vectors[0]

    question_lower = search_query.lower()
    asks_historical = any(
        indicator in question_lower
        for indicator in HISTORICAL_INDICATORS
    )

    scored_chunks = []
    for chunk in chunks:
        emb = chunk.get("embedding")
        if not emb:
            continue

        status = chunk.get("status", "unspecified").lower()
        if status == "superseded" and not asks_historical:
            continue

        sim = cosine_similarity(q_emb, emb)
        authority = AUTHORITY_WEIGHTS.get(
            chunk.get("doc_type", "unspecified").lower(), 0.7
        )

        score = max(0.0, sim) * authority
        if status == "superseded":
            score *= 0.7

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {
            key: value
            for key, value in chunk.items()
            if key != "embedding"
        } | {"score": round(score, 3)}
        for score, chunk in scored_chunks[:top_k]
    ]


def search_documents(
    question: str,
    top_k: int = 5,
    mode: str = "hybrid",
    client=None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: Path | str = DEFAULT_CACHE_FILE,
) -> list[dict]:
    """Retrieve top-k relevant knowledge-base sections.

    Supports:
    - mode="hybrid" (default): Combines lexical matching and semantic embeddings.
    - mode="lexical": Uses purely offline lexical keyword overlap.
    - mode="semantic": Uses vector embedding cosine similarity.

    Automatically and gracefully falls back to lexical retrieval if the embedding
    API is unavailable, quota is exhausted, or credentials are missing.
    """
    if not question or not question.strip():
        return []

    if mode == "lexical":
        return search_documents_lexical(question, top_k=top_k)

    # For semantic mode, try semantic search; fallback to lexical if unavailable
    if mode == "semantic":
        res = search_documents_semantic(
            question=question,
            top_k=top_k,
            client=client,
            model=model,
            cache_path=cache_path,
        )
        if res is not None:
            return res
        return search_documents_lexical(question, top_k=top_k)

    # Hybrid mode
    try:
        active_client = client or get_embedding_client()
        chunks = get_embedded_chunks(
            cache_path=cache_path,
            client=active_client,
            model=model,
        )

        has_embeddings = any(c.get("embedding") for c in chunks)
        if active_client is None or not has_embeddings:
            return search_documents_lexical(question, top_k=top_k, chunks=chunks)

        expansion = get_product_policy_expansion(question)
        search_query = f"{question}{expansion}" if expansion else question

        q_vectors = embed_texts(active_client, [search_query], model=model)
        if not q_vectors or not q_vectors[0]:
            return search_documents_lexical(question, top_k=top_k, chunks=chunks)

        q_emb = q_vectors[0]
        question_lower = search_query.lower()
        asks_historical = any(
            indicator in question_lower
            for indicator in HISTORICAL_INDICATORS
        )

        query_terms = tokenize(search_query, filter_stops=True)
        if not query_terms:
            query_terms = tokenize(search_query, filter_stops=False)

        scored_chunks = []
        for chunk in chunks:
            status = chunk.get("status", "unspecified").lower()
            if status == "superseded" and not asks_historical:
                continue

            authority = AUTHORITY_WEIGHTS.get(
                chunk.get("doc_type", "unspecified").lower(), 0.7
            )

            # Lexical component
            doc_terms = tokenize(
                chunk["heading"] + " " + chunk["text"],
                filter_stops=False,
            )
            overlap = query_terms & doc_terms if query_terms else set()
            if overlap:
                overlap_score = len(overlap) / len(query_terms)
                heading_terms = tokenize(chunk["heading"], filter_stops=False)
                heading_score = (
                    len(query_terms & heading_terms) / len(query_terms)
                )
                lex_score = overlap_score * authority + heading_score * 0.5
                if status == "superseded":
                    lex_score *= 0.7
            else:
                lex_score = 0.0

            # Semantic component
            chunk_emb = chunk.get("embedding")
            if chunk_emb:
                sim = cosine_similarity(q_emb, chunk_emb)
                sem_score = max(0.0, sim) * authority
                if status == "superseded":
                    sem_score *= 0.7
            else:
                sim = 0.0
                sem_score = 0.0

            # Include candidates that have lexical overlap or notable semantic similarity
            if lex_score > 0 or sim > 0.65:
                combined_score = 0.7 * lex_score + 0.3 * sem_score
                scored_chunks.append((combined_score, chunk))

        if not scored_chunks:
            return search_documents_lexical(question, top_k=top_k, chunks=chunks)

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            {
                key: value
                for key, value in chunk.items()
                if key != "embedding"
            } | {"score": round(score, 3)}
            for score, chunk in scored_chunks[:top_k]
        ]
    except Exception:
        # Transparent fallback to lexical retrieval on any failure
        return search_documents_lexical(question, top_k=top_k)


if __name__ == "__main__":
    chunks = load_documents()
    sample_query = "What is the return policy and return window?"
    results = search_documents(sample_query)

    print(f"Knowledge base: {KB_DIR}")
    print(f"Sections found: {len(chunks)}")
    print("Superseded sections excluded unless historical intent is detected.")
    print(f"\nTop hybrid search results for '{sample_query}':")

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
        print(result["text"][:300])
