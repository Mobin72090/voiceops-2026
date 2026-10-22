"""Product catalog lookup and cross-referencing module for ZentroMart VoiceOps 2026.

Provides verified product facts from data/products.csv, supporting:
- Explicit SKU extraction and lookup (e.g. ELE-0023).
- Reliable, unambiguous product-title matching without guessing.
- Policy cross-referencing for high-value items (> ₹10,000 threshold under DWM-3 & ESC-1)
  versus standard damage resolution (DWM-1 & DWM-2).
"""

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRODUCTS_CSV = (
    PROJECT_ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "data"
    / "products.csv"
)

# Verified boundary from KB policies:
# - pol_damaged_wrong_missing.md [DWM-3]: "Claims for items worth more than ₹10,000..."
# - agent_escalation_handoff.md [ESC-1]: "Damaged, defective or wrong item worth more than ₹10,000."
HIGH_VALUE_THRESHOLD = 10000

# Damage indicators used across e-commerce support queries (English, Hindi, Hinglish)
DAMAGE_KEYWORDS = (
    "crack",
    "cracked",
    "damage",
    "damaged",
    "broken",
    "torn",
    "defective",
    "defect",
    "faulty",
    "टूटा",
    "टूट",
    "खराब",
)

_CATALOG_CACHE: dict[str, dict] = {}
_CACHED_CATALOG_PATH: Path | None = None


def load_products(catalog_path: Path | str = DEFAULT_PRODUCTS_CSV) -> dict[str, dict]:
    """Load product catalog CSV into a dictionary indexed by SKU."""
    global _CATALOG_CACHE, _CACHED_CATALOG_PATH

    path = Path(catalog_path)
    if not path.is_file():
        return {}

    # Return cached dictionary if same path
    if _CATALOG_CACHE and _CACHED_CATALOG_PATH == path:
        return _CATALOG_CACHE

    products = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sku_raw = row.get("sku")
                if not sku_raw:
                    continue
                sku = sku_raw.strip().upper()
                price_str = row.get("price_inr", "").strip()
                price = int(price_str) if price_str.isdigit() else 0
                warranty_str = row.get("warranty_months", "").strip()
                warranty = int(warranty_str) if warranty_str.isdigit() else 0

                products[sku] = {
                    "sku": sku,
                    "title": row.get("title", "").strip(),
                    "brand": row.get("brand", "").strip(),
                    "return_category": row.get("return_category", "").strip().lower(),
                    "price_inr": price,
                    "fulfilled_by": row.get("fulfilled_by", "").strip(),
                    "assured": row.get("assured", "").strip().lower() == "true",
                    "warranty_months": warranty,
                    "description": row.get("description", "").strip(),
                }
    except Exception:
        return {}

    if path == DEFAULT_PRODUCTS_CSV:
        _CATALOG_CACHE = products
        _CACHED_CATALOG_PATH = path

    return products


def lookup_product(
    query: str,
    catalog: dict[str, dict] | None = None,
    catalog_path: Path | str = DEFAULT_PRODUCTS_CSV,
) -> dict | None:
    """Find a product in the catalog via explicit SKU or unambiguous title matching.

    Returns None if:
    - No product is mentioned.
    - An explicit SKU is provided but not found in the catalog (no guessing).
    - A product title matches multiple products with conflicting price tiers (> ₹10,000 vs <= ₹10,000)
      or conflicting categories (ambiguous; not guessed).
    """
    if not query or not query.strip():
        return None

    if catalog is None:
        catalog = load_products(catalog_path)
    if not catalog:
        return None

    # 1. Explicit SKU pattern (e.g. ELE-0023, FSH-0029, TST-0001)
    sku_match = re.search(r"\b([A-Za-z]{3}-[A-Za-z0-9]{4,})\b", query)
    if sku_match:
        sku = sku_match.group(1).upper()
        # Look up exact SKU. If unknown, do not guess another product.
        return catalog.get(sku)

    # 2. Reliable, unambiguous product-title matching
    query_norm = re.sub(r"[^\w\s]", " ", query.lower())
    query_words = set(query_norm.split())

    matches = []
    for prod in catalog.values():
        title = prod["title"]
        title_clean = re.sub(r"[^\w\s]", " ", title.lower()).strip()
        if title_clean and title_clean in query_norm:
            matches.append(prod)
            continue

        # Check base model name before color/variant comma (e.g. "Kestrel Book 14 Laptop")
        base_title = title.split(",")[0].strip()
        base_clean = re.sub(r"[^\w\s]", " ", base_title.lower()).strip()
        if len(base_clean) >= 6:
            base_words = set(base_clean.split())
            if base_words and base_words.issubset(query_words):
                matches.append(prod)

    if not matches:
        return None

    # Deduplicate matches by SKU
    unique_matches = {m["sku"]: m for m in matches}
    matched_prods = list(unique_matches.values())

    if len(matched_prods) == 1:
        return matched_prods[0]

    # Multiple matching products: verify no ambiguity across price threshold and category
    high_value_flags = [p["price_inr"] > HIGH_VALUE_THRESHOLD for p in matched_prods]
    categories = {p["return_category"] for p in matched_prods}

    # If variants straddle the ₹10,000 boundary or belong to different categories,
    # it is ambiguous and must not be guessed.
    if len(set(high_value_flags)) > 1 or len(categories) > 1:
        return None

    # If all variants agree on high-value status and category, return first matching variant
    return matched_prods[0]


def get_product_policy_expansion(
    question: str,
    catalog: dict[str, dict] | None = None,
    catalog_path: Path | str = DEFAULT_PRODUCTS_CSV,
) -> str:
    """Return additional policy domain terms based on verified catalog data and intent.

    Applies:
    - High-value damage escalation (DWM-3, ESC-1) when price_inr > ₹10,000.
    - Standard damage resolution (DWM-1, DWM-2) when price_inr <= ₹10,000.
    """
    product = lookup_product(question, catalog=catalog, catalog_path=catalog_path)
    if not product:
        return ""

    question_lower = question.lower()
    is_damage = any(word in question_lower for word in DAMAGE_KEYWORDS)

    if is_damage:
        if product["price_inr"] > HIGH_VALUE_THRESHOLD:
            return " high value damage priority resolution escalation human review [DWM-3] [ESC-1]"
        else:
            return " damaged replacement refund reporting window [DWM-1] [DWM-2]"

    return ""
