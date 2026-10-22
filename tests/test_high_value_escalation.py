"""End-to-end unit tests for high-value damaged-product escalation workflow.

Verifies:
1. High-value damaged products dynamically look up catalog facts and retrieve DWM-3 & ESC-1.
2. Lower-value damaged products retrieve standard DWM-1 & DWM-2 without triggering DWM-3.
3. Unverified order access strictly gates private order information.
4. Ticket creation failure never produces a success claim or invented ticket ID.
5. Successful ticket creation is reported only when the tool returns created=True with the exact ID.

Uses mocked tools only to ensure no real tickets or data modifications occur on disk.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026"))

from catalog import load_products, HIGH_VALUE_THRESHOLD
from retriever import search_documents
from conversation_agent import handle_order_lookup, handle_ticket_creation
from tools import lookup_order


@pytest.fixture(scope="module")
def catalog():
    """Load product catalog dynamically without hardcoding prices or SKUs."""
    cat = load_products()
    assert cat, "Catalog must not be empty"
    return cat


def test_high_value_damaged_product_escalation_workflow(catalog):
    """Verify high-value damaged product retrieves DWM-3/ESC-1 and coordinates human handover."""
    # Dynamically select a high-value product (> ₹10,000 threshold under DWM-3/ESC-1)
    high_val_prod = next(
        p for p in catalog.values() if p["price_inr"] > HIGH_VALUE_THRESHOLD
    )
    assert high_val_prod["price_inr"] > HIGH_VALUE_THRESHOLD

    # 1. Knowledge retrieval: Customer reports physical damage on the high-value item
    query = (
        f"The {high_val_prod['title']} (SKU {high_val_prod['sku']}) I received "
        "has a crack and is damaged. I want it sorted."
    )
    results = search_documents(query, top_k=5)
    section_ids = {r["section_id"] for r in results}

    # Under DWM-3 and ESC-1, items > ₹10,000 require Priority Resolution review / human escalation
    assert "DWM-3" in section_ids or "ESC-1" in section_ids, (
        f"Expected DWM-3 or ESC-1 for high-value product {high_val_prod['sku']} "
        f"(price: {high_val_prod['price_inr']}), got {section_ids}"
    )

    # 2. Handover action: Create ticket with mock tool
    mock_tid = "TKT_HIGH_VALUE_ESC_001"
    inputs = iter([
        f"Caller reports high-value item {high_val_prod['sku']} damaged; requires review under DWM-3.",
        "89782837",
    ])
    history = []

    with patch("builtins.input", lambda _: next(inputs)), \
         patch("conversation_agent.create_ticket", return_value={"created": True, "ticket_id": mock_tid}) as mock_create:
        ticket_result = handle_ticket_creation(history=history)

    assert ticket_result.get("created") is True
    assert ticket_result.get("ticket_id") == mock_tid
    mock_create.assert_called_once()
    assert len(history) == 1
    assert mock_tid in history[0]["content"]


def test_lower_value_damaged_product_avoids_high_value_escalation(catalog):
    """Verify lower-value damaged product retrieves standard DWM-1/DWM-2 without triggering DWM-3."""
    # Dynamically select a product with price <= ₹10,000
    low_val_prod = next(
        p for p in catalog.values() if 0 < p["price_inr"] <= HIGH_VALUE_THRESHOLD
    )
    assert low_val_prod["price_inr"] <= HIGH_VALUE_THRESHOLD

    query = (
        f"The {low_val_prod['title']} (SKU {low_val_prod['sku']}) came damaged and torn. "
        "Can I get a replacement?"
    )
    results = search_documents(query, top_k=5)
    section_ids = {r["section_id"] for r in results}

    # Standard damage policies apply
    assert "DWM-1" in section_ids or "DWM-2" in section_ids
    # Must NOT trigger high-value escalation policy
    assert "DWM-3" not in section_ids, (
        f"Lower-value product {low_val_prod['sku']} (price: {low_val_prod['price_inr']}) "
        f"should not retrieve high-value policy DWM-3; got {section_ids}"
    )


def test_unverified_order_access_withholds_private_details(capsys):
    """Verify calling order lookup without verification hides all private order data."""
    # 1. Direct tool-level check: unverified call returns verification_required only
    unverified = lookup_order("89782837")
    assert unverified.get("verification_required") is True
    assert "status" not in unverified
    assert "items" not in unverified
    assert "total_inr" not in unverified
    assert "placed_at" not in unverified
    assert "customer_first_name" not in unverified
    assert "delivery_pincode" not in unverified

    # 2. Agent-level check: unverified user fails verification attempts
    inputs = iter(["89782837", "0000", "1111"])
    history = []

    with patch("builtins.input", lambda _: next(inputs)):
        res = handle_order_lookup(history=history)

    assert res is None, "Order lookup must return None when unverified"
    captured = capsys.readouterr().out
    assert "Verification failed" in captured
    assert "I can't disclose order details" in captured
    # History must remain empty of private order information
    assert len(history) == 0


def test_ticket_creation_failure_never_claims_success(capsys):
    """Verify tool failure never claims ticket creation or invents a ticket ID."""
    inputs = iter([
        "High-value damaged product report requiring manual review by priority team.",
        "89782837",
    ])
    history = []

    # Mock tool returns failure
    with patch("builtins.input", lambda _: next(inputs)), \
         patch("conversation_agent.create_ticket", return_value={"created": False, "error": "Queue service unavailable"}):
        result = handle_ticket_creation(history=history)

    assert result.get("created") is False
    captured = capsys.readouterr().out

    # Must NOT claim success or provide a ticket ID
    assert "Your support ticket was created successfully." not in captured
    assert "Ticket ID:" not in captured
    # Must explicitly inform the customer of unconfirmed handover
    assert "I couldn't confirm that the support ticket was created." in captured
    # History must not record unconfirmed ticket
    assert len(history) == 0


def test_ticket_creation_success_reported_only_with_exact_tool_id(capsys):
    """Verify confirmation is reported only when created=True, displaying the exact returned ID."""
    mock_ticket_id = "TKT_VERIFIED_EXACT_998877"
    inputs = iter([
        "Customer reports laptop display shattered upon delivery; priority resolution required.",
        "89782837",
    ])
    history = []

    # Mock tool returns success with exact synthetic ID
    with patch("builtins.input", lambda _: next(inputs)), \
         patch("conversation_agent.create_ticket", return_value={"created": True, "ticket_id": mock_ticket_id}):
        result = handle_ticket_creation(history=history)

    assert result.get("created") is True
    assert result.get("ticket_id") == mock_ticket_id
    captured = capsys.readouterr().out

    # Must confirm success with the exact returned ID
    assert "Your support ticket was created successfully." in captured
    assert f"Ticket ID: {mock_ticket_id}" in captured
    # History must record the confirmed ticket
    assert len(history) == 1
    assert mock_ticket_id in history[0]["content"]
