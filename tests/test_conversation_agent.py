import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026"))

from conversation_agent import (
    call_model,
    get_previous_interaction_id,
    handle_order_lookup,
    handle_ticket_creation,
    MODEL,
    SYSTEM_PROMPT,
)
from tools import lookup_order


class MockInteraction:
    def __init__(self, interaction_id, output_text):
        self.id = interaction_id
        self.output_text = output_text


# ---------------------------------------------------------------------------
# 1. Normal Gemini message handling
# ---------------------------------------------------------------------------

def test_normal_gemini_message_handling():
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MockInteraction(
        interaction_id="int-norm-001",
        output_text="Hello! I am ZentroMart's AI support assistant.",
    )

    history = []
    answer = call_model(mock_client, history, "Hello there")

    assert answer == "Hello! I am ZentroMart's AI support assistant."
    mock_client.interactions.create.assert_called_once_with(
        model=MODEL,
        input="Hello there",
        system_instruction=SYSTEM_PROMPT,
    )
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello there"}
    assert history[1] == {
        "role": "assistant",
        "content": "Hello! I am ZentroMart's AI support assistant.",
        "interaction_id": "int-norm-001",
    }


def test_normal_gemini_message_empty_output_fallback():
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MockInteraction(
        interaction_id="int-empty-001",
        output_text=None,
    )

    history = []
    answer = call_model(mock_client, history, "Test question")

    assert "couldn't generate a response" in answer
    assert history[-1]["content"] == answer


# ---------------------------------------------------------------------------
# 2. Multi-turn memory
# ---------------------------------------------------------------------------

def test_multi_turn_memory_chaining():
    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = [
        MockInteraction("turn-1-id", "Hello! How may I assist you today?"),
        MockInteraction("turn-2-id", "Pleased to meet you, Alice."),
        MockInteraction("turn-3-id", "Your name is Alice."),
    ]

    history = []

    # Turn 1: No previous interaction
    ans1 = call_model(mock_client, history, "Hello")
    assert ans1 == "Hello! How may I assist you today?"
    assert mock_client.interactions.create.call_args_list[0].kwargs == {
        "model": MODEL,
        "input": "Hello",
        "system_instruction": SYSTEM_PROMPT,
    }

    # Turn 2: Should use previous_interaction_id="turn-1-id"
    ans2 = call_model(mock_client, history, "My name is Alice")
    assert ans2 == "Pleased to meet you, Alice."
    assert mock_client.interactions.create.call_args_list[1].kwargs == {
        "model": MODEL,
        "input": "My name is Alice",
        "system_instruction": SYSTEM_PROMPT,
        "previous_interaction_id": "turn-1-id",
    }

    # Turn 3: Should use previous_interaction_id="turn-2-id"
    ans3 = call_model(mock_client, history, "What is my name?")
    assert ans3 == "Your name is Alice."
    assert mock_client.interactions.create.call_args_list[2].kwargs == {
        "model": MODEL,
        "input": "What is my name?",
        "system_instruction": SYSTEM_PROMPT,
        "previous_interaction_id": "turn-2-id",
    }


def test_explicit_previous_interaction_id():
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = MockInteraction(
        "turn-custom-id", "Response"
    )

    history = []
    call_model(
        mock_client,
        history,
        "Direct question",
        previous_interaction_id="forced-prev-id",
    )

    mock_client.interactions.create.assert_called_once_with(
        model=MODEL,
        input="Direct question",
        system_instruction=SYSTEM_PROMPT,
        previous_interaction_id="forced-prev-id",
    )


# ---------------------------------------------------------------------------
# 3. Tool commands do not break conversation history or memory
# ---------------------------------------------------------------------------

def test_tool_command_preserves_conversation_history():
    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = [
        MockInteraction("int-before-tool", "Initial answer"),
        MockInteraction("int-after-tool", "Subsequent answer"),
    ]

    history = []

    # Turn 1: Gemini interaction
    call_model(mock_client, history, "I need help with my order")
    assert get_previous_interaction_id(history) == "int-before-tool"

    # Interleaved tool command (e.g. order lookup or ticket creation)
    history.append({
        "role": "assistant",
        "content": "The caller successfully verified their identity for order 69520867.",
    })

    # Tool entry has no interaction_id, but the helper correctly finds the last Gemini interaction
    assert get_previous_interaction_id(history) == "int-before-tool"

    # Turn 2: Gemini interaction following tool command
    call_model(mock_client, history, "What is the delivery date?")
    assert mock_client.interactions.create.call_args_list[1].kwargs["previous_interaction_id"] == "int-before-tool"
    assert get_previous_interaction_id(history) == "int-after-tool"

    # Verify conversation history length and structure
    contents = [entry["content"] for entry in history]
    assert "I need help with my order" in contents
    assert "The caller successfully verified their identity for order 69520867." in contents
    assert "What is the delivery date?" in contents


# ---------------------------------------------------------------------------
# 4. Order verification failure and retry limits
# ---------------------------------------------------------------------------

def test_order_verification_failed_and_retry_limits(capsys):
    # Order 69520867 exists in dataset.
    # We supply: order_id, wrong proof 1, wrong proof 2 (max 1 retry allowed).
    inputs = iter(["69520867", "0000", "9999"])

    history = []
    with patch("builtins.input", lambda _: next(inputs)):
        result = handle_order_lookup(history=history)

    assert result is None
    captured = capsys.readouterr().out

    # Must alert the caller that they have one more attempt after the first failure
    assert "Verification failed. You have one more attempt." in captured
    # Must refuse to disclose order details after two failed attempts
    assert "I couldn't verify the order. For your privacy, I can't disclose order details." in captured
    # Must not reveal expected phone suffix (0101) or delivery pincode (160017)
    assert "0101" not in captured
    assert "160017" not in captured
    # History must not contain verified order details
    assert len(history) == 0


# ---------------------------------------------------------------------------
# 5. Successful order verification without data leak before verification
# ---------------------------------------------------------------------------

def test_order_lookup_unverified_gates_details():
    # Calling lookup_order without verification must never reveal order data
    unverified = lookup_order("69520867")
    assert unverified.get("verification_required") is True
    assert "status" not in unverified
    assert "items" not in unverified
    assert "placed_at" not in unverified
    assert "total_inr" not in unverified


def test_order_verification_successful_via_phone_last4(capsys):
    # Customer C100124 registered phone is +91 59596 10101 -> last 4 digits: 0101
    inputs = iter(["69520867", "0101"])
    history = []

    with patch("builtins.input", lambda _: next(inputs)):
        safe_order = handle_order_lookup(history=history)

    assert safe_order is not None
    assert safe_order["order_id"] == "69520867"
    assert safe_order["status"] == "shipped"
    # Ensure real CSV column names are present, not invented ones
    assert "placed_at" in safe_order
    assert safe_order["placed_at"] == "2026-10-12T08:37"
    assert "promised_date" in safe_order
    assert safe_order["promised_date"] == "2026-10-23"
    assert "shipped_on" in safe_order
    assert safe_order["shipped_on"] == "2026-10-13"
    assert "total_inr" in safe_order
    assert safe_order["total_inr"] == 3778
    assert "items" in safe_order
    assert len(safe_order["items"]) == 2

    # Verify no invented field names
    assert "order_date" not in safe_order
    assert "delivery_date" not in safe_order
    assert "expected_delivery_date" not in safe_order

    captured = capsys.readouterr().out
    assert "Verification successful." in captured
    assert len(history) == 1
    assert "verified their identity for order 69520867" in history[0]["content"]


def test_order_verification_successful_via_pincode_on_retry(capsys):
    # Order 69520867 delivery pincode is 160017
    # First attempt: wrong phone digits "9999"
    # Second attempt (retry 1): correct delivery pincode "160017"
    inputs = iter(["69520867", "9999", "160017"])
    history = []

    with patch("builtins.input", lambda _: next(inputs)):
        safe_order = handle_order_lookup(history=history)

    assert safe_order is not None
    assert safe_order["order_id"] == "69520867"
    captured = capsys.readouterr().out
    assert "Verification failed. You have one more attempt." in captured
    assert "Verification successful." in captured
    assert len(history) == 1


# ---------------------------------------------------------------------------
# 6. Ticket creation success and failure handling
# ---------------------------------------------------------------------------

def test_ticket_creation_too_short_summary(capsys):
    inputs = iter(["Damaged item", "69520867"])
    history = []

    with patch("builtins.input", lambda _: next(inputs)):
        result = handle_ticket_creation(history=history)

    assert result.get("created") is False
    captured = capsys.readouterr().out
    assert "The summary needs at least 20 characters." in captured
    assert len(history) == 0


def test_ticket_creation_failure_from_tool(capsys):
    inputs = iter([
        "Customer reports television screen broken upon delivery.",
        "69520867",
    ])
    history = []

    with patch("builtins.input", lambda _: next(inputs)), \
         patch("conversation_agent.create_ticket", return_value={"created": False, "error": "System failure"}):
        result = handle_ticket_creation(history=history)

    assert result.get("created") is False
    captured = capsys.readouterr().out
    assert "I couldn't confirm that the support ticket was created." in captured
    assert "Ticket ID:" not in captured
    assert len(history) == 0


def test_ticket_creation_success_reported(capsys):
    inputs = iter([
        "Customer reports television screen broken upon delivery.",
        "69520867",
    ])
    history = []

    with patch("builtins.input", lambda _: next(inputs)), \
         patch("conversation_agent.create_ticket", return_value={"created": True, "ticket_id": "TKT_MOCK_998877"}):
        result = handle_ticket_creation(history=history)

    assert result.get("created") is True
    assert result.get("ticket_id") == "TKT_MOCK_998877"
    captured = capsys.readouterr().out
    assert "Your support ticket was created successfully." in captured
    assert "Ticket ID: TKT_MOCK_998877" in captured
    assert len(history) == 1
    assert "TKT_MOCK_998877" in history[0]["content"]
