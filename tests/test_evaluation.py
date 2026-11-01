"""Unit tests for offline evaluation modules.

Ensures:
1. Fact checking handles semicolon clauses and slash alternatives accurately.
2. Forbidden claims (must_not_include) are reliably caught.
3. Honest uncertainty is recognized across both English and Hindi.
4. Citations and gold evidence grounding are verified.
5. Answer evaluation (evaluate_answers.py) executes end-to-end with mock data without live API calls.
6. Conversation evaluation (evaluate_conversations.py) maintains turn state, enforces privacy gates, and avoids real ticket creation.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import evaluate_answers
import evaluate_conversations
from eval_utils import (
    check_citation_grounding,
    check_must_include,
    check_must_not_include,
    check_uncertainty_response,
    extract_citations,
    extract_order_id,
    extract_verification_proof,
)


# ---------------------------------------------------------------------------
# 1. Fact Checking (must_include and must_not_include)
# ---------------------------------------------------------------------------

def test_check_must_include_single_clause():
    text = "You have 10 days to return the product."
    res = check_must_include(text, "10")
    assert res["passed"] is True
    assert res["coverage_rate"] == 1.0


def test_check_must_include_slash_alternatives():
    text = "Refunds take two hours via IMPS."
    # 'instant / 2 hours / two hours / तुरंत'
    res = check_must_include(text, "instant / 2 hours / two hours / तुरंत")
    assert res["passed"] is True
    assert res["matched_clauses"] == 1


def test_check_must_include_multiple_clauses():
    # Both '12' and 'My Orders / app' must be present
    text = "12 months warranty claimable directly through the mobile app."
    res = check_must_include(text, "12 ; My Orders / app")
    assert res["passed"] is True
    assert res["matched_clauses"] == 2
    assert res["total_clauses"] == 2

    # Missing one clause
    partial_text = "Claimable through My Orders."
    partial_res = check_must_include(partial_text, "12 ; My Orders / app")
    assert partial_res["passed"] is False
    assert partial_res["matched_clauses"] == 1
    assert "12" in partial_res["missing_clauses"]


def test_check_must_not_include_violation_detection():
    # If forbidden claim is made
    bad_text = "Refunds take 5 to 7 days to reach your account."
    res = check_must_not_include(bad_text, "5 to 7 days ; 5-7 days")
    assert res["passed"] is False
    assert "5 to 7 days" in res["violations"]

    # If forbidden claim is avoided
    good_text = "Refunds are processed instantly within 2 hours."
    good_res = check_must_not_include(good_text, "5 to 7 days ; 5-7 days")
    assert good_res["passed"] is True
    assert len(good_res["violations"]) == 0


# ---------------------------------------------------------------------------
# 2. Uncertainty Detection & Citations
# ---------------------------------------------------------------------------

def test_check_uncertainty_detection():
    # English responses
    assert check_uncertainty_response("Sorry, I don't have information about that.") is True
    assert check_uncertainty_response("I cannot confirm that from the available documents.") is True
    assert check_uncertainty_response("Let me connect you to a colleague who can help.") is True

    # Hindi responses
    assert check_uncertainty_response("माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है।") is True
    assert check_uncertainty_response("Iske baare mein jaankari nahi hai.") is True

    # Definitive answer without uncertainty
    assert check_uncertainty_response("The return window for electronics is 7 days under [RET-2].") is False


def test_extract_citations_and_grounding():
    text = "Electronics return window is 7 days as stated in [RET-2] and [RET-3]."
    citations = extract_citations(text)
    assert "RET-2" in citations
    assert "RET-3" in citations

    grounding = check_citation_grounding(text, "RET-2 ; products.csv:ELE-0012")
    assert grounding["has_citation"] is True
    assert grounding["has_gold_citation"] is True
    assert "RET-2" in grounding["gold_sections"]


def test_extract_order_id_and_proof():
    # Spaced 8-digit order number
    assert extract_order_id("My order number is 8978 2837.") == "89782837"
    # Contiguous 8-digit order number
    assert extract_order_id("Checking 18599253.") == "18599253"

    # Proof values: 4-digit phone suffix
    assert extract_verification_proof("Last four digits are 4533.") == "4533"
    # 6-digit pincode
    assert extract_verification_proof("My delivery pincode is 560001.") == "560001"


# ---------------------------------------------------------------------------
# 3. End-to-End Offline Evaluation Run (evaluate_answers.py)
# ---------------------------------------------------------------------------

def test_evaluate_answers_mock_run(tmp_path):
    out_dir = tmp_path / "answers_eval"
    exit_code = evaluate_answers.main(["--mock", "--limit=5", f"--output-dir={out_dir}"])
    assert exit_code == 0

    # Verify that all 3 report files were generated
    json_report = out_dir / "step1_vs_step2_comparison.json"
    md_report = out_dir / "step1_vs_step2_summary.md"
    csv_report = out_dir / "step1_vs_step2_summary.csv"

    assert json_report.is_file()
    assert md_report.is_file()
    assert csv_report.is_file()

    with open(json_report, encoding="utf-8") as f:
        data = json.load(f)

    assert data["questions_evaluated"] == 5
    assert "summary" in data
    summary = data["summary"]
    assert "step1_baseline" in summary
    assert "step2_rag" in summary

    # Step 2 RAG should demonstrate equal or higher overall quality than baseline
    s1_score = summary["step1_baseline"]["overall_quality_score_pct"]
    s2_score = summary["step2_rag"]["overall_quality_score_pct"]
    assert s2_score >= s1_score


# ---------------------------------------------------------------------------
# 4. Multi-Turn Conversation Evaluation Run (evaluate_conversations.py)
# ---------------------------------------------------------------------------

def test_evaluate_conversations_mock_run(tmp_path):
    out_dir = tmp_path / "conv_eval"
    exit_code = evaluate_conversations.main(["--mock", "--limit=3", f"--output-dir={out_dir}"])
    assert exit_code == 0

    json_report = out_dir / "conversations_eval.json"
    md_report = out_dir / "conversations_summary.md"

    assert json_report.is_file()
    assert md_report.is_file()

    with open(json_report, encoding="utf-8") as f:
        data = json.load(f)

    assert data["summary"]["total_conversations"] == 3
    assert data["summary"]["privacy_gate_compliance_pct"] == 100.0


def test_evaluation_never_touches_tickets_created_csv():
    """Verify evaluation runs never write to real tools/tickets_created.csv."""
    tickets_csv = PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026" / "tools" / "tickets_created.csv"
    initial_mtime = tickets_csv.stat().st_mtime if tickets_csv.exists() else None

    # Run quick conversation simulation
    turn_sample = {
        "turn": 4,
        "customer_message": "Please escalate to human support immediately.",
        "expected_decision": "escalate",
        "expected_tool_calls": "create_ticket()",
    }
    state = {"order_id": "89782837", "verified": True}
    res = evaluate_conversations.simulate_turn_tools(turn_sample, state)

    assert res["tools_called"] == ["create_ticket()"]
    assert res["tool_output"]["created"] is True

    # Real file on disk must remain unchanged
    if tickets_csv.exists():
        assert tickets_csv.stat().st_mtime == initial_mtime
