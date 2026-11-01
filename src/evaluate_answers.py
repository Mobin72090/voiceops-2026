
"""Step 1 (Baseline Prompting) vs Step 2 (RAG) Answer-Quality Evaluation.

Benchmarks baseline chatbot answers versus RAG chatbot answers on
practice/questions.csv as specified in the official Capstone Brief.

Metrics:
1. Fact Inclusion Rate
2. Hallucination / Forbidden Claim Rate
3. Honest Uncertainty Rate
4. Knowledge-base Citation Rate
5. Retrieval Recall@5
6. Overall Answer Quality Score
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chatbot
import rag_chatbot
from eval_utils import (
    check_citation_grounding,
    check_must_include,
    check_must_not_include,
    check_uncertainty_response,
)
from google import genai
from google.genai import types
from retriever import search_documents

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def load_practice_questions(
    csv_path: Path | str = DEFAULT_DATASET,
) -> list[dict]:
    """Load practice questions, preserving column values as strings."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Practice questions file not found: {path}"
        )

    records = []
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append({
                key: (value or "").strip()
                for key, value in row.items()
            })

    return records


def evaluate_single_answer(
    question_data: dict,
    answer_text: str,
    retrieved_chunks: list[dict] | None = None,
) -> dict:
    """Evaluate one generated answer against the dataset criteria."""
    is_answerable = (
        question_data.get("answerable", "").lower() in ("true", "1")
    )
    must_include = question_data.get("must_include", "")
    must_not_include = question_data.get("must_not_include", "")
    gold_evidence = question_data.get("gold_evidence", "")

    inclusion = check_must_include(answer_text, must_include)
    exclusion = check_must_not_include(answer_text, must_not_include)
    uncertainty = check_uncertainty_response(answer_text)
    citations = check_citation_grounding(answer_text, gold_evidence)

    retrieval_hit = False
    retrieved_section_ids = []

    if retrieved_chunks:
        retrieved_section_ids = [
            chunk.get("section_id", "")
            for chunk in retrieved_chunks
        ]

        if gold_evidence:
            gold_sections = {
                item.split(":")[0].strip()
                for item in gold_evidence.split(";")
                if item.strip()
                and not item.strip().endswith(".csv")
            }
            retrieval_hit = bool(
                gold_sections & set(retrieved_section_ids)
            )

    # Preserve the original scoring rules.
    if not is_answerable:
        answer_score = (
            1.0
            if uncertainty and exclusion["passed"]
            else 0.0
        )
    else:
        base_score = inclusion["coverage_rate"]

        if not exclusion["passed"]:
            base_score = max(0.0, base_score - 0.5)

        answer_score = round(base_score, 3)

    return {
        "id": question_data.get("id"),
        "answerable": is_answerable,
        "answer_text": answer_text,
        "must_include_passed": inclusion["passed"],
        "fact_coverage": inclusion["coverage_rate"],
        "missing_facts": inclusion["missing_clauses"],
        "must_not_include_passed": exclusion["passed"],
        "violations": exclusion["violations"],
        "uncertainty_detected": uncertainty,
        "uncertainty_accurate": (
            (uncertainty and not is_answerable)
            or (not uncertainty and is_answerable)
        ),
        "citations": citations["cited_sections"],
        "has_gold_citation": citations["has_gold_citation"],
        "retrieval_hit": retrieval_hit,
        "retrieved_sections": retrieved_section_ids,
        "answer_score": answer_score,
    }


def run_evaluation(
    questions: list[dict],
    client: Any = None,
    mode: str = "both",
    mock_responses: dict | None = None,
) -> dict:
    """Evaluate baseline and/or RAG answers with visible progress."""
    results = []
    total = len(questions)

    for idx, question_data in enumerate(questions, 1):
        question_id = question_data.get("id", f"Q{idx}")
        question_text = question_data.get("question", "")
        question_start = time.monotonic()

        print(
            f"\n[{idx}/{total}] Processing question {question_id}",
            flush=True,
        )

        # Step 1: Baseline prompting.
        baseline_answer = ""

        if mode in ("both", "baseline"):
            if mock_responses and "baseline" in mock_responses:
                print("  Using baseline mock response.", flush=True)
                baseline_answer = mock_responses["baseline"].get(
                    question_id, "Baseline mock answer"
                )

            elif client:
                print(
                    "  Generating baseline answer via Gemini...",
                    flush=True,
                )
                request_start = time.monotonic()

                try:
                    baseline_answer = chatbot.answer_question(
                        client, question_text
                    )
                    print(
                        "  Baseline completed in "
                        f"{time.monotonic() - request_start:.1f}s.",
                        flush=True,
                    )
                except Exception as exc:
                    baseline_answer = (
                        f"[Generation Error: {type(exc).__name__}: {exc}]"
                    )
                    print(
                        f"  Baseline request failed: {baseline_answer}",
                        flush=True,
                    )

        # Step 2: Retrieval-augmented generation.
        rag_answer = ""
        retrieved_chunks = []

        if mode in ("both", "rag"):
            if mock_responses and "rag" in mock_responses:
                print("  Using RAG mock response.", flush=True)
                mock_entry = mock_responses["rag"].get(
                    question_id, {}
                )
                rag_answer = mock_entry.get(
                    "answer", "RAG mock answer [RET-2]"
                )
                retrieved_chunks = mock_entry.get("chunks", [])

            elif client:
                print(
                    "  Retrieving evidence and generating RAG answer...",
                    flush=True,
                )
                request_start = time.monotonic()

                try:
                    rag_answer, retrieved_chunks = (
                        rag_chatbot.answer_question(
                            client, question_text
                        )
                    )
                    print(
                        "  RAG completed in "
                        f"{time.monotonic() - request_start:.1f}s.",
                        flush=True,
                    )
                except Exception as exc:
                    rag_answer = (
                        f"[Generation Error: {type(exc).__name__}: {exc}]"
                    )
                    print(
                        f"  RAG request failed: {rag_answer}",
                        flush=True,
                    )
                    retrieved_chunks = search_documents(
                        question_text, top_k=5
                    )

            else:
                # Offline retrieval metric when no client is available.
                retrieved_chunks = search_documents(
                    question_text, top_k=5
                )

        baseline_result = (
            evaluate_single_answer(question_data, baseline_answer)
            if baseline_answer
            else None
        )

        rag_result = (
            evaluate_single_answer(
                question_data,
                rag_answer,
                retrieved_chunks=retrieved_chunks,
            )
            if rag_answer
            else None
        )

        results.append({
            "id": question_id,
            "question": question_text,
            "intent": question_data.get("intent", ""),
            "answerable": (
                question_data.get("answerable", "").lower()
                in ("true", "1")
            ),
            "reference_answer": question_data.get(
                "reference_answer", ""
            ),
            "step1_baseline": baseline_result,
            "step2_rag": rag_result,
        })

        print(
            f"  Question finished in "
            f"{time.monotonic() - question_start:.1f}s.",
            flush=True,
        )

    summary = compute_summary_metrics(results)

    return {
        "summary": summary,
        "questions_evaluated": len(results),
        "results": results,
    }


def compute_summary_metrics(results: list[dict]) -> dict:
    """Compute aggregate baseline and RAG metrics."""
    total = len(results)

    if total == 0:
        return {}

    answerable_total = sum(
        1 for result in results if result["answerable"]
    )
    unanswerable_total = total - answerable_total

    def calculate_stats(step_key: str) -> dict:
        entries = [
            result[step_key]
            for result in results
            if result.get(step_key)
        ]

        if not entries:
            return {}

        answerable_entries = [
            entry for entry in entries if entry["answerable"]
        ]
        unanswerable_entries = [
            entry for entry in entries if not entry["answerable"]
        ]

        fact_coverage = (
            sum(entry["fact_coverage"] for entry in answerable_entries)
            / len(answerable_entries)
            if answerable_entries
            else 0.0
        )

        hallucinations = sum(
            1 for entry in entries
            if not entry["must_not_include_passed"]
        )

        honest_uncertainty = sum(
            1 for entry in unanswerable_entries
            if entry["uncertainty_detected"]
        )

        uncertainty_rate = (
            honest_uncertainty / len(unanswerable_entries)
            if unanswerable_entries
            else 0.0
        )

        average_score = (
            sum(entry["answer_score"] for entry in entries)
            / len(entries)
        )

        retrieval_hits = sum(
            1 for entry in answerable_entries
            if entry.get("retrieval_hit")
        )

        retrieval_recall = (
            retrieval_hits / len(answerable_entries)
            if answerable_entries
            else 0.0
        )

        citation_count = sum(
            1 for entry in entries if entry.get("citations")
        )

        citation_rate = citation_count / len(entries)

        return {
            "evaluated_count": len(entries),
            "fact_coverage_pct": round(fact_coverage * 100, 1),
            "hallucination_violations": hallucinations,
            "unanswerable_uncertainty_rate_pct": round(
                uncertainty_rate * 100, 1
            ),
            "retrieval_recall_pct": round(
                retrieval_recall * 100, 1
            ),
            "citation_rate_pct": round(citation_rate * 100, 1),
            "overall_quality_score_pct": round(
                average_score * 100, 1
            ),
        }

    return {
        "total_questions": total,
        "answerable_questions": answerable_total,
        "unanswerable_questions": unanswerable_total,
        "step1_baseline": calculate_stats("step1_baseline"),
        "step2_rag": calculate_stats("step2_rag"),
    }


def save_reports(
    eval_data: dict,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Save JSON, Markdown, and CSV evaluation reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_file = output_path / "step1_vs_step2_comparison.json"

    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(eval_data, file, indent=2, ensure_ascii=False)

    summary = eval_data.get("summary", {})
    baseline = summary.get("step1_baseline", {})
    rag = summary.get("step2_rag", {})

    markdown_file = output_path / "step1_vs_step2_summary.md"

    markdown_content = f"""# Step 1 (Prompting) vs Step 2 (RAG) Evaluation Summary

Evaluated on: `{eval_data.get('questions_evaluated', 0)}` questions from `practice/questions.csv`.
- Answerable questions: `{summary.get('answerable_questions', 0)}`
- Unanswerable / unsupported questions: `{summary.get('unanswerable_questions', 0)}`

| Metric | Step 1 Baseline (Prompting Only) | Step 2 RAG (Knowledge Grounded) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Fact Inclusion Coverage** | {baseline.get('fact_coverage_pct', 'N/A')}% | {rag.get('fact_coverage_pct', 'N/A')}% | {round(rag.get('fact_coverage_pct', 0) - baseline.get('fact_coverage_pct', 0), 1)} percentage points |
| **Hallucination Violations** | {baseline.get('hallucination_violations', 'N/A')} | {rag.get('hallucination_violations', 'N/A')} | {rag.get('hallucination_violations', 0) - baseline.get('hallucination_violations', 0)} |
| **Honest Uncertainty Rate (Unanswerable)** | {baseline.get('unanswerable_uncertainty_rate_pct', 'N/A')}% | {rag.get('unanswerable_uncertainty_rate_pct', 'N/A')}% | {round(rag.get('unanswerable_uncertainty_rate_pct', 0) - baseline.get('unanswerable_uncertainty_rate_pct', 0), 1)} percentage points |
| **Retrieval Recall@5** | N/A (No retrieval) | {rag.get('retrieval_recall_pct', 'N/A')}% | N/A |
| **Section ID Citation Rate** | {baseline.get('citation_rate_pct', 'N/A')}% | {rag.get('citation_rate_pct', 'N/A')}% | {round(rag.get('citation_rate_pct', 0) - baseline.get('citation_rate_pct', 0), 1)} percentage points |
| **Overall Answer Quality Score** | {baseline.get('overall_quality_score_pct', 'N/A')}% | {rag.get('overall_quality_score_pct', 'N/A')}% | {round(rag.get('overall_quality_score_pct', 0) - baseline.get('overall_quality_score_pct', 0), 1)} percentage points |

Note: These metrics are automated checks against the dataset criteria, not a guarantee of real-world answer quality.
"""

    markdown_file.write_text(markdown_content, encoding="utf-8")

    csv_file = output_path / "step1_vs_step2_summary.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Step 1 Baseline", "Step 2 RAG"])
        writer.writerow([
            "fact_coverage_pct",
            baseline.get("fact_coverage_pct", ""),
            rag.get("fact_coverage_pct", ""),
        ])
        writer.writerow([
            "hallucination_violations",
            baseline.get("hallucination_violations", ""),
            rag.get("hallucination_violations", ""),
        ])
        writer.writerow([
            "unanswerable_uncertainty_rate_pct",
            baseline.get("unanswerable_uncertainty_rate_pct", ""),
            rag.get("unanswerable_uncertainty_rate_pct", ""),
        ])
        writer.writerow([
            "retrieval_recall_pct",
            "N/A",
            rag.get("retrieval_recall_pct", ""),
        ])
        writer.writerow([
            "citation_rate_pct",
            baseline.get("citation_rate_pct", ""),
            rag.get("citation_rate_pct", ""),
        ])
        writer.writerow([
            "overall_quality_score_pct",
            baseline.get("overall_quality_score_pct", ""),
            rag.get("overall_quality_score_pct", ""),
        ])

    return {
        "json": str(json_file),
        "markdown": str(markdown_file),
        "csv": str(csv_file),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline prompting versus RAG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions evaluated",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "rag", "baseline"],
        default="both",
        help="Evaluate both models, RAG only, or baseline only",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using mock responses without live API calls",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory in which reports are saved",
    )

    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    questions = load_practice_questions()

    if args.limit is not None:
        questions = questions[:args.limit]

    print(
        f"Loaded {len(questions)} questions for evaluation.",
        flush=True,
    )

    client = None
    mock_responses = None

    if args.mock:
        print(
            "Running in MOCK mode (no live API calls).",
            flush=True,
        )

        mock_responses = {"baseline": {}, "rag": {}}

        for question in questions:
            question_id = question["id"]

            if question["answerable"].lower() in ("true", "1"):
                mock_responses["baseline"][question_id] = (
                    f"I think {question['question']} "
                    "is allowed under company policy."
                )

                facts = (
                    question["must_include"].split(";")[0].split("/")[0].strip()
                    if question["must_include"]
                    else "standard terms"
                )

                mock_responses["rag"][question_id] = {
                    "answer": (
                        f"According to ZentroMart policy [RET-2], "
                        f"{facts} applies."
                    ),
                    "chunks": search_documents(
                        question["question"], top_k=5
                    ),
                }

            else:
                mock_responses["baseline"][question_id] = (
                    "Yes, you can schedule a delivery at midnight."
                )

                mock_responses["rag"][question_id] = {
                    "answer": (
                        "Sorry, I don't have information about that "
                        "in the knowledge base. I can connect you "
                        "to human support."
                    ),
                    "chunks": [],
                }

    else:
        # Load .env variables for standalone CLI use.
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print(
                "GEMINI_API_KEY not found. Reverting to mock evaluation.",
                flush=True,
            )
            fallback_args = ["--mock", "--output-dir", args.output_dir]

            if args.limit is not None:
                fallback_args.extend(["--limit", str(args.limit)])

            if args.mode != "both":
                fallback_args.extend(["--mode", args.mode])

            return main(fallback_args)

        # Timeout is in milliseconds. A single HTTP operation is
        # limited to approximately 30 seconds; retries or multiple
        # requests can make total evaluation time longer.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000),
        )

    try:
        eval_data = run_evaluation(
            questions,
            client=client,
            mode=args.mode,
            mock_responses=mock_responses,
        )
    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted. No final summary was saved.",
            flush=True,
        )
        return 130
    except Exception as exc:
        print(
            f"\nEvaluation failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 1

    saved_paths = save_reports(
        eval_data,
        output_dir=args.output_dir,
    )

    print("\n=== EVALUATION COMPLETE ===", flush=True)

    summary = eval_data.get("summary", {})
    baseline = summary.get("step1_baseline", {})
    rag = summary.get("step2_rag", {})

    print(
        f"Questions Evaluated: {eval_data.get('questions_evaluated')}",
        flush=True,
    )

    print("\nMetric Comparison:", flush=True)
    print(
        f"  Fact Coverage: Step 1: "
        f"{baseline.get('fact_coverage_pct', 'N/A')}% -> "
        f"Step 2: {rag.get('fact_coverage_pct', 'N/A')}%",
        flush=True,
    )
    print(
        f"  Hallucinations: Step 1: "
        f"{baseline.get('hallucination_violations', 'N/A')} -> "
        f"Step 2: {rag.get('hallucination_violations', 'N/A')}",
        flush=True,
    )
    print(
        f"  Honest Uncertainty: Step 1: "
        f"{baseline.get('unanswerable_uncertainty_rate_pct', 'N/A')}% -> "
        f"Step 2: {rag.get('unanswerable_uncertainty_rate_pct', 'N/A')}%",
        flush=True,
    )
    print(
        f"  Retrieval Recall@5: Step 1: N/A -> "
        f"Step 2: {rag.get('retrieval_recall_pct', 'N/A')}%",
        flush=True,
    )
    print(
        f"  Overall Score: Step 1: "
        f"{baseline.get('overall_quality_score_pct', 'N/A')}% -> "
        f"Step 2: {rag.get('overall_quality_score_pct', 'N/A')}%",
        flush=True,
    )

    print("\nArtifacts saved to:", flush=True)
    print(f"  {saved_paths['markdown']}", flush=True)
    print(f"  {saved_paths['json']}", flush=True)
    print(f"  {saved_paths['csv']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
