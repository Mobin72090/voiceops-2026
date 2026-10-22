
import pandas as pd

from src.retriever import search_documents

DATA = "Capstone_Project_Dataset_FTTT_2026/practice/questions.csv"


def main():
    questions = pd.read_csv(DATA, dtype=str).fillna("")
    total = 0
    hits = 0
    skipped = 0
    misses = []

    for _, row in questions.iterrows():
        expected = {
            item.strip().split(":")[0]
            for item in row["gold_evidence"].split(";")
            if item.strip()
            and ".csv:" not in item
            and not item.strip().endswith(".csv")
        }

        if not expected:
            skipped += 1
            continue

        results = search_documents(row["question"], top_k=5)
        retrieved = {
            result.get("section_id", "")
            for result in results
        }

        total += 1
        if expected & retrieved:
            hits += 1
        else:
            misses.append(
                (row["id"], row["question"], sorted(expected))
            )

    print(f"Questions evaluated: {total}")
    print(f"Skipped (no KB evidence): {skipped}")
    print(f"Questions with at least one expected section in top 5: {hits}/{total}")
    if total:
        print(f"Recall@5 (question-level): {hits / total:.1%}")

    print("\nQuestions with no expected section in top 5:")
    for question_id, question, expected in misses:
        print(f"- {question_id}: expected {expected}")
        print(f"  {question}")


if __name__ == "__main__":
    main()
