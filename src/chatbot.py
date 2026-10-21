
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"

SYSTEM_PROMPT = """
You are ZentroMart's AI customer-support assistant.
ZentroMart is a fictional Indian online store selling groceries and electronics.

Rules:
1. Identify yourself as an AI assistant.
2. Be polite, concise, and helpful.
3. Help with delivery, return, refund, cancellation, product, and warranty questions.
4. You do not have access to company documents or order records yet.
5. Never invent company policies, order statuses, dates, or refund amounts.
6. If you cannot verify an answer, explain that you cannot confirm it yet.
7. Never claim to have performed an action you did not perform.
8. Ask one clear question when essential information is missing.
"""


def main():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your local .env file."
        )

    client = genai.Client(api_key=API_KEY)

    print("ZentroMart AI Support - Step 1 baseline")
    print("Type 'exit' to quit.\n")

    print(
        "AI assistant: Hello! I'm ZentroMart's AI support assistant. "
        "How can I help you today?\n"
    )

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            interaction = client.interactions.create(
                model=MODEL,
                input=question,
                system_instruction=SYSTEM_PROMPT,
            )

            answer = interaction.output_text

            if answer:
                print(f"\nZentroMart AI: {answer.strip()}\n")
            else:
                print(
                    "\nZentroMart AI: Sorry, I couldn't generate a reply. "
                    "Please try again.\n"
                )

        except Exception as exc:
            print(f"\nRequest failed: {type(exc).__name__}: {exc}")
            print(
                "Check your API key, model name, internet connection, "
                "and API quota.\n"
            )


if __name__ == "__main__":
    main()
