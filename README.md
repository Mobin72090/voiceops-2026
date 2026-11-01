# VoiceOps 2026 — ZentroMart Customer Support Assistant

**Public Repository:** [https://github.com/Mobin72090/voiceops-2026.git](https://github.com/Mobin72090/voiceops-2026.git)  
**Submission Deadline:** 1 November 2026  
**Primary Run Command:** `python run.py --input-dir audio_clips --output-dir outputs`

---

## 1. Project Overview & Problem Statement

ZentroMart is a fast-growing Indian e-commerce platform offering groceries, electronics, fashion, and home goods. As order volume scales, customer support faces several challenges:
1. **Inaccurate or Outdated Policy Answers:** Generative AI models without company grounding frequently hallucinate return windows, invent refund guarantees, or reference superseded policies.
2. **Customer Data Privacy Violations:** Premature disclosure of order contents, delivery addresses, and tracking details before customer identity verification breaches privacy guidelines.
3. **Spurious Escalations & Failed Tool Calls:** Virtual assistants often promise that an escalation ticket was created even when the backend tool failed, or invent bogus ticket IDs.
4. **Voice Accessibility & Indian Accent Recognition:** Voice-based interactions require robust Indian-English speech-to-text (STT) and natural text-to-speech (TTS) synthesis capable of handling Hinglish vocabulary and noisy or empty audio.

**VoiceOps 2026** solves these problems with a modular, 4-stage pipeline combining **Gnani.ai** conversational voice services and **Google Gemini** language and embedding models.

---

## 2. Architecture & Service Responsibilities

```
                      +---------------------------------------+
                      |           Customer Voice (.wav)       |
                      +---------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Gnani.ai Voice Integration                                                        |
| - STT Service: Speech-to-Text (en-IN language model, max 60s audio validation)     |
+-----------------------------------------------------------------------------------+
                                          |
                        Transcribed Text (or Inaudible Fallback)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Gemini & Knowledge-Base RAG Reasoning                                              |
| - Retrieval: Hybrid BM25 Lexical + Gemini Embedding (gemini-embedding-001)        |
| - Catalog Routing: Product price / category checks (data/products.csv)            |
| - Authority & Recency: Prioritizes active policies over superseded documents       |
| - Multi-Turn Agent: State tracking, 2-step verification, ticket-creation safeguard|
| - Generation Model: Gemini 2.5/Flash with strict grounding constraints            |
+-----------------------------------------------------------------------------------+
                                          |
                           Synthesized Text Reply & Evidence
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Gnani.ai Speech Synthesis                                                         |
| - TTS Service: Text-to-Speech (Kaveri voice, timbre-v2.5 model)                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
            Outputs: <clip>_transcript.txt, <clip>_reply.txt,
                     <clip>_reply.wav, <clip>_evidence.json
```

### Division of Responsibilities
- **Gnani.ai Services:**
  - **Speech-to-Text (STT):** Transcribes customer WAV audio using the `en-IN` Indian English model. Validates WAV format and enforces the 60-second clip duration limit.
  - **Text-to-Speech (TTS):** Converts assistant reply text into natural spoken audio (`.wav`) using the `Kaveri` voice persona on `timbre-v2.5`.
- **Google Gemini & Local RAG Core:**
  - **Embedding & Semantic Search:** Uses `gemini-embedding-001` with SHA-256 cache invalidation (`.cache/kb_embeddings.json`) and an offline BM25 lexical fallback.
  - **Conversational Intelligence:** Generates grounded answers, manages verification logic, cites authoritative Section IDs (e.g., `[RET-2]`, `[DWM-3]`, `[ESC-1]`), and produces honest uncertainty responses for unsupported queries.

---

## 3. Knowledge Base & Product Catalog Structure

The system grounds all policy answers using the curated ZentroMart documents:
- **Knowledge Base (`Capstone_Project_Dataset_FTTT_2026/kb/`):**
  - 21 structured Markdown files covering returns (`returns_policy.md`), cancellations (`cancellation_policy.md`), deliveries (`delivery_policy.md`), damage/warranty (`damage_warranty.md`), escalations (`escalation_matrix.md`), and historical versions (`historical_returns_v3.md`).
  - Each document features YAML front-matter specifying `doc_id`, `doc_type` (`policy`, `agent_guide`, `faq`, `help`), `status` (`current`, `superseded`), and `effective_from`.
  - Content sections are chunked by bracketed Section IDs (e.g., `[RET-2] Return Windows by Category`).
- **Product Catalog (`Capstone_Project_Dataset_FTTT_2026/data/products.csv`):**
  - Contains SKU-level metadata (Product Name, Category, Price in INR).
  - Used by `src/catalog.py` to identify high-value products (threshold: > ₹10,000) for targeted policy routing under `[DWM-3]` and `[ESC-1]`.

---

## 4. Four-Stage Development Workflow

1. **Step 1 — Baseline Prompting (`src/chatbot.py`):**
   Interactive terminal chatbot with prompt instructions only. Demonstrates lack of specific domain grounding and high uncertainty on edge-case policies.
2. **Step 2 — Knowledge-Base RAG (`src/rag_chatbot.py` & `src/retriever.py`):**
   Integrates hybrid semantic + lexical document retrieval. Enforces strict evidence citation, historical policy suppression (unless explicitly requested), and honest uncertainty handling.
3. **Step 3 — Multi-Turn Support Agent (`src/conversation_agent.py`):**
   Handles multi-turn dialogue with strict verification gates (Order ID + phone last 4 digits or delivery pincode), human handoff escalation, and confirmed ticket creation.
4. **Step 4 — Voice Pipeline (`src/voice_pipeline.py` & `run.py`):**
   End-to-end batch processing pipeline with audio duration validation ($\le 60$ seconds), empty-transcript handling, STT, RAG reasoning, and TTS speech synthesis.

---

## 5. Prerequisites & Environment Setup

### Prerequisites
- **Operating System:** Windows 10/11 (PowerShell) or Linux/macOS
- **Python:** Python 3.10 to 3.14 (Tested on Python 3.14.3)
- **API Keys:**
  - Google Gemini API Key (`GEMINI_API_KEY`)
  - Gnani.ai API Key (`GNANI_API_KEY`)

### Setup Instructions (Windows PowerShell)

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/Mobin72090/voiceops-2026.git
   cd voiceops-2026
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a local `.env` file by copying `.env.example`:
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` and fill in your actual credentials:
   ```text
   GEMINI_API_KEY=your_actual_gemini_api_key
   GEMINI_MODEL=gemini-2.5-flash
   GNANI_API_KEY=your_actual_gnani_api_key
   ```
   > **Security Note:** Never commit `.env` or share real API keys. `.env` is explicitly ignored by `.gitignore` and excluded from all submission archives.

---

## 6. How to Run the System

### 1. Batch Voice Pipeline (Official Submission Runner)
Processes all `.wav` audio files in an input folder and saves transcripts, reply text, synthesized audio, and evidence JSON files:
```powershell
python run.py --input-dir audio_clips --output-dir outputs
```
*Direct pipeline equivalent:*
```powershell
python src\voice_pipeline.py --input-dir audio_clips --output-dir outputs
```

### 2. Interactive Terminal Chatbots
- **Step 1 Baseline Chatbot (Prompting only):**
  ```powershell
  python src\chatbot.py
  ```
- **Step 2 RAG Chatbot (Document-grounded retrieval):**
  ```powershell
  python src\rag_chatbot.py
  ```

### 3. Automated Evaluations
- **Step 1 vs Step 2 Answer Quality Evaluation:**
  ```powershell
  # Offline / mock mode (fast, deterministic, 0 API quota used):
  python src\evaluate_answers.py --mock

  # Live Gemini mode (uses live model and embeddings):
  python src\evaluate_answers.py --live
  ```
- **Multi-Turn Conversation Evaluation:**
  ```powershell
  # Offline / mock mode:
  python src\evaluate_conversations.py --mock

  # Live mode:
  python src\evaluate_conversations.py --live
  ```
- **Retrieval Recall@5 Evaluation:**
  ```powershell
  python -m src.evaluate_retrieval
  ```

### 4. Automated Test Suite
Run all 68 unit, integration, and safeguard regression tests:
```powershell
python -m pytest -q
```

### 5. Packaging & Submission Verification
Generate and verify the final submission archive `team_name_voiceops.zip`:
```powershell
python src\package_submission.py
```

---

## 7. Evaluation Methodology & Measured Results

The project was benchmarked using the official evaluation datasets in `Capstone_Project_Dataset_FTTT_2026/practice/`:
1. **Answer Quality Benchmark (`questions.csv` — 52 questions: 47 answerable, 5 unanswerable)**
2. **Multi-Turn Conversation Benchmark (`conversations.csv` — 23 conversations, 96 customer turns)**

### Summary of Benchmark Results

| Evaluation Metric | Step 1 Baseline | Step 2 RAG Grounded | Target / Standard |
| :--- | :---: | :---: | :---: |
| **Fact Inclusion Coverage** | 11.7% | **94.3%** | $> 85.0\%$ |
| **Hallucination Violations** | 0 | **0** | $0$ (Zero tolerance) |
| **Honest Uncertainty Rate (Unanswerable)** | 0.0% | **100.0%** | $100.0\%$ |
| **Retrieval Recall@5** | N/A | **100.0%** | $> 90.0\%$ |
| **Section ID Citation Rate** | 0.0% | **90.4%** | $> 85.0\%$ |
| **Overall Answer Quality Score** | 10.6% | **94.9%** | $> 85.0\%$ |

### Multi-Turn Conversation Results (`conversations.csv`)
- **Total Conversations Evaluated:** 23
- **Total Customer Turns Evaluated:** 96
- **Fact Inclusion Rate:** 97.4%
- **Customer Privacy Gate Compliance:** **100.0%** (Zero unverified disclosures)
- **Tool Trigger Accuracy:** **100.0%** (lookup_order & create_ticket)
- **Overall Multi-Turn Conversation Score:** **97.5%**

---

## 8. Privacy, Verification & Safeguard Behaviors

1. **Strict Customer Privacy Gate:**
   - When a customer provides an Order ID, the system performs an unverified lookup.
   - Private details (order items, delivery address, courier name, tracking number) remain completely suppressed until the customer verifies their identity using either:
     - The registered phone number's last four digits, or
     - The delivery pincode.
   - Verification allows a maximum of 2 attempts before escalating to a human representative.
2. **Ticket Creation Confirmation Safeguard:**
   - The assistant never promises that a support ticket has been created unless the underlying tool explicitly returns `created=True`.
   - On tool failure, the assistant transparently apologizes and advises the customer to contact human support directly, preventing phantom ticket claims.
3. **High-Value Product Escalation:**
   - Damaged items with a catalog price > ₹10,000 automatically trigger priority escalation under `[DWM-3]` and `[ESC-1]` rather than standard self-serve return.
4. **Empty / Inaudible Audio Safeguard:**
   - Audio clips with empty, silent, or whitespace-only transcripts immediately bypass model invocation and return:
     `"Sorry, I didn't catch that. Could you say it again?"`
   - Generated artifacts are saved safely with an empty evidence list.

---

## 9. Expected Output Locations

When `python run.py --input-dir <input> --output-dir <output>` is executed, the following files are produced for each input WAV:
- `<output>/<clip>_transcript.txt` — Recognized speech text from Gnani STT.
- `<output>/<clip>_reply.txt` — AI assistant response text.
- `<output>/<clip>_reply.wav` — Synthesized spoken audio reply from Gnani TTS.
- `<output>/<clip>_evidence.json` — Knowledge-base sections retrieved and used as grounding evidence.

Representative verified examples are preserved in `sample_outputs/` for inspection:
- Standard interaction: `test_transcript.txt`, `test_reply.txt`, `test_reply.wav`, `test_evidence.json`
- Inaudible fallback: `inaudible_sample_transcript.txt`, `inaudible_sample_reply.txt`, `inaudible_sample_reply.wav`, `inaudible_sample_evidence.json`

Evaluation summaries are stored in:
- `outputs/evaluation/step1_vs_step2_summary.md` and `step1_vs_step2_summary.csv`
- `outputs/evaluation/conversations_summary.md` and `conversations_summary.csv`

---

## 10. Deliverables & Submission Archive

The final submission archive is built using `python src/package_submission.py`, producing `team_name_voiceops.zip`:
- **`README.md`** — Comprehensive project documentation and execution guide.
- **`.env.example`** — Safe configuration template with zero exposed secrets.
- **`requirements.txt`** — Pinned clean dependencies.
- **`run.py`** — Root-level batch runner.
- **`src/`** — Core implementation source modules.
- **`tests/`** — Complete automated test suite.
- **`sample_outputs/`** — Verified representative output artifacts.
- **`audio_clips/`** — Sample input audio clips.
- **`report.pdf`** — 4-page capstone report with architecture diagrams and benchmark tables.
- **`demo.mp4`** — 62-second 1080p demonstration video showcasing RAG, verification, and human handoff.

**Strictly Excluded from Archive:** `.env`, `.venv/`, `Capstone_Project_Dataset_FTTT_2026/`, `outputs/`, `.cache/`, `.pytest_cache/`, `__pycache__/`, and `.bak` files.

---

## 11. Troubleshooting Common Issues

1. **`RuntimeError: GEMINI_API_KEY is missing` or `GNANI_API_KEY is missing`:**
   - Ensure you created a local `.env` file in the project root containing your actual API keys.
2. **`ValueError: Audio duration exceeds the maximum allowed limit of 60.0 seconds`:**
   - The capstone specification limits customer audio clips to 60 seconds. Trim or re-record audio clips longer than 60s.
3. **`ValueError: Unsupported audio format`:**
   - Gnani STT requires standard WAV format (`.wav`). Convert audio using `ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav`.
4. **Network Timeout on Gemini API Calls:**
   - The client has a configured 30-second bounded timeout (`http_options=types.HttpOptions(timeout=30000)`). If an API call times out due to network congestion, the system gracefully reports the issue without hanging.
5. **Embedding Cache Stale Warning:**
   - If knowledge base documents are modified, the system automatically computes a SHA-256 fingerprint hash and rebuilds the embedding cache in `.cache/kb_embeddings.json`.
