# VoiceOps 2026 — ZentroMart Customer Support Assistant

A voice-enabled AI customer-support assistant built for the VoiceOps 2026 capstone. It combines speech recognition, retrieval-augmented generation (RAG), multi-turn conversation handling, order-verification safeguards, ticket creation, and speech synthesis.

## Features

- **Speech-to-text:** Gnani STT using `en-IN`.
- **AI responses:** Gemini through the Google GenAI SDK.
- **Knowledge retrieval:** Retrieves relevant sections from the supplied ZentroMart knowledge base and records evidence.
- **Conversation handling:** Supports follow-up messages and conversation history.
- **Order verification:** Requires an Order ID and either the registered phone number's last four digits or the delivery pincode before disclosing order details.
- **Ticket creation:** Uses the supplied mock tool and reports success only when the tool confirms creation.
- **Text-to-speech:** Gnani TTS using the Kaveri voice and `timbre-v2.5`.
- **Audio validation:** Rejects unsupported audio and clips longer than 60 seconds.

## Requirements

- Python 3.14 or another version compatible with the installed dependencies
- Gemini API key
- Gnani API key with access to the required STT and TTS services

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your own credentials:

```powershell
Copy-Item .env.example .env
```

Set these variables in `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
GNANI_API_KEY=your_gnani_api_key
```

Never commit `.env` or share API keys.

## Run the voice pipeline

Place supported WAV recordings in an input folder. Each recording must be no longer than 60 seconds.

```powershell
python src\voice_pipeline.py --input-dir audio_clips --output-dir outputs
```

The pipeline transcribes each WAV file, retrieves knowledge-base evidence, generates an assistant response, and synthesizes a spoken reply.

For each processed clip, the output folder contains:

- `<clip>_transcript.txt` — recognized speech
- `<clip>_reply.txt` — assistant response
- `<clip>_reply.wav` — generated speech
- `<clip>_evidence.json` — retrieved evidence metadata

## Run tests

```powershell
python -m pytest -v
```

## Validate the dataset

```powershell
python Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
```

## Project structure

```text
voiceops-2026/
├── audio_clips/
├── Capstone_Project_Dataset_FTTT_2026/
├── outputs/
├── src/
│   ├── chatbot.py
│   ├── conversation_agent.py
│   ├── rag_chatbot.py
│   ├── retriever.py
│   └── voice_pipeline.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Security and data handling

This project uses the supplied synthetic capstone dataset and mock support tools. Order details must not be disclosed until verification succeeds. Verification should allow no more than one retry after the initial attempt. A support handover must not be reported as successful unless ticket creation is confirmed by the tool.

Do not commit API credentials, virtual environments, caches, or generated output files.

## Limitations

This is a capstone prototype, not a production customer-support system. The supplied dataset and mock tools are intended for educational testing.
