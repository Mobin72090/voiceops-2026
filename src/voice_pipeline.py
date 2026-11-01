"""Stage 4 Voice Pipeline for ZentroMart VoiceOps 2026.

Integrates Gnani.ai STT (en-IN) and TTS (Kaveri, timbre-v2.5) with the
conversation agent and knowledge-base retriever.
"""

import argparse
import json
import os
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conversation_agent import call_model, create_client
from rag_chatbot import build_evidence
from retriever import search_documents

DEFAULT_VOICE = "Kaveri"
DEFAULT_MODEL = "timbre-v2.5"
DEFAULT_LANGUAGE = "en-IN"
MAX_AUDIO_DURATION_SECONDS = 60.0
UNCLEAR_AUDIO_FALLBACK = "Sorry, I didn't catch that. Could you say it again?"


def validate_audio(audio_path: Path | str, max_duration: float = MAX_AUDIO_DURATION_SECONDS) -> float:
    """Validate that audio file exists, is a valid WAV, and does not exceed max_duration.

    Returns the duration in seconds.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. Only .wav files are supported."
        )

    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                raise ValueError(f"Invalid sample rate ({rate}) in WAV file: {path}")
            duration = frames / float(rate)
    except wave.Error as exc:
        raise ValueError(f"Corrupted or invalid WAV file: {path}") from exc

    if duration <= 0:
        raise ValueError(f"Audio file has non-positive duration ({duration:.2f}s): {path}")

    if duration > max_duration:
        raise ValueError(
            f"Audio duration ({duration:.2f}s) exceeds the maximum allowed limit "
            f"of {max_duration:.1f} seconds: {path}"
        )

    return duration


def create_stt_client(api_key: str | None = None):
    """Create a live Gnani STT client or raise an informative error if unavailable."""
    api_key = api_key or os.getenv("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GNANI_API_KEY is missing. Check your local .env file."
        )
    try:
        from gnani.stt import GnaniSTTClient
        return GnaniSTTClient(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError(
            "gnani-vachana package is not installed. Run 'pip install gnani-vachana'."
        ) from exc


def create_tts_client(api_key: str | None = None):
    """Create a live Gnani TTS client or raise an informative error if unavailable."""
    api_key = api_key or os.getenv("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GNANI_API_KEY is missing. Check your local .env file."
        )
    try:
        from gnani.tts import GnaniTTSClient
        return GnaniTTSClient(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError(
            "gnani-vachana package is not installed. Run 'pip install gnani-vachana'."
        ) from exc


def transcribe_audio(
    audio_path: Path | str,
    stt_client=None,
    language_code: str = DEFAULT_LANGUAGE,
    max_duration: float = MAX_AUDIO_DURATION_SECONDS,
) -> str:
    """Validate audio duration and transcribe speech using Gnani STT."""
    audio_path = Path(audio_path)
    validate_audio(audio_path, max_duration=max_duration)

    client = stt_client or create_stt_client()
    result = client.transcribe(str(audio_path), language_code=language_code)

    if isinstance(result, dict):
        transcript = result.get("transcript") or result.get("text") or ""
    elif isinstance(result, str):
        transcript = result
    else:
        transcript = getattr(result, "transcript", getattr(result, "text", str(result)))

    return str(transcript).strip()


def normalize_text_for_speech(text: str) -> str:
    """Normalize text for ear / TTS synthesis per Capstone Brief Page 11.

    - Breaks long numeric order IDs (6+ digits) into readable grouped digits.
    - Strips bracketed section citations (e.g. [RET-2]) and markdown asterisks
      so synthesized speech sounds natural.
    """
    if not text:
        return text

    import re

    def _group_digits(match):
        digits = match.group(0)
        return " ".join([digits[i:i + 4] for i in range(0, len(digits), 4)])

    normalized = re.sub(r"\b\d{6,12}\b", _group_digits, text)
    normalized = re.sub(r"\[(RET|DWM|CAN|REF|ESC|WAR|FAQ|HLP)-\d+\]", "", normalized)
    normalized = re.sub(r"\*+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def synthesize_speech(
    text: str,
    output_audio_path: Path | str,
    tts_client=None,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> bytes:
    """Synthesize text to speech using Gnani TTS and save to a .wav file."""
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    spoken_text = normalize_text_for_speech(text)

    client = tts_client or create_tts_client()
    audio_response = client.synthesize(
        text=spoken_text,
        voice=voice,
        model=model,
        language=language,
    )

    if hasattr(audio_response, "content"):
        audio_bytes = audio_response.content
    elif isinstance(audio_response, bytes):
        audio_bytes = audio_response
    elif isinstance(audio_response, dict) and "audio" in audio_response:
        audio_bytes = audio_response["audio"]
    else:
        audio_bytes = bytes(audio_response) if audio_response else b""

    if not output_audio_path.is_file() or output_audio_path.stat().st_size == 0:
        output_audio_path.write_bytes(audio_bytes)
    return audio_bytes


def voice_turn(
    audio_path: Path | str,
    agent=None,
    stt_client=None,
    tts_client=None,
    output_file: Path | str = "reply.wav",
    language_code: str = DEFAULT_LANGUAGE,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """Execute a single voice turn matching the official capstone brief specification (Page 10).

    Transcribes audio with Gnani STT, queries Step 3 agent, and synthesizes spoken reply with Gnani TTS.
    Returns (heard, reply).
    """
    stt = stt_client or create_stt_client()
    tts = tts_client or create_tts_client()

    heard = transcribe_audio(audio_path, stt_client=stt, language_code=language_code)

    if not heard.strip():
        reply = UNCLEAR_AUDIO_FALLBACK
    else:
        if agent is not None and hasattr(agent, "respond"):
            reply = agent.respond(heard)
        elif agent is not None and callable(agent):
            reply = agent(heard)
        else:
            retrieved_sections = search_documents(heard, top_k=5)
            agent_prompt = format_rag_prompt(heard, retrieved_sections)
            gemini = create_client()
            reply = call_model(gemini, [], agent_prompt)

    synthesize_speech(
        text=reply,
        output_audio_path=output_file,
        tts_client=tts,
        voice=voice,
        model=model,
        language=language_code,
    )

    return heard, reply


def format_rag_prompt(question: str, results: list) -> str:
    """Format customer question and retrieved knowledge-base evidence into an agent prompt."""
    if not results:
        return question

    evidence = build_evidence(results)
    return (
        f"Today's date is 20 October 2026. Business days run Mon–Sat excluding declared holidays; return windows use calendar days.\n\n"
        f"Customer query:\n{question}\n\n"
        f"Retrieved knowledge-base evidence:\n{evidence}\n\n"
        f"Answer the customer clearly and concisely using only relevant, authoritative evidence. "
        f"Include supporting section IDs in square brackets where applicable."
    )


def process_audio_clip(
    audio_path: Path | str,
    output_dir: Path | str = "outputs",
    history: list | None = None,
    gemini_client=None,
    stt_client=None,
    tts_client=None,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    top_k: int = 5,
) -> dict:
    """Process a single audio clip: STT -> RAG + Agent -> TTS -> Save artifacts."""
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_stem = audio_path.stem

    if history is None:
        history = []

    # 1. Transcribe audio
    transcript = transcribe_audio(
        audio_path,
        stt_client=stt_client,
        language_code=language,
    )
    transcript_file = output_dir / f"{clip_stem}_transcript.txt"
    transcript_file.write_text(transcript, encoding="utf-8")

    # 2. Handle empty or whitespace-only transcripts safely
    if not transcript or not transcript.strip():
        evidence_data = []
        evidence_file = output_dir / f"{clip_stem}_evidence.json"
        evidence_file.write_text(json.dumps(evidence_data, indent=2), encoding="utf-8")

        reply_text = UNCLEAR_AUDIO_FALLBACK
        reply_text_file = output_dir / f"{clip_stem}_reply.txt"
        reply_text_file.write_text(reply_text, encoding="utf-8")
    else:
        # Retrieve knowledge-base evidence
        retrieved_sections = search_documents(transcript, top_k=top_k)
        evidence_data = [
            {
                "section_id": item["section_id"],
                "source": item["source"],
                "heading": item["heading"],
                "doc_type": item.get("doc_type", "unspecified"),
                "status": item.get("status", "unspecified"),
                "score": item.get("score", 0.0),
            }
            for item in retrieved_sections
        ]
        evidence_file = output_dir / f"{clip_stem}_evidence.json"
        evidence_file.write_text(json.dumps(evidence_data, indent=2), encoding="utf-8")

        # 3. Generate agent reply with multi-turn memory
        agent_prompt = format_rag_prompt(transcript, retrieved_sections)
        client_for_agent = gemini_client or create_client()
        reply_text = call_model(client_for_agent, history, agent_prompt)

        reply_text_file = output_dir / f"{clip_stem}_reply.txt"
        reply_text_file.write_text(reply_text, encoding="utf-8")

    # 4. Synthesize reply audio
    reply_audio_file = output_dir / f"{clip_stem}_reply.wav"
    synthesize_speech(
        text=reply_text,
        output_audio_path=reply_audio_file,
        tts_client=tts_client,
        voice=voice,
        model=model,
        language=language,
    )

    return {
        "clip": audio_path.name,
        "transcript": transcript,
        "reply_text": reply_text,
        "evidence": evidence_data,
        "transcript_file": str(transcript_file),
        "reply_text_file": str(reply_text_file),
        "reply_audio_file": str(reply_audio_file),
        "evidence_file": str(evidence_file),
    }


def process_audio_folder(
    input_dir: Path | str,
    output_dir: Path | str = "outputs",
    continuous_conversation: bool = True,
    gemini_client=None,
    stt_client=None,
    tts_client=None,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """Process all supported audio files in input_dir, preserving conversation memory if requested."""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return []

    all_files = sorted(f for f in input_path.iterdir() if f.is_file())
    audio_files = [f for f in all_files if f.suffix.lower() == ".wav"]
    unsupported_files = [f for f in all_files if f.suffix.lower() != ".wav" and not f.name.startswith(".")]

    for unf in unsupported_files:
        print(f"Skipping unsupported audio file '{unf.name}'. Only .wav files are supported.")

    if not audio_files:
        print(f"No supported audio files (.wav) found in directory: {input_dir}")
        return []

    shared_history = [] if continuous_conversation else None
    results = []

    for audio_file in audio_files:
        history_for_turn = shared_history if continuous_conversation else []
        try:
            res = process_audio_clip(
                audio_path=audio_file,
                output_dir=output_dir,
                history=history_for_turn,
                gemini_client=gemini_client,
                stt_client=stt_client,
                tts_client=tts_client,
                voice=voice,
                model=model,
                language=language,
            )
            results.append(res)
        except Exception as exc:
            print(f"Error processing '{audio_file.name}': {exc}")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ZentroMart Stage 4 Voice Pipeline (Gnani STT/TTS + Support Agent)"
    )
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="Directory containing customer voice message clips (.wav)",
    )
    parser.add_argument(
        "positional_output",
        nargs="?",
        default=None,
        help="Directory where transcripts, replies, audio, and evidence will be saved (optional positional)",
    )
    parser.add_argument(
        "--input-dir",
        dest="flag_input_dir",
        default=None,
        help="Directory containing customer voice message clips (.wav)",
    )
    parser.add_argument(
        "--output-dir",
        dest="flag_output_dir",
        default=None,
        help="Directory where transcripts, replies, audio, and evidence will be saved (default: outputs)",
    )
    parser.add_argument(
        "--separate-conversations",
        action="store_true",
        help="Do not maintain continuous conversation memory across audio clips in the folder",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline using mock STT/TTS and agent responses (no live API calls)",
    )
    args = parser.parse_args(argv)

    input_dir_str = args.flag_input_dir or args.positional_input
    if not input_dir_str:
        parser.error("the following arguments are required: --input-dir (or positional input directory)")

    output_dir_str = args.flag_output_dir or args.positional_output or "outputs"
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    gemini_client = None
    stt_client = None
    tts_client = None

    if args.mock:
        from unittest.mock import MagicMock

        class MockInteraction:
            def __init__(self, interaction_id, output_text):
                self.id = interaction_id
                self.output_text = output_text

        def mock_transcribe(path, **kwargs):
            p = Path(path)
            if "inaudible" in p.name.lower() or "silence" in p.name.lower() or "empty" in p.name.lower():
                return ""
            return "What is the return policy for electronics?"

        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = mock_transcribe

        mock_tts = MagicMock()
        # Valid RIFF header with 1s of 16kHz mono 16-bit silence
        mock_tts.synthesize.return_value = MagicMock(
            content=b"RIFF\x24\x7d\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x7d\x00\x00" + b"\x00" * 32000
        )

        mock_gemini = MagicMock()
        mock_gemini.interactions.create.return_value = MockInteraction(
            "int-mock-turn",
            "Under ZentroMart's policies [RET-3] and [RET-2], the return window for electronics is 7 days from delivery.",
        )

        stt_client = mock_stt
        tts_client = mock_tts
        gemini_client = mock_gemini

    results = process_audio_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        continuous_conversation=not args.separate_conversations,
        gemini_client=gemini_client,
        stt_client=stt_client,
        tts_client=tts_client,
    )

    if not results:
        return 0

    print(f"\nSuccessfully processed {len(results)} clip(s). Outputs saved to: {output_dir}")
    for res in results:
        print(f"\n- Clip: {res['clip']}")
        print(f"  Transcript: {res['transcript']}")
        preview = res['reply_text'][:100] + "..." if len(res['reply_text']) > 100 else res['reply_text']
        print(f"  Reply: {preview}")
        print(f"  Evidence items: {len(res['evidence'])}")
        print(f"  Audio: {res['reply_audio_file']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
