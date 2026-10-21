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

    client = tts_client or create_tts_client()
    audio_response = client.synthesize(
        text=text,
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
        audio_bytes = bytes(audio_response)

    output_audio_path.write_bytes(audio_bytes)
    return audio_bytes


def format_rag_prompt(question: str, results: list) -> str:
    """Format customer question and retrieved knowledge-base evidence into an agent prompt."""
    if not results:
        return question

    evidence = build_evidence(results)
    return (
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

    # 2. Retrieve knowledge-base evidence
    retrieved_sections = search_documents(transcript, top_k=top_k) if transcript else []
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

    audio_files = sorted(f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == ".wav")
    if not audio_files:
        print(f"No supported audio files (.wav) found in directory: {input_dir}")
        return []

    shared_history = [] if continuous_conversation else None
    results = []

    for audio_file in audio_files:
        history_for_turn = shared_history if continuous_conversation else []
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

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ZentroMart Stage 4 Voice Pipeline (Gnani STT/TTS + Support Agent)"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing customer voice message clips (.wav)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where transcripts, replies, audio, and evidence will be saved (default: outputs)",
    )
    parser.add_argument(
        "--separate-conversations",
        action="store_true",
        help="Do not maintain continuous conversation memory across audio clips in the folder",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    results = process_audio_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        continuous_conversation=not args.separate_conversations,
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
