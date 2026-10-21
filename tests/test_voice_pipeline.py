import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voice_pipeline import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    create_stt_client,
    create_tts_client,
    main,
    process_audio_clip,
    process_audio_folder,
    synthesize_speech,
    transcribe_audio,
    validate_audio,
)


def create_test_wav(file_path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """Helper to create a valid test WAV file with silence."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * duration_seconds)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return file_path


class MockInteraction:
    def __init__(self, interaction_id, output_text):
        self.id = interaction_id
        self.output_text = output_text


# ---------------------------------------------------------------------------
# 1. Audio Duration & Format Validation
# ---------------------------------------------------------------------------

def test_validate_audio_valid(tmp_path):
    wav_path = create_test_wav(tmp_path / "valid.wav", duration_seconds=2.5)
    duration = validate_audio(wav_path)
    assert pytest.approx(duration, 0.05) == 2.5


def test_validate_audio_exceeds_60_seconds(tmp_path):
    wav_path = create_test_wav(tmp_path / "too_long.wav", duration_seconds=61.0)
    with pytest.raises(ValueError, match="exceeds the maximum allowed limit of 60.0 seconds"):
        validate_audio(wav_path)


def test_validate_audio_corrupted(tmp_path):
    corrupted_path = tmp_path / "corrupted.wav"
    corrupted_path.write_bytes(b"NOT_A_VALID_RIFF_HEADER_123456789")
    with pytest.raises(ValueError, match="Corrupted or invalid WAV file"):
        validate_audio(corrupted_path)


def test_validate_audio_non_existent(tmp_path):
    missing_path = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        validate_audio(missing_path)


def test_validate_audio_unsupported_format(tmp_path):
    mp3_path = tmp_path / "sample.mp3"
    mp3_path.write_bytes(b"dummy mp3 data")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        validate_audio(mp3_path)


# ---------------------------------------------------------------------------
# 2. STT Transcription with Gnani settings
# ---------------------------------------------------------------------------

def test_transcribe_audio_uses_en_in_and_calls_client(tmp_path):
    wav_path = create_test_wav(tmp_path / "speech.wav", duration_seconds=1.5)

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {"transcript": "What is the return policy for electronics?"}

    transcript = transcribe_audio(
        wav_path,
        stt_client=mock_stt,
        language_code="en-IN",
    )

    assert transcript == "What is the return policy for electronics?"
    mock_stt.transcribe.assert_called_once_with(
        str(wav_path),
        language_code="en-IN",
    )


# ---------------------------------------------------------------------------
# 3. TTS Synthesis with Kaveri, timbre-v2.5, en-IN
# ---------------------------------------------------------------------------

def test_synthesize_speech_settings(tmp_path):
    output_wav = tmp_path / "reply.wav"
    dummy_audio_bytes = b"RIFF_MOCK_WAV_BYTES_DATA"

    mock_tts = MagicMock()
    mock_audio_resp = MagicMock()
    mock_audio_resp.content = dummy_audio_bytes
    mock_tts.synthesize.return_value = mock_audio_resp

    result_bytes = synthesize_speech(
        text="Your return request is approved.",
        output_audio_path=output_wav,
        tts_client=mock_tts,
        voice=DEFAULT_VOICE,
        model=DEFAULT_MODEL,
        language=DEFAULT_LANGUAGE,
    )

    assert result_bytes == dummy_audio_bytes
    assert output_wav.read_bytes() == dummy_audio_bytes
    mock_tts.synthesize.assert_called_once_with(
        text="Your return request is approved.",
        voice="Kaveri",
        model="timbre-v2.5",
        language="en-IN",
    )


# ---------------------------------------------------------------------------
# 4. Missing Dependencies and Credentials Error Handling
# ---------------------------------------------------------------------------

def test_create_stt_client_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GNANI_API_KEY is missing"):
            create_stt_client(api_key=None)


def test_create_tts_client_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GNANI_API_KEY is missing"):
            create_tts_client(api_key=None)


def test_create_stt_client_missing_package():
    with patch.dict("os.environ", {"GNANI_API_KEY": "fake_key"}):
        with patch.dict("sys.modules", {"gnani": None, "gnani.stt": None}):
            with pytest.raises(RuntimeError, match="gnani-vachana package is not installed"):
                create_stt_client()


def test_create_tts_client_missing_package():
    with patch.dict("os.environ", {"GNANI_API_KEY": "fake_key"}):
        with patch.dict("sys.modules", {"gnani": None, "gnani.tts": None}):
            with pytest.raises(RuntimeError, match="gnani-vachana package is not installed"):
                create_tts_client()


def test_create_clients_when_package_installed():
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    stt = create_stt_client(api_key="mock_key")
    tts = create_tts_client(api_key="mock_key")
    assert isinstance(stt, GnaniSTTClient)
    assert isinstance(tts, GnaniTTSClient)


# ---------------------------------------------------------------------------
# 5. End-to-End Audio Clip & Folder Processing
# ---------------------------------------------------------------------------

def test_process_audio_clip_end_to_end(tmp_path):
    wav_path = create_test_wav(tmp_path / "query.wav", duration_seconds=2.0)
    out_dir = tmp_path / "outputs"

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {"transcript": "What is the return window for electronics?"}

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = MagicMock(content=b"RIFF_AUDIO_CONTENT")

    mock_gemini = MagicMock()
    mock_gemini.interactions.create.return_value = MockInteraction(
        interaction_id="int-voice-001",
        output_text="Electronics have a 7-day return window as per [RET-2].",
    )

    history = []
    res = process_audio_clip(
        audio_path=wav_path,
        output_dir=out_dir,
        history=history,
        gemini_client=mock_gemini,
        stt_client=mock_stt,
        tts_client=mock_tts,
    )

    assert res["clip"] == "query.wav"
    assert res["transcript"] == "What is the return window for electronics?"
    assert "7-day return window" in res["reply_text"]

    # Verify artifacts were written to output_dir
    transcript_file = Path(res["transcript_file"])
    reply_text_file = Path(res["reply_text_file"])
    reply_audio_file = Path(res["reply_audio_file"])
    evidence_file = Path(res["evidence_file"])

    assert transcript_file.is_file()
    assert reply_text_file.is_file()
    assert reply_audio_file.is_file()
    assert evidence_file.is_file()

    assert transcript_file.read_text(encoding="utf-8") == res["transcript"]
    assert reply_text_file.read_text(encoding="utf-8") == res["reply_text"]
    assert reply_audio_file.read_bytes() == b"RIFF_AUDIO_CONTENT"

    evidence_json = json.loads(evidence_file.read_text(encoding="utf-8"))
    assert len(evidence_json) > 0
    section_ids = {item["section_id"] for item in evidence_json}
    assert "RET-2" in section_ids or "RET-3" in section_ids


def test_process_audio_folder_continuous_conversation(tmp_path):
    input_dir = tmp_path / "audio_input"
    out_dir = tmp_path / "outputs"

    clip1 = create_test_wav(input_dir / "clip_01.wav", duration_seconds=1.0)
    clip2 = create_test_wav(input_dir / "clip_02.wav", duration_seconds=1.0)

    mock_stt = MagicMock()
    mock_stt.transcribe.side_effect = [
        {"transcript": "Hello, my name is Priya."},
        {"transcript": "Can you remind me of my name?"},
    ]

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = MagicMock(content=b"RIFF_AUDIO_RESPONSE")

    mock_gemini = MagicMock()
    mock_gemini.interactions.create.side_effect = [
        MockInteraction("int-turn-1", "Hello Priya! How can I help you today?"),
        MockInteraction("int-turn-2", "Your name is Priya."),
    ]

    results = process_audio_folder(
        input_dir=input_dir,
        output_dir=out_dir,
        continuous_conversation=True,
        gemini_client=mock_gemini,
        stt_client=mock_stt,
        tts_client=mock_tts,
    )

    assert len(results) == 2
    assert results[0]["clip"] == "clip_01.wav"
    assert results[1]["clip"] == "clip_02.wav"

    # Multi-turn check: turn 1 has no previous_interaction_id; turn 2 passes int-turn-1
    call_args_turn1 = mock_gemini.interactions.create.call_args_list[0].kwargs
    assert "previous_interaction_id" not in call_args_turn1

    call_args_turn2 = mock_gemini.interactions.create.call_args_list[1].kwargs
    assert call_args_turn2.get("previous_interaction_id") == "int-turn-1"


def test_process_audio_folder_empty_dir(tmp_path, capsys):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    results = process_audio_folder(input_dir=empty_dir, output_dir=tmp_path / "out")
    assert results == []

    captured = capsys.readouterr().out
    assert "No supported audio files (.wav) found" in captured


def test_cli_empty_directory(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"

    exit_code = main(["--input-dir", str(empty_dir), "--output-dir", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "No supported audio files (.wav) found" in captured
