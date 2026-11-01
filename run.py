#!/usr/bin/env python3
"""VoiceOps 2026 Root Submission Runner.

Serves as the root execution entry point specified in the official capstone brief:
    python run.py --input-dir <path> --output-dir <path>

Delegates directly to src.voice_pipeline.main to avoid duplicating pipeline logic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voice_pipeline import main


def run(argv: list[str] | None = None) -> int:
    """Execute the voice pipeline batch runner."""
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
