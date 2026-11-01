"""Submission Packaging & Deliverables Validator for VoiceOps 2026.

Creates team_name_voiceops.zip in the project root containing:
- README.md (with GitHub repository URL and run instructions)
- .env.example (clean placeholders, zero secrets)
- .gitignore
- requirements.txt
- run.py
- src/ (core implementation modules, clean source files)
- tests/ (automated test suites)
- sample_outputs/ (representative outputs: transcript, reply, wav, evidence)
- audio_clips/ (sample input audio clips)
- report.pdf (verified <= 4 pages)
- demo.mp4 (verified <= 120 seconds)

Strictly excludes:
- .env and secrets
- .venv/
- .cache/
- .git/
- Capstone_Project_Dataset_FTTT_2026/
- outputs/
- __pycache__/, *.pyc
"""

import os
import re
import sys
import zipfile
from pathlib import Path
import pypdf
import subprocess
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_FILENAME = "team_name_voiceops.zip"
ZIP_PATH = PROJECT_ROOT / ZIP_FILENAME

# Required Deliverable Paths
REQUIRED_DELIVERABLES = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "run.py",
    "src",
    "report.pdf",
    "demo.mp4",
    "sample_outputs",
]

# Explicit Exclusions
EXCLUDED_DIR_NAMES = {
    ".venv",
    ".cache",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "Capstone_Project_Dataset_FTTT_2026",
    "outputs",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ZIP_FILENAME,
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".bak",
}


def verify_readme():
    """Verify README contains public GitHub URL and run command."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md missing"
    content = readme_path.read_text(encoding="utf-8")
    assert "https://github.com/Mobin72090/voiceops-2026.git" in content, "README missing GitHub URL"
    assert "python run.py --input-dir audio_clips --output-dir outputs" in content, "README missing run command"
    print("[OK] README.md verified (contains GitHub URL and exact run command)")


def verify_env_example():
    """Verify .env.example contains no real secrets."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), ".env.example missing"
    content = env_example.read_text(encoding="utf-8")
    for line in content.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            val = val.strip()
            assert val == "", f".env.example contains value for {key}: '{val}'"
    print("[OK] .env.example verified (contains clean placeholders with 0 secrets)")


def verify_report_pdf():
    """Verify report.pdf page count <= 4."""
    pdf_path = PROJECT_ROOT / "report.pdf"
    assert pdf_path.exists(), "report.pdf missing"
    reader = pypdf.PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    assert num_pages <= 4, f"report.pdf has {num_pages} pages, which exceeds the 4-page limit"
    print(f"[OK] report.pdf verified ({num_pages} pages <= 4 pages)")
    return num_pages


def verify_demo_mp4():
    """Verify demo.mp4 duration <= 120 seconds and valid playback."""
    video_path = PROJECT_ROOT / "demo.mp4"
    assert video_path.exists(), "demo.mp4 missing"
    
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=codec_name,width,height",
        "-of", "json", str(video_path)
    ]
    res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    meta = json.loads(res.stdout)
    duration = float(meta["format"]["duration"])
    size = int(meta["format"]["size"])
    assert duration <= 120.0, f"demo.mp4 duration {duration:.2f}s exceeds 120s limit"
    
    # Quick integrity decode check
    decode_cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"]
    subprocess.run(decode_cmd, check=True)
    
    print(f"[OK] demo.mp4 verified (duration: {duration:.2f}s <= 120s, size: {size:,} bytes, decode: PASS)")
    return duration, size


def verify_sample_outputs():
    """Verify sample_outputs contains matching transcript, reply, wav, and evidence."""
    s_dir = PROJECT_ROOT / "sample_outputs"
    assert s_dir.exists() and s_dir.is_dir(), "sample_outputs/ missing"
    required_files = [
        "test_transcript.txt",
        "test_reply.txt",
        "test_reply.wav",
        "test_evidence.json",
        "inaudible_sample_transcript.txt",
        "inaudible_sample_reply.txt",
        "inaudible_sample_reply.wav",
        "inaudible_sample_evidence.json",
    ]
    for rf in required_files:
        p = s_dir / rf
        assert p.exists(), f"sample_outputs/{rf} missing"
    print(f"[OK] sample_outputs verified (all {len(required_files)} representative artifacts present)")


def collect_files_to_pack():
    """Collect all valid files to include, strictly respecting exclusions."""
    files_to_pack = []
    
    # Specific top-level files
    top_files = [
        "README.md",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "run.py",
        "report.pdf",
        "demo.mp4",
    ]
    if (PROJECT_ROOT / "run.sh").exists():
        top_files.append("run.sh")
    for tf in top_files:
        p = PROJECT_ROOT / tf
        if p.exists():
            files_to_pack.append(p)
        else:
            raise RuntimeError(f"Missing top-level file: {tf}")
            
    # Directories to pack recursively
    include_dirs = ["src", "tests", "sample_outputs", "audio_clips"]
    for d_name in include_dirs:
        d_path = PROJECT_ROOT / d_name
        if not d_path.exists():
            raise RuntimeError(f"Missing directory: {d_name}")
        for root, dirs, files in os.walk(d_path):
            # Exclude unwanted directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
            for f in files:
                f_path = Path(root) / f
                if f in EXCLUDED_FILE_NAMES:
                    continue
                if f_path.suffix in EXCLUDED_EXTENSIONS or f.endswith(".bak") or ".bak" in f:
                    continue
                if "_remux.wav" in f or "_fixed.wav" in f:
                    continue
                files_to_pack.append(f_path)
                
    return files_to_pack


def create_zip():
    """Create the submission zip file."""
    print(f"\n--- Packaging {ZIP_FILENAME} ---")
    files = collect_files_to_pack()
    print(f"Total files selected for packaging: {len(files)}")
    
    # Secret check before packaging
    secret_pat = re.compile(r'AIza[0-9A-Za-z-_]{35}|gnani_[0-9a-zA-Z]{15,}')
    for f_path in files:
        if f_path.suffix in [".py", ".txt", ".json", ".md", ".example"]:
            try:
                txt = f_path.read_text(encoding="utf-8", errors="ignore")
                matches = secret_pat.findall(txt)
                if matches:
                    raise RuntimeError(f"POTENTIAL SECRET FOUND in {f_path}: {matches}")
            except Exception as e:
                if "POTENTIAL SECRET" in str(e):
                    raise
    print("[OK] Pre-packaging secret scan passed (0 secret tokens found)")

    # Build ZIP
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f_path in sorted(files):
            arcname = f_path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(f_path, arcname)
            
    zip_size = ZIP_PATH.stat().st_size
    print(f"[OK] Created {ZIP_PATH.name} ({zip_size:,} bytes / {zip_size / (1024*1024):.2f} MB)")
    return ZIP_PATH


def inspect_and_verify_zip():
    """Inspect and verify the contents of the generated ZIP file."""
    print(f"\n--- Inspecting {ZIP_FILENAME} Manifest ---")
    assert ZIP_PATH.exists(), f"{ZIP_PATH} does not exist"
    
    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        namelist = zf.namelist()
        infolist = zf.infolist()
        total_uncompressed = sum(info.file_size for info in infolist)
        total_compressed = sum(info.compress_size for info in infolist)
        
        print(f"Total entries inside ZIP: {len(namelist)}")
        print(f"Total uncompressed size: {total_uncompressed:,} bytes ({total_uncompressed / (1024*1024):.2f} MB)")
        print(f"Total compressed size:   {total_compressed:,} bytes ({total_compressed / (1024*1024):.2f} MB)")
        
        # Check for nested root folder duplicates
        for name in namelist:
            parts = name.split("/")
            assert parts[0] not in ["voiceops-2026", "team_name_voiceops"], f"Duplicate nested project directory found: {name}"
            
        # Check required deliverables are in namelist
        for req in REQUIRED_DELIVERABLES:
            found = any(name == req or name.startswith(req + "/") for name in namelist)
            assert found, f"Required deliverable '{req}' is MISSING in ZIP manifest"
            
        # Check explicit exclusions
        for name in namelist:
            if name == ".env":
                raise AssertionError(f"FATAL: Secret file '.env' found inside ZIP!")
            if name == ".git" or name.startswith(".git/"):
                raise AssertionError(f"FATAL: Git directory found inside ZIP: {name}")
            if name == ".venv" or name.startswith(".venv/"):
                raise AssertionError(f"FATAL: .venv directory found inside ZIP: {name}")
            if name == ".cache" or name.startswith(".cache/"):
                raise AssertionError(f"FATAL: .cache directory found inside ZIP: {name}")
            if name == ".pytest_cache" or name.startswith(".pytest_cache/"):
                raise AssertionError(f"FATAL: .pytest_cache directory found inside ZIP: {name}")
            if name == "outputs" or name.startswith("outputs/"):
                raise AssertionError(f"FATAL: Excluded outputs directory found inside ZIP: {name}")
            if "Capstone_Project_Dataset_FTTT_2026" in name:
                raise AssertionError(f"FATAL: Excluded dataset directory found inside ZIP: {name}")
            if "__pycache__" in name:
                raise AssertionError(f"FATAL: __pycache__ found inside ZIP: {name}")
                
        print("[OK] All required deliverables confirmed present in archive")
        print("[OK] All exclusion checks passed (0 secrets, 0 cache, 0 dataset, 0 venv, 0 unnecessary outputs)")
        print("[OK] Directory structure verified: cleanly anchored at project root without nested folder duplicates")
        
        print("\nComplete Archive Manifest:")
        for idx, info in enumerate(sorted(infolist, key=lambda x: x.filename), 1):
            ratio = (1.0 - info.compress_size / info.file_size) * 100 if info.file_size > 0 else 0
            print(f"  {idx:2d}. {info.filename:<48} | {info.file_size:>9,} B -> {info.compress_size:>9,} B ({ratio:4.1f}% saved)")


def main():
    print("=======================================================")
    print("   VoiceOps 2026 Capstone -- Final Packaging & Audit   ")
    print("=======================================================")
    
    # 1. Verify Deliverables
    verify_readme()
    verify_env_example()
    verify_report_pdf()
    verify_demo_mp4()
    verify_sample_outputs()
    
    # 2. Create Zip
    create_zip()
    
    # 3. Inspect Zip
    inspect_and_verify_zip()


if __name__ == "__main__":
    main()
