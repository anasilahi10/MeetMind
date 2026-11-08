import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_BINARY = PROJECT_ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = PROJECT_ROOT / "whisper.cpp" / "models" / "ggml-base.en.bin"

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}
TRANSCRIPTION_TIMEOUT_SECONDS = 1800


def _get_audio_bytes(audio_file):
    """Read an uploaded audio file safely into bytes."""
    try:
        audio_bytes = audio_file.getvalue()
    except AttributeError:
        try:
            audio_bytes = audio_file.read()
        except Exception as error:
            raise ValueError(
                "The uploaded audio could not be read. Please upload it again."
            ) from error

    if not audio_bytes:
        raise ValueError(
            "The uploaded audio file is empty. Please choose a valid recording."
        )
    return audio_bytes


def _contains_spoken_content(transcript):
    """
    Reject output made only of non-speech annotations, e.g. '(breathing heavily)'.
    This is a lightweight transcript check, not full voice activity detection.
    """
    without_annotations = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", transcript)
    return bool(re.search(r"[A-Za-z0-9]", without_annotations))


def transcribe_audio(audio_file):
    """Transcribe an uploaded recording with the local whisper.cpp executable."""
    if not WHISPER_BINARY.is_file():
        raise FileNotFoundError(
            f"Whisper executable was not found at: {WHISPER_BINARY}. "
            "Check that whisper.cpp has been built successfully."
        )

    if not WHISPER_MODEL.is_file():
        raise FileNotFoundError(
            f"Whisper model was not found at: {WHISPER_MODEL}. "
            "Check that ggml-base.en.bin is present in the models folder."
        )

    if WHISPER_MODEL.stat().st_size == 0:
        raise RuntimeError(
            "The Whisper model file is empty or incomplete. Download it again."
        )

    filename = getattr(audio_file, "name", "")
    if not filename:
        raise ValueError(
            "The uploaded file has no valid filename. Please upload it again."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported audio format: {suffix or 'unknown'}. "
            f"Supported formats are: {supported}."
        )

    audio_bytes = _get_audio_bytes(audio_file)

    with tempfile.TemporaryDirectory(prefix="meetmind_") as temp_dir:
        audio_path = Path(temp_dir) / f"meeting{suffix}"
        audio_path.write_bytes(audio_bytes)

        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise RuntimeError(
                "Could not prepare the uploaded recording for transcription."
            )

        command = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", str(audio_path),
            "-nt",
            "-l", "en",
            "-t", "2"
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
                check=False
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "Transcription took longer than 30 minutes and was stopped. "
                "Try a shorter recording."
            ) from error
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "The Whisper executable could not be started. Check its location."
            ) from error
        except PermissionError as error:
            raise PermissionError(
                "Permission denied while running Whisper. "
                "Check that whisper-cli has execute permission."
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Operating-system error while starting Whisper: {error}"
            ) from error

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            if not details:
                details = "Whisper returned no additional error details."
            raise RuntimeError(
                f"Whisper transcription failed (exit code {result.returncode}). "
                f"Details: {details[-1500:]}"
            )

        transcript = (result.stdout or "").strip()
        if not transcript:
            raise RuntimeError(
                "Whisper did not detect any speech in the recording."
            )

        if not _contains_spoken_content(transcript):
            raise RuntimeError(
                "No spoken words were detected. The recording may contain "
                "silence, breathing, music, or other non-speech sounds. "
                "Please upload a recording with people speaking."
            )

        return transcript
