# MeetMind — Local AI Meeting Intelligence

MeetMind is a local-first meeting assistant. It transcribes an audio recording with `whisper.cpp`, lets the user review and correct the transcript, analyzes it with a local Ollama model, stores meetings and action items in SQLite, and provides meeting chat, task tracking, and analytics through Streamlit.

## Features

- Local audio transcription using `whisper.cpp`
- Editable transcript review before analysis
- AI-generated summary, objectives, discussion points, decisions, risks, unresolved questions, facts, and action items
- Transcript-quote evidence checks for extracted facts and action items
- Human review gate before saving an analysis
- SQLite persistence for meetings, tasks, and chat history
- Meeting search, report/transcript downloads, and deletion
- Chat with saved meeting transcripts
- Action-item status management and analytics

## Architecture

```text
Audio file
   |
   v
whisper.cpp (local speech-to-text)
   |
   v
Transcript review/editing (Streamlit)
   |
   v
Ollama local LLM via LangChain
   |
   v
JSON parsing + evidence validation (Python)
   |
   +----> Human review
   |          |
   v          v
SQLite <--- Save meeting and verified action items
   |
   +----> Meeting history
   +----> Chat with meetings
   +----> Task manager
   +----> Analytics
```

## Requirements

- macOS, Linux, or Windows (the transcription executable must be built for the host OS)
- Python 3.13 (matches the current `pyproject.toml`)
- [Ollama](https://ollama.com/) installed and running
- A local Ollama model installed
- `whisper.cpp` built locally, plus a Whisper GGML model

### Hardware note

The development machine used for this project is a 2017 Intel MacBook Air with a dual-core Intel i5 and 8 GB RAM. Keep the model small. `qwen3:4b` is the current baseline; do not assume a larger model will run acceptably. Benchmark alternatives on the same set of meeting transcripts before switching. Close other memory-heavy applications during inference.

## Setup

1. Install Python 3.13, Git, CMake, and Ollama.
2. Start Ollama and pull the current baseline model:

   ```bash
   ollama pull qwen3:4b
   ollama list
   ```

3. Create and activate a virtual environment from the project root:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure the local model (optional):

   ```bash
   cp .env.example .env
   ```

   `MEETMIND_MODEL` controls the analysis model. `MEETMIND_CHAT_MODEL` controls the chat model. Both default to `qwen3:4b`. Only change these values after pulling the corresponding Ollama model.

5. Build `whisper.cpp` for your machine and download a compatible GGML model. The transcription service currently expects:

   ```text
   whisper.cpp/build/bin/whisper-cli
   whisper.cpp/models/ggml-base.en.bin
   ```

   The `whisper.cpp` repository's build instructions are maintained upstream: <https://github.com/ggml-org/whisper.cpp>.

6. Start MeetMind:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Open **New Meeting**.
2. Enter a meeting title and upload WAV, MP3, FLAC, or OGG audio (up to 100 MB).
3. Wait for transcription.
4. Review and correct names, dates, numbers, and unclear words in the editable transcript.
5. Select **Analyze Corrected Transcript**.
6. Review the summary, action items, evidence quotes, and warnings.
7. Confirm human review and save the meeting.
8. Use Meeting History, Chat with Meetings, Task Manager, and Analytics.

## Model configuration

The model is configurable through environment variables:

```dotenv
MEETMIND_MODEL=qwen3:4b
MEETMIND_CHAT_MODEL=qwen3:4b
```

The model is deliberately not hard-coded to a larger checkpoint. On an 8 GB Intel laptop, memory pressure and inference speed matter. Compare candidate models using the same fixed test set and record extraction precision, recall, owner accuracy, deadline accuracy, JSON validity, and latency.

## Testing

Run offline parsing and evidence-validation tests:

```bash
python -m unittest discover -s tests -v
```

Compile-check the Python source:

```bash
python -m compileall -q app.py services utils tests
```

These tests do not prove that Ollama, Whisper, the audio pipeline, or the Streamlit UI work end-to-end. Those require local runtime tests with the model and audio dependencies installed.

## Data and privacy

- Meeting transcripts, analysis, tasks, and chat are stored in `data/meetmind.db`.
- Audio is written to a temporary directory for transcription and removed when the transcription call completes.
- `.env`, local database files, generated outputs, virtual environments, and caches should not be committed to Git.
- Review all meeting content before sharing reports or screenshots. Meeting transcripts can contain confidential information.

## Known limitations

- Transcription is currently configured for English (`-l en`) and uses the `base.en` model.
- Speaker diarization is not implemented; owner attribution depends on what the transcript explicitly says.
- Evidence matching confirms that a quote occurs in the transcript; it does not prove the model's interpretation of that quote is correct.
- Human review is required before saving AI analysis.
- Long transcripts above the configured analysis/chat character limits are rejected rather than silently truncated.

## Project structure

```text
MeetMind/
├── app.py
├── services/
│   ├── transcription.py
│   ├── meeting_ai.py
│   └── database.py
├── utils/
├── tests/
├── data/                 # local SQLite data; do not commit
├── outputs/              # generated outputs; do not commit
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```
