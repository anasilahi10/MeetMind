# MeetMind — AI-Powered Meeting Intelligence

MeetMind is an AI-powered meeting assistant built with Python and Streamlit. It converts meeting recordings into structured meeting intelligence, including summaries, decisions, risks, deadlines, and action items.

MeetMind uses **whisper.cpp** for local audio transcription and **OpenRouter** for cloud-based AI analysis and meeting chat. It validates extracted information against transcript evidence and allows users to review results before saving them.

## Features

- Local audio transcription using `whisper.cpp`
- Editable transcript review before AI analysis
- AI-generated executive summaries
- Automatic extraction of meeting objectives and discussion points
- Decision, risk, and unresolved-question identification
- Action-item extraction with responsible persons and deadlines
- Transcript evidence validation for extracted facts and action items
- Human review before saving meeting analysis
- SQLite database for meetings, tasks, and chat history
- Chat with saved meeting transcripts
- Meeting history, search, and report/transcript downloads
- Task status management and analytics

## Architecture

```text
Audio Recording
      |
      v
whisper.cpp
(Local Speech-to-Text)
      |
      v
Editable Transcript
(Streamlit)
      |
      v
OpenRouter API
(LangChain + ChatOpenAI)
      |
      v
Structured AI Analysis
(JSON Output)
      |
      v
Python Validation
(Evidence + Action Items)
      |
      v
Human Review
      |
      v
SQLite Database
      |
      +----> Meeting History
      +----> Meeting Chat
      +----> Task Manager
      +----> Analytics
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.13 |
| Frontend | Streamlit |
| AI model provider | OpenRouter |
| AI integration | LangChain |
| Model interface | langchain-openai |
| Speech-to-text | whisper.cpp |
| Database | SQLite |
| Database toolkit | SQLAlchemy |
| Data processing | Pandas |
| Validation | Python |

## Requirements

- Python 3.13
- Git
- `uv` package manager
- CMake
- OpenRouter API key
- `whisper.cpp` built locally
- Compatible Whisper GGML model

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/anasilahi10/MeetMind.git
cd MeetMind
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create your local environment file:

```bash
cp .env.example .env
```

Open `.env` and configure:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
MEETMIND_MODEL=openrouter/free
MEETMIND_CHAT_MODEL=openrouter/free
```

Replace the placeholder with your actual OpenRouter API key.

Never commit your `.env` file or expose your API key.

### 4. Configure Whisper

Build `whisper.cpp` and download a compatible GGML model.

MeetMind currently expects:

```text
whisper.cpp/
├── build/
│   └── bin/
│       └── whisper-cli
└── models/
    └── ggml-base.en.bin
```

Refer to the official repository for build instructions:

https://github.com/ggml-org/whisper.cpp

### 5. Run MeetMind

```bash
uv run streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

## How to Use

1. Open the **New Meeting** section.
2. Enter a meeting title and upload an audio recording.
3. Wait for Whisper to generate the transcript.
4. Review and correct names, dates, numbers, and unclear phrases.
5. Run AI analysis on the transcript.
6. Review the executive summary, objectives, decisions, risks, and action items.
7. Check supporting transcript evidence and validation warnings.
8. Review and save the meeting.
9. Access saved meetings, chat, tasks, and analytics.

## Model Configuration

MeetMind uses OpenRouter for AI analysis and meeting chat.

The model is configurable through environment variables:

```dotenv
MEETMIND_MODEL=openrouter/free
MEETMIND_CHAT_MODEL=openrouter/free
```

- `MEETMIND_MODEL` controls meeting analysis.
- `MEETMIND_CHAT_MODEL` controls chat with saved meetings.
- If the chat model is not specified, it defaults to the analysis model.

The default model is `openrouter/free`. You can configure other models supported by your OpenRouter account.

## Validation and Reliability

MeetMind includes a Python-based validation layer to improve the reliability of AI-generated meeting intelligence.

It checks:

- Whether extracted evidence quotes appear in the transcript.
- Whether action-item owners are explicitly supported by the evidence.
- Whether deadlines are supported by the corresponding evidence.
- Whether action items are duplicated or too vague.
- Whether extracted facts have supporting transcript evidence.

Unverified information is flagged for review rather than silently treated as confirmed.

**Important:** Evidence matching checks whether a quotation exists in the transcript. It does not independently prove that the AI interpreted the quotation correctly.

## Testing

Run the project's unit tests:

```bash
uv run python -m unittest discover -s tests -v
```

Compile-check the application:

```bash
uv run python -m compileall -q app.py services utils tests
```

These checks do not replace end-to-end testing with actual audio recordings and OpenRouter requests.

## Data and Privacy

- Meeting records, transcripts, analysis, tasks, and chat history are stored in a local SQLite database.
- Audio files are processed using local `whisper.cpp` and written to a temporary directory during transcription.
- The AI analysis and meeting-chat features send transcript content to OpenRouter.
- An internet connection and a valid OpenRouter API key are required for AI analysis and chat.
- API data handling is subject to OpenRouter's applicable policies.
- Never upload confidential meeting recordings without appropriate authorization.
- Local environment files, databases, generated outputs, and virtual environments should not be committed to Git.

## Known Limitations

- Transcription is currently configured for English using the `base.en` Whisper model.
- Speaker diarization is not implemented.
- Owner attribution depends on explicit names or assignments in the transcript.
- First-person statements such as "I will complete this" may have an unknown owner if the speaker's identity is not established.
- Transcription errors can affect downstream AI analysis.
- Evidence matching confirms quotation presence, not semantic correctness.
- Human review is recommended before saving meeting analysis.
- Long transcripts exceeding configured character limits are rejected rather than silently truncated.

## Project Structure

```text
MeetMind/
├── app.py
├── services/
│   ├── transcription.py
│   ├── meeting_ai.py
│   └── database.py
├── utils/
├── tests/
├── data/                 # Local SQLite data
├── outputs/              # Generated outputs
├── whisper.cpp/          # Local transcription engine
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Future Improvements

- Speaker diarization and improved speaker identification
- Better action-item owner attribution
- Improved transcription accuracy
- Support for additional languages
- More robust meeting analytics

---

**MeetMind — Turning conversations into actionable meeting intelligence.**