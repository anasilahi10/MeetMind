from dotenv import load_dotenv
import requests
from io import BytesIO
from functools import lru_cache
import time
import subprocess
import re
import os
import json
import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_os.environ["OMP_NUM_THREADS"] = "1"


load_dotenv()

# ======================= Providers =======================

PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": ["openrouter/free"],
        "needs_key": True,
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b"],
        "needs_key": True,
    },
    "Ollama (local)": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
        "models": ["llama3.2"],
        "needs_key": False,
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "env_key": "CUSTOM_API_KEY",
        "models": [],
        "needs_key": True,
    },
}


def make_cfg(provider, model, api_key="", base_url=""):
    p = PROVIDERS[provider]
    key = (api_key or "").strip() or (
        os.getenv(p["env_key"]) if p["env_key"] else "") or ""
    url = ((base_url or "").strip() or p["base_url"]).rstrip("/")
    return {"provider": provider, "model": (model or "").strip(), "api_key": key.strip(), "base_url": url}


# ======================= Audio =======================

def to_wav(src, dst, enhance=True):
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000"]
    if enhance:
        cmd += ["-af", "highpass=f=80,dynaudnorm"]
    cmd.append(dst)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")


@lru_cache(maxsize=2)
def load_whisper(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def load_diarizer():
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing in your .env file.")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipe is None:
        raise RuntimeError(
            "Could not load the speaker model. Accept the terms on huggingface.co/pyannote/"
            "speaker-diarization-3.1 and huggingface.co/pyannote/segmentation-3.0 and check HF_TOKEN."
        )
    return pipe


# ======================= Transcription & speakers =======================

def transcribe(wav_path, model_size="small", language=None, vocab="", progress_cb=None):
    """Returns (words, duration, detected_language). Words carry timestamps."""
    model = load_whisper(model_size)
    segments, info = model.transcribe(
        wav_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
        hotwords=(vocab.strip() or None),
    )
    words = []
    for seg in segments:
        unclear = seg.avg_logprob < -1.0
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end,
                         "text": w.word, "unclear": unclear})
        if progress_cb and info.duration:
            progress_cb(min(seg.end / info.duration, 1.0))
    return words, info.duration, info.language


def diarize(wav_path, num_speakers=None):
    pipe = load_diarizer()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipe(wav_path, **kwargs)
    return [(t.start, t.end, spk) for t, _, spk in result.itertracks(yield_label=True)]


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _speaker_at(t, turns):
    best, best_d = None, 1e12
    for start, end, spk in turns:
        if start <= t <= end:
            return spk
        d = min(abs(t - start), abs(t - end))
        if d < best_d:
            best, best_d = spk, d
    return best or "SPEAKER_00"


def build_transcript(words, turns, gap=4.0):
    """Assign every WORD to a speaker, smooth tiny flips, then group into turns."""
    if not words:
        return ""
    labeled = [dict(w, spk=_speaker_at((w["start"] + w["end"]) / 2, turns))
               for w in words]

    # smooth: a 1-2 word blip between two runs of the same speaker is probably a mistake
    n, i = len(labeled), 0
    while i < n:
        j = i
        while j < n and labeled[j]["spk"] == labeled[i]["spk"]:
            j += 1
        if (j - i) <= 2 and i > 0 and j < n and labeled[i - 1]["spk"] == labeled[j]["spk"]:
            for k in range(i, j):
                labeled[k]["spk"] = labeled[i - 1]["spk"]
        i = j

    grouped = []
    for w in labeled:
        last = grouped[-1] if grouped else None
        if last and last["spk"] == w["spk"] and w["start"] - last["end"] < gap:
            last["words"].append(w["text"])
            last["end"] = w["end"]
            last["unclear"] += int(w["unclear"])
            last["n"] += 1
        else:
            grouped.append({"spk": w["spk"], "start": w["start"], "end": w["end"],
                            "words": [w["text"]], "unclear": int(w["unclear"]), "n": 1})

    lines = []
    for g in grouped:
        text = re.sub(r"\s+", " ", "".join(g["words"])).strip()
        if g["unclear"] / g["n"] >= 0.5:
            text += " [unclear]"
        lines.append(f"[{fmt_time(g['start'])}] {g['spk']}: {text}")
    return "\n".join(lines)


def list_speakers(transcript):
    seen = []
    for name in re.findall(r"^\[[\d:]+\] ([^:\n]+): ", transcript, flags=re.M):
        if name not in seen:
            seen.append(name)
    return seen


def speaking_share(transcript):
    counts = {}
    for m in re.finditer(r"^\[[\d:]+\] ([^:\n]+): (.*)$", transcript, flags=re.M):
        counts[m.group(1)] = counts.get(
            m.group(1), 0) + len(m.group(2).split())
    return counts


def rename_speakers(text, mapping):
    for old, new in mapping.items():
        new = new.replace(":", "").strip()
        if new and new != old:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


# ======================= LLM =======================

def llm(messages, cfg, temperature=0.2, max_tokens=None):
    if not cfg.get("model"):
        raise RuntimeError(
            "No model selected. Choose one in the sidebar (AI model).")
    if not cfg.get("base_url"):
        raise RuntimeError("No base URL set for this provider.")
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    elif PROVIDERS[cfg["provider"]]["needs_key"]:
        raise RuntimeError(
            f"No API key for {cfg['provider']}. Paste one in the sidebar or add it to .env.")

    payload = {"model": cfg["model"],
               "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    for attempt in range(4):
        try:
            r = requests.post(f"{cfg['base_url']}/chat/completions",
                              headers=headers, json=payload, timeout=180)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach {cfg['base_url']}: {e}")
        if r.status_code in (429, 502, 503):
            time.sleep(6 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(
                f"{cfg['provider']} error {r.status_code}: {r.text[:300]}")
        try:
            content = r.json()["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise RuntimeError(f"Unexpected response: {r.text[:300]}")
        content = re.sub(r"<think>.*?</think>", "",
                         content, flags=re.S).strip()
        if not content:
            raise RuntimeError(
                "The model returned an empty answer. Try again or pick another model.")
        return content
    raise RuntimeError(
        "This model is rate-limiting you. Wait a minute or switch model.")


def test_connection(cfg):
    try:
        out = llm(
            [{"role": "user", "content": "Reply with the single word OK."}], cfg, temperature=0)
        return True, f"Connected to {cfg['provider']} ✓  (model replied: {out[:40]})"
    except Exception as e:
        return False, str(e)


def chunk_text(text, max_chars=9000):
    chunks, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) + 1 > max_chars:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


SUMMARY_FORMAT = """Write in Markdown with exactly these sections:
## Summary
(3-6 short lines: the main topics and what each speaker argued. Do not retell the transcript line by line.)
## Key Decisions
(only things the group clearly agreed on; otherwise "None mentioned")
## Action Items
(only clear commitments such as "I will..." or tasks explicitly assigned to someone. Format: **Person** - task - deadline if mentioned. Suggestions and ideas are NOT action items. If none: "None mentioned")
## Open Questions
(only questions actually asked in the transcript or topics left unresolved; otherwise "None mentioned")

Rules: Keep speaker labels exactly as written. The transcript comes from speech recognition and may contain errors; passages marked [unclear] are unreliable, so say they are unclear instead of guessing. Never invent facts, names or tasks."""


def _summarize_raw(transcript, cfg, context=""):
    system = {"role": "system",
              "content": "You are an expert, careful meeting assistant."}
    ctx = f"\nBackground (use these spellings for names/terms): {context.strip()}\n" if context.strip(
    ) else ""
    chunks = chunk_text(transcript)

    if len(chunks) == 1:
        return llm([system, {"role": "user",
                             "content": f"{SUMMARY_FORMAT}\n{ctx}\nTranscript:\n{transcript}"}], cfg)

    notes = []
    for i, chunk in enumerate(chunks, 1):
        notes.append(llm([system, {"role": "user", "content":
                                   "Make detailed notes on this part of a meeting. Keep speaker labels, decisions, and who "
                                   f"must do what. {ctx}\nPart {i}/{len(chunks)}:\n{chunk}"}], cfg))
    return llm([system, {"role": "user", "content":
                         f"{SUMMARY_FORMAT}\n{ctx}\nThese are notes from all parts of the meeting:\n" + "\n\n".join(notes)}], cfg)


# ======================= Action items =======================

def _section(md, title):
    m = re.search(
        rf"##\s*{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)", md, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def _norm(items):
    out = []
    for it in items:
        if isinstance(it, dict):
            task = str(it.get("task", "")).strip()
            if not task:
                continue
            deadline = str(it.get("deadline", "")).strip()
            if "no deadline" in deadline.lower() or deadline.lower() in ("none", "n/a"):
                deadline = ""
            out.append({"owner": str(it.get("owner", "")).strip() or "Unassigned",
                        "task": task, "deadline": deadline})
    return out


def _parse_bullets(section):
    items = []
    for line in section.splitlines():
        line = re.sub(r"^\s*[-*•]\s*", "", line).replace("**", "").strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"\s[-–—]\s", line, maxsplit=2)]
        if len(parts) >= 2:
            items.append({"owner": parts[0], "task": parts[1],
                          "deadline": parts[2] if len(parts) > 2 else ""})
        else:
            items.append({"owner": "Unassigned", "task": line, "deadline": ""})
    return _norm(items)


def get_action_items(summary, cfg):
    section = _section(summary, "Action Items")
    if not section or "none mentioned" in section.lower():
        return []
    prompt = ('Convert these meeting action items into a JSON array. Each element must be an object with '
              'keys "owner", "task", "deadline" (use "" if none). Output ONLY the JSON array.\n\n' + section)
    try:
        raw = llm([{"role": "user", "content": prompt}], cfg, temperature=0)
        data = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        items = _norm(data)
        if items:
            return items
    except Exception:
        pass
    return _parse_bullets(section)


# ======================= Q&A =======================

STOP = set("""the a an and or of to in on at for with is are was were be been it this that these those i you
he she we they what who whom when where why how did do does can could should would will about from by as not
no yes so but if then than there here have has had my your our their his her its into over under also just
tell please meeting said say""".split())


def _tokens(s):
    out = set()
    for w in re.findall(r"\w+", s.lower()):
        if len(w) > 2 and w not in STOP:
            out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _pick(chunks_text, question, max_chars):
    q = _tokens(question)
    ranked = sorted(range(len(chunks_text)),
                    key=lambda i: len(q & _tokens(chunks_text[i])), reverse=True)
    picked, total = [], 0
    for i in ranked:
        if total + len(chunks_text[i]) > max_chars:
            break
        picked.append(i)
        total += len(chunks_text[i])
    return sorted(picked)


def select_context(question, transcript, max_chars=18000):
    if len(transcript) <= max_chars:
        return transcript
    chunks = chunk_text(transcript, 2500)
    return "\n...\n".join(chunks[i] for i in _pick(chunks, question, max_chars))


def answer_question(question, transcript, summary, history, cfg):
    system = {"role": "system", "content": (
        "You answer questions about ONE meeting using only the material below. If the answer is not "
        "in the meeting, say so. Passages marked [unclear] are unreliable. Mention speakers and "
        "timestamps like [05:12] when helpful. Be concise.\n\n"
        f"MEETING SUMMARY:\n{summary}\n\nTRANSCRIPT (may be partial):\n{select_context(question, transcript)}"
    )}
    messages = [system] + [{"role": h["role"],
                            "content": h["content"]} for h in history[-6:]]
    messages.append({"role": "user", "content": question})
    return llm(messages, cfg)


def answer_across(question, meetings, history, cfg, max_chars=16000):
    chunks, labels = [], []
    for m in meetings:
        for c in chunk_text(m["transcript"], 2500):
            chunks.append(c)
            labels.append(f"{m['title']} ({m['created_at'][:10]})")
    if not chunks:
        raise RuntimeError("No saved meetings yet.")
    picked = _pick(chunks, question, max_chars)
    context = "\n\n".join(f"### {labels[i]}\n{chunks[i]}" for i in picked)
    summaries = "\n\n".join(f"### {m['title']} ({m['created_at'][:10]})\n{m['summary'][:1200]}"
                            for m in meetings[:10] if m["summary"].strip())
    system = {"role": "system", "content": (
        "You answer questions across SEVERAL saved meetings using only the material below. Always say "
        "which meeting an answer comes from. If it is not in the material, say so. Be concise.\n\n"
        f"SUMMARIES:\n{summaries}\n\nTRANSCRIPT EXCERPTS:\n{context}"
    )}
    messages = [system] + [{"role": h["role"],
                            "content": h["content"]} for h in history[-6:]]
    messages.append({"role": "user", "content": question})
    return llm(messages, cfg)


# ======================= Word export =======================

def _add_markdown(doc, text):
    for raw in text.splitlines():
        if not raw.strip():
            continue
        clean = raw.replace("**", "").rstrip()
        if raw.startswith("## "):
            doc.add_heading(clean[3:], level=2)
        elif raw.startswith("# "):
            doc.add_heading(clean[2:], level=1)
        elif re.match(r"^\s*[-*] ", raw):
            doc.add_paragraph(
                re.sub(r"^\s*[-*] ", "", clean), style="List Bullet")
        else:
            doc.add_paragraph(clean)


def build_docx(m):
    from docx import Document
    doc = Document()
    doc.add_heading(m["title"], 0)
    doc.add_paragraph(f"{m['created_at']}  ·  {m['filename']}")
    if m["summary"].strip():
        _add_markdown(doc, m["summary"])
    if m["action_items"]:
        doc.add_heading("Action items table", level=2)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        for cell, name in zip(table.rows[0].cells, ["Owner", "Task", "Deadline"]):
            cell.text = name
        for it in m["action_items"]:
            row = table.add_row().cells
            row[0].text, row[1].text, row[2].text = it["owner"], it["task"], it["deadline"]
    doc.add_heading("Transcript", level=2)
    for line in m["transcript"].splitlines():
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _fix_action_section(summary):
    """Put each action item on its own bullet line, if the model squashed them together."""
    m = re.search(r"(##\s*Action Items\s*\n)(.*?)(?=\n##\s|\Z)",
                  summary, flags=re.S | re.I)
    if not m:
        return summary
    body = m.group(2)
    if re.search(r"^\s*[-*] ", body, flags=re.M):
        return summary  # already bulleted
    fixed = re.sub(r"\s*(\*\*[^*\n]+\*\*\s+[-–—])", r"\n- \1", body).strip()
    return summary[: m.start(2)] + fixed + summary[m.end(2):]


def summarize(transcript, cfg, context=""):
    return _fix_action_section(_summarize_raw(transcript, cfg, context))
