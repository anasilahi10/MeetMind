import pipeline
import db
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import re
import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_os.environ["OMP_NUM_THREADS"] = "1"


load_dotenv()
st.set_page_config(page_title="MeetMind", page_icon="🧠", layout="wide")
db.init_db()

st.markdown("""
<style>
.block-container {padding-top: 4.5rem; max-width: 1200px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border-radius: 12px; padding: 12px 16px;}
</style>
""", unsafe_allow_html=True)

for k, v in {"view": "new", "selected": None, "all_chat": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

LANGS = {"Auto-detect": None, "English": "en", "Hindi": "hi", "Urdu": "ur",
         "Spanish": "es", "French": "fr", "German": "de"}
SIZES = ["tiny", "base", "small", "medium"]
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def md_safe(text):
    """Stop '$' from turning text into math formulas."""
    return text.replace("$", "\\$")


def fmt_duration(sec):
    if not sec:
        return "unknown"
    m, s = divmod(int(sec), 60)
    return f"{m} min {s} s"


# ======================= Sidebar =======================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 MeetMind")
        if st.button("➕ New meeting", key="nav_new"):
            st.session_state.view, st.session_state.selected = "new", None
            st.rerun()
        if st.button("🔎 Ask all meetings", key="nav_all"):
            st.session_state.view = "all"
            st.rerun()

        st.divider()
        with st.expander("🤖 AI model (summary & chat)"):
            names = list(pipeline.PROVIDERS)
            saved_provider = db.get_setting("provider", names[0])
            provider = st.selectbox("Provider", names,
                                    index=names.index(
                                        saved_provider) if saved_provider in names else 0,
                                    key="provider_sel")
            db.set_setting("provider", provider)
            info = pipeline.PROVIDERS[provider]

            presets = info["models"]
            options = presets + ["Custom..."]
            saved_model = db.get_setting(
                f"model_{provider}", presets[0] if presets else "")
            default_choice = saved_model if saved_model in presets else "Custom..."
            choice = st.selectbox("Model", options, index=options.index(default_choice),
                                  key=f"mc_{provider}")
            if choice == "Custom...":
                model = st.text_input("Custom model ID",
                                      value="" if saved_model in presets else saved_model,
                                      key=f"mt_{provider}", placeholder="provider/model-name")
            else:
                model = choice
            if model.strip():
                db.set_setting(f"model_{provider}", model.strip())

            base_url = ""
            if provider in ("Ollama (local)", "Custom (OpenAI-compatible)"):
                base_url = st.text_input("Base URL", value=db.get_setting(f"url_{provider}", info["base_url"]),
                                         key=f"url_{provider}")
                db.set_setting(f"url_{provider}", base_url)

            api_key = ""
            if info["needs_key"]:
                api_key = st.text_input("API key", type="password", key=f"apikey_{provider}",
                                        help="Leave empty to use the key in your .env file. Kept only for this "
                                             "browser session, never saved.")

            cfg = pipeline.make_cfg(provider, model, api_key, base_url)
            if st.button("🔌 Test connection", key="test_conn"):
                with st.spinner("Testing..."):
                    ok, msg = pipeline.test_connection(cfg)
                (st.success if ok else st.error)(msg)
            st.caption(
                "Free models can make small mistakes. A stronger model gives better summaries.")

        st.divider()
        st.markdown("### Past meetings")
        search = st.text_input("Search", key="title_search", label_visibility="collapsed",
                               placeholder="Search meetings...")
        meetings = db.list_meetings()
        if not meetings:
            st.caption("No meetings saved yet.")
        for m in meetings:
            if search and search.lower() not in m["title"].lower():
                continue
            if st.button(f"{m['title']} · {m['created_at'][:10]}", key=f"open_{m['id']}"):
                st.session_state.view, st.session_state.selected = "meeting", m["id"]
                st.rerun()
    return cfg


# ======================= New meeting =======================

def new_meeting_view(cfg):
    st.title("New meeting")
    st.caption("Upload or record a meeting. MeetMind transcribes it, labels who is speaking, "
               "and writes the summary and action items.")

    tab_up, tab_rec = st.tabs(["📁 Upload a file", "🎙️ Record now"])
    with tab_up:
        uploaded = st.file_uploader("Audio or video file",
                                    type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"])
    recorded = None
    with tab_rec:
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Record a meeting (click the mic)")
        else:
            st.info(
                "Recording needs a newer Streamlit. Run: pip install -U streamlit")
    source = uploaded or recorded

    c1, c2 = st.columns([3, 1])
    title = c1.text_input("Meeting title", placeholder="e.g. Weekly team sync")
    num_speakers = c2.number_input(
        "Speakers (0 = auto)", min_value=0, max_value=20, value=0)

    with st.expander("⚙️ Accuracy settings"):
        saved_size = db.get_setting("whisper_size", "small")
        size = st.selectbox("Transcription model", SIZES,
                            index=SIZES.index(
                                saved_size) if saved_size in SIZES else 2,
                            help="Bigger = more accurate but slower. 'small' is a good balance on a laptop.")
        lang_names = list(LANGS)
        saved_lang = db.get_setting("language", "Auto-detect")
        lang = st.selectbox("Language", lang_names,
                            index=lang_names.index(
                                saved_lang) if saved_lang in lang_names else 0,
                            help="Choosing the language is more accurate than auto-detect.")
        vocab = st.text_area("Names & special terms (comma-separated)",
                             placeholder="e.g. GitLab, dogfooding, Priya, Rahul",
                             help="Helps with names and jargon that speech recognition often gets wrong.")
        enhance = st.checkbox("Clean up audio first (recommended)", value=True)
        db.set_setting("whisper_size", size)
        db.set_setting("language", lang)

    if st.button("🚀 Process meeting", type="primary", disabled=source is None):
        name = getattr(source, "name", None) or "recording.wav"
        run_pipeline(source, name, title.strip() or Path(name).stem, int(num_speakers),
                     size, LANGS[lang], vocab, enhance, cfg)


def run_pipeline(source, name, title, num_speakers, size, lang_code, vocab, enhance, cfg):
    meeting_id = None
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{Path(name).suffix or '.wav'}"
        src.write_bytes(source.getbuffer())
        wav = Path(tmp) / "audio.wav"
        try:
            with st.status("Processing your meeting...", expanded=True) as status:
                st.write("1/5 Preparing audio...")
                pipeline.to_wav(str(src), str(wav), enhance)

                st.write(
                    "2/5 Transcribing (slowest step; the first run downloads models)...")
                bar = st.progress(0.0)
                words, duration, _ = pipeline.transcribe(
                    str(wav), size, lang_code, vocab, bar.progress)

                st.write("3/5 Identifying speakers...")
                turns = pipeline.diarize(str(wav), num_speakers or None)
                transcript = pipeline.build_transcript(words, turns)
                if not transcript.strip():
                    raise RuntimeError(
                        "No speech was detected in this recording.")

                meeting_id = db.save_meeting(title, name, duration, transcript)

                st.write("4/5 Writing the summary...")
                try:
                    summary = pipeline.summarize(
                        transcript, cfg, context=f"{title}. {vocab}")
                    st.write("5/5 Extracting action items...")
                    items = pipeline.get_action_items(summary, cfg)
                    db.update_summary(meeting_id, summary, items)
                    status.update(label="Done!", state="complete")
                except Exception as e:
                    st.warning(
                        f"Transcript saved, but the summary failed: {e}")
                    status.update(
                        label="Saved (summary failed)", state="error")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
    st.session_state.view, st.session_state.selected = "meeting", meeting_id
    st.rerun()


# ======================= Saved meeting =======================

def meeting_view(meeting_id, cfg):
    m = db.get_meeting(meeting_id)
    if m is None:
        st.session_state.view, st.session_state.selected = "new", None
        st.rerun()
        return

        st.title(m["title"])
    st.caption(f"{m['created_at']}  ·  {m['filename']}")
    speakers = pipeline.list_speakers(m["transcript"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_duration(m["duration_sec"]))
    c2.metric("Speakers", len(speakers))
    c3.metric(
        "Words", f"{sum(pipeline.speaking_share(m['transcript']).values()):,}")
    c4.metric("Action items", len(m["action_items"]))

    tab_over, tab_tr, tab_ask, tab_manage = st.tabs(
        ["📝 Overview", "💬 Transcript", "🤖 Ask AI", "⚙️ Manage & export"])

    # ---------- Overview ----------
    with tab_over:
        if m["summary"].strip():
            st.markdown(md_safe(m["summary"]))
        else:
            st.warning("No summary yet. Click the button below to generate it.")

        if m["action_items"]:
            st.subheader("✅ Action items")
            df = pd.DataFrame(m["action_items"]).rename(
                columns={"owner": "Owner", "task": "Task", "deadline": "Deadline"})
            st.dataframe(df, hide_index=True)
            st.download_button("⬇️ Action items (.csv)", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{m['title']}_actions.csv", mime="text/csv", key="dl_csv")

        share = pipeline.speaking_share(m["transcript"])
        if share:
            st.subheader("🗣️ Who spoke how much")
            st.caption("Share of the conversation, measured in words spoken.")
            st.bar_chart(pd.Series(share))

        if st.button("🔄 Generate / regenerate summary", key="regen"):
            with st.spinner("Summarizing..."):
                try:
                    summary = pipeline.summarize(
                        m["transcript"], cfg, context=m["title"])
                    items = pipeline.get_action_items(summary, cfg)
                    db.update_summary(meeting_id, summary, items)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # ---------- Transcript ----------
    with tab_tr:
        query = st.text_input("Search in transcript", placeholder="Type a word or a speaker name...",
                              key=f"tsearch_{meeting_id}")
        lines = m["transcript"].splitlines()
        if query:
            lines = [l for l in lines if query.lower() in l.lower()]
            st.caption(f"{len(lines)} matching line(s)")
        rendered = []
        for line in lines:
            mt = re.match(r"^(\[[\d:]+\] [^:]+:) (.*)$", line)
            rendered.append(
                f"**{mt.group(1)}** {md_safe(mt.group(2))}" if mt else md_safe(line))
        with st.container(height=520, border=True):
            st.markdown("\n\n".join(rendered))
        st.download_button("⬇️ Transcript (.txt)", m["transcript"],
                           file_name=f"{m['title']}_transcript.txt", key="dl_tr")

    # ---------- Ask AI ----------
    with tab_ask:
        history = db.get_chat(meeting_id)
        for h in history:
            with st.chat_message(h["role"]):
                st.markdown(md_safe(h["content"]))

        suggestions = ["What are the action items for each person?",
                       "What decisions were made?",
                       "What did each speaker mainly say?"]
        cols = st.columns(len(suggestions))
        clicked = None
        for col, s in zip(cols, suggestions):
            if col.button(s, key=f"sg_{meeting_id}_{s}"):
                clicked = s
        question = st.chat_input("Ask about this meeting...") or clicked

        if question:
            with st.chat_message("user"):
                st.markdown(md_safe(question))
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = pipeline.answer_question(
                            question, m["transcript"], m["summary"], history, cfg)
                    except Exception as e:
                        answer = None
                        st.error(str(e))
                if answer:
                    st.markdown(md_safe(answer))
                    db.add_chat(meeting_id, "user", question)
                    db.add_chat(meeting_id, "assistant", answer)
        if history and st.button("🗑️ Clear chat", key="clear_chat"):
            db.clear_chat(meeting_id)
            st.rerun()

    # ---------- Manage & export ----------
    with tab_manage:
        st.subheader("Rename meeting")
        new_title = st.text_input(
            "Title", value=m["title"], key=f"title_{meeting_id}")
        if st.button("Save title") and new_title.strip():
            db.update_title(meeting_id, new_title.strip())
            st.rerun()

        st.subheader("Rename speakers")
        st.caption(
            "Replaces the names in the transcript, summary and action items.")
        with st.form(f"rename_{meeting_id}"):
            mapping = {s: st.text_input(
                s, value=s, key=f"rn_{meeting_id}_{s}") for s in speakers}
            if st.form_submit_button("Save names"):
                items = [dict(it, owner=pipeline.rename_speakers(it["owner"], mapping))
                         for it in m["action_items"]]
                db.update_text(meeting_id,
                               pipeline.rename_speakers(
                                   m["transcript"], mapping),
                               pipeline.rename_speakers(m["summary"], mapping), items)
                st.rerun()

        st.subheader("Export")
        full = (f"# {m['title']}\n\n{m['created_at']} · {m['filename']}\n\n"
                f"{m['summary']}\n\n---\n\n## Transcript\n\n{m['transcript']}\n")
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ Full meeting (.md)", full,
                           file_name=f"{m['title']}_full.md", key="dl_full")
        e2.download_button(
            "⬇️ Summary (.md)", m["summary"], file_name=f"{m['title']}_summary.md", key="dl_sum")
        try:
            e3.download_button("⬇️ Word report (.docx)", pipeline.build_docx(m),
                               file_name=f"{m['title']}.docx", mime=DOCX_MIME, key="dl_docx")
        except ImportError:
            e3.info("Run: pip install python-docx")

        st.subheader("Delete")
        if st.checkbox("I want to permanently delete this meeting"):
            if st.button("Delete meeting", type="primary"):
                db.delete_meeting(meeting_id)
                st.session_state.view, st.session_state.selected = "new", None
                st.rerun()


# ======================= Ask all meetings =======================

def all_meetings_view(cfg):
    st.title("🔎 Ask all your meetings")
    st.caption("Ask across every saved meeting, e.g. “What tasks does Rahul have?” or "
               "“Where did we discuss the budget?”")
    meetings = db.all_meetings()
    if not meetings:
        st.info("No meetings saved yet.")
        return
    for h in st.session_state.all_chat:
        with st.chat_message(h["role"]):
            st.markdown(md_safe(h["content"]))
    question = st.chat_input("Ask across all meetings...")
    if question:
        with st.chat_message("user"):
            st.markdown(md_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("Searching your meetings..."):
                try:
                    answer = pipeline.answer_across(
                        question, meetings, st.session_state.all_chat, cfg)
                except Exception as e:
                    answer = None
                    st.error(str(e))
            if answer:
                st.markdown(md_safe(answer))
                st.session_state.all_chat += [{"role": "user", "content": question},
                                              {"role": "assistant", "content": answer}]
    if st.session_state.all_chat and st.button("🗑️ Clear conversation"):
        st.session_state.all_chat = []
        st.rerun()


# ======================= Router =======================

cfg = render_sidebar()
view = st.session_state.view
if view == "meeting" and st.session_state.selected:
    meeting_view(st.session_state.selected, cfg)
elif view == "all":
    all_meetings_view(cfg)
else:
    new_meeting_view(cfg)
