import json
from pathlib import Path

import pandas as pd
import streamlit as st

from services.transcription import transcribe_audio
from services.meeting_ai import analyze_meeting, chat_with_meeting
from services.database import (
    save_meeting_with_tasks,
    get_meetings,
    delete_meeting,
    get_tasks,
    update_task_status,
    get_chat_history,
    save_chat_exchange,
    clear_chat_history
)


# --------------------------------------------------
# APP CONFIGURATION AND SAFETY LIMITS
# --------------------------------------------------

st.set_page_config(
    page_title="MeetMind AI",
    page_icon="🎙️",
    layout="wide"
)

MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}

st.title("🎙️ MeetMind")
st.caption("AI-powered meeting intelligence and productivity")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "transcript": None,
    "analysis": None,
    "analysis_error": None,
    "processed_title": None,
    "meeting_saved": False,
    "saved_meeting_id": None,
    "chat_loaded_meeting_id": None,
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def safe_get_meetings():
    """Load meetings without allowing a database error to crash the whole app."""
    try:
        return get_meetings()
    except Exception as error:
        st.error(f"Could not load meetings from the database: {error}")
        return []


def safe_get_tasks():
    """Load tasks without allowing a database error to crash the whole app."""
    try:
        return get_tasks()
    except Exception as error:
        st.error(f"Could not load tasks from the database: {error}")
        return []


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎙️ New Meeting",
    "📚 Meeting History",
    "💬 Chat with Meetings",
    "📌 Task Manager",
    "📊 Analytics"
])


# ==================================================
# TAB 1: NEW MEETING
# ==================================================

with tab1:
    st.header("Create New Meeting")

    title = st.text_input(
        "Meeting Title",
        placeholder="Enter meeting title...",
        max_chars=255
    )

    audio = st.file_uploader(
        "Upload Meeting Recording",
        type=sorted(ALLOWED_AUDIO_EXTENSIONS),
        help="Supported formats: WAV, MP3, FLAC and OGG. Maximum size: 100 MB."
    )

    if st.button("Transcribe Audio", type="primary"):
        clean_title = (title or "").strip()

        if not clean_title:
            st.warning("Please enter a meeting title.")
        elif audio is None:
            st.warning("Please upload a meeting recording.")
        elif audio.size == 0:
            st.error("The uploaded file is empty.")
        elif audio.size > MAX_AUDIO_BYTES:
            st.error(
                "This recording exceeds the 100 MB upload limit. "
                "Please use a smaller audio file."
            )
        elif Path(audio.name).suffix.lower().lstrip(".") not in ALLOWED_AUDIO_EXTENSIONS:
            st.error("Unsupported audio format.")
        else:
            st.session_state.transcript = None
            st.session_state.editable_transcript = ""
            st.session_state.analysis = None
            st.session_state.analysis_error = None
            st.session_state.processed_title = None
            st.session_state.meeting_saved = False
            st.session_state.saved_meeting_id = None
            st.session_state.pop("confirm_analysis_review", None)

            try:
                with st.spinner("Transcribing audio with Whisper..."):
                    transcript = transcribe_audio(audio)

                if not isinstance(transcript, str) or not transcript.strip():
                    st.error(
                        "No spoken words were detected. The recording may "
                        "contain silence, music, or non-speech sounds."
                    )
                else:
                    st.session_state.transcript = transcript.strip()
                    st.session_state.editable_transcript = transcript.strip()
                    st.session_state.processed_title = clean_title
                    st.success("Transcription completed successfully.")

                    # Automatically analyse every successful transcription.
                    # Keep transcription success separate from analysis failure
                    # so users never have to upload the audio again if Ollama fails.
                    try:
                        with st.spinner("MeetMind is analysing the meeting with your local AI model..."):
                            st.session_state.analysis = analyze_meeting(transcript.strip())
                        st.session_state.analysis_error = None
                        st.success("Meeting analysis completed. Review the results below.")
                    except Exception as analysis_error:
                        st.session_state.analysis = None
                        st.session_state.analysis_error = str(analysis_error)
                        st.warning(
                            "Transcription succeeded, but automatic AI analysis failed. "
                            "Your transcript is preserved in this session. Check that Ollama "
                            "is running, then use the analysis button below to retry."
                        )
            except Exception as error:
                st.error(f"Transcription failed: {error}")

    if st.session_state.transcript:
        st.divider()
        st.subheader("Review and Correct Transcript")
        st.caption(
            "MeetMind analyses the transcript automatically. You can correct names, "
            "dates, numbers or unclear phrases below and rerun the analysis if needed."
        )

        st.text_area(
            "Editable transcript",
            key="editable_transcript",
            height=240,
            help="Correct transcription mistakes here before analysis."
        )

        corrected_transcript = (st.session_state.editable_transcript or "").strip()
        transcript_changed = corrected_transcript != (st.session_state.transcript or "").strip()
        if transcript_changed and st.session_state.analysis:
            st.info("You have edited the transcript. Re-analyze it before saving so the report matches the corrected text.")

        col_transcript, col_analyze = st.columns(2)
        with col_transcript:
            st.download_button(
                "Download Corrected Transcript",
                corrected_transcript,
                file_name="meeting_transcript_corrected.txt",
                mime="text/plain",
                disabled=not corrected_transcript,
                key="corrected_transcript_download"
            )

        with col_analyze:
            analyze_clicked = st.button(
                "Re-analyze Edited Transcript" if st.session_state.analysis else "Run AI Analysis",
                type="primary",
                disabled=not corrected_transcript,
                key="analyze_corrected_transcript"
            )

        if st.session_state.analysis_error and not st.session_state.analysis:
            st.info(
                "The transcript is available. Once Ollama is ready, use the "
                "button above to run or retry AI analysis."
            )

        if analyze_clicked:
            try:
                st.session_state.analysis = None
                st.session_state.analysis_error = None
                st.session_state.meeting_saved = False
                st.session_state.saved_meeting_id = None
                st.session_state.transcript = corrected_transcript
                st.session_state.pop("confirm_analysis_review", None)

                with st.spinner("Re-running meeting analysis with your local AI model..."):
                    st.session_state.analysis = analyze_meeting(corrected_transcript)

                transcript_changed = False
                st.success("Meeting analysis completed. Review the results before saving.")
            except Exception as error:
                st.session_state.analysis_error = str(error)
                st.error(f"Analysis failed: {error}")

    if st.session_state.analysis:
        analysis = st.session_state.analysis

        st.divider()
        st.header("Meeting Intelligence")
        st.subheader("Executive Summary")
        st.write(analysis.get("summary") or "No summary available.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Meeting Objectives")
            objectives = analysis.get("objectives", [])
            if objectives:
                for item in objectives:
                    st.write(f"• {item}")
            else:
                st.info("No objectives identified.")

            st.subheader("Decisions")
            decisions = analysis.get("decisions", [])
            if decisions:
                for item in decisions:
                    st.write(f"• {item}")
            else:
                st.info("No decisions identified.")

            st.subheader("Risks and Blockers")
            risks = analysis.get("risks", [])
            if risks:
                for item in risks:
                    st.write(f"• {item}")
            else:
                st.info("No risks identified.")

        with col2:
            st.subheader("Key Discussion Points")
            key_points = analysis.get("key_points", [])
            if key_points:
                for item in key_points:
                    st.write(f"• {item}")
            else:
                st.info("No key points identified.")

            st.subheader("Unresolved Questions")
            questions = analysis.get("unresolved_questions", [])
            if questions:
                for item in questions:
                    st.write(f"• {item}")
            else:
                st.info("No unresolved questions identified.")

        st.subheader("Action Items")
        action_items = analysis.get("action_items", [])

        if action_items:
            st.dataframe(
                pd.DataFrame(action_items),
                width="stretch",
                hide_index=True
            )
            with st.expander("View supporting transcript evidence"):
                for item in action_items:
                    st.markdown(f"**{item.get('task', 'Task')}**")
                    st.caption(
                        f"Owner: {item.get('responsible_person', 'Unknown')} | "
                        f"Deadline: {item.get('deadline', 'Not specified')}"
                    )
                    st.write(item.get("evidence") or "No evidence quote returned.")
        else:
            st.info("No action items identified.")

        validation = analysis.get("validation", {})
        warnings = validation.get("warnings", [])
        blocking_issues = validation.get("blocking_issues", [])

        if blocking_issues:
            st.warning(
                "Some extracted information could not be matched to the transcript. "
                "Review the flagged items before saving."
            )
            for issue in blocking_issues:
                st.write(f"• {issue}")
            review_confirmed = st.checkbox(
                "I reviewed the transcript and accept the flagged verification issues.",
                key="confirm_analysis_review"
            )
        elif warnings:
            st.info(
                "Review recommended: some action items have an unknown owner or "
                "an unspecified deadline. MeetMind has kept those fields explicit "
                "instead of guessing. You can still save the meeting."
            )
            for warning in warnings:
                st.write(f"• {warning}")
            review_confirmed = True
        else:
            st.success("Automated evidence checks passed.")
            review_confirmed = True

        with st.expander("View extracted fact ledger"):
            facts = analysis.get("facts", [])
            if facts:
                st.dataframe(
                    pd.DataFrame(facts),
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("No separate facts were extracted.")

        if not st.session_state.meeting_saved:
            if st.button(
                "💾 Save Meeting",
                type="primary",
                disabled=(not review_confirmed) or transcript_changed
            ):
                try:
                    meeting_id = save_meeting_with_tasks(
                        title=st.session_state.processed_title,
                        transcript=st.session_state.transcript,
                        summary=analysis.get("summary", ""),
                        analysis=json.dumps(analysis, ensure_ascii=False),
                        action_items=action_items
                    )

                    st.session_state.meeting_saved = True
                    st.session_state.saved_meeting_id = meeting_id

                    st.success(
                        f"Meeting and action items saved successfully. "
                        f"Meeting ID: {meeting_id}"
                    )
                except Exception as error:
                    st.error(f"Save failed. No partial meeting/task save was kept: {error}")
        else:
            st.success(
                f"Meeting saved. ID: {st.session_state.saved_meeting_id}"
            )

        st.download_button(
            "Download Complete AI Report",
            json.dumps(analysis, indent=4, ensure_ascii=False),
            file_name="meetmind_report.json",
            mime="application/json",
            key="new_report_download"
        )


# ==================================================
# TAB 2: MEETING HISTORY
# ==================================================

with tab2:
    st.header("Meeting History")

    try:
        meetings = safe_get_meetings()
        search = st.text_input(
            "Search meetings",
            placeholder="Search by title...",
            key="meeting_history_search"
        )

        filtered = [
            meeting for meeting in meetings
            if search.casefold() in (meeting.title or "").casefold()
        ]

        st.caption(
            f"{len(filtered)} matching meetings out of {len(meetings)}"
        )

        if not filtered:
            st.info("No matching meetings found.")

        for meeting in filtered:
            with st.expander(
                f"{meeting.title} | ID: {meeting.id}"
            ):
                st.write("**Date:**", meeting.created_at)
                st.write("**Summary:**", meeting.summary or "No summary available.")

                st.text_area(
                    "Transcript",
                    meeting.transcript or "",
                    height=180,
                    disabled=True,
                    key=f"history_transcript_{meeting.id}"
                )

                try:
                    analysis = json.loads(meeting.analysis or "{}")
                    if not isinstance(analysis, dict):
                        analysis = {}
                except (json.JSONDecodeError, TypeError):
                    analysis = {}

                for section, label in [
                    ("objectives", "Meeting Objectives"),
                    ("key_points", "Key Discussion Points"),
                    ("decisions", "Decisions"),
                    ("risks", "Risks and Blockers"),
                    ("unresolved_questions", "Unresolved Questions")
                ]:
                    values = analysis.get(section, [])
                    if isinstance(values, list) and values:
                        st.subheader(label)
                        for item in values:
                            st.write(f"• {item}")

                st.subheader("Action Items")
                items = analysis.get("action_items", [])
                if isinstance(items, list) and items:
                    st.dataframe(
                        pd.DataFrame(items),
                        width="stretch",
                        hide_index=True
                    )
                    with st.expander("View action-item evidence"):
                        for item in items:
                            if isinstance(item, dict):
                                st.markdown(f"**{item.get('task', 'Task')}**")
                                st.write(
                                    item.get("evidence") or
                                    "No evidence was stored for this item."
                                )
                else:
                    st.info("No action items stored for this meeting.")

                validation = analysis.get("validation", {})
                if validation.get("warnings"):
                    st.warning("This saved analysis contains review warnings.")
                    for warning in validation["warnings"]:
                        st.write(f"• {warning}")

                st.download_button(
                    "Download Transcript",
                    meeting.transcript or "",
                    file_name=f"meeting_{meeting.id}.txt",
                    mime="text/plain",
                    key=f"download_transcript_{meeting.id}"
                )

                st.download_button(
                    "Download AI Report",
                    json.dumps(analysis, indent=4, ensure_ascii=False),
                    file_name=f"meeting_{meeting.id}.json",
                    mime="application/json",
                    key=f"download_report_{meeting.id}"
                )

                if st.button("Delete Meeting", key=f"delete_{meeting.id}"):
                    try:
                        if delete_meeting(meeting.id):
                            st.success("Meeting and its tasks/chat history deleted.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Delete failed: {error}")

    except Exception as error:
        st.error(f"Could not load meeting history: {error}")


# ==================================================
# TAB 3: CHAT WITH MEETINGS
# ==================================================

with tab3:
    st.header("💬 Chat with Your Meetings")
    meetings = safe_get_meetings()

    if not meetings:
        st.info("Save a meeting before using meeting chat.")
    else:
        meeting_options = {
            f"{meeting.title} (ID: {meeting.id})": meeting.id
            for meeting in meetings
        }

        selected = st.selectbox(
            "Select a meeting",
            list(meeting_options.keys()),
            key="chat_meeting_selector"
        )

        meeting_id = meeting_options[selected]
        meeting = next(m for m in meetings if m.id == meeting_id)

        if st.session_state.chat_loaded_meeting_id != meeting_id:
            st.session_state.chat_history = get_chat_history(meeting_id)
            st.session_state.chat_loaded_meeting_id = meeting_id

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask something about this meeting...")

        if question:
            previous_history = list(st.session_state.chat_history)

            try:
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = chat_with_meeting(
                            meeting.transcript,
                            question,
                            previous_history
                        )
                    st.write(answer)

                # Persist the question and answer atomically.
                save_chat_exchange(meeting_id, question, answer)

                st.session_state.chat_history = (
                    previous_history
                    + [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                )

            except Exception as error:
                st.error(f"Chat failed: {error}")

        if st.button("Clear Chat", key=f"clear_chat_{meeting_id}"):
            try:
                clear_chat_history(meeting_id)
                st.session_state.chat_history = []
                st.session_state.chat_loaded_meeting_id = meeting_id
                st.success("Chat history cleared.")
                st.rerun()
            except Exception as error:
                st.error(f"Could not clear chat: {error}")


# ==================================================
# TAB 4: TASK MANAGER
# ==================================================

with tab4:
    st.header("📌 Action Item Management")
    tasks = safe_get_tasks()

    if not tasks:
        st.info("No action items yet. Analyze and save a meeting first.")
    else:
        task_data = [
            {
                "ID": task.id,
                "Meeting ID": task.meeting_id,
                "Task": task.task,
                "Responsible Person": task.responsible_person,
                "Deadline": task.deadline,
                "Priority": task.priority,
                "Status": task.status
            }
            for task in tasks
        ]

        st.dataframe(
            pd.DataFrame(task_data),
            width="stretch",
            hide_index=True
        )

        st.divider()
        st.subheader("Update Task Status")

        selected_task = st.selectbox(
            "Select Task",
            tasks,
            format_func=lambda task: f"#{task.id} — {task.task[:65]}",
            key="task_selector"
        )

        statuses = ["Pending", "In Progress", "Completed"]
        current_index = (
            statuses.index(selected_task.status)
            if selected_task.status in statuses else 0
        )

        new_status = st.selectbox(
            "New Status",
            statuses,
            index=current_index,
            key=f"task_status_{selected_task.id}"
        )

        if st.button("Update Task"):
            try:
                if update_task_status(selected_task.id, new_status):
                    st.success("Task status updated!")
                else:
                    st.error("Could not find that task.")
                st.rerun()
            except Exception as error:
                st.error(f"Task update failed: {error}")


# ==================================================
# TAB 5: ANALYTICS DASHBOARD
# ==================================================

with tab5:
    st.header("📊 Meeting Analytics")
    meetings = safe_get_meetings()
    tasks = safe_get_tasks()

    total_meetings = len(meetings)
    total_tasks = len(tasks)
    completed = sum(task.status == "Completed" for task in tasks)
    pending = sum(task.status == "Pending" for task in tasks)
    in_progress = sum(task.status == "In Progress" for task in tasks)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Meetings", total_meetings)
    col2.metric("Total Tasks", total_tasks)
    col3.metric("Completed", completed)
    col4.metric("Pending", pending)

    st.divider()
    st.subheader("Task Status Distribution")

    status_data = pd.DataFrame({
        "Status": ["Completed", "In Progress", "Pending"],
        "Tasks": [completed, in_progress, pending]
    })

    if total_tasks:
        st.bar_chart(status_data.set_index("Status"))
        completion_rate = completed / total_tasks * 100
        st.metric("Task Completion Rate", f"{completion_rate:.1f}%")
    else:
        st.info("Save meetings containing action items to see analytics.")

    st.divider()
    st.subheader("Task Priority Distribution")

    if tasks:
        priority_data = pd.DataFrame([
            {"Priority": task.priority or "Medium"}
            for task in tasks
        ])
        priority_counts = (
            priority_data["Priority"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("Priority")
            .to_frame("Tasks")
        )
        st.bar_chart(priority_counts)
    else:
        st.info("No task priority data available yet.")

    st.divider()
    st.subheader("Meeting-wise Task Statistics")

    if meetings:
        task_rows = [
            {"Meeting ID": task.meeting_id, "Status": task.status}
            for task in tasks
        ]
        task_frame = pd.DataFrame(
            task_rows,
            columns=["Meeting ID", "Status"]
        )

        meeting_stats = []
        for meeting in meetings:
            meeting_tasks = (
                task_frame[task_frame["Meeting ID"] == meeting.id]
                if not task_frame.empty else pd.DataFrame()
            )

            meeting_stats.append({
                "Meeting": meeting.title,
                "Meeting ID": meeting.id,
                "Total Tasks": len(meeting_tasks),
                "Completed": (
                    int((meeting_tasks["Status"] == "Completed").sum())
                    if not meeting_tasks.empty else 0
                ),
                "In Progress": (
                    int((meeting_tasks["Status"] == "In Progress").sum())
                    if not meeting_tasks.empty else 0
                ),
                "Pending": (
                    int((meeting_tasks["Status"] == "Pending").sum())
                    if not meeting_tasks.empty else 0
                )
            })

        st.dataframe(
            pd.DataFrame(meeting_stats),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No meetings available yet.")

    st.divider()
    st.subheader("Meeting Activity")

    if meetings:
        activity = pd.DataFrame([
            {
                "Meeting": meeting.title,
                "Created": meeting.created_at.strftime("%Y-%m-%d")
            }
            for meeting in meetings
        ])
        st.dataframe(
            activity,
            width="stretch",
            hide_index=True
        )
