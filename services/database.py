from pathlib import Path
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, ForeignKey, event
)
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIR / "meetmind.db"

engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True
)


# SQLite does not enforce foreign keys unless enabled per connection.
@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)
Base = declarative_base()


# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    transcript = Column(Text, nullable=False)
    summary = Column(Text)
    analysis = Column(Text)
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )


class MeetingTask(Base):
    __tablename__ = "meeting_tasks"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(
        Integer,
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    task = Column(Text, nullable=False)
    responsible_person = Column(String(255), default="Unknown")
    deadline = Column(String(255), default="Not specified")
    priority = Column(String(30), default="Medium")
    status = Column(String(30), default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class MeetingChat(Base):
    __tablename__ = "meeting_chats"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(
        Integer,
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Adds new tables without deleting existing data.
Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# MEETING OPERATIONS
# --------------------------------------------------

def save_meeting(title, transcript, summary, analysis):
    title = (title or "").strip()
    transcript = (transcript or "").strip()

    if not title:
        raise ValueError("Meeting title cannot be empty.")
    if not transcript:
        raise ValueError("Meeting transcript cannot be empty.")

    with SessionLocal() as session:
        try:
            meeting = Meeting(
                title=title,
                transcript=transcript,
                summary=summary or "",
                analysis=analysis or "{}"
            )
            session.add(meeting)
            session.commit()
            session.refresh(meeting)
            return meeting.id
        except Exception:
            session.rollback()
            raise


def save_meeting_with_tasks(title, transcript, summary, analysis, action_items):
    """Save a meeting and all its action items in one transaction."""
    title = (title or "").strip()
    transcript = (transcript or "").strip()

    if not title:
        raise ValueError("Meeting title cannot be empty.")
    if not transcript:
        raise ValueError("Meeting transcript cannot be empty.")

    action_items = action_items or []

    with SessionLocal() as session:
        try:
            meeting = Meeting(
                title=title,
                transcript=transcript,
                summary=summary or "",
                analysis=analysis or "{}"
            )
            session.add(meeting)
            session.flush()  # Assign meeting.id before creating tasks.

            for item in action_items:
                if not isinstance(item, dict):
                    continue

                task_text = str(item.get("task", "")).strip()
                if not task_text:
                    continue

                priority = str(item.get("priority", "Medium")).title()
                if priority not in ("High", "Medium", "Low"):
                    priority = "Medium"

                session.add(MeetingTask(
                    meeting_id=meeting.id,
                    task=task_text,
                    responsible_person=str(
                        item.get("responsible_person") or "Unknown"
                    ),
                    deadline=str(item.get("deadline") or "Not specified"),
                    priority=priority,
                    status="Pending"
                ))

            session.commit()
            return meeting.id
        except Exception:
            session.rollback()
            raise


def get_meetings():
    with SessionLocal() as session:
        return (
            session.query(Meeting)
            .order_by(Meeting.created_at.desc(), Meeting.id.desc())
            .all()
        )


def get_meeting_by_id(meeting_id):
    with SessionLocal() as session:
        return session.query(Meeting).filter(
            Meeting.id == meeting_id
        ).first()


def delete_meeting(meeting_id):
    with SessionLocal() as session:
        try:
            meeting = session.query(Meeting).filter(
                Meeting.id == meeting_id
            ).first()

            if not meeting:
                return False

            # Explicit deletes are retained for compatibility with old DBs.
            session.query(MeetingTask).filter(
                MeetingTask.meeting_id == meeting_id
            ).delete(synchronize_session=False)

            session.query(MeetingChat).filter(
                MeetingChat.meeting_id == meeting_id
            ).delete(synchronize_session=False)

            session.delete(meeting)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise


# --------------------------------------------------
# TASK OPERATIONS
# --------------------------------------------------

def create_meeting_tasks(meeting_id, action_items):
    action_items = action_items or []

    with SessionLocal() as session:
        try:
            meeting_exists = session.query(Meeting.id).filter(
                Meeting.id == meeting_id
            ).first()

            if not meeting_exists:
                raise ValueError("Cannot create tasks: meeting does not exist.")

            existing = session.query(MeetingTask).filter(
                MeetingTask.meeting_id == meeting_id
            ).count()

            if existing:
                return existing

            created = 0
            for item in action_items:
                if not isinstance(item, dict):
                    continue

                task_text = str(item.get("task", "")).strip()
                if not task_text:
                    continue

                priority = str(item.get("priority", "Medium")).title()
                if priority not in ("High", "Medium", "Low"):
                    priority = "Medium"

                session.add(MeetingTask(
                    meeting_id=meeting_id,
                    task=task_text,
                    responsible_person=str(
                        item.get("responsible_person") or "Unknown"
                    ),
                    deadline=str(item.get("deadline") or "Not specified"),
                    priority=priority,
                    status="Pending"
                ))
                created += 1

            session.commit()
            return created
        except Exception:
            session.rollback()
            raise


def get_tasks(meeting_id=None):
    with SessionLocal() as session:
        query = session.query(MeetingTask)

        if meeting_id is not None:
            query = query.filter(MeetingTask.meeting_id == meeting_id)

        return query.order_by(
            MeetingTask.created_at.desc(),
            MeetingTask.id.desc()
        ).all()


def update_task_status(task_id, status):
    allowed = {"Pending", "In Progress", "Completed"}
    if status not in allowed:
        raise ValueError("Invalid task status.")

    with SessionLocal() as session:
        try:
            task = session.query(MeetingTask).filter(
                MeetingTask.id == task_id
            ).first()

            if not task:
                return False

            task.status = status
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise


# --------------------------------------------------
# CHAT OPERATIONS
# --------------------------------------------------

def save_chat_message(meeting_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("Invalid chat role.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat message cannot be empty.")

    with SessionLocal() as session:
        try:
            meeting_exists = session.query(Meeting.id).filter(
                Meeting.id == meeting_id
            ).first()

            if not meeting_exists:
                raise ValueError("Meeting does not exist.")

            session.add(MeetingChat(
                meeting_id=meeting_id,
                role=role,
                content=content.strip()
            ))
            session.commit()
        except Exception:
            session.rollback()
            raise


def save_chat_exchange(meeting_id, question, answer):
    """Save a user question and AI answer together or save neither."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not answer or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    with SessionLocal() as session:
        try:
            meeting_exists = session.query(Meeting.id).filter(
                Meeting.id == meeting_id
            ).first()

            if not meeting_exists:
                raise ValueError("Meeting does not exist.")

            session.add_all([
                MeetingChat(
                    meeting_id=meeting_id,
                    role="user",
                    content=question.strip()
                ),
                MeetingChat(
                    meeting_id=meeting_id,
                    role="assistant",
                    content=answer.strip()
                )
            ])
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_chat_history(meeting_id):
    with SessionLocal() as session:
        messages = session.query(MeetingChat).filter(
            MeetingChat.meeting_id == meeting_id
        ).order_by(
            MeetingChat.created_at.asc(),
            MeetingChat.id.asc()
        ).all()

        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]


def clear_chat_history(meeting_id):
    with SessionLocal() as session:
        try:
            session.query(MeetingChat).filter(
                MeetingChat.meeting_id == meeting_id
            ).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
