"""MeetMind meeting analysis and transcript chat.

Drop-in replacement for the existing services/meeting_ai.py module.
Requires: langchain-ollama, python-dotenv, and a running Ollama server.

Environment variables:
    MEETMIND_MODEL       Ollama model for structured meeting analysis
    MEETMIND_CHAT_MODEL  Ollama model for meeting chat (defaults to analysis model)
"""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

MAX_ANALYSIS_CHARS = 18000
MAX_CHAT_TRANSCRIPT_CHARS = 18000
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_MESSAGES = 8

DEFAULT_MODEL = "gemma:2b"
OLLAMA_MODEL = os.getenv("MEETMIND_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
OLLAMA_CHAT_MODEL = (
    os.getenv("MEETMIND_CHAT_MODEL", OLLAMA_MODEL).strip() or OLLAMA_MODEL
)

analysis_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    format="json",
)
chat_llm = ChatOllama(
    model=OLLAMA_CHAT_MODEL,
    temperature=0,
)


# ---------------------------------------------------------------------------
# JSON and text helpers
# ---------------------------------------------------------------------------


def _response_text(response: Any) -> str:
    """Extract text from LangChain message content across common versions."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _clean_json(content: str) -> str:
    content = (content or "").strip()
    content = re.sub(r"\A```(?:json|text)?\s*", "", content, flags=re.I)
    content = re.sub(r"\s*```\Z", "", content)
    return content.strip()


def parse_json(content: str) -> dict:
    """Parse a JSON object, tolerating surrounding prose/code fences."""
    content = _clean_json(content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            raise ValueError("The local AI did not return a JSON object.")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("The local AI returned incomplete or invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("The local AI returned JSON, but it was not an object.")
    return parsed


def _normalise(text: Any) -> str:
    """Normalize whitespace and case while preserving the original words."""
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _safe_string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return fallback
    return str(value).strip() or fallback


def _evidence_is_present(transcript: str, evidence: str) -> bool:
    evidence = _safe_string(evidence)
    return bool(evidence) and _normalise(evidence) in _normalise(transcript)


def _clean_list(values: Any) -> list[str]:
    cleaned = []
    for value in _as_list(values):
        if isinstance(value, str) and value.strip():
            cleaned.append(value.strip())
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(cleaned))


def _normalise_owner(owner: str) -> str:
    owner = _safe_string(owner, "Unknown")
    # A first-person pronoun does not identify the speaker by name.
    if owner.casefold() in {
        "unknown", "not specified", "none", "n/a", "i", "me", "myself",
        "the speaker", "speaker", "someone", "unspecified",
    }:
        return "Unknown"
    return owner


def _normalise_deadline(deadline: str) -> str:
    deadline = _safe_string(deadline, "Not specified")
    if deadline.casefold() in {"unknown", "none", "n/a", "not mentioned", "unspecified"}:
        return "Not specified"
    return deadline


# ---------------------------------------------------------------------------
# Analysis prompt
# ---------------------------------------------------------------------------


def _build_prompt(transcript: str) -> str:
    schema = {
        "summary": "A concise but informative summary of the meeting",
        "objectives": ["A stated meeting objective"],
        "key_points": ["An important fact, update, date, number, or discussion point"],
        "decisions": ["A decision explicitly agreed in the meeting"],
        "action_items": [
            {
                "task": "A clear verb-led task",
                "responsible_person": "Exact person's name or Unknown",
                "deadline": "Exact spoken deadline or Not specified",
                "priority": "High, Medium, or Low",
                "evidence": "An exact continuous quotation copied from the transcript",
            }
        ],
        "risks": ["An explicitly discussed risk, blocker, or dependency"],
        "unresolved_questions": ["A question or issue explicitly left unresolved"],
        "facts": [
            {
                "category": "date, number, person, budget, status, or other",
                "fact": "One specific fact stated in the transcript",
                "evidence": "An exact continuous quotation copied from the transcript",
            }
        ],
        "next_meeting": "Exact next-meeting date/time, or Not specified",
    }

    return f"""You are MeetMind, a precise meeting-minutes assistant.
Analyze ONLY the transcript. Do not use outside knowledge or make assumptions.

Return ONLY one valid JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

EXTRACTION RULES:
1. Preserve names, dates, times, percentages, counts, currencies and amounts exactly as spoken.
2. Do not invent a year, timezone, currency, deadline, owner, or missing detail.
3. Do not convert currencies or silently correct a number.
4. Include an action item only when someone explicitly accepts, is assigned, or is clearly instructed to do a task.
5. Do not turn general discussion, suggestions, or questions into action items.
6. The owner must be a clearly identified person. If the speaker says "I" but their identity is not established in the transcript, use "Unknown" rather than "I".
7. A deadline must be explicitly connected to that task. Do not borrow a nearby deadline from another task.
8. Use "Not specified" when the transcript does not state a deadline.
9. Use High priority only when urgency or criticality is explicit. Otherwise use Medium. Use Low only when explicitly described as low priority.
10. Each action item's evidence must be an exact, continuous quote from the transcript that supports the task and any extracted owner/deadline. Do not paraphrase evidence.
11. Each fact's evidence must be an exact, continuous quote supporting that fact.
12. Include the next meeting only if its date/time is explicitly stated.
13. Keep the summary informative: include the meeting's purpose, important progress, decisions, material deadlines and budget constraints when present.
14. Keep objectives, key points, decisions, risks and unresolved questions in their correct categories.
15. A payment approval awaiting follow-up can be an action item; do not call it unresolved if the transcript says it has already been resolved.
16. Return empty arrays when the transcript does not support entries. Do not fabricate content to fill the schema.
17. Do not infer an action from a noun phrase or topic fragment (for example, "professional posts by Friday") unless an action or assignment is explicitly stated.
18. Keep separate commitments as separate action items, and attach each deadline only to the clause that states it.

TRANSCRIPT:
<<<
{transcript}
>>>
"""


# ---------------------------------------------------------------------------
# Deterministic action-item coverage pass
# ---------------------------------------------------------------------------

_MONTH = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE = rf"(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?"
_DATE_RE = re.compile(_DATE, re.IGNORECASE)


def _sentences(transcript: str) -> list[str]:
    """Split transcript into sentence-like units without changing their text."""
    return [m.group(0).strip() for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", transcript) if m.group(0).strip()]


def _task_from_fragment(fragment: str, inherited_verb: str = "") -> str:
    task = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", fragment).strip()
    task = re.sub(r"^(?:that\s+)?", "", task, flags=re.IGNORECASE)
    task = re.sub(r"\s+(?:by|on)\s+" + _DATE + r"\s*$", "", task, flags=re.IGNORECASE).strip()
    task = re.sub(r"\s+", " ", task).strip(" ,;:")
    if not task:
        return ""

    # Normalize common gerund-form commitments into clear task labels.
    gerund_prefixes = {
        "resolving ": "Resolve ",
        "reviewing ": "Review ",
        "publishing ": "Publish ",
        "launching ": "Launch ",
        "sending ": "Send ",
        "approving ": "Approve ",
        "preparing ": "Prepare ",
        "creating ": "Create ",
        "finishing ": "Finish ",
        "completing ": "Complete ",
        "following up ": "Follow up ",
    }
    for gerund, infinitive in gerund_prefixes.items():
        if task.casefold().startswith(gerund):
            task = infinitive + task[len(gerund):]
            break

    # In coordinated speech, speakers often omit the verb in later clauses:
    # "finish the module by Monday and testing by Tuesday".
    if inherited_verb and not re.match(
        r"^(?:complete|finish|test|review|prepare|publish|send|launch|create|resolve|approve|submit|deliver|update|follow|check|finali[sz]e|implement|fix|build|provide|share|confirm|schedule|organize|organise|draft|discuss|contact|call|email|pay|review)\b",
        task,
        re.IGNORECASE,
    ):
        return f"{inherited_verb} {task}".strip()

    # Noun/gerund fragments with a deadline are retained as clear, conservative
    # task labels rather than silently discarded.
    if re.match(r"^(?:testing|posts|professional posts|expenses|the email campaign|email campaign)\b", task, re.IGNORECASE):
        if re.match(r"^testing\b", task, re.IGNORECASE):
            return f"Complete {task}"
        if re.match(r"^(?:posts|professional posts)\b", task, re.IGNORECASE):
            return f"Prepare {task}"
        if re.match(r"^expenses\b", task, re.IGNORECASE):
            return f"Review {task}"
    return task[0].upper() + task[1:]


def _extract_explicit_action_candidates(transcript: str) -> list[dict]:
    """Recover explicit commitments that a small local model may omit.

    This is deliberately conservative: it only handles clear first-person
    commitments, named assignments, and named requests with a stated task.
    It does not guess who said 'I'.
    """
    candidates: list[dict] = []

    def add(task: str, owner: str, deadline: str, evidence: str) -> None:
        task = _safe_string(task)
        evidence = _safe_string(evidence)
        if not task or not evidence or not _evidence_is_present(transcript, evidence):
            return
        candidates.append({
            "task": task,
            "responsible_person": owner or "Unknown",
            "deadline": deadline or "Not specified",
            "priority": "Medium",
            "evidence": evidence,
        })

    for sentence in _sentences(transcript):
        # Explicit named assignment: "Raul, you are responsible for resolving ... by ..."
        assigned = re.search(
            r"\b([A-Z][A-Za-z'-]+)\s*,\s*you are responsible for\s+(.+?)(?:\s+(?:by|on)\s+(" + _DATE + r"))(?=[,;.]|$)",
            sentence,
            re.IGNORECASE,
        )
        if assigned:
            add(_task_from_fragment(assigned.group(2)), assigned.group(1), assigned.group(3), assigned.group(0))

        # Explicit named request: "Raul, please follow up by November 10th."
        requested = re.search(
            r"\b([A-Z][A-Za-z'-]+)\s*,\s*please\s+(.+?)(?:\s+(?:by|on)\s+(" + _DATE + r"))(?=[,;.]|$)",
            sentence,
            re.IGNORECASE,
        )
        if requested:
            add(_task_from_fragment(requested.group(2)), requested.group(1), requested.group(3), requested.group(0))

        # First-person commitments. The speaker's name is not guessed from
        # nearby dialogue; the UI can show Unknown until a person is confirmed.
        first_person = re.search(r"\b(I['’]ll|I will|we['’]ll|we will)\s+(.+)$", sentence, re.IGNORECASE)
        if first_person:
            body = first_person.group(2).strip().rstrip(".!? ")
            # Split coordinated actions so each deadline stays with its own task.
            fragments = re.split(r"\s+and\s+", body, flags=re.IGNORECASE)
            inherited_verb = ""
            for fragment in fragments:
                date_match = _DATE_RE.search(fragment)
                deadline = date_match.group(0) if date_match else "Not specified"
                clean_fragment = fragment[:date_match.start()].strip() if date_match else fragment.strip()
                clean_fragment = re.sub(r"\s+(?:by|on)\s*$", "", clean_fragment, flags=re.IGNORECASE).strip()
                verb_match = re.match(
                    r"(finish|complete|test|review|prepare|publish|send|launch|create|resolve|approve|submit|deliver|update|follow up|check|finali[sz]e|implement|fix|build|provide|share|confirm|schedule|organize|organise|draft|discuss|contact|call|email|pay)\b",
                    clean_fragment,
                    re.IGNORECASE,
                )
                if verb_match:
                    inherited_verb = verb_match.group(1).capitalize()
                task = _task_from_fragment(clean_fragment, inherited_verb if not verb_match else "")
                if task:
                    add(task, "Unknown", deadline, fragment.strip())

        # Do not infer an action from a deadline-bearing noun phrase alone. Only explicit commitments, assignments, or requests are recovered here.

    # De-duplicate by task + deadline while preserving the first exact evidence.
    unique: list[dict] = []
    seen = set()
    for item in candidates:
        key = (_normalise(item["task"]), _normalise(item["deadline"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _merge_action_items(transcript: str, model_items: Any) -> list[dict]:
    """Merge model output with conservative rule-based coverage candidates."""
    model_list = [x for x in _as_list(model_items) if isinstance(x, dict)]
    candidates = _extract_explicit_action_candidates(transcript)
    merged = list(model_list)

    def similarity(a: str, b: str) -> float:
        from difflib import SequenceMatcher
        na, nb = _normalise(a), _normalise(b)
        if not na or not nb:
            return 0.0
        if na in nb or nb in na:
            return 1.0
        return SequenceMatcher(None, na, nb).ratio()

    for candidate in candidates:
        match_index = None
        for i, existing in enumerate(merged):
            if similarity(candidate["task"], _safe_string(existing.get("task"))) >= 0.72:
                match_index = i
                break
        if match_index is None:
            merged.append(candidate)
            continue

        # The deterministic clause-level date takes precedence over a model
        # date that may have leaked from an adjacent task. Owner stays Unknown
        # unless the source sentence explicitly names one.
        existing = dict(merged[match_index])
        existing["deadline"] = candidate["deadline"]
        if candidate["responsible_person"] != "Unknown":
            existing["responsible_person"] = candidate["responsible_person"]
        else:
            existing["responsible_person"] = "Unknown"
        # Keep the task label no more specific than the exact clause-level
        # evidence supports (e.g. "Follow up", not an inferred object).
        existing["task"] = candidate["task"]
        existing["evidence"] = candidate["evidence"]
        existing["priority"] = candidate["priority"]
        merged[match_index] = existing

    return merged


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------


def _validate_analysis(transcript: str, result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Meeting analysis returned an invalid structure.")

    summary = _safe_string(result.get("summary"))
    if not summary or summary.casefold() in {
        "meeting summary", "summary", "no summary available", "not specified"
    }:
        summary = "A reliable executive summary could not be generated. Please review the transcript."

    for field in ("objectives", "key_points", "decisions", "risks", "unresolved_questions"):
        result[field] = _clean_list(result.get(field))

    warnings: list[str] = []
    blocking_issues: list[str] = []

    if not result["objectives"]:
        warnings.append("No explicit meeting objectives were extracted.")

    # Validate facts individually. Unverified facts are retained for inspection,
    # but clearly marked and never represented as verified.
    verified_facts = []
    for index, fact in enumerate(_as_list(result.get("facts")), start=1):
        if not isinstance(fact, dict):
            continue
        fact_text = _safe_string(fact.get("fact"))
        evidence = _safe_string(fact.get("evidence"))
        if not fact_text:
            continue
        verified = _evidence_is_present(transcript, evidence)
        verified_facts.append({
            "category": _safe_string(fact.get("category"), "other"),
            "fact": fact_text,
            "evidence": evidence,
            "evidence_verified": verified,
        })
        if not verified:
            warnings.append(f"Fact {index} has missing or non-matching transcript evidence.")
            blocking_issues.append(f"Fact {index} could not be verified against the transcript.")

    # Action items are eligible for storage only when their evidence is present
    # in the transcript. Missing owner/deadline is incomplete, not fabricated.
    verified_tasks = []
    for index, item in enumerate(_as_list(result.get("action_items")), start=1):
        if not isinstance(item, dict):
            continue

        task = _safe_string(item.get("task"))
        if not task:
            continue

        evidence = _safe_string(item.get("evidence"))
        if not _evidence_is_present(transcript, evidence):
            warnings.append(
                f"Action item {index} ({task}) was excluded because its evidence "
                "is missing or does not match the transcript."
            )
            blocking_issues.append(f"Action item {index} could not be verified.")
            continue

        owner = _normalise_owner(_safe_string(item.get("responsible_person"), "Unknown"))
        deadline = _normalise_deadline(_safe_string(item.get("deadline"), "Not specified"))
        priority = _safe_string(item.get("priority"), "Medium").title()
        if priority not in {"High", "Medium", "Low"}:
            priority = "Medium"

        evidence_norm = _normalise(evidence)
        owner_verified = owner != "Unknown" and _normalise(owner) in evidence_norm
        if not owner_verified:
            owner = "Unknown"
            warnings.append(f"Action item has no confirmed owner: {task}")

        deadline_verified = deadline != "Not specified" and _normalise(deadline) in evidence_norm
        if not deadline_verified:
            deadline = "Not specified"
            warnings.append(f"Action item has no confirmed deadline: {task}")

        verified_tasks.append({
            "task": task,
            "responsible_person": owner,
            "owner_verified": owner_verified,
            "deadline": deadline,
            "deadline_verified": deadline_verified,
            "priority": priority,
            "evidence": evidence,
            "evidence_verified": True,
        })

    next_meeting = _safe_string(result.get("next_meeting"), "Not specified")
    if next_meeting.casefold() in {"unknown", "none", "n/a", "not mentioned"}:
        next_meeting = "Not specified"

    result["summary"] = summary
    result["facts"] = verified_facts
    result["action_items"] = verified_tasks
    result["next_meeting"] = next_meeting

    # Missing metadata (e.g. a task deadline) is a non-blocking review note.
    # Only evidence failures are classified as blocking verification problems.
    result["validation"] = {
        "status": "review_required" if blocking_issues else "ready",
        "review_required": bool(blocking_issues),
        "review_recommended": bool(warnings),
        "warnings": list(dict.fromkeys(warnings)),
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "verified_action_item_count": len(verified_tasks),
        "unverified_action_items_excluded": True,
    }
    return result


# ---------------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------------


def analyze_meeting(transcript: str) -> dict:
    """Analyze a transcript and return the structured MeetMind result."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Transcript is empty.")

    transcript = transcript.strip()
    if len(transcript) > MAX_ANALYSIS_CHARS:
        raise ValueError(
            f"Transcript is {len(transcript):,} characters. The current limit is "
            f"{MAX_ANALYSIS_CHARS:,}. The transcript was not truncated; analyze "
            "shorter sections separately."
        )

    try:
        response = analysis_llm.invoke(_build_prompt(transcript))
        result = parse_json(_response_text(response))
        # A second, deterministic pass recovers explicit commitments that the
        # local model may omit and prevents adjacent deadlines being borrowed.
        result["action_items"] = _merge_action_items(
            transcript, result.get("action_items", [])
        )
        return _validate_analysis(transcript, result)
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        raise RuntimeError(
            "Meeting analysis failed. Confirm Ollama is running and that "
            f"'{OLLAMA_MODEL}' is installed. Details: {error}"
        ) from error


def chat_with_meeting(transcript: str, question: str, history: list | None = None) -> str:
    """Answer a question using only the supplied meeting transcript."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Meeting transcript is unavailable.")
    if len(transcript) > MAX_CHAT_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Transcript exceeds the chat limit of {MAX_CHAT_TRANSCRIPT_CHARS:,} "
            "characters. It was not truncated."
        )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Please keep your question under {MAX_QUESTION_CHARS:,} characters.")

    safe_history = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            safe_history.append(f"{role}: {content[:1500]}")

    conversation = "\n".join(safe_history) or "No previous conversation."
    prompt = f"""You are MeetMind, a professional meeting assistant.
Answer using ONLY the meeting transcript. The conversation history is included
only to understand follow-up questions; it is not a source of meeting facts.

Rules:
- Be precise with names, dates, times, numbers, currencies, owners and status.
- Do not invent missing information or infer a person's identity from "I".
- Distinguish completed work from pending work.
- If the transcript does not contain the answer, say that it was not specified.
- If two statements conflict, explain the conflict instead of choosing silently.
- Answer naturally and concisely. Do not return JSON.

Conversation history:
{conversation}

Meeting transcript:
<<<
{transcript}
>>>

User question:
{question}

Answer:
"""

    try:
        response = chat_llm.invoke(prompt)
    except Exception as error:
        raise RuntimeError(
            f"Local meeting chat failed. Check Ollama and '{OLLAMA_CHAT_MODEL}'. "
            f"Details: {error}"
        ) from error

    answer = _response_text(response)
    if not answer:
        raise ValueError("The AI returned an empty answer.")
    return answer
