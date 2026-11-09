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
OLLAMA_MODEL = os.getenv(
    "MEETMIND_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
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
    reasoning=False,
    num_ctx=2048,
    num_predict=256,
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
            raise ValueError(
                "The local AI returned incomplete or invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "The local AI returned JSON, but it was not an object.")
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
        "the", "a", "an", "this", "that",
    }:
        return "Unknown"
    return owner


def _normalise_deadline(deadline: str) -> str:
    deadline = _safe_string(deadline, "Not specified")
    if deadline.casefold() in {"unknown", "none", "n/a", "not mentioned", "unspecified"}:
        return "Not specified"
    return deadline


def _is_vague_action_task(task: str) -> bool:
    """Reject action labels that contain no identifiable task object.

    Examples: "Fix it", "Complete that", and "Review". This is a
    conservative quality filter; it does not try to resolve pronouns.
    """
    words = re.findall(r"[a-z]+", _normalise(task))
    if not words:
        return True

    non_content = {
        "a", "an", "and", "the", "to", "for", "with", "you", "your",
        "it", "this", "that", "them", "these", "those", "me", "us",
        "i", "we", "by", "on", "up",
        "fix", "complete", "finish", "review", "test", "prepare",
        "publish", "send", "launch", "create", "resolve", "approve",
        "submit", "deliver", "update", "follow", "check", "finalize",
        "finalise", "implement", "build", "provide", "share", "confirm",
        "schedule", "organize", "organise", "draft", "discuss",
        "contact", "call", "pay", "monitor", "circulate",
        "process", "correct",
    }
    return not any(word not in non_content for word in words)


def _is_status_statement_task(task: str) -> bool:
    """Reject status statements mistakenly formatted as action items."""
    text = _normalise(task)
    return bool(re.match(
        r"^.+?\s+(?:is|are|was|were)\s+"
        r"(?:\d+(?:\.\d+)?\s*%\s*)?(?:complete|completed|done|finished|pending)\b",
        text,
    ))


def _normalise_date_text(value: str) -> str:
    """Normalize common written date variants for evidence comparison."""
    value = _normalise(value)
    value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value)
    value = re.sub(r"\bsept\b", "september", value)
    month_aliases = {
        "jan": "january", "feb": "february", "mar": "march",
        "apr": "april", "jun": "june", "jul": "july",
        "aug": "august", "sep": "september", "oct": "october",
        "nov": "november", "dec": "december",
    }
    for short, full in month_aliases.items():
        value = re.sub(rf"\b{short}\b", full, value)
    return re.sub(r"[,\s]+", " ", value).strip()


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
    """Split transcript into sentence-like units, ignoring timestamp decimals."""
    # Streamlit's editable transcript can include [HH:MM:SS.mmm --> ...] prefixes.
    # Remove only those metadata prefixes for rule matching; evidence is still
    # checked against the untouched original transcript.
    without_timestamps = re.sub(
        r"\[\d{2}:\d{2}:\d{2}\.\d+\s*-->\s*\d{2}:\d{2}:\d{2}\.\d+\]\s*",
        "",
        transcript,
    )
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", without_timestamps)
        if match.group(0).strip()
    ]


def _task_from_fragment(fragment: str, inherited_verb: str = "") -> str:
    task = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", fragment).strip()
    task = re.sub(r"^(?:that\s+)?", "", task, flags=re.IGNORECASE)
    task = re.sub(r"\s+(?:by|on)\s+" + _DATE + r"\s*$",
                  "", task, flags=re.IGNORECASE).strip()
    task = re.sub(r"\s+", " ", task).strip(" ,;:.!?\t\n")
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
            r"\b([A-Z][A-Za-z'-]+)\s*,\s*you are responsible for\s+(.+?)"
            r"(?:\s+(?:by|on)\s+(" + _DATE + r"))?(?=[,;.]|$)",
            sentence,
            re.IGNORECASE,
        )
        if assigned:
            add(
                _task_from_fragment(assigned.group(2)),
                assigned.group(1),
                assigned.group(3) or "Not specified",
                assigned.group(0),
            )

        # Explicit named requests. When a sentence contains multiple dated
        # clauses, keep each task paired with its own deadline.
        requested = re.match(
            r"^\s*([A-Z][A-Za-z'-]+)(?:,\s*(?:please\s+)?|\s+please\s+)(.+)$",
            sentence,
        )
        if requested:
            owner = requested.group(1)
            request_body = requested.group(2).strip()
            dated_clauses = re.split(
                r"\s+and\s+(?=(?:follow up|resolve|finish|complete|test|review|prepare|"
                r"publish|send|launch|create|approve|submit|deliver|update|fix|provide|"
                r"share|confirm|schedule|organize|organise|draft|contact|call|email|pay|"
                r"monitor|circulate|process|correct)\b)",
                request_body,
                flags=re.IGNORECASE,
            )
            clause_dates = [_DATE_RE.search(clause)
                            for clause in dated_clauses]
            # Split only when each coordinated clause has its own explicit date.
            # Otherwise preserve the complete compound commitment as one task.
            split_dated_tasks = (
                len(dated_clauses) > 1
                and all(match is not None for match in clause_dates)
            )
            clauses_to_add = dated_clauses if split_dated_tasks else [
                request_body]
            for clause in clauses_to_add:
                date_match = _DATE_RE.search(clause)
                deadline = date_match.group(
                    0) if date_match else "Not specified"
                task_fragment = clause
                if date_match:
                    task_fragment = clause[:date_match.start()].strip()
                    task_fragment = re.sub(
                        r"\s+(?:by|on)\s*$", "", task_fragment, flags=re.IGNORECASE
                    ).strip()
                task = _task_from_fragment(task_fragment)
                if task:
                    add(task, owner, deadline, sentence)

        # First-person commitments. The speaker's name is not guessed from
        # nearby dialogue; the UI can show Unknown until a person is confirmed.
        first_person = re.search(
            r"\b(I['’]ll|I will|we['’]ll|we will)\s+(.+)$", sentence, re.IGNORECASE)
        if first_person:
            body = first_person.group(2).strip().rstrip(".!? ")

            # Preserve coordinated commitments as one complete action.
            date_match = _DATE_RE.search(body)

            deadline = date_match.group(0) if date_match else "Not specified"

            clean_fragment = (
                body[:date_match.start()].strip()
                if date_match
                else body.strip()
            )

            clean_fragment = re.sub(
                r"\s+(?:by|on)\s*$",
                "",
                clean_fragment,
                flags=re.IGNORECASE,
            ).strip()

            task = _task_from_fragment(clean_fragment)

            if task and not _is_vague_action_task(task):
                add(task, "Unknown", deadline, body.strip())

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
    """Combine model results with explicit transcript candidates conservatively."""
    from difflib import SequenceMatcher

    def task_similarity(a: str, b: str) -> float:
        na, nb = _normalise(a), _normalise(b)
        if not na or not nb:
            return 0.0
        if na == nb or na in nb or nb in na:
            return 1.0
        return SequenceMatcher(None, na, nb).ratio()

    def merge_metadata(existing: dict, incoming: dict) -> dict:
        merged_item = dict(existing)
        for field in ("task", "responsible_person", "deadline", "priority", "evidence"):
            old_value = _safe_string(merged_item.get(field))
            new_value = _safe_string(incoming.get(field))
            if not old_value or old_value.casefold() in {"unknown", "not specified"}:
                if new_value:
                    merged_item[field] = new_value
        # Prefer the richer task label when one clearly contains the other.
        old_task = _safe_string(existing.get("task"))
        new_task = _safe_string(incoming.get("task"))
        if len(new_task) > len(old_task) and task_similarity(old_task, new_task) >= 0.88:
            merged_item["task"] = new_task
        return merged_item

    merged: list[dict] = []
    for raw_item in _as_list(model_items):
        if not isinstance(raw_item, dict):
            continue
        task = _safe_string(raw_item.get("task"))
        evidence = _safe_string(raw_item.get("evidence"))
        if not task or _is_vague_action_task(task) or _is_status_statement_task(task):
            continue
        if evidence and not _evidence_is_present(transcript, evidence):
            continue

        duplicate_index = None
        for index, existing in enumerate(merged):
            same_deadline = (
                _normalise_deadline(_safe_string(
                    existing.get("deadline"), "Not specified"))
                == _normalise_deadline(_safe_string(raw_item.get("deadline"), "Not specified"))
            )
            if same_deadline and task_similarity(task, _safe_string(existing.get("task"))) >= 0.90:
                duplicate_index = index
                break
        if duplicate_index is None:
            # If the model split one quoted assignment into multiple verbs,
            # consolidate only when evidence, owner, and deadline all match.
            compound_index = None
            for index, existing in enumerate(merged):
                same_evidence = (
                    _normalise(_safe_string(existing.get("evidence")))
                    == _normalise(evidence)
                    and bool(evidence)
                )
                same_owner = (
                    _normalise_owner(_safe_string(
                        existing.get("responsible_person"), "Unknown"))
                    == _normalise_owner(_safe_string(raw_item.get("responsible_person"), "Unknown"))
                )
                same_deadline = (
                    _normalise_date_text(_safe_string(
                        existing.get("deadline"), "Not specified"))
                    == _normalise_date_text(_safe_string(raw_item.get("deadline"), "Not specified"))
                )
                if same_evidence and same_owner and same_deadline:
                    compound_index = index
                    break
            if compound_index is None:
                merged.append(dict(raw_item))
            else:
                existing = dict(merged[compound_index])
                old_task = _safe_string(existing.get("task"))
                new_task = task
                old_words = old_task.split()
                new_words = new_task.split()
                old_verb = old_words[0] if old_words else ""
                new_verb = new_words[0] if new_words else ""
                old_object = " ".join(old_words[1:]).casefold()
                new_object = " ".join(new_words[1:]).casefold()
                if old_verb.casefold() != new_verb.casefold() and old_object == new_object and old_object:
                    existing["task"] = f"{old_verb} and {new_verb.lower()} {' '.join(old_words[1:])}"
                elif len(new_task) > len(old_task):
                    existing["task"] = new_task
                merged[compound_index] = existing
        else:
            merged[duplicate_index] = merge_metadata(
                merged[duplicate_index], raw_item)

    for candidate in _extract_explicit_action_candidates(transcript):
        if _is_vague_action_task(candidate["task"]):
            continue
        match_index = None
        for index, existing in enumerate(merged):
            existing_task = _safe_string(existing.get("task"))
            candidate_deadline = _normalise_deadline(candidate["deadline"])
            existing_deadline = _normalise_deadline(
                _safe_string(existing.get("deadline"), "Not specified")
            )
            deadline_compatible = (
                candidate_deadline == "not specified"
                or existing_deadline == "not specified"
                or _normalise_date_text(candidate_deadline) == _normalise_date_text(existing_deadline)
            )
            candidate_evidence = _normalise(candidate["evidence"])
            existing_evidence = _normalise(
                _safe_string(existing.get("evidence")))
            evidence_related = bool(
                candidate_evidence and existing_evidence
                and (candidate_evidence in existing_evidence or existing_evidence in candidate_evidence)
            )
            if deadline_compatible and (
                task_similarity(candidate["task"], existing_task) >= 0.84
                or evidence_related
            ):
                match_index = index
                break
        if match_index is None:
            merged.append(candidate)
            continue

        existing = dict(merged[match_index])
        if candidate["responsible_person"] != "Unknown":
            existing["responsible_person"] = candidate["responsible_person"]
        if candidate["deadline"] != "Not specified":
            existing["deadline"] = candidate["deadline"]
        # Candidate evidence is an exact transcript quote and supports its
        # deterministic owner/deadline fields. Preserve model evidence otherwise.
        if (candidate["responsible_person"] != "Unknown"
                or candidate["deadline"] != "Not specified"):
            existing["evidence"] = candidate["evidence"]
        if len(candidate["task"]) > len(_safe_string(existing.get("task"))):
            existing["task"] = candidate["task"]
        existing["priority"] = _safe_string(existing.get("priority"), "Medium")
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
            warnings.append(
                f"Fact {index} has missing or non-matching transcript evidence.")
            blocking_issues.append(
                f"Fact {index} could not be verified against the transcript.")

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
            blocking_issues.append(
                f"Action item {index} could not be verified.")
            continue

        owner = _normalise_owner(_safe_string(
            item.get("responsible_person"), "Unknown"))
        deadline = _normalise_deadline(_safe_string(
            item.get("deadline"), "Not specified"))
        priority = _safe_string(item.get("priority"), "Medium").title()
        if priority not in {"High", "Medium", "Low"}:
            priority = "Medium"

        evidence_norm = _normalise(evidence)
        owner_verified = owner != "Unknown" and _normalise(
            owner) in evidence_norm
        if not owner_verified:
            owner = "Unknown"
            warnings.append(f"Action item has no confirmed owner: {task}")

        deadline_verified = (
            deadline != "Not specified"
            and _normalise_date_text(deadline) in _normalise_date_text(evidence_norm)
        )
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
    if (
        next_meeting.casefold() in {"unknown", "none", "n/a", "not mentioned"}
        or (next_meeting.casefold() != "not specified"
            and _normalise_date_text(next_meeting) not in _normalise_date_text(transcript))
    ):
        if next_meeting.casefold() != "not specified":
            warnings.append(
                "Next-meeting date could not be matched to the transcript.")
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
    """Answer questions using only the supplied meeting transcript."""

    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Meeting transcript is unavailable.")

    if len(transcript) > MAX_CHAT_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Transcript exceeds the chat limit of "
            f"{MAX_CHAT_TRANSCRIPT_CHARS:,} characters. It was not truncated."
        )

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question.")

    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Please keep your question under {MAX_QUESTION_CHARS:,} characters."
        )

    safe_history = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            safe_history.append(f"{role}: {content[:1000]}")

    conversation = "\n".join(safe_history) or "No previous conversation."

    prompt = f"""You are MeetMind, an accurate meeting assistant.

Answer the user's question using ONLY the meeting transcript.

Rules:
- Identify ALL relevant tasks, responsibilities and follow-ups.
- Include names, deadlines, dates, numbers and status accurately.
- Distinguish completed work from pending work.
- Include external dependencies and provider follow-ups.
- Never invent missing information.
- If information is unavailable, explicitly say it was not specified.
- Use conversation history only to understand follow-up questions.
- Do not treat conversation history as a source of facts.
- Be direct and concise. Do not show internal reasoning.
- Use numbered points when listing multiple tasks.
- Do not omit relevant tasks simply to make the answer shorter.

Conversation history:
{conversation}

Meeting transcript:
<<<
{transcript}
>>>

User question:
{question}

Provide the final answer:
"""

    try:
        response = chat_llm.invoke(prompt)
    except Exception as error:
        raise RuntimeError(
            f"Local meeting chat failed. Check Ollama and "
            f"'{OLLAMA_CHAT_MODEL}'. Details: {error}"
        ) from error

    answer = _response_text(response)

    if not answer:
        raise ValueError("The AI returned an empty answer.")

    return answer
