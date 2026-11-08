"""Offline unit tests for MeetMind's JSON and evidence validation helpers."""
import sys
import types
import unittest

# Unit tests exercise pure parsing/validation logic without requiring a live
# Ollama server or the optional LangChain integration at import time.
langchain_ollama_stub = types.ModuleType("langchain_ollama")
class _UnusedChatOllama:
    def __init__(self, *args, **kwargs):
        pass
    def invoke(self, *args, **kwargs):
        raise AssertionError("LLM invocation is not part of these unit tests")
langchain_ollama_stub.ChatOllama = _UnusedChatOllama
sys.modules.setdefault("langchain_ollama", langchain_ollama_stub)

from services.meeting_ai import (
    _evidence_is_present,
    _validate_analysis,
    parse_json,
)


class ParseJsonTests(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(parse_json('{"ok": true}'), {"ok": True})

    def test_parses_fenced_json(self):
        self.assertEqual(parse_json('```json\n{"count": 2}\n```'), {"count": 2})

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_json("not json")


class EvidenceTests(unittest.TestCase):
    def test_evidence_matches_case_insensitively_and_ignores_whitespace(self):
        self.assertTrue(_evidence_is_present("Rahul will finish by Nov 12", "RAHUL  will finish by Nov 12"))

    def test_evidence_must_be_present(self):
        self.assertFalse(_evidence_is_present("Rahul will finish the module", "Priya will launch campaign"))

    def test_validation_excludes_action_without_verbatim_evidence(self):
        transcript = "Rahul will finish the payment module by November 12th."
        result = {
            "summary": "Payment module deadline reviewed.",
            "objectives": ["Complete payment module"],
            "key_points": [],
            "decisions": [],
            "action_items": [
                {
                    "task": "finish the payment module",
                    "responsible_person": "Rahul",
                    "deadline": "November 12th",
                    "priority": "High",
                    "evidence": "Rahul will finish the payment module by November 12th."
                },
                {
                    "task": "launch the email campaign",
                    "responsible_person": "Priya",
                    "deadline": "November 17th",
                    "priority": "Medium",
                    "evidence": "Priya will launch the email campaign by November 17th."
                }
            ],
            "risks": [],
            "unresolved_questions": [],
            "facts": []
        }
        validated = _validate_analysis(transcript, result)
        self.assertEqual(len(validated["action_items"]), 1)
        self.assertEqual(validated["action_items"][0]["responsible_person"], "Rahul")
        self.assertTrue(validated["action_items"][0]["evidence_verified"])
        self.assertTrue(any("excluded" in warning for warning in validated["validation"]["warnings"]))

    def test_missing_owner_is_flagged(self):
        transcript = "Complete the deployment by Friday."
        result = {
            "summary": "Deployment was discussed.",
            "objectives": ["Complete deployment"],
            "action_items": [{
                "task": "Complete deployment",
                "responsible_person": "Unknown",
                "deadline": "Friday",
                "priority": "Medium",
                "evidence": "Complete the deployment by Friday."
            }]
        }
        validated = _validate_analysis(transcript, result)
        self.assertEqual(validated["action_items"][0]["responsible_person"], "Unknown")
        self.assertTrue(any("no confirmed owner" in w for w in validated["validation"]["warnings"]))


if __name__ == "__main__":
    unittest.main()
