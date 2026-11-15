import db
import pipeline


def _w(text, start, end, unclear=False):
    return {"text": text, "start": start, "end": end, "unclear": unclear}


def test_build_transcript_assigns_speakers_by_word():
    words = [_w(" Hello", 0.0, 0.5), _w(" there", 0.5, 1.0),
             _w(" Hi", 5.0, 5.5), _w(" back", 5.5, 6.0)]
    turns = [(0.0, 2.0, "SPEAKER_00"), (4.0, 7.0, "SPEAKER_01")]
    lines = pipeline.build_transcript(words, turns).splitlines()
    assert lines[0] == "[00:00] SPEAKER_00: Hello there"
    assert lines[1] == "[00:05] SPEAKER_01: Hi back"


def test_single_word_blip_is_smoothed():
    words = [_w(" we", 0.0, 0.5), _w(" agree", 0.5, 1.0), _w(" yes", 1.0, 1.3),
             _w(" and", 1.3, 1.8), _w(" more", 1.8, 2.3), _w(" words", 2.3, 2.8)]
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 1.3,
                                        "SPEAKER_01"), (1.3, 3.0, "SPEAKER_00")]
    out = pipeline.build_transcript(words, turns)
    assert out == "[00:00] SPEAKER_00: we agree yes and more words"


def test_real_interruption_is_kept():
    words = [_w(" so", 0.0, 0.5), _w(" we", 0.5, 1.0),
             _w(" yeah", 1.0, 1.3), _w(" exactly",
                                       1.3, 1.8), _w(" right", 1.8, 2.2),
             _w(" ok", 2.2, 2.6), _w(" good", 2.6, 3.0)]
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.2,
                                        "SPEAKER_01"), (2.2, 4.0, "SPEAKER_00")]
    lines = pipeline.build_transcript(words, turns).splitlines()
    assert len(lines) == 3
    assert "SPEAKER_01: yeah exactly right" in lines[1]


def test_unclear_marker():
    words = [_w(" mumble", 0.0, 0.5, True), _w(" noise", 0.5, 1.0, True)]
    out = pipeline.build_transcript(words, [(0.0, 2.0, "SPEAKER_00")])
    assert out.endswith("[unclear]")


def test_rename_speakers_whole_word_only():
    text = "SPEAKER_01 said hi. SPEAKER_010 stayed."
    result = pipeline.rename_speakers(text, {"SPEAKER_01": "Rahul"})
    assert result == "Rahul said hi. SPEAKER_010 stayed."


TRANSCRIPT = ("[00:00] SPEAKER_00: hello there\n"
              "[00:05] SPEAKER_01: hi\n"
              "[00:09] SPEAKER_00: bye now friend")


def test_list_speakers():
    assert pipeline.list_speakers(TRANSCRIPT) == ["SPEAKER_00", "SPEAKER_01"]


def test_speaking_share():
    assert pipeline.speaking_share(TRANSCRIPT) == {
        "SPEAKER_00": 5, "SPEAKER_01": 1}


def test_chunk_text_respects_limit():
    text = "\n".join(["x" * 100] * 30)
    chunks = pipeline.chunk_text(text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_parse_action_bullets():
    section = ("- **Rahul** - fix bugs - Friday\n"
               "- **Daniel** - send newsletter - no deadline mentioned")
    items = pipeline._parse_bullets(section)
    assert items[0] == {"owner": "Rahul",
                        "task": "fix bugs", "deadline": "Friday"}
    assert items[1]["owner"] == "Daniel" and items[1]["deadline"] == ""


def test_database_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    mid = db.save_meeting("Test", "f.wav", 12.0, "[00:00] SPEAKER_00: hi", "summary",
                          [{"owner": "A", "task": "x", "deadline": ""}])
    meeting = db.get_meeting(mid)
    assert meeting["title"] == "Test"
    assert meeting["action_items"][0]["owner"] == "A"
    db.add_chat(mid, "user", "question")
    assert db.get_chat(mid)[0]["content"] == "question"
    db.delete_meeting(mid)
    assert db.get_meeting(mid) is None
