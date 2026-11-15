import subprocess
import tempfile
import wave
from pathlib import Path

# (voice, line). Three different voices = three "people".
SCRIPT = [
    ("Samantha", "Alright everyone, let's start the weekly project sync. Today we need to cover the website launch, the budget, and the customer feedback."),
    ("Alex", "Sure, I'll start. Hi, I'm Rahul. The website design is finished, but the payment page still has a few bugs. I will fix those bugs by Friday."),
    ("Samantha", "Great, thanks Rahul. Daniel, how is the marketing side going?"),
    ("Daniel", "Hi, this is Daniel. The email campaign is ready, but I can't send it until the launch date is confirmed. Once it is, I will send the newsletter the following Tuesday."),
    ("Samantha", "Okay. Let's confirm the launch date then. Rahul, if you finish the fixes by Friday, can we launch on the twentieth?"),
    ("Alex", "Yes, the twentieth works for me, as long as testing finishes on Monday."),
    ("Samantha", "Perfect. Then we have decided that the launch date is the twentieth. Now the budget. We need an extra two thousand dollars for advertising."),
    ("Daniel", "That is correct. Without the extra budget, I can only run ads for one week instead of three."),
    ("Samantha", "I'm Priya, by the way, and I will ask the finance team to approve the extra two thousand dollars by Wednesday."),
    ("Alex", "Another thing. Last month's customer survey results are still not summarized. I can do that, and I will share a short report by next Monday."),
    ("Daniel", "That would be helpful. I would like to use those results in the newsletter, so please include the top three customer complaints."),
    ("Samantha", "Good. So to recap. Rahul fixes the payment page by Friday and sends the survey report on Monday. Daniel sends the newsletter on Tuesday after launch. I will get the budget approved by Wednesday. Thanks everyone, meeting closed."),
]


def main():
    tmp = Path(tempfile.mkdtemp())
    wavs = []
    for i, (voice, text) in enumerate(SCRIPT):
        aiff, wav = tmp / f"{i}.aiff", tmp / f"{i}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", str(wav)],
            check=True, capture_output=True,
        )
        wavs.append(wav)

    silence = b"\x00\x00" * int(16000 * 0.7)
    with wave.open("demo_meeting.wav", "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        for w in wavs:
            with wave.open(str(w), "rb") as f:
                out.writeframes(f.readframes(f.getnframes()))
            out.writeframes(silence)
    print("Created demo_meeting.wav")


main()
