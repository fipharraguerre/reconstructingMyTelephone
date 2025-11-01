import whisper
import json
import os
from pathlib import Path

# Config
src = Path(r"C:\temp\reconstructingMyTelephone")
dst = src / "speeches"
dst.mkdir(exist_ok=True)
model = whisper.load_model("medium")

# Loop
for audio_path in src.glob("*.m4a"):
    out_json = dst / (audio_path.stem + ".json")
    if out_json.exists():
        print(f"Skip {audio_path.name} (already done)")
        continue

    print(f"Transcribing {audio_path.name}...")
    try:
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            fp16=False,
            language="en"
        )
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error with {audio_path.name}: {e}")

print("✅ All done.")
