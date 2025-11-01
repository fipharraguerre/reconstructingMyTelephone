import librosa
import numpy as np
import pandas as pd
from pathlib import Path

src = Path(r"C:\temp\reconstructingMyTelephone")
out_csv = src / "audio_features.csv"

records = []

for audio_path in src.glob("*.m4a"):
    try:
        # Cargar audio (mono)
        y, sr = librosa.load(path=audio_path, sr=44100, mono=True)

        # Tempo (nuevo API en librosa 0.10)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # Chroma y key estimate
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_idx = int(np.argmax(chroma_mean))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]

        # Energía y brillo espectral
        rms = float(librosa.feature.rms(y=y).mean())
        spec_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

        records.append({
            "track": audio_path.stem,
            "tempo": float(tempo),
            "key": key,
            "rms": rms,
            "spectral_centroid": spec_centroid
        })
        print(f"Processed {audio_path.name}")

    except Exception as e:
        print(f"Error {audio_path.name}: {e}")

df = pd.DataFrame(records)
df.to_csv(out_csv, index=False)
print(f"\n✅ Saved {out_csv} ({len(df)} tracks)")
