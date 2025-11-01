from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

src = Path(r"C:\temp\reconstructingMyTelephone\speeches")
dst = Path(r"C:\temp\reconstructingMyTelephone\embeddings")
dst.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

for json_file in src.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data.get("text", "").strip()
    if not text:
        continue

    emb = model.encode(text, normalize_embeddings=True)
    out_path = dst / (json_file.stem + ".npy")
    np.save(out_path, emb)
    print(f"Saved {out_path.name} ({emb.shape})")
