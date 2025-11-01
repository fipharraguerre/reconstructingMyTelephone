import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths
emb_dir = Path(r"C:\temp\reconstructingMyTelephone\embeddings")
order_file = emb_dir / "semantic_order.csv"

# Leer embeddings y orden
df = pd.read_csv(order_file)
embeddings = []
for name in df["track"]:
    emb = np.load(emb_dir / f"{name}.npy")
    embeddings.append(emb)
embeddings = np.vstack(embeddings)

# Calcular matriz de similitud coseno (ya están normalizados)
sim = np.dot(embeddings, embeddings.T)

# Graficar
plt.figure(figsize=(8, 8))
plt.imshow(sim, cmap="magma", origin="lower")
plt.title("Self-Similarity Matrix (semantic order)")
plt.xlabel("Track index")
plt.ylabel("Track index")
plt.colorbar(label="Cosine similarity")
plt.tight_layout()
plt.savefig(emb_dir / "SSM_semantic_order.png", dpi=300)
plt.show()
