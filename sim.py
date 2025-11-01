import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Config
emb_dir = Path(r"C:\temp\reconstructingMyTelephone\embeddings")

# Leer todos los embeddings
embeddings = []
names = []

for npy_file in sorted(emb_dir.glob("*.npy")):
    emb = np.load(npy_file)
    embeddings.append(emb)
    names.append(npy_file.stem)

embeddings = np.vstack(embeddings)  # matriz (N, 384)
print(f"Loaded {len(names)} embeddings")

# Matriz de similitud coseno
sim = cosine_similarity(embeddings)

# Calcular una "ruta" simple: empezamos desde el primero y vamos siempre al más parecido no visitado
n = len(names)
visited = [0]
while len(visited) < n:
    last = visited[-1]
    unvisited = [i for i in range(n) if i not in visited]
    next_idx = max(unvisited, key=lambda i: sim[last, i])
    visited.append(next_idx)

# Exportar orden resultante
ordered_names = [names[i] for i in visited]
df = pd.DataFrame({"order": range(1, n + 1), "track": ordered_names})
df.to_csv(emb_dir / "semantic_order.csv", index=False)

print("✅ Saved semantic_order.csv")
