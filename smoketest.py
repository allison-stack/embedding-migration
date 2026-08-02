from sentence_transformers import SentenceTransformer
import numpy as np, time

m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
t = time.time()
v = m.encode(["hello world"] * 512, batch_size=128, show_progress_bar=True)
print(v.shape, v.dtype, f"{time.time()-t:.1f}s")
np.save("/workspace/data/embeddings/_smoketest.npy", v)
