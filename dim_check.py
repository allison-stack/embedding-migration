from embed import EXPECTED_DIM
# If there's no EXPECTED_DIM dict, just do a quick manual check:
from sentence_transformers import SentenceTransformer
import os
models = {
    'intfloat/e5-base-v2': 768,
    'BAAI/bge-base-en-v1.5': 768,
    'Alibaba-NLP/gte-base-en-v1.5': 768,
    'nomic-ai/nomic-embed-text-v1.5': 768,
    'Qwen/Qwen3-Embedding-0.6B': 1024,
    'Qwen/Qwen3-Embedding-4B': 2560,
}
for name, expected in models.items():
    m = SentenceTransformer(name, cache_folder=os.environ['HF_HOME']+'/hub', trust_remote_code=True)
    actual = m.get_sentence_embedding_dimension()
    status = '✓' if actual == expected else '✗ MISMATCH'
    print(f'{status}  {name}: expected {expected}, got {actual}')
    del m
