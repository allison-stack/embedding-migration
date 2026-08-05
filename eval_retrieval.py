"""Compute recall@10 ceilings for each model natively."""
import numpy as np
import pandas as pd
import json, argparse, sys
from pathlib import Path

def load_embeddings(emb_dir, split):
    """Load and concatenate shards, return float32."""
    shards = sorted(Path(emb_dir).glob(f"{split}_shard_*.npy"))
    assert shards, f"No {split} shards in {emb_dir}"
    arrs = [np.load(s) for s in shards]
    return np.concatenate(arrs, axis=0).astype(np.float32)

def check_hashes(emb_dir, manifest_path):
    """Assert all sidecar hashes match the corpus manifest."""
    manifest = json.load(open(manifest_path))
    for sidecar_path in Path(emb_dir).glob("*_manifest.json"):
        sc = json.load(open(sidecar_path))
        for key in ["corpus_hash", "queries_hash"]:
            if key in sc and key in manifest:
                assert str(sc[key]) == str(manifest[key]), \
                    f"Hash mismatch in {sidecar_path.name}: {key}"

def recall_at_k(Q, D, qrels_df, k=10):
    """Q: (n_q, dim), D: (n_d, dim). Both L2-normalized → dot = cosine."""
    scores = Q @ D.T                          # (n_q, n_d)
    top_k = np.argsort(-scores, axis=1)[:, :k]  # (n_q, k)
    grouped = qrels_df.groupby("query_row")["corpus_row"].apply(set)
    total_recall = 0.0
    for q_idx, relevant in grouped.items():
        retrieved = set(top_k[q_idx].tolist())
        total_recall += len(relevant & retrieved) / len(relevant)
    return total_recall / len(grouped)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--corpus-dir", default="/workspace/data/corpus")
    args = p.parse_args()

    manifest_path = Path(args.corpus_dir) / "manifest.json"
    check_hashes(args.emb_dir, manifest_path)

    Q = load_embeddings(args.emb_dir, "queries")
    D = load_embeddings(args.emb_dir, "corpus")
    qrels = pd.read_parquet(Path(args.corpus_dir) / "qrels.parquet")

    r10 = recall_at_k(Q, D, qrels, k=10)
    print(f"recall@10 = {r10:.5f}  ({args.emb_dir})")
