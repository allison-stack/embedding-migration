"""
procrustes.py

First cross-model translation: learn a linear map from model A's space
to model B's space using paired document embeddings, then translate
A's queries and retrieve against B's native doc index.

Reports recall@10(translated) / recall@10(native B) — the key ratio.

At 10k corpus this ratio should be ~0.99 (pipeline validation).
At 1M it's where models actually separate.

Usage:
    # Orthogonal Procrustes (same dim, e.g. E5 → BGE, both 768)
    python procrustes.py --source intfloat/e5-base-v2 --target BAAI/bge-base-en-v1.5

    # Least-squares (mismatched dim, e.g. E5 → Qwen3-0.6B, 768 → 1024)
    python procrustes.py --source intfloat/e5-base-v2 --target Qwen/Qwen3-Embedding-0.6B

    # All pairs
    python procrustes.py --all-pairs

    # Custom train/test split
    python procrustes.py --source ... --target ... --train-size 5000
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

EMBED_DIR = Path("/workspace/data/embeddings")
CORPUS_DIR = Path("/workspace/data/corpus")
RESULTS_DIR = Path("/workspace/data/procrustes_10k")

MODELS = [
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "Alibaba-NLP/gte-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
]


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def load_embeddings(model_name: str, split: str) -> np.ndarray:
    """Load all shards for a model/split, cast to float32."""
    slug = model_slug(model_name)
    d = EMBED_DIR / slug
    shards = sorted(d.glob(f"{split}_shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"No shards for {slug}/{split}")
    arrs = [np.load(s).astype(np.float32) for s in shards]
    return np.concatenate(arrs, axis=0)


def recall_at_k(query_emb, doc_emb, qrels_df, k=10):
    """Brute-force cosine recall@k (embeddings assumed L2-normalized)."""
    scores = query_emb @ doc_emb.T  # (n_queries, n_docs)
    hits = 0
    total = 0
    for _, row in qrels_df.iterrows():
        qi = row["query_row"]
        di = row["corpus_row"]
        top_k = np.argsort(scores[qi])[::-1][:k]
        if di in top_k:
            hits += 1
        total += 1
    return hits / total


def orthogonal_procrustes(A, B):
    """Solve M = argmin ||A @ M - B|| s.t. M orthogonal. A,B same dim."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def least_squares(A, B):
    """Solve M = argmin ||A @ M - B|| via least squares. Any dims."""
    M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return M


def run_pair(source: str, target: str, train_size: int = 5000, seed: int = 42):
    """Run Procrustes for one source→target pair. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  {source}  →  {target}")
    print(f"{'='*60}")

    # Load embeddings
    src_docs = load_embeddings(source, "corpus")
    src_queries = load_embeddings(source, "queries")
    tgt_docs = load_embeddings(target, "corpus")
    tgt_queries = load_embeddings(target, "queries")

    n_docs = src_docs.shape[0]
    assert src_docs.shape[0] == tgt_docs.shape[0], "doc count mismatch"
    assert src_queries.shape[0] == tgt_queries.shape[0], "query count mismatch"

    qrels_df = pd.read_parquet(CORPUS_DIR / "qrels.parquet")

    # Train/test split on corpus indices
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_docs)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    print(f"  train: {len(train_idx)} docs, test: {len(test_idx)} docs")

    # Filter qrels to test set only
    test_set = set(test_idx.tolist())
    test_qrels = qrels_df[qrels_df["corpus_row"].isin(test_set)].copy()

    # Remap corpus_row to position in test subset
    test_idx_list = test_idx.tolist()
    old_to_new = {old: new for new, old in enumerate(test_idx_list)}
    test_qrels = test_qrels.copy()
    test_qrels["corpus_row"] = test_qrels["corpus_row"].map(old_to_new)

    # Drop any qrels that reference docs not in test set
    test_qrels = test_qrels.dropna(subset=["corpus_row"])
    test_qrels["corpus_row"] = test_qrels["corpus_row"].astype(int)

    n_test_queries = test_qrels["query_row"].nunique()
    print(f"  test qrels: {len(test_qrels)} pairs, {n_test_queries} unique queries")

    if len(test_qrels) == 0:
        print("  SKIP — no qrels in test split")
        return None

    # Choose method based on dimensionality
    src_dim = src_docs.shape[1]
    tgt_dim = tgt_docs.shape[1]
    same_dim = src_dim == tgt_dim

    t0 = time.time()
    if same_dim:
        method = "orthogonal_procrustes"
        M = orthogonal_procrustes(src_docs[train_idx], tgt_docs[train_idx])
    else:
        method = "least_squares"
        M = least_squares(src_docs[train_idx], tgt_docs[train_idx])

    train_time = time.time() - t0
    print(f"  method: {method} ({src_dim}→{tgt_dim}), solved in {train_time:.3f}s")

    # Translate source queries into target space
    translated_queries = src_queries @ M
    # Re-normalize after translation
    norms = np.linalg.norm(translated_queries, axis=1, keepdims=True)
    translated_queries = translated_queries / np.clip(norms, 1e-8, None)

    # Retrieve: translated queries against native target test docs
    tgt_test_docs = tgt_docs[test_idx]

    recall_translated = recall_at_k(translated_queries, tgt_test_docs, test_qrels, k=10)

    # Ceiling: native target queries against native target test docs
    recall_native = recall_at_k(tgt_queries, tgt_test_docs, test_qrels, k=10)

    ratio = recall_translated / recall_native if recall_native > 0 else 0.0

    print(f"  recall@10 (translated): {recall_translated:.5f}")
    print(f"  recall@10 (native tgt): {recall_native:.5f}")
    print(f"  ratio:                  {ratio:.5f}")

    result = {
        "source": source,
        "target": target,
        "method": method,
        "src_dim": src_dim,
        "tgt_dim": tgt_dim,
        "train_docs": len(train_idx),
        "test_docs": len(test_idx),
        "test_qrels": len(test_qrels),
        "test_queries": n_test_queries,
        "recall@10_translated": round(recall_translated, 5),
        "recall@10_native_target": round(recall_native, 5),
        "ratio": round(ratio, 5),
        "train_seconds": round(train_time, 3),
        "seed": seed,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Cross-model Procrustes translation")
    ap.add_argument("--source", type=str, help="Source model name")
    ap.add_argument("--target", type=str, help="Target model name")
    ap.add_argument("--all-pairs", action="store_true",
                    help="Run all ordered pairs of models")
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_pairs:
        pairs = [(a, b) for a, b in itertools.permutations(MODELS, 2)]
    elif args.source and args.target:
        pairs = [(args.source, args.target)]
    else:
        # Default: E5 → BGE (the handoff's first requested pair)
        pairs = [("intfloat/e5-base-v2", "BAAI/bge-base-en-v1.5")]

    all_results = []
    for src, tgt in pairs:
        result = run_pair(src, tgt, train_size=args.train_size, seed=args.seed)
        if result:
            all_results.append(result)

    # Save results
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Source':<20} {'Target':<20} {'Trans':>7} {'Native':>7} {'Ratio':>7}")
    print(f"{'-'*70}")
    for r in all_results:
        src_short = r['source'].split('/')[-1][:18]
        tgt_short = r['target'].split('/')[-1][:18]
        print(f"{src_short:<20} {tgt_short:<20} "
              f"{r['recall@10_translated']:>7.5f} "
              f"{r['recall@10_native_target']:>7.5f} "
              f"{r['ratio']:>7.5f}")


if __name__ == "__main__":
    main()
