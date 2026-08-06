"""
sanity_checks.py

Four checks that catch silent embedding errors before they get baked
into expensive 1M-scale runs.

Usage:
    python sanity_checks.py --check prefix          # most critical
    python sanity_checks.py --check dim determinism prefix scifact
    python sanity_checks.py --check all

Runs on cheapest GPU (RTX 2000 Ada / RTX A4500, $0.24/hr). ~20 min total.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── shared constants (mirrored from embed.py) ────────────────────────

MODELS = [
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "Alibaba-NLP/gte-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
]

EXPECTED_DIM = {
    "intfloat/e5-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "Alibaba-NLP/gte-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
}

QWEN_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

PREFIXES = {
    "intfloat/e5-base-v2":            {"query": "query: ",  "doc": "passage: "},
    "BAAI/bge-base-en-v1.5":          {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "Alibaba-NLP/gte-base-en-v1.5":   {"query": "",         "doc": ""},
    "nomic-ai/nomic-embed-text-v1.5": {"query": "search_query: ", "doc": "search_document: "},
    "Qwen/Qwen3-Embedding-0.6B":      {"query": QWEN_PROMPT, "doc": ""},
    "Qwen/Qwen3-Embedding-4B":        {"query": QWEN_PROMPT, "doc": ""},
}

# Models where prefix matters for retrieval quality
# GTE has no prefix so skip it; Qwen uses prompt_name internally so
# a manual prefix test doesn't apply cleanly.
PREFIX_TESTABLE = [
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]

EMBED_DIR = Path("/workspace/data/embeddings")
CORPUS_DIR = Path("/workspace/data/corpus")


def model_slug(name: str) -> str:
    return name.replace("/", "_")


# ── CHECK 1: dim ─────────────────────────────────────────────────────

def check_dim():
    """Assert every saved embedding has the expected dimensionality."""
    print("\n=== CHECK: dim ===")
    ok = True
    for m in MODELS:
        slug = model_slug(m)
        for split in ["corpus", "queries"]:
            npy = EMBED_DIR / slug / f"{split}_shard_0000.npy"
            if not npy.exists():
                print(f"  SKIP {slug}/{split} — file missing")
                continue
            arr = np.load(npy, mmap_mode="r")
            actual = arr.shape[1]
            expected = EXPECTED_DIM[m]
            status = "OK" if actual == expected else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {status}  {slug}/{split}  dim={actual}  expected={expected}  rows={arr.shape[0]}")
    return ok


# ── CHECK 2: determinism ─────────────────────────────────────────────

def check_determinism():
    """Embed 10 texts twice with each model; cosine similarity should be ~1.0."""
    from sentence_transformers import SentenceTransformer

    print("\n=== CHECK: determinism ===")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Embedding models must produce consistent outputs.",
        "Quantum computing may transform cryptography.",
        "The capital of France is Paris.",
        "Machine learning requires large datasets.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Neural networks are loosely inspired by biology.",
        "The stock market closed higher on Tuesday.",
        "Photosynthesis converts light into chemical energy.",
        "A well-written README saves hours of onboarding.",
    ]

    ok = True
    for m in MODELS:
        prefix = PREFIXES[m]["doc"]
        prefixed = [prefix + t for t in texts]

        model = SentenceTransformer(m, trust_remote_code=True, device="cuda")
        model.max_seq_length = 512

        a = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        b = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)

        # cosine of L2-normed vectors = dot product
        cosines = np.sum(a * b, axis=1)
        min_cos = cosines.min()
        mean_cos = cosines.mean()
        status = "OK" if min_cos > 0.9999 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status}  {model_slug(m)}  min_cos={min_cos:.6f}  mean_cos={mean_cos:.6f}")

        del model
        import torch; torch.cuda.empty_cache()

    return ok


# ── CHECK 3: prefix ──────────────────────────────────────────────────

def check_prefix():
    """
    For models with query/doc prefixes, embed 100 query-doc pairs
    WITH and WITHOUT the correct prefixes. The prefixed version should
    produce higher mean query-doc cosine similarity.

    This is the most important check — a wrong or missing prefix is
    silent and produces plausible-but-degraded embeddings.
    """
    from sentence_transformers import SentenceTransformer

    print("\n=== CHECK: prefix (most critical) ===")

    # Load 100 queries and their positive docs
    queries_df = pd.read_parquet(CORPUS_DIR / "queries.parquet")
    corpus_df = pd.read_parquet(CORPUS_DIR / "corpus.parquet")
    qrels_df = pd.read_parquet(CORPUS_DIR / "qrels.parquet")

    # Take first 100 unique queries that have a positive
    subset = qrels_df.head(100)
    q_texts = queries_df.iloc[subset["query_row"].values]["text"].tolist()
    d_texts = corpus_df.iloc[subset["corpus_row"].values]["text"].tolist()

    ok = True
    for m in PREFIX_TESTABLE:
        q_prefix = PREFIXES[m]["query"]
        d_prefix = PREFIXES[m]["doc"]

        # Skip if both prefixes are empty — nothing to test
        if q_prefix == "" and d_prefix == "":
            print(f"  SKIP {model_slug(m)} — no prefixes to test")
            continue

        model = SentenceTransformer(m, trust_remote_code=True, device="cuda")
        model.max_seq_length = 512

        # WITH prefix
        q_with = model.encode([q_prefix + t for t in q_texts],
                              normalize_embeddings=True, convert_to_numpy=True)
        d_with = model.encode([d_prefix + t for t in d_texts],
                              normalize_embeddings=True, convert_to_numpy=True)
        cos_with = np.sum(q_with * d_with, axis=1).mean()

        # WITHOUT prefix (raw text)
        q_without = model.encode(q_texts,
                                 normalize_embeddings=True, convert_to_numpy=True)
        d_without = model.encode(d_texts,
                                 normalize_embeddings=True, convert_to_numpy=True)
        cos_without = np.sum(q_without * d_without, axis=1).mean()

        gap = cos_with - cos_without
        # E5 and Nomic should show a visible gap; BGE query prefix also helps
        status = "OK" if gap > 0.005 else "WARN"
        if status == "WARN":
            ok = False
        print(f"  {status}  {model_slug(m)}  with_prefix={cos_with:.4f}  "
              f"without={cos_without:.4f}  gap={gap:.4f}")

        del model
        import torch; torch.cuda.empty_cache()

    return ok


# ── CHECK 4: scifact ─────────────────────────────────────────────────

def check_scifact():
    """
    Embed SciFact (small BEIR dataset), compute NDCG@10, compare to
    each model's MTEB leaderboard entry. Within ~2 points = correct
    pooling and prefixes. This is the check that catches
    plausible-but-wrong vectors.
    """
    print("\n=== CHECK: scifact (MTEB reproduction) ===")

    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print("  Installing beir...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "beir", "--break-system-packages", "-q"])
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader

    from sentence_transformers import SentenceTransformer

    # Load expected scores
    expected_path = Path(__file__).parent / "expected_mteb.json"
    if expected_path.exists():
        with open(expected_path) as f:
            expected_scores = json.load(f)
    else:
        expected_scores = {}
        print("  WARN: expected_mteb.json not found, will report scores without comparison")

    # Download SciFact
    dataset = "scifact"
    data_path = beir_util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
        "/workspace/data/beir"
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    ok = True
    for m in MODELS:
        q_prefix = PREFIXES[m]["query"]
        d_prefix = PREFIXES[m]["doc"]

        model = SentenceTransformer(m, trust_remote_code=True, device="cuda")
        model.max_seq_length = 512

        # Check if Qwen uses prompt_name
        use_prompt_name = (
            m.startswith("Qwen/Qwen3-Embedding")
            and "query" in getattr(model, "prompts", {})
        )

        # Encode corpus
        doc_ids = sorted(corpus.keys())
        doc_texts = [d_prefix + (corpus[did].get("title", "") + " " + corpus[did]["text"]).strip()
                     for did in doc_ids]
        doc_emb = model.encode(doc_texts, batch_size=128, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)

        # Encode queries
        q_ids = sorted(queries.keys())
        if use_prompt_name:
            q_texts = [queries[qid] for qid in q_ids]
            q_emb = model.encode(q_texts, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 prompt_name="query")
        else:
            q_texts = [q_prefix + queries[qid] for qid in q_ids]
            q_emb = model.encode(q_texts, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

        # Compute NDCG@10
        # scores: (n_queries, n_docs)
        scores = q_emb @ doc_emb.T
        ndcg_values = []
        for i, qid in enumerate(q_ids):
            if qid not in qrels:
                continue
            relevant = qrels[qid]  # {doc_id: score}
            top_k_idx = np.argsort(scores[i])[::-1][:10]
            top_k_docs = [doc_ids[j] for j in top_k_idx]

            # DCG@10
            dcg = 0.0
            for rank, did in enumerate(top_k_docs):
                if did in relevant:
                    dcg += relevant[did] / np.log2(rank + 2)

            # IDCG@10
            ideal = sorted(relevant.values(), reverse=True)[:10]
            idcg = sum(rel / np.log2(r + 2) for r, rel in enumerate(ideal))

            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

        ndcg = np.mean(ndcg_values) * 100  # as percentage

        slug = model_slug(m)
        expected_key = m  # or slug — depends on expected_mteb.json format
        # Try both key formats
        exp = expected_scores.get(m) or expected_scores.get(slug)
        if exp and "SciFact" in exp:
            exp_val = exp["SciFact"]
            diff = abs(ndcg - exp_val)
            status = "OK" if diff < 3.0 else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {status}  {slug}  NDCG@10={ndcg:.1f}  expected~{exp_val:.1f}  diff={diff:.1f}")
        else:
            print(f"  ??    {slug}  NDCG@10={ndcg:.1f}  (no expected value to compare)")

        del model
        import torch; torch.cuda.empty_cache()

    return ok


# ── main ──────────────────────────────────────────────────────────────

CHECKS = {
    "dim": check_dim,
    "determinism": check_determinism,
    "prefix": check_prefix,
    "scifact": check_scifact,
}


def main():
    ap = argparse.ArgumentParser(description="Embedding sanity checks")
    ap.add_argument("--check", nargs="+", required=True,
                    choices=list(CHECKS.keys()) + ["all"],
                    help="Which checks to run")
    args = ap.parse_args()

    to_run = list(CHECKS.keys()) if "all" in args.check else args.check

    results = {}
    for name in to_run:
        try:
            results[name] = CHECKS[name]()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  {icon}  {name}")

    if not all_ok:
        print("\n⚠ Some checks failed. Fix before scaling to 1M.")
        sys.exit(1)
    else:
        print("\n✓ All checks passed.")


if __name__ == "__main__":
    main()
