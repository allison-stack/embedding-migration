"""
embed.py

Encodes a frozen parquet (corpus.parquet or queries.parquet) with one
embedding model and writes float16 .npy shards plus a JSON sidecar.

ROW ORDER IS THE ONLY THING LINKING VECTORS TO TEXT. Nothing here may
shuffle, drop, or reorder rows. The manifest hash check and the row-count
assertion exist to make a violation crash instead of quietly costing days.

Usage:
    python embed.py --model intfloat/e5-base-v2 \
        --input /workspace/data/corpus/corpus.parquet \
        --output /workspace/data/embeddings/e5-base-v2 \
        --batch-size 256 --prefix-type doc
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from prepare_corpus import id_hash  # import, never reimplement

SHARD_ROWS = 50_000
MAX_SEQ_LENGTH = 512

QWEN_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

# "" means "verified: this model takes no prefix".
# A missing key means "nobody has checked" and must raise.
PREFIXES = {
    "intfloat/e5-base-v2":            {"query": "query: ", "doc": "passage: "},
    "BAAI/bge-base-en-v1.5":          {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
    "Alibaba-NLP/gte-base-en-v1.5":   {"query": "", "doc": ""},
    "nomic-ai/nomic-embed-text-v1.5": {"query": "search_query: ", "doc": "search_document: "},
    "Qwen/Qwen3-Embedding-0.6B":      {"query": QWEN_PROMPT, "doc": ""},
    "Qwen/Qwen3-Embedding-4B":        {"query": QWEN_PROMPT, "doc": ""},
}

EXPECTED_DIM = {
    "intfloat/e5-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "Alibaba-NLP/gte-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--prefix-type", required=True, choices=["query", "doc"])
    ap.add_argument("--force", action="store_true", help="re-embed even if sidecar exists")
    args = ap.parse_args()

    if args.model not in PREFIXES:
        raise SystemExit(f"unknown model {args.model}; add it to PREFIXES and EXPECTED_DIM")
    prefix = PREFIXES[args.model][args.prefix_type]

    inpath = Path(args.input)
    split = inpath.stem                      # "corpus" or "queries"
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    sidecar_path = outdir / f"{split}_manifest.json"

    if sidecar_path.exists() and not args.force:
        raise SystemExit(f"{sidecar_path} exists; pass --force to re-embed")

    # ---- 1. load and verify alignment -----------------------------------
    df = pd.read_parquet(inpath)
    id_col = "doc_id" if "doc_id" in df.columns else "query_id"
    key = "corpus_hash" if id_col == "doc_id" else "queries_hash"

    observed = id_hash(df[id_col])
    manifest = json.loads((inpath.parent / "manifest.json").read_text())
    if observed != manifest[key]:
        raise SystemExit(
            f"{key} mismatch: {observed} != {manifest[key]}\n"
            "The corpus has drifted. Do NOT embed against it."
        )
    print(f"{split}: {len(df):,} rows, {key}={observed} OK")

    texts = [prefix + t for t in df["text"].tolist()]

    # ---- 2. load model ---------------------------------------------------
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda")
    model.max_seq_length = MAX_SEQ_LENGTH
    revision = model_info(args.model).sha

    # Qwen3 carries its own query prompt in config_sentence_transformers.json.
    # Prefer the model's own mechanism over hand-concatenation.
    use_prompt_name = (
        args.model.startswith("Qwen/Qwen3-Embedding")
        and args.prefix_type == "query"
        and "query" in getattr(model, "prompts", {})
    )
    if use_prompt_name:
        texts = df["text"].tolist()          # undo manual prefix
        prefix = model.prompts["query"]      # record what ST will actually use
        print("using model.prompts['query'] instead of manual prefix")

    # ---- 3. encode in shard-sized chunks (resumable) ---------------------
    t0 = time.time()
    shards, total = [], 0
    for i, start in enumerate(range(0, len(texts), SHARD_ROWS)):
        chunk = texts[start:start + SHARD_ROWS]
        path = outdir / f"{split}_shard_{i:04d}.npy"

        if path.exists() and np.load(path, mmap_mode="r").shape[0] == len(chunk):
            print(f"  shard {i:04d} present, skipping")
            shards.append(path)
            total += len(chunk)
            continue

        kwargs = dict(
            batch_size=args.batch_size,
            normalize_embeddings=True,       # done in float32, before the cast
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        if use_prompt_name:
            kwargs["prompt_name"] = "query"

        emb = model.encode(chunk, **kwargs)
        assert emb.dtype == np.float32, f"expected float32 from encode, got {emb.dtype}"
        assert emb.shape[0] == len(chunk), f"{emb.shape[0]} vectors for {len(chunk)} texts"
        assert emb.shape[1] == EXPECTED_DIM[args.model], (
            f"dim {emb.shape[1]} != expected {EXPECTED_DIM[args.model]}; wrong revision?"
        )
        np.save(path, emb.astype(np.float16))
        shards.append(path)
        total += emb.shape[0]

    elapsed = time.time() - t0

    # ---- 4. final alignment assertion ------------------------------------
    assert total == len(df), f"wrote {total} vectors for {len(df)} rows"

    # ---- 5. sidecar ------------------------------------------------------
    sidecar = {
        "model": args.model,
        "revision": revision,
        "split": split,
        "dim": EXPECTED_DIM[args.model],
        "normalized": True,
        "dtype": "float16",
        "prefix_type": args.prefix_type,
        "prefix": prefix,                    # literal string, not a boolean
        "n_rows": total,
        "n_shards": len(shards),
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": args.batch_size,
        key: observed,
        "seconds": round(elapsed, 1),
        "rows_per_sec": round(total / elapsed, 1),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(json.dumps(sidecar, indent=2))
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()
