"""
prepare_corpus_1m.py -- Scale corpus to ~1M passages for embedding-migration.

Same logic as prepare_corpus.py (query-first sampling, frozen manifest),
but targets 1M docs instead of 10k. Writes to a SEPARATE directory so
the 10k corpus is untouched.

Key differences from prepare_corpus.py:
  - Outputs to /workspace/data/corpus_1m/ (not corpus/)
  - Samples 1,000,000 passages (not 10,000)
  - Applies ftfy.fix_text inline (the 10k script didn't; it was post-hoc)
  - Reuses the SAME 500 queries and their positives

Outputs to --outdir:
    corpus.parquet    row_idx, doc_id, text     (~1M passages)
    queries.parquet   row_idx, query_id, text   (500 queries, same as 10k)
    qrels.parquet     query_row, corpus_row     (integer index pairs)
    manifest.json     hashes + counts

Usage:
    python prepare_corpus_1m.py --outdir /workspace/data/corpus_1m
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import ftfy
import pandas as pd
from datasets import load_dataset

SEED = 42


def id_hash(ids) -> str:
    """Order-sensitive fingerprint of an ID column."""
    h = hashlib.sha256()
    for x in ids:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/workspace/data/corpus_1m")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-docs", type=int, default=1_000_000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if (outdir / "manifest.json").exists():
        raise SystemExit(
            f"{outdir}/manifest.json already exists. The corpus is frozen.\n"
            "Delete the directory by hand if you really mean to rebuild it, "
            "and re-run ALL embedding jobs afterwards."
        )

    t0 = time.time()

    # ---- 1. load ---------------------------------------------------------
    # Full MS MARCO: ~8.8M passages. This is a large download (~3GB) but
    # should be mostly cached from the 10k run in /workspace/hf_cache.
    print("loading msmarco corpus (8.8M passages, uses HF cache if available) ...")
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    print(f"  loaded corpus in {time.time()-t0:.0f}s")

    print("loading queries and qrels ...")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")

    corpus_df = corpus_ds.to_pandas()
    queries_df = queries_ds.to_pandas()
    qrels_df = qrels_ds.to_pandas()

    # normalise id columns to str
    corpus_df["_id"] = corpus_df["_id"].astype(str)
    queries_df["_id"] = queries_df["_id"].astype(str)
    qrels_df["query-id"] = qrels_df["query-id"].astype(str)
    qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

    print(f"  corpus={len(corpus_df):,}  queries={len(queries_df):,}  qrels={len(qrels_df):,}")

    # ---- 2. sample queries, then docs ------------------------------------
    # Same 500 queries as the 10k corpus (same seed, same sort order).
    qrels_df = qrels_df[qrels_df["score"] > 0]
    all_qids = sorted(qrels_df["query-id"].unique())
    rng = pd.Series(all_qids).sample(n=args.n_queries, random_state=SEED)
    sampled_qids = set(rng)

    sub_qrels = qrels_df[qrels_df["query-id"].isin(sampled_qids)]
    positive_ids = set(sub_qrels["corpus-id"])
    print(f"  {len(sampled_qids)} queries -> {len(positive_ids)} positive passages")

    positives = corpus_df[corpus_df["_id"].isin(positive_ids)]

    missing = positive_ids - set(positives["_id"])
    if missing:
        print(f"  WARNING: {len(missing)} positives absent from corpus, dropping their qrels")
        sub_qrels = sub_qrels[~sub_qrels["corpus-id"].isin(missing)]

    n_distractors = args.n_docs - len(positives)
    if n_distractors < 0:
        raise SystemExit("more positives than --n-docs; raise --n-docs or lower --n-queries")

    print(f"  sampling {n_distractors:,} distractors (this may take a moment) ...")
    distractors = corpus_df[~corpus_df["_id"].isin(positive_ids)].sample(
        n=n_distractors, random_state=SEED
    )

    # ---- 3. freeze order -------------------------------------------------
    print("  shuffling and freezing row order ...")
    corpus_out = (
        pd.concat([positives, distractors])
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    corpus_out = corpus_out.rename(columns={"_id": "doc_id"})

    # join title + text (MS MARCO has a title column)
    if "title" in corpus_out.columns:
        corpus_out["text"] = (
            corpus_out["title"].fillna("").str.strip()
            + " "
            + corpus_out["text"].str.strip()
        ).str.strip()
    else:
        corpus_out["text"] = corpus_out["text"].str.strip()

    corpus_out = corpus_out[["doc_id", "text"]]
    corpus_out.insert(0, "row_idx", range(len(corpus_out)))

    queries_out = (
        queries_df[queries_df["_id"].isin(sampled_qids)]
        .rename(columns={"_id": "query_id"})
        .sort_values("query_id")
        .reset_index(drop=True)[["query_id", "text"]]
    )
    queries_out.insert(0, "row_idx", range(len(queries_out)))

    # ---- 4. ftfy cleaning ------------------------------------------------
    # MS MARCO has mojibake (donâ€™t -> don't). The 10k corpus was cleaned
    # post-hoc; here we bake it in so the script is the single source of truth.
    print("  applying ftfy text cleaning ...")
    corpus_out["text"] = corpus_out["text"].map(ftfy.fix_text)
    queries_out["text"] = queries_out["text"].map(ftfy.fix_text)

    # ---- 5. qrels in pure integer index space ----------------------------
    doc_pos = dict(zip(corpus_out["doc_id"], corpus_out["row_idx"]))
    qry_pos = dict(zip(queries_out["query_id"], queries_out["row_idx"]))
    qrels_out = pd.DataFrame({
        "query_row": sub_qrels["query-id"].map(qry_pos),
        "corpus_row": sub_qrels["corpus-id"].map(doc_pos),
    }).dropna().astype(int).drop_duplicates().reset_index(drop=True)

    # ---- 6. assertions ---------------------------------------------------
    assert len(corpus_out) == args.n_docs, (
        f"expected {args.n_docs} docs, got {len(corpus_out)}"
    )
    assert corpus_out["doc_id"].is_unique, "duplicate doc_id"
    assert queries_out["query_id"].is_unique, "duplicate query_id"
    assert (corpus_out["row_idx"] == range(len(corpus_out))).all()
    assert (queries_out["row_idx"] == range(len(queries_out))).all()
    assert corpus_out["text"].str.len().min() > 0, "empty document text"
    assert len(qrels_out) > 0, "no qrels survived the join"
    assert qrels_out["corpus_row"].max() < len(corpus_out)
    assert qrels_out["query_row"].max() < len(queries_out)
    assert qrels_out["query_row"].nunique() == len(queries_out), (
        "some query lost all positives"
    )

    # ---- 7. write --------------------------------------------------------
    print(f"  writing parquet files to {outdir} ...")
    corpus_out.to_parquet(outdir / "corpus.parquet", index=False)
    queries_out.to_parquet(outdir / "queries.parquet", index=False)
    qrels_out.to_parquet(outdir / "qrels.parquet", index=False)

    manifest = {
        "seed": SEED,
        "source": "BeIR/msmarco",
        "n_corpus": len(corpus_out),
        "n_queries": len(queries_out),
        "n_qrels": len(qrels_out),
        "corpus_hash": id_hash(corpus_out["doc_id"]),
        "queries_hash": id_hash(queries_out["query_id"]),
        "ftfy_cleaned": True,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\n{json.dumps(manifest, indent=2)}")
    print(f"\nDone in {elapsed:.0f}s -> {outdir}")
    print("Do not run this script again.")


if __name__ == "__main__":
    main()
