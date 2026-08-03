"""
prepare_corpus.py

Builds a FROZEN 10k-passage corpus from MS MARCO plus the queries and
relevance judgments that point into it.

Run this ONCE. After it succeeds, never run it again -- every embedding
array you produce is aligned to these files by ROW ORDER alone.

Outputs to --outdir:
    corpus.parquet    row_idx, doc_id, text     (10k passages)
    queries.parquet   row_idx, query_id, text   (~500 queries)
    qrels.parquet     query_row, corpus_row     (integer index pairs)
    manifest.json     hashes + counts, used to detect drift later

Usage:
    python prepare_corpus.py --outdir /workspace/data/corpus
"""

import argparse
import hashlib
import json
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
        h.update(b"\x00")  # delimiter, so ["ab","c"] != ["a","bc"]
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-docs", type=int, default=10_000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if (outdir / "manifest.json").exists():
        raise SystemExit(
            f"{outdir}/manifest.json already exists. The corpus is frozen.\n"
            "Delete the directory by hand if you really mean to rebuild it, "
            "and re-run ALL embedding jobs afterwards."
        )

    # ---- 1. load ---------------------------------------------------------
    # Verify these config names against the dataset page before your first
    # run; HF repos get restructured and these have moved before.
    print("loading msmarco (this pulls several GB, be patient) ...")
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")

    corpus_df = corpus_ds.to_pandas()
    queries_df = queries_ds.to_pandas()
    qrels_df = qrels_ds.to_pandas()

    # normalise id columns to str -- they arrive as a mix of int and str
    corpus_df["_id"] = corpus_df["_id"].astype(str)
    queries_df["_id"] = queries_df["_id"].astype(str)
    qrels_df["query-id"] = qrels_df["query-id"].astype(str)
    qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

    print(f"  corpus={len(corpus_df):,}  queries={len(queries_df):,}  qrels={len(qrels_df):,}")

    # ---- 2. sample queries FIRST, then the docs they point at ------------
    # Sampling docs at random would give a ~0.1% chance of catching any
    # labelled passage, and every recall number would come out at zero.
    qrels_df = qrels_df[qrels_df["score"] > 0]
    all_qids = sorted(qrels_df["query-id"].unique())          # sorted => reproducible
    rng = pd.Series(all_qids).sample(n=args.n_queries, random_state=SEED)
    sampled_qids = set(rng)

    sub_qrels = qrels_df[qrels_df["query-id"].isin(sampled_qids)]
    positive_ids = set(sub_qrels["corpus-id"])
    print(f"  {len(sampled_qids)} queries -> {len(positive_ids)} positive passages")

    positives = corpus_df[corpus_df["_id"].isin(positive_ids)]

    # sanity: every positive must actually exist in the corpus
    missing = positive_ids - set(positives["_id"])
    if missing:
        print(f"  WARNING: {len(missing)} positives absent from corpus, dropping their qrels")
        sub_qrels = sub_qrels[~sub_qrels["corpus-id"].isin(missing)]

    n_distractors = args.n_docs - len(positives)
    if n_distractors < 0:
        raise SystemExit("more positives than --n-docs; raise --n-docs or lower --n-queries")

    distractors = corpus_df[~corpus_df["_id"].isin(positive_ids)].sample(
        n=n_distractors, random_state=SEED
    )

    # ---- 3. freeze order -------------------------------------------------
    corpus_out = (
        pd.concat([positives, distractors])
        .sample(frac=1.0, random_state=SEED)   # shuffle so positives aren't all at the top
        .reset_index(drop=True)
    )
    corpus_out = corpus_out.rename(columns={"_id": "doc_id"})
    title = corpus_out["title"] if "title" in corpus_out.columns else pd.Series("", index=corpus_out.index)
    corpus_out["text"] = (
        title.fillna("").str.strip() + " " + corpus_out["text"].str.strip()
    ).str.strip()
    corpus_out["text"] = corpus_out["text"].map(ftfy.fix_text)
    corpus_out = corpus_out[["doc_id", "text"]]
    corpus_out.insert(0, "row_idx", range(len(corpus_out)))

    queries_out = (
        queries_df[queries_df["_id"].isin(sampled_qids)]
        .rename(columns={"_id": "query_id"})
        .sort_values("query_id")               # deterministic order
        .reset_index(drop=True)[["query_id", "text"]]
    )
    queries_out["text"] = queries_out["text"].map(ftfy.fix_text)
    queries_out.insert(0, "row_idx", range(len(queries_out)))

    # ---- 4. qrels in pure integer index space ----------------------------
    doc_pos = dict(zip(corpus_out["doc_id"], corpus_out["row_idx"]))
    qry_pos = dict(zip(queries_out["query_id"], queries_out["row_idx"]))
    qrels_out = pd.DataFrame({
        "query_row": sub_qrels["query-id"].map(qry_pos),
        "corpus_row": sub_qrels["corpus-id"].map(doc_pos),
    }).dropna().astype(int).drop_duplicates().reset_index(drop=True)

    # ---- 5. assertions ---------------------------------------------------
    assert corpus_out["doc_id"].is_unique, "duplicate doc_id"
    assert queries_out["query_id"].is_unique, "duplicate query_id"
    assert (corpus_out["row_idx"] == range(len(corpus_out))).all()
    assert (queries_out["row_idx"] == range(len(queries_out))).all()
    assert corpus_out["text"].str.len().min() > 0, "empty document text"
    assert len(qrels_out) > 0, "no qrels survived the join"
    assert qrels_out["corpus_row"].max() < len(corpus_out)
    assert qrels_out["query_row"].max() < len(queries_out)
    # every sampled query must keep at least one positive
    assert qrels_out["query_row"].nunique() == len(queries_out), "some query lost all positives"

    # ---- 6. write --------------------------------------------------------
    corpus_out.to_parquet(outdir / "corpus.parquet", index=False)
    queries_out.to_parquet(outdir / "queries.parquet", index=False)
    qrels_out.to_parquet(outdir / "qrels.parquet", index=False)

    manifest = {
        "seed": SEED,
        "source": "BeIR/msmarco",
        "corpus_hash": id_hash(corpus_out["doc_id"]),
        "queries_hash": id_hash(queries_out["query_id"]),
        "n_corpus": len(corpus_out),
        "n_queries": len(queries_out),
        "n_qrels": len(qrels_out),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\nfrozen -> {outdir}\nDo not run this script again.")


if __name__ == "__main__":
    main()
