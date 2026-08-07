#!/bin/bash
set +e
cd /workspace/code/embedding-migration

# 1M corpus embedding on RTX PRO 6000 WK (96GB VRAM)
# Batch sizes bumped vs 10k run to exploit the extra VRAM.
# Each run writes to /workspace/data/embeddings_1m/{model_slug}/
# Expect ~4-6 hours total wall time.

CORPUS="/workspace/data/corpus_1m/corpus.parquet"
QUERIES="/workspace/data/corpus_1m/queries.parquet"
OUTBASE="/workspace/data/embeddings_1m"

mkdir -p "$OUTBASE"

models=(
  "intfloat/e5-base-v2|512"
  "BAAI/bge-base-en-v1.5|512"
  "Alibaba-NLP/gte-base-en-v1.5|512"
  "nomic-ai/nomic-embed-text-v1.5|512"
  "Qwen/Qwen3-Embedding-0.6B|256"
  "Qwen/Qwen3-Embedding-4B|64"
)

FORCE=""
if [ "$1" = "--force" ]; then
  FORCE="--force"
  echo "Force mode: will re-embed all models"
fi

start_time=$(date +%s)

for entry in "${models[@]}"; do
  IFS='|' read -r model bs <<< "$entry"
  slug=$(echo "$model" | tr '/' '_')
  outdir="$OUTBASE/$slug"

  for split in doc query; do
    if [ "$split" = "doc" ]; then
      input="$CORPUS"
    else
      input="$QUERIES"
    fi

    echo ""
    echo "=== $model : $split (batch=$bs) ==="
    echo "    started: $(date)"
    python3 embed.py --model "$model" \
      --input "$input" \
      --output "$outdir/" \
      --batch-size "$bs" \
      --prefix-type "$split" $FORCE || echo "  (skipped or failed)"
    echo "    finished: $(date)"
  done
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "=== All done in ${elapsed}s ==="
echo ""

# Sidecar summary
echo "=== Sidecar summary ==="
for f in "$OUTBASE"/*/; do
  echo "--- $(basename "$f") ---"
  for s in "$f"*_manifest.json; do
    [ -f "$s" ] && python3 -c "
import json; d=json.load(open('$s'))
print(f\"  {d['split']:>7s}: {d['n_rows']} rows, dim {d['dim']}, {d.get('rows_per_sec','?')} rows/sec\")
"
  done
done
