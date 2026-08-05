#!/bin/bash
set +e

models=(
  "intfloat/e5-base-v2|256"
  "BAAI/bge-base-en-v1.5|256"
  "Alibaba-NLP/gte-base-en-v1.5|256"
  "nomic-ai/nomic-embed-text-v1.5|256"
  "Qwen/Qwen3-Embedding-0.6B|128"
  "Qwen/Qwen3-Embedding-4B|32"
)

FORCE=""
if [ "$1" = "--force" ]; then
  FORCE="--force"
  echo "Force mode: will re-embed all models"
fi

for entry in "${models[@]}"; do
  IFS='|' read -r model bs <<< "$entry"
  slug=$(echo "$model" | tr '/' '_')
  outdir="/workspace/data/embeddings/$slug"

  for split in doc query; do
    if [ "$split" = "doc" ]; then
      input="/workspace/data/corpus/corpus.parquet"
    else
      input="/workspace/data/corpus/queries.parquet"
    fi

    echo ""
    echo "=== $model : $split ==="
    python3 embed.py --model "$model" \
      --input "$input" \
      --output "$outdir/" \
      --batch-size "$bs" \
      --prefix-type "$split" $FORCE || echo "  (skipped or failed)"
  done
done

echo ""
echo "=== All done. Sidecar summary: ==="
for f in /workspace/data/embeddings/*/; do
  echo "--- $(basename "$f") ---"
  for s in "$f"*_manifest.json; do
    [ -f "$s" ] && python3 -c "import json; d=json.load(open('$s')); print(f\"  {d['split']:>7s}: {d['n_rows']} rows, dim {d['dim']}, {d['rows_per_sec']} rows/sec\")"
  done
done
