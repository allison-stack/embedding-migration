import pandas as pd
frozen = pd.read_parquet("/workspace/data/corpus/corpus.parquet")
fresh  = pd.read_parquet("/tmp/ftfy_check/corpus.parquet")

print(len(frozen), len(fresh))
print((frozen.doc_id.values == fresh.doc_id.values).all())   # order-sensitive
print((frozen.text.values  == fresh.text.values).all())      # the real test

# if False, look at what differs
d = frozen.text.values != fresh.text.values
print(d.sum())
for f, n in list(zip(frozen.text[d], fresh.text[d]))[:5]:
    print(repr(f[:120])); print(repr(n[:120])); print("---")
