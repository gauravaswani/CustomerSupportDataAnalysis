import pandas as pd
from keyword_bucket import bucket

df = pd.read_csv("data/processed/eval_pool.csv")
df["bucket"] = df["customer_message"].apply(bucket)
print(df["bucket"].value_counts())

PER_BUCKET = 28
frames = []
for b, g in df.groupby("bucket"):
    n = min(PER_BUCKET, len(g))
    frames.append(g.sample(n=n, random_state=7))
sample = pd.concat(frames).sample(frac=1.0, random_state=7).reset_index(drop=True)
sample.insert(0, "cand_id", range(1, len(sample) + 1))
sample.to_csv("data/processed/golden_candidates.csv", index=False)
print(f"\nWrote {len(sample)} candidates to data/processed/golden_candidates.csv")
print(sample["bucket"].value_counts())
