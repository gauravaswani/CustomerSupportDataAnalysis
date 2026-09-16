"""Split examples into a grounding corpus (used for retrieval + few-shot) and a
held-out eval pool (golden set is sampled only from here, so the agent never
sees an eval example's own historical reply during retrieval)."""
import pandas as pd

df = pd.read_csv("data/processed/spotify_examples.csv")
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

n_eval_pool = 3000
eval_pool = df.iloc[:n_eval_pool].copy()
corpus = df.iloc[n_eval_pool:].copy()

eval_pool.to_csv("data/processed/eval_pool.csv", index=False)
corpus.to_csv("data/processed/corpus.csv", index=False)
print(f"corpus: {len(corpus)}  eval_pool: {len(eval_pool)}")
