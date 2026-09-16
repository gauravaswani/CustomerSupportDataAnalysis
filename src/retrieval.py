"""TF-IDF retrieval over the grounding corpus of historically-resolved
SpotifyCares threads. No LLM involved -- this is what grounds the agent's
reply in "how this brand has actually resolved similar issues", and it's also
reused as-is for the simple (non-LLM) baseline's reply generation.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def _norm(t):
    t = URL_RE.sub(" ", str(t))
    t = MENTION_RE.sub(" ", t)
    return t.lower()


class Retriever:
    def __init__(self, corpus_csv="data/processed/corpus.csv"):
        self.corpus = pd.read_csv(corpus_csv)
        self.corpus["norm_text"] = self.corpus["customer_message"].map(_norm)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.corpus["norm_text"])

    def top_k(self, query_text, k=3):
        q = self.vectorizer.transform([_norm(query_text)])
        sims = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            row = self.corpus.iloc[i]
            out.append({
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
                "similarity": float(sims[i]),
            })
        return out

    def nearest(self, query_text):
        return self.top_k(query_text, k=1)[0]
