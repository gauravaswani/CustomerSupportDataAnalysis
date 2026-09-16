"""Simple baseline: classic ML, no LLM.

- intent: TF-IDF + multinomial Logistic Regression, trained on the corpus
  using keyword-bucket weak labels (we don't have hand labels for 24.7k
  corpus rows -- only the golden set is hand-labeled -- so this is a
  realistic "what you'd ship with a weekend of weak supervision" baseline).
- reply: verbatim reply from the single most similar historical message
  (TF-IDF nearest neighbor) -- no generation, so it can only ever repeat a
  real past reply, right or wrong.
- escalate: rule-based -- escalate if predicted intent is in
  ESCALATE_INTENTS, or an escalation keyword is present. This is the fixed
  intent->escalation mapping the LLM agent deliberately does NOT use (see
  taxonomy.py) -- keeping it here lets the eval show the cost of that
  simplification.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from keyword_bucket import bucket
from retrieval import Retriever
from taxonomy import ESCALATE_INTENTS, ESCALATION_KEYWORDS


class SimpleBaseline:
    def __init__(self, corpus_csv="data/processed/corpus.csv"):
        self.retriever = Retriever(corpus_csv)
        df = self.retriever.corpus.copy()
        df["weak_label"] = df["customer_message"].map(bucket)
        df = df[df["weak_label"] != "unmatched"]  # can't train on "no signal"

        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        X = self.vec.fit_transform(df["norm_text"] if "norm_text" in df else df["customer_message"].str.lower())
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(X, df["weak_label"])

    def handle(self, customer_message, context_before=""):
        X = self.vec.transform([customer_message.lower()])
        intent = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X).max()

        nn = self.retriever.nearest(customer_message)
        reply = nn["brand_reply"]

        low = customer_message.lower()
        escalate = intent in ESCALATE_INTENTS or any(kw in low for kw in ESCALATION_KEYWORDS)
        reason = "intent requires account/payment access" if intent in ESCALATE_INTENTS else (
            "escalation keyword matched" if escalate else "")

        return {
            "intent": intent,
            "confidence": float(proba),
            "escalate": bool(escalate),
            "escalate_reason": reason,
            "draft_reply": reply,
        }
