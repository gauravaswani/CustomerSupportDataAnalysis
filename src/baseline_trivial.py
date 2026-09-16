"""Trivial baseline: no ML, no LLM. Whatever a support team would ship on day
one with zero engineering effort.

- intent: always predict the single most common intent in the corpus
- reply: a fixed canned template, same for every message
- escalate: always False (never escalates -- this is the point: it makes the
  automated reply-quality average look deceptively OK while its escalation
  recall is exactly 0, which is why we don't just report reply-quality alone)
"""
import pandas as pd

CANNED_REPLY = (
    "Hey! Thanks for reaching out. Could you tell us a bit more about what's "
    "going on -- what device/app version you're using, or DM us your account "
    "email if this is about your account -- and we'll take a look?"
)


def majority_intent(corpus_csv="data/processed/corpus.csv"):
    df = pd.read_csv(corpus_csv)
    from keyword_bucket import bucket
    counts = df["customer_message"].sample(n=min(5000, len(df)), random_state=0).map(bucket).value_counts()
    top = counts.index[0]
    return top if top != "unmatched" else counts.index[1]


class TrivialBaseline:
    def __init__(self, corpus_csv="data/processed/corpus.csv"):
        self.majority = majority_intent(corpus_csv)

    def handle(self, customer_message, context_before=""):
        return {
            "intent": self.majority,
            "confidence": None,
            "escalate": False,
            "escalate_reason": "",
            "draft_reply": CANNED_REPLY,
        }
