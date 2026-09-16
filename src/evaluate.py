"""Compute automated metrics for one or more systems against the golden set:
- intent classification: accuracy, macro-F1, per-class report
- escalation decision: precision/recall/F1, confusion counts
- reply quality (automated proxy only -- NOT a substitute for the LLM judge):
  ROUGE-L-ish token overlap and TF-IDF cosine similarity against the
  brand's own historical reply to a similar/same message.

Usage: python src/evaluate.py trivial simple agent
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, "src")


def rouge_l_f1(a, b):
    a_tokens, b_tokens = a.lower().split(), b.lower().split()
    if not a_tokens or not b_tokens:
        return 0.0
    # LCS length
    m, n = len(a_tokens), len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def tfidf_cosine(texts_a, texts_b):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    all_text = list(texts_a) + list(texts_b)
    m = vec.fit_transform(all_text)
    n = len(texts_a)
    a, b = m[:n], m[n:]
    sims = np.array((a.multiply(b)).sum(axis=1)).ravel()
    norms_a = np.sqrt(np.array(a.multiply(a).sum(axis=1)).ravel())
    norms_b = np.sqrt(np.array(b.multiply(b).sum(axis=1)).ravel())
    denom = norms_a * norms_b
    denom[denom == 0] = 1e-9
    return sims / denom


def evaluate_system(name, golden, preds_path):
    preds = pd.read_csv(preds_path)
    df = golden.merge(preds, on="id", how="inner")
    if len(df) < len(golden):
        print(f"  WARNING: only {len(df)}/{len(golden)} golden rows have predictions")

    n_error = int((df["pred_intent"] == "ERROR").sum())
    if n_error:
        print(f"  NOTE: excluding {n_error}/{len(df)} rows where the system errored "
              f"(no prediction was made) -- see report for why (API quota).")
        df = df[df["pred_intent"] != "ERROR"].reset_index(drop=True)

    print(f"\n{'='*60}\nSYSTEM: {name}  (n={len(df)})\n{'='*60}")

    # --- intent classification ---
    acc = accuracy_score(df["intent_gold"], df["pred_intent"])
    macro_f1 = f1_score(df["intent_gold"], df["pred_intent"], average="macro", zero_division=0)
    print(f"\nIntent classification: accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")
    print(classification_report(df["intent_gold"], df["pred_intent"], zero_division=0))

    # --- escalation decision ---
    gold_esc = df["escalate_gold"].astype(bool)
    pred_esc = df["pred_escalate"].astype(bool)
    esc_p = precision_score(gold_esc, pred_esc, zero_division=0)
    esc_r = recall_score(gold_esc, pred_esc, zero_division=0)
    esc_f1 = f1_score(gold_esc, pred_esc, zero_division=0)
    cm = confusion_matrix(gold_esc, pred_esc, labels=[True, False])
    print(f"\nEscalation decision: precision={esc_p:.3f}  recall={esc_r:.3f}  F1={esc_f1:.3f}")
    print("Confusion matrix [rows=gold(True,False), cols=pred(True,False)]:")
    print(cm)
    missed_escalations = df[gold_esc & ~pred_esc]
    print(f"Missed escalations (gold=True, pred=False): {len(missed_escalations)} "
          f"({len(missed_escalations)/max(gold_esc.sum(),1):.1%} of true escalations)")

    # --- reply quality (automated proxy) ---
    df["rouge_l"] = [rouge_l_f1(a, b) for a, b in zip(df["pred_reply"].fillna(""), df["historical_brand_reply"].fillna(""))]
    cos = tfidf_cosine(df["pred_reply"].fillna(""), df["historical_brand_reply"].fillna(""))
    df["tfidf_cosine"] = cos
    print(f"\nReply vs. historical-reply similarity (automated proxy, NOT a quality judgment):")
    print(f"  mean ROUGE-L F1  = {df['rouge_l'].mean():.3f}")
    print(f"  mean TF-IDF cos  = {df['tfidf_cosine'].mean():.3f}")

    summary = {
        "system": name, "n": len(df),
        "intent_accuracy": acc, "intent_macro_f1": macro_f1,
        "escalation_precision": esc_p, "escalation_recall": esc_r, "escalation_f1": esc_f1,
        "missed_escalations": int(len(missed_escalations)),
        "mean_rouge_l": float(df["rouge_l"].mean()),
        "mean_tfidf_cosine": float(df["tfidf_cosine"].mean()),
    }
    df.to_csv(f"data/eval/scored_{name}.csv", index=False, encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", choices=["trivial", "simple", "agent"])
    ap.add_argument("--golden", default="data/eval/golden_set.csv")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    summaries = []
    for name in args.systems:
        summaries.append(evaluate_system(name, golden, f"data/eval/predictions_{name}.csv"))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("data/eval/summary_metrics.csv", index=False)


if __name__ == "__main__":
    main()
