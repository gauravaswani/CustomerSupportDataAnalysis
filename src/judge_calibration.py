"""Two-step judge/human calibration.

Step 1 (prepare): sample N examples from a judged system's output into a CSV
with an empty human_overall_quality column.
Step 2 (score): after a human fills in human_overall_quality by hand, compute
agreement (Spearman correlation + quadratic-weighted Cohen's kappa) between
the LLM judge's overall_quality and the human's score.
"""
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

sys.path.insert(0, "src")


def prepare(system, n, seed, golden_path, out_path):
    golden = pd.read_csv(golden_path)
    preds = pd.read_csv(f"data/eval/predictions_{system}.csv")
    judge = pd.read_csv(f"data/eval/judge_{system}.csv")
    df = golden.merge(preds, on="id").merge(judge, on="id")
    sample = df.sample(n=min(n, len(df)), random_state=seed)[
        ["id", "customer_message", "pred_reply", "pred_escalate",
         "historical_brand_reply", "overall_quality", "rationale"]
    ].rename(columns={"overall_quality": "judge_overall_quality"})
    sample["human_overall_quality"] = ""
    sample.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(sample)} rows to {out_path}. Fill in human_overall_quality (1-5), then run --score.")


def score(path):
    df = pd.read_csv(path)
    df = df[df["human_overall_quality"].notna() & (df["human_overall_quality"] != "")]
    n_before = len(df)
    df = df[df["judge_overall_quality"].notna()]
    if len(df) < n_before:
        print(f"Dropping {n_before - len(df)} rows where the judge itself errored "
              f"(no judge_overall_quality) -- can't compute agreement without both scores")
    if len(df) < 5:
        print(f"Only {len(df)} scored rows -- fill in more before computing agreement.")
        return
    h = df["human_overall_quality"].astype(int)
    j = df["judge_overall_quality"].astype(int)
    rho, p = spearmanr(h, j)
    kappa = cohen_kappa_score(h, j, weights="quadratic")
    mae = float(np.abs(h - j).mean())
    exact = float((h == j).mean())
    print(f"n={len(df)}  Spearman rho={rho:.3f} (p={p:.3f})  "
          f"quadratic-weighted kappa={kappa:.3f}  MAE={mae:.2f}  exact-match={exact:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score-file", default=None)
    ap.add_argument("--system", default="agent")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--golden", default="data/eval/golden_set.csv")
    ap.add_argument("--out", default="data/eval/judge_calibration_sample.csv")
    args = ap.parse_args()

    if args.prepare:
        prepare(args.system, args.n, args.seed, args.golden, args.out)
    else:
        score(args.score_file or args.out)
