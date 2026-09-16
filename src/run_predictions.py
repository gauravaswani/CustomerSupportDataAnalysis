"""Run a given system over the golden set and cache its predictions to CSV.

Usage: python src/run_predictions.py <system> [--limit N]
  system in {trivial, simple, agent}

Predictions are cached so evaluate.py never needs to re-call any model, and
so LLM cost is paid exactly once per system per golden-set run.
"""
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, "src")


def get_system(name):
    if name == "trivial":
        from baseline_trivial import TrivialBaseline
        return TrivialBaseline()
    if name == "simple":
        from baseline_simple import SimpleBaseline
        return SimpleBaseline()
    if name == "agent":
        from agent import Agent
        return Agent(k=3)  # uses llm_client.DEFAULT_MODEL
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["trivial", "simple", "agent"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--golden", default="data/eval/golden_set.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                     help="Keep existing non-ERROR rows from --out (or the default predictions "
                          "file) and only (re-)call the model for missing/ERROR rows.")
    ap.add_argument("--max-new", type=int, default=None,
                     help="Cap the number of NEW model calls this invocation makes (existing "
                          "cached rows don't count against this) -- use this to stay under a "
                          "rate/quota limit across multiple runs.")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    if args.limit:
        golden = golden.head(args.limit)

    out_path = args.out or f"data/eval/predictions_{args.system}.csv"

    cached = {}
    if args.resume:
        try:
            prev = pd.read_csv(out_path)
            for _, r in prev.iterrows():
                if r["pred_intent"] != "ERROR":
                    cached[r["id"]] = r.to_dict()
            print(f"Resuming: {len(cached)} cached good rows from {out_path}")
        except FileNotFoundError:
            print(f"--resume given but {out_path} doesn't exist yet, starting fresh")

    system = get_system(args.system)
    rows = []
    new_calls = 0
    t0 = time.time()
    for i, r in golden.iterrows():
        if r["id"] in cached:
            rows.append(cached[r["id"]])
            continue
        if args.max_new is not None and new_calls >= args.max_new:
            print(f"Hit --max-new={args.max_new}, stopping early ({len(golden) - len(rows)} rows left undone)")
            break
        try:
            result = system.handle(r["customer_message"], r.get("context_before", "") or "")
        except Exception as e:
            result = {"intent": "ERROR", "confidence": None, "escalate": None,
                      "escalate_reason": str(e), "draft_reply": ""}
        new_calls += 1
        rows.append({
            "id": r["id"],
            "pred_intent": result["intent"],
            "pred_confidence": result.get("confidence"),
            "pred_escalate": result["escalate"],
            "pred_escalate_reason": result.get("escalate_reason", ""),
            "pred_reply": result["draft_reply"],
            "model_used": result.get("_model_used", ""),
        })
        if new_calls % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {len(rows)}/{len(golden)}  ({new_calls} new calls, {elapsed:.0f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8")
    n_error = (out["pred_intent"] == "ERROR").sum()
    print(f"Wrote {len(out)} predictions ({new_calls} new calls, {n_error} still ERROR) "
          f"to {out_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
