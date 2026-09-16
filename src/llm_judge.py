"""LLM-as-judge for reply quality. Scores each (customer_message, draft_reply)
pair on a 1-5 rubric across four axes, using the historical brand reply as
grounding reference (not as a target to match verbatim -- paraphrase and
alternate valid solutions should not be penalized).

Deliberately a SEPARATE prompt/role from the agent's own system prompt, and
uses a different Gemini snapshot than the agent by default, to reduce
self-preference bias (see report/decision_log.md).
"""
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, "src")
from llm_client import call_json, call_json_pool, JUDGE_MODEL, GEMINI_AGENT_MODEL_POOL, BACKEND

JUDGE_SYSTEM = """You are an impartial quality auditor for a customer support \
team (SpotifyCares on Twitter). You did NOT write the draft reply you are \
about to review -- judge it as a skeptical third party, not its author.

Score the draft reply on four axes, each 1-5 (5=best):
- groundedness: does it avoid inventing facts (prices, limits, policies) not \
supported by the reference historical resolution or generic safe knowledge \
(e.g. asking for device/OS info is always safe)?
- correctness: is the classification/handling appropriate for what the \
customer actually asked?
- tone: does it match a friendly, casual brand-support voice (not stiff, not rude)?
- actionability: does the customer know what happens next (info requested, \
link given, or a clear next step)?

Also score overall_quality 1-5 as your holistic judgment (not simply the \
average of the four axes).

Respond with ONLY a JSON object, no markdown fences:
{"groundedness": <1-5 int>, "correctness": <1-5 int>, "tone": <1-5 int>, \
"actionability": <1-5 int>, "overall_quality": <1-5 int>, "rationale": "<one sentence>"}
"""


def _build_user_prompt(customer_message, draft_reply, historical_reply, escalate, escalate_reason):
    # NOTE: the agent always drafts a real public reply, even when escalate=True --
    # "escalate" means the underlying issue needs a human/account-system lookup
    # (matching how the brand itself operates: reply publicly asking to DM account
    # details, then continue privately), not "send nothing." An earlier version of
    # this prompt substituted "[N/A -- escalated instead of replying]" whenever
    # escalate=True, which caused the judge to penalize correctly-escalated replies
    # for "failing to respond" -- discovered via human calibration (see decision log).
    escalate_note = f" (this message was ALSO escalated for human/account follow-up: {escalate_reason})" if escalate else ""
    return (
        f"Customer message: {customer_message}\n\n"
        f"Agent's draft reply{escalate_note}: {draft_reply}\n\n"
        f"Reference: how this brand has resolved similar issues historically: {historical_reply}"
    )


def judge_one(customer_message, draft_reply, historical_reply, escalate, escalate_reason,
              model=JUDGE_MODEL, use_pool=False):
    user = _build_user_prompt(customer_message, draft_reply, historical_reply, escalate, escalate_reason)
    if use_pool:
        result, used_model = call_json_pool(GEMINI_AGENT_MODEL_POOL, JUDGE_SYSTEM, user, max_tokens=400)
        result["_model_used"] = used_model
        return result
    result = call_json(model, JUDGE_SYSTEM, user, max_tokens=400)
    result["_model_used"] = model
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["trivial", "simple", "agent"])
    ap.add_argument("--golden", default="data/eval/golden_set.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--match-ids-from", default=None,
                     help="Path to another predictions_*.csv; restrict judging to the ids that "
                          "have a non-ERROR prediction there, for a fair matched-N comparison "
                          "across systems when one system has fewer usable rows than another.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--use-pool", action="store_true",
                     help="Round-robin across GEMINI_AGENT_MODEL_POOL instead of a single "
                          "fixed judge model -- use when the single judge model's quota is tight.")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    preds = pd.read_csv(f"data/eval/predictions_{args.system}.csv")
    df = golden.merge(preds, on="id", how="inner")
    n_error = int((df["pred_intent"] == "ERROR").sum())
    if n_error:
        print(f"Excluding {n_error}/{len(df)} errored (no-prediction) rows from judging")
        df = df[df["pred_intent"] != "ERROR"].reset_index(drop=True)

    if args.match_ids_from:
        other = pd.read_csv(args.match_ids_from)
        good_ids = set(other[other["pred_intent"] != "ERROR"]["id"])
        before = len(df)
        df = df[df["id"].isin(good_ids)].reset_index(drop=True)
        print(f"Matched to {args.match_ids_from}: {len(df)}/{before} rows kept")

    if args.limit:
        df = df.head(args.limit)

    out_path = f"data/eval/judge_{args.system}.csv"
    cached = {}
    if args.resume:
        try:
            prev = pd.read_csv(out_path)
            for _, r in prev.iterrows():
                if pd.notna(r.get("overall_quality")):
                    cached[r["id"]] = r.to_dict()
            print(f"Resuming: {len(cached)} cached judged rows from {out_path}")
        except FileNotFoundError:
            print(f"--resume given but {out_path} doesn't exist yet, starting fresh")

    rows = []
    new_calls = 0
    t0 = time.time()
    for i, r in df.iterrows():
        if r["id"] in cached:
            rows.append(cached[r["id"]])
            continue
        if args.max_new is not None and new_calls >= args.max_new:
            print(f"Hit --max-new={args.max_new}, stopping early ({len(df) - len(rows)} rows left undone)")
            break
        try:
            j = judge_one(r["customer_message"], r["pred_reply"], r["historical_brand_reply"],
                          r["pred_escalate"], r.get("pred_escalate_reason", ""), use_pool=args.use_pool)
        except Exception as e:
            j = {"groundedness": None, "correctness": None, "tone": None,
                 "actionability": None, "overall_quality": None, "rationale": f"ERROR: {e}",
                 "_model_used": ""}
        j["id"] = r["id"]
        new_calls += 1
        rows.append(j)
        if new_calls % 10 == 0:
            print(f"  judged {len(rows)}/{len(df)}  ({new_calls} new calls, {time.time()-t0:.0f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8")
    n_scored = out["overall_quality"].notna().sum()
    print(f"Wrote {len(out)} rows ({n_scored} scored, {len(out)-n_scored} ERROR) to {out_path}")
    numeric = out[["groundedness", "correctness", "tone", "actionability", "overall_quality"]]
    print(numeric.mean())


if __name__ == "__main__":
    main()
