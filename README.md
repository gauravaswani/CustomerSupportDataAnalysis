# SpotifyCares support agent

An AI support agent for **SpotifyCares** (Spotify's Twitter support account), built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an incoming customer message, it:

1. classifies the message into one of 9 intents,
2. drafts a reply grounded in how SpotifyCares has historically resolved similar issues
   (TF-IDF retrieval over 24.7k real historical resolutions), and
3. decides auto-handle vs. escalate-to-human, with a stated reason.

See [`report/report.md`](report/report.md) for the full write-up (problem framing, results vs.
baselines, failure analysis, what's misleading about the headline number, decision log at
[`report/decision_log.md`](report/decision_log.md)), and
[`data/eval/README.md`](data/eval/README.md) for how the 214-example golden evaluation set was
built.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (gitignored) in the repo root with your key:

```
GEMINI_API_KEY=...
```

The agent and LLM judge default to Gemini (`gemini-3.6-flash` for the agent,
`gemini-3.5-flash` for the judge -- two distinct snapshots for partial role separation;
see `report/decision_log.md`). An Anthropic backend is also supported and used automatically
if `ANTHROPIC_API_KEY` is set instead (or set `LLM_BACKEND=api`/`gemini` explicitly to force
one). See `src/llm_client.py`.

No Kaggle account is needed to reproduce the headline results: `data/processed/` and
`data/eval/golden_set.csv` are checked into the repo, already built from the raw dataset.
(`src/build_threads.py` / `prepare_examples.py` / `split_data.py` regenerate them from scratch
from the raw Kaggle CSV, if you want to re-derive everything -- that step needs
`kaggle datasets download -d thoughtvector/customer-support-on-twitter` and does take longer
than 15 minutes due to the 3M-row scan.)

## Current status

All code runs end-to-end. Numbers currently in `report/report.md` were produced with a
**free-tier Gemini key**, which caps each model at ~5 requests/minute and ~20/day -- even after
round-robining across 6 model snapshots and 2 separate keys over 3 days, only 71/214 golden-set
examples got a real agent prediction, and only 47/71 (agent) and 62/71 (trivial) got judged
(simple has 0 judge scores; that run was stopped before any calls landed). The report treats
this honestly as a partial sample rather than pretending it's a full run -- see report.md
Section 5. **With a paid/higher-quota key, the full pipeline reproduces in well under 15 minutes**
(the per-call latency itself is a few seconds; quota, not compute, was the bottleneck).

## Reproduce the headline results (< 15 min, given adequate API quota)

```bash
# 1. Non-LLM baselines (seconds)
python src/run_predictions.py trivial
python src/run_predictions.py simple

# 2. The agent (LLM calls -- a few minutes for 214 examples)
python src/run_predictions.py agent

# 3. Score everything
python src/evaluate.py trivial simple agent

# 4. LLM-as-judge reply-quality scoring (separate from the automated metrics above)
python src/llm_judge.py trivial
python src/llm_judge.py simple
python src/llm_judge.py agent
```

`evaluate.py` prints intent-classification accuracy/macro-F1, escalation precision/recall/F1,
and an automated reply-similarity proxy for each system, and writes
`data/eval/summary_metrics.csv`. `llm_judge.py` writes per-example 1-5 rubric scores to
`data/eval/judge_<system>.csv`. Both scripts automatically exclude any row where the system
errored (e.g. quota exhaustion) rather than counting it as wrong.

**Running under a tight quota:** `run_predictions.py` and `llm_judge.py` both support
`--resume` (keep existing good rows, only retry ERROR/missing ones) and `--max-new N` (cap how
many fresh model calls this invocation makes). `agent.handle()`/`llm_judge.judge_one()` can also
round-robin across `llm_client.GEMINI_AGENT_MODEL_POOL` (pass `--use-pool` to `llm_judge.py`,
or construct `Agent(use_model_pool=True)`) so one exhausted model doesn't block the whole run.

To check the judge itself: `python src/judge_calibration.py --prepare` samples 40 judged
examples into a CSV with a blank `human_overall_quality` column; fill it in by hand, then
`python src/judge_calibration.py --score-file data/eval/judge_calibration_sample.csv` reports
Spearman correlation and quadratic-weighted Cohen's kappa between the judge and the human.

## Repo layout

```
src/                   pipeline code (see individual files for docstrings)
  build_threads.py      raw Kaggle CSV -> reconstructed conversation threads (one brand)
  prepare_examples.py   threads -> clean (customer_message, brand_reply, full_conversation) rows
  split_data.py         corpus (grounding) / eval_pool (golden-set source) split
  taxonomy.py           the 9-intent taxonomy + escalation policy notes
  keyword_bucket.py      cheap deterministic bucketer (stratified sampling + simple baseline)
  stratified_sample.py  eval_pool -> golden_candidates.csv
  build_golden_set.py    hand labels -> data/eval/golden_set.csv
  retrieval.py           TF-IDF retrieval over the grounding corpus
  agent.py                the LLM agent (classify + draft reply + escalate, one call)
  baseline_trivial.py    majority-class intent, canned reply, never escalates
  baseline_simple.py     TF-IDF+LogReg intent, nearest-neighbor reply, rule-based escalation
  run_predictions.py     run any system over the golden set, cache predictions
  evaluate.py             automated metrics (classification, escalation, reply-similarity proxy)
  llm_judge.py            LLM-as-judge reply-quality rubric
  judge_calibration.py    human-vs-judge agreement check
data/
  raw/                   untouched Kaggle download (gitignored if large)
  processed/              reconstructed threads, cleaned examples, corpus/eval_pool split
  eval/                   golden_set.csv + all prediction/judge/score outputs
report/
  report.md               the write-up
  decision_log.md          non-obvious decisions and why
```
