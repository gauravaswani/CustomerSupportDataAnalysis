# SpotifyCares support agent -- report

## 1. Problem framing

**Brand:** SpotifyCares. Chosen over Amazon/Apple/airlines because it's a single consumer
product with a bounded set of issue types (playback, account, billing, catalog) and no
shipping/safety-critical traffic, at a volume (43k historical replies) large enough to build
both a grounding corpus and a golden set.

**What "good" means here:** SpotifyCares' own historical behavior, read across ~250 real
threads, has a very consistent pattern: troubleshoot in public for anything client-side
(device/OS/app-version questions, "have you tried logging out and back in"), answer policy/
content questions with a link, redirect out-of-scope questions (artist/developer/jobs), and
the moment anything requires looking at the customer's *actual* account or payment record,
ask for a DM -- but critically, that DM request is *itself* a real public reply, not silence.
"Good" for an AI version of this agent means reproducing that same split reliably: confidently
self-serve the first category in the brand's own voice, and correctly recognize -- not guess
at -- the second category rather than hallucinating an account-specific answer it has no way
to actually know.

**What I chose not to build:**
- **No actual account/billing system integration.** The agent never claims to have looked up
  a real account; when it decides `escalate=True` it still drafts the public "please DM us
  your account email" reply the brand always sends, then hands the private follow-up to a
  human. Building a real account-lookup tool was out of scope for a take-home and would mask
  the actual hard problem (deciding *when* escalation is needed).
- **No multi-turn dialogue management.** Each classification/reply/escalate decision is made
  from a single customer message (+ any turns before it in the same thread), not a simulated
  back-and-forth. Real deployments would need conversation state; this scopes to the
  first-contact decision, which is also what's gradeable against the historical data.
- **No fine-tuning.** Everything is prompting + retrieval over a frozen model. A production
  system handling this volume would likely fine-tune a small classifier for the intent step
  alone (cheaper, faster, more consistent) and reserve the LLM for reply drafting.
- **No embeddings-based retrieval.** TF-IDF over historical customer messages, not a vector
  index -- these messages are short and keyword-heavy, so lexical overlap is already a strong
  signal, and it needs no extra API calls.

## 2. Method

- **Data:** 28,277 reconstructed customer<->SpotifyCares threads from the raw 3M-row Kaggle
  dataset. Split into a 24,745-example grounding corpus and a held-out 3,000-example eval pool
  (thread-level split, so no leakage).
- **Taxonomy:** 9 intents, derived by manually reading ~250 real threads: `playback_technical_issue`,
  `account_billing`, `login_account_access`, `content_availability`, `content_error_report`,
  `feature_request_or_limitation`, `redirect_other_team`, `praise_or_gratitude`,
  `unclear_or_low_signal`.
- **Agent:** one Gemini call per message. Retrieves the 3 most similar historical
  (customer message, brand reply) pairs via TF-IDF, then classifies + drafts a reply in
  SpotifyCares' voice + decides escalate/auto with a stated reason, all in one JSON response.
  A deterministic keyword safety-net forces escalation on explicit legal/self-harm/fraud
  language regardless of the model's own output.
- **Golden set:** 214 hand-labeled examples, stratified across the 9 intents via a
  keyword heuristic (not an LLM) over the held-out eval pool, each hand-labeled by reading the
  full historical conversation. Full methodology: `data/eval/README.md`.
- **LLM backend:** Gemini (`gemini-3.6-flash` family), not Anthropic, because the available API
  key for this run was a Gemini key -- `src/llm_client.py` supports both.

## 3. Results vs. baselines

Baselines: **trivial** (majority-class intent from the corpus, one fixed canned reply, never
escalates) and **simple** (TF-IDF + logistic regression intent classifier trained on
keyword-bucket weak labels, nearest-neighbor historical reply, rule-based escalation from a
fixed intent->escalate lookup table).

**Classification and escalation, full golden set (trivial/simple) vs. agent's achieved subset:**

| System  | n   | Intent accuracy | Intent macro-F1 | Escalation P / R / F1 | Missed escalations |
|---------|-----|------------------|-------------------|-------------------------|----------------------|
| trivial | 214 | 0.178            | 0.034             | 0.000 / 0.000 / 0.000   | 70/70 (100%)         |
| simple  | 214 | 0.636            | 0.569             | 0.817 / 0.829 / 0.823   | 12/70 (17%)          |
| agent   | 71  | **0.873**        | **0.885**         | **0.889 / 0.889 / 0.889** | 3/27 (11%)         |

The agent's n=71, not 214: Gemini's free tier caps each model at ~5 requests/minute and ~20
requests/day, and even after round-robining across 6 flash snapshots and 2 separate API keys
over 3 days, 143/214 calls still ended in `RESOURCE_EXHAUSTED`. See Section 5 -- this is not a
detail to bury, it's a real finding.

**Same 71 messages, all three systems (apples-to-apples):**

| System  | Intent accuracy | Escalation P / R / F1 |
|---------|------------------|-------------------------|
| trivial | 0.211            | 0.000 / 0.000 / 0.000 (0 tp, 27 missed) |
| simple  | 0.690            | 0.893 / 0.926 / 0.909 (25 tp, 3 fp, 2 fn) |
| agent   | 0.873            | 0.889 / 0.889 / 0.889 (24 tp, 3 fp, 3 fn) |

On this exact matched subset, `simple`'s escalation F1 (0.909) is actually marginally *higher*
than the agent's (0.889) -- driven by 2 fewer missed escalations. The agent wins clearly on
intent accuracy (0.873 vs 0.690) but the escalation gap is much smaller than the headline
numbers above suggest, because `simple`'s escalation rule is a direct function of intent
(`ESCALATE_INTENTS` lookup) and gets a "free" boost on this particular subsample. Worth reading
both tables, not just the first one.

**Reply quality (LLM-judge, 1-5 rubric, same 71-message subset where obtainable):**

| System  | n judged | Groundedness | Correctness | Tone | Actionability | Overall |
|---------|----------|---------------|--------------|------|------------------|---------|
| trivial | 62/71    | 4.97          | 3.16         | 4.53 | 3.92             | 3.18    |
| agent   | 47/71    | 4.47          | 3.64         | 3.38 | 3.00             | 3.30    |
| simple  | 0/71     | --            | --           | --   | --               | not obtained (quota ran out before any calls landed) |

Read this table with Section 5's judge-calibration finding before drawing conclusions --
the agent's tone/actionability numbers here are measurably deflated by a bug in the judge
harness itself, not by the agent.

## 4. Failure analysis

Five real failure modes, pulled directly from `data/eval/scored_agent.csv` (the 71 messages
the agent actually processed):

1. **Sarcasm and vague-but-real complaints get bucketed as "too vague to classify."** 3 of the
   9 intent misclassifications are the agent retreating to `unclear_or_low_signal` on messages
   that do have a real underlying intent: id 16 ("Wow, the android app is getting worse and
   worse...thanks for that, guys" -- gold `playback_technical_issue`), id 74 ("can u get ur shit
   together with my discover weekly. Thanks" -- gold `feature_request_or_limitation`), and id 20
   (see #2). Being cautious about ambiguity is the right instinct in general, but here it's
   triggering on messages a human agent would read as clearly annoyed-but-specific complaints.
2. **Referential ambiguity compounds into a missed escalation.** id 20, "Changed password, And
   it's still happening?? Help please! This is driving me nuts!", never says what "it" refers
   to without the customer's earlier tweets (which our extraction doesn't carry forward from
   before the thread's start). The agent classified this `unclear_or_low_signal` /
   `escalate=False`; gold is `login_account_access` / `escalate=True`. This is a real limitation
   of scoring first-contact messages in isolation, and it's the one case where an intent miss
   directly caused an escalation miss too -- a risk of the single-combined-call design (one bad
   call affects both decisions at once).
3. **`account_billing` <-> `login_account_access` boundary confusion.** ids 4 and 7 swap these
   two categories in opposite directions -- unsurprising, since they're the two most
   semantically adjacent intents in the taxonomy (both often *look* identical at the surface
   level: "I can't get into my account/plan"). Both got escalated correctly regardless (the
   escalation policy for these two intents is nearly identical), so the practical damage from
   this specific confusion is limited to a wrong dashboard label, not a wrong customer outcome.
4. **A judge-harness bug, not an agent bug, deflates the agent's judged reply quality.** The
   original `llm_judge.py` substituted `"[N/A -- escalated instead of replying]"` for the
   agent's actual reply whenever `escalate=True`, even though the agent *always* drafts a real
   public reply (matching the brand's own pattern of asking for a DM before continuing
   privately). Hand-scoring a 25-example calibration sample against the judge's own scores
   surfaced this directly: ids 27, 10, 32, 3, 31, 23, 30, 19 (all `escalate=True`) got judge
   scores of 1-2 with rationales like "the agent failed to draft any public reply," when the
   actual `pred_reply` for every one of those was a substantive, on-brand message asking for
   account details. Fixed in `llm_judge.py` (see decision log), but not re-run due to quota.
5. **Systematic, moderate over-escalation on ambiguous or "recurring" language.** All 3 of the
   agent's false-positive escalations (ids 26, 34, 44) involve either a non-English message
   getting escalated instead of auto-handled via the deterministic language-redirect that the
   brand actually uses (id 26), or language like "keeps happening" / "recurring" reading as
   "needs a human" when the historical pattern is actually a self-serve troubleshooting doc
   (id 34, id 44). The agent is trading deflection rate for safety in genuinely ambiguous
   cases -- defensible, but it's the reason `escalation precision` (0.889) isn't higher despite
   `recall` being identical.

## 5. What's misleading about my headline number

- **"87.3% intent accuracy" is measured on 71 examples, not 214, and that 71 isn't a random
  sample of the golden set -- it's whatever got through before Gemini's free-tier quota ran
  out.** The first ~20 calls each day hit whichever model was tried first in the round-robin
  pool; later calls in a batch increasingly landed on whichever models still had headroom. If
  harder examples cluster anywhere in golden-set order (they don't appear to, but I can't fully
  rule it out), the achieved subset could be systematically easier or harder than the full 214.
- **The escalation story flips depending on which table you read.** The headline table (agent
  n=71 vs. baselines at n=214) makes the agent look dominant on escalation (0.889 F1 vs.
  simple's 0.823). The matched-n=71 table tells a different story: on the *exact same*
  messages, `simple`'s escalation F1 (0.909) is actually slightly higher than the agent's
  (0.889). Comparing systems across different N inflates or deflates gaps that don't survive
  an apples-to-apples check -- this is the single most important thing in this report to
  actually read past the headline number for.
- **The LLM-judge reply-quality numbers are not directly comparable to intent/escalation
  accuracy in trustworthiness.** Spearman correlation between the judge and my own hand-scores
  on a 14-example calibration sample (the only 14 where the judge itself didn't error) is 0.884
  -- strong *relative* agreement, the judge can tell good replies from bad ones. But
  quadratic-weighted kappa is only 0.343 and exact-match is 42.9%, because the judge and I
  disagree by 1-3 points on absolute scale specifically on escalated replies (the bug in
  failure mode #4). A judge that ranks well but calibrates poorly will make a system look much
  worse (or better) than it is on any report that quotes its raw averages without this check --
  which is exactly why this project requires running that check at all, not just trusting the
  judge.
- **The simple baseline's 63.6%/69.0% intent accuracy is inflated by label leakage-by-
  construction.** Its training labels (keyword-bucket weak supervision) and my own gold labels
  were both produced by a human (me) reading similar lexical cues -- they aren't independently
  sourced. A classifier trained on genuinely independent labels would likely score lower.
- **A reply-similarity score (ROUGE-L/TF-IDF cosine against the historical reply) rewards
  copying the brand's own habits, including its bad ones.** SpotifyCares' historical replies
  sometimes over-ask for DM/account-email even when unnecessary (e.g. id 1, id 39 in the golden
  set); a system that mimics this faithfully scores *well* on similarity metrics while actually
  escalating too much. Similarity-to-history is a proxy for "sounds like this brand," not for
  "made the right call."
- **The quota saga is itself a finding, not just an excuse.** Building "an AI system that
  works" against a free-tier LLM API means the operational failure mode isn't just model
  quality -- it's whether the system degrades gracefully under real external constraints
  (rate limits, partial outages, multiple keys with independent budgets). The `--resume`/
  `--max-new`/model-pool machinery this project needed to build just to get a usable N is
  exactly the kind of infrastructure a real deployment against a free/cheap tier would also
  need, and "prove it works" has to include proving it works *under* those constraints, not
  just in an idealized unlimited-budget run.
- **214 examples from one brand, one time window, one language-dominant dataset.** Headline
  numbers here say nothing about non-English traffic (several golden examples are Malay/
  Indonesian/Portuguese and got redirected rather than resolved, and the agent's one true
  over-escalation on a language case shows this is a live weak spot), other Spotify support
  channels (DM, email, in-app), or brands with fundamentally different risk profiles (financial
  services, airlines).

## 6. What I'd do next with one more week

1. Get a paid/higher-quota key (or spread the eval run across enough real days) and finish all
   214 agent predictions + full judge scoring for all three systems, including `simple`, which
   currently has zero judge coverage.
2. Re-run `llm_judge.py` with the N/A-substitution bug fixed (already patched in code) and
   re-check judge/human calibration -- I'd expect kappa to jump substantially once escalated
   replies are shown to the judge honestly.
3. Expand the golden set beyond one sampling pass -- specifically oversample
   `content_error_report`, `redirect_other_team`, and `praise_or_gratitude` (7, 11, 4 examples
   respectively), which have the fewest examples and the widest per-class metric variance.
4. Investigate whether achieved-subset order bias (Section 5) is real by checking if the 71
   completed IDs differ systematically (message length, intent distribution) from the 143 that
   errored out -- if quota exhaustion correlates with anything about the message itself
   (e.g. longer prompts costing more retries), the 87.3% number could be biased in a specific
   direction.
5. Replace TF-IDF retrieval with embeddings and measure whether it changes reply grounding on
   paraphrase-heavy queries.
6. Add a second, independently-sourced set of intent labels for the corpus to get a
   simple-baseline accuracy number that isn't inflated by the weak-label/gold-label correlation
   flagged in Section 5.
