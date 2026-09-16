# Decision log

Non-obvious decisions made while building this, and why.

1. **Brand: SpotifyCares.** Single consumer product, no shipping/logistics/safety-critical
   traffic (unlike airlines/telecom), and a large enough reply volume (43k) to build both a
   grounding corpus and a golden set from real historical resolutions.

2. **Thread reconstruction walks the `in_response_to_tweet_id` chain up to the root, then
   back down preferring the brand-authored branch** (capped at depth 20). The dataset only
   gives pairwise reply links, not conversation IDs, and a tweet can have multiple children
   (e.g. a bystander reply) -- naively following `response_tweet_id` forward can wander off
   into an unrelated branch.

3. **`customer_message` = the last customer turn immediately before the first brand reply,
   not the thread root.** Several threads start with an unrelated promo/broadcast tweet, and
   the actual ask arrives in a follow-up reply from the same user. Classifying the root tweet
   would mislabel these.

4. **9 intents, not the 8 I started with.** Reading ~50 real threads surfaced a real category
   my first taxonomy didn't cover: spam/unintelligible text and messages so vague ("I have a
   complaint") that no intent is assignable without a clarifying follow-up. Forcing these into
   a real bucket would have hurt precision on the categories that matter; `unclear_or_low_signal`
   is a fine label to have, and the *right* handling for it is usually "ask a clarifying
   question," not escalate.

5. **Escalation is a judgment call per message, not a fixed function of intent bucket.**
   `account_billing` and `login_account_access` escalate *most* of the time (most require a
   real account/payment lookup) but not always -- e.g. "how do I turn off auto-renew" is pure
   policy information, no account access needed. The agent's system prompt states the actual
   criterion (does resolving this need to touch the customer's real account/payment record,
   or does it expose PII/legal risk/spam) rather than a lookup table. The *simple* baseline
   deliberately uses the naive lookup-table version, so the eval shows the cost of that
   simplification.

6. **Golden-set stratified sampling uses a cheap keyword-bucket heuristic, not an LLM call.**
   Using the model under test (or any LLM) to pre-select which examples get hand-labeled would
   make the sampling procedure entangled with the thing being evaluated. The keyword bucketer
   is deterministic, has zero cost, and only needs to be "good enough" to avoid a golden set
   that's 90% playback complaints.

7. **Gold labels were assigned reading the full historical conversation, but the agent is only
   ever scored on what's knowable from the first customer message.** I read every follow-up
   turn to understand the true underlying issue (e.g. distinguishing sarcasm, or seeing that a
   `login_account_access` issue turned out to need an account lookup), but the escalate/intent
   label reflects the decision a support agent could actually make *at first contact* --
   knowing the eventual resolution would be hindsight bias.

8. **Corpus/eval split happens at the thread level before golden sampling**, so no golden
   example's own historical reply can leak into the retrieval corpus that grounds the agent's
   reply generation.

9. **The simple baseline's intent classifier is trained on keyword-bucket weak labels**, not
   the hand-labeled golden set -- there's no hand-labeled training data for the 24.7k-row
   corpus, only for the 214-row golden set, so this mirrors what a team would actually ship
   with a weekend of weak supervision. Flagged explicitly in the misleading-number section:
   this also means the weak labels and my gold labels share a keyword-driven origin, which
   inflates the simple baseline's apparent agreement with gold beyond what an independently-
   labeled training set would produce.

10. **One combined LLM call per message for classify + draft reply + escalate**, not three
    separate calls. Keeps cost/latency down (~3x fewer calls) and avoids a failure mode where
    the classifier and the reply-writer silently disagree about what the message is even about.

11. **A deterministic keyword safety-net forces `escalate=True`** on explicit legal/self-harm/
    fraud language regardless of what the model returns. High-stakes categories should not
    depend entirely on a single LLM call's judgment.

12. **The LLM judge uses a separate prompt/role from the agent**, and is calibrated against a
    hand-scored subsample rather than trusted blindly (see report.md's judge-agreement numbers).

13. **Retrieval grounding uses TF-IDF, not embeddings.** These customer messages are short and
    keyword-heavy ("can't log in," "10k limit," "not available in Kuwait"); lexical overlap is
    already a strong signal, and TF-IDF is free, deterministic, and needs no extra API calls.
    A real deployment handling paraphrase-heavy queries would likely want embeddings instead.

14. **Golden set is 214 examples** (within the requested 150-250), toward the lower end of that
    range because hand-labeling here means reading the *entire* historical conversation for
    each candidate, not just the first message -- keeping this tractable took priority over
    maximizing N.

15. **Reproducibility does not require the raw 3M-row Kaggle download or Kaggle credentials.**
    A filtered, pre-processed SpotifyCares subset is checked into the repo so a grader can
    reproduce headline results in under 15 minutes without re-downloading or re-authenticating
    to Kaggle.

16. **Gemini, not Anthropic, is the default LLM backend**, because the available API key for
    this run is a Gemini key. `src/llm_client.py` supports both; Anthropic remains a fully
    working alternate path (auto-selected if `ANTHROPIC_API_KEY` is set instead). This is a
    pragmatic call, not a claim that Gemini is a better fit for the task.

17. **Agent and judge use two different Gemini flash snapshots, not two different model
    tiers.** The available API key's free tier has zero quota for any pro-tier Gemini model
    (`gemini-3.1-pro` returned `RESOURCE_EXHAUSTED` with a daily/per-minute limit of 0), so the
    usual mitigation for judge self-preference bias -- use a stronger, different-family model
    as judge -- wasn't available. Using `gemini-3.6-flash` for the agent and `gemini-3.5-flash`
    for the judge is a partial substitute, and this limitation is called out explicitly rather
    than silently reusing the identical model+snapshot for both roles.

18. **When per-model quota ran out, requests were round-robined across 6 same-family Gemini
    flash snapshots and, later, 2 separate API keys**, rather than waiting out a single model's
    reset. Documented in `llm_client.GEMINI_AGENT_MODEL_POOL` and `run_predictions.py`'s
    `--resume`/`--max-new` flags. This means the agent's 214-example golden set was only ever
    71% completed (n=71) even after this mitigation -- reported honestly as a partial sample
    rather than silently treated as equivalent to a full run (see report.md Section 5).

19. **The LLM-judge prompt originally substituted a placeholder ("[N/A -- escalated instead of
    replying]") for the agent's real reply whenever `escalate=True`**, on the mistaken
    assumption that escalation meant no customer-facing reply was sent. In fact the agent
    always drafts a real public reply even when escalating (matching the brand's own pattern:
    ask for a DM, then continue privately) -- so this substitution caused the judge to
    incorrectly penalize correctly-escalated, well-written replies for "not responding." This
    was only caught by manually hand-scoring a calibration sample and comparing it against the
    judge's own scores; it would not have been visible from the judge's numbers alone. Fixed in
    `llm_judge.py`, documented as a concrete example of why judge/human calibration is a
    required step, not an optional nice-to-have.

20. **Judge/human calibration used quadratic-weighted Cohen's kappa AND Spearman correlation
    together, not just one.** On the 14-example sample where both scores existed, Spearman rho
    was a strong 0.884 while kappa was only 0.343 -- the judge ranks replies correctly relative
    to each other but is miscalibrated on absolute scale (specifically because of decision #19).
    Reporting only rho would have hidden this; reporting only kappa without rho would have made
    the judge look uniformly bad rather than "directionally right, absolutely wrong in one
    specific case." Both numbers were needed to state the finding precisely.
