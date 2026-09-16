# Golden evaluation set

`golden_set.csv` -- 214 hand-labeled examples.

## Sampling

1. Reconstructed 28,277 full customer<->SpotifyCares conversation threads from the raw
   Kaggle dataset (`src/build_threads.py`), then extracted one example per thread
   (`src/prepare_examples.py`): the customer's message immediately preceding the brand's
   first reply, plus the full conversation for reference.
2. Split into a 24,745-row grounding corpus and a 3,000-row held-out eval pool
   (`src/split_data.py`, thread-level split, seed=42) -- the golden set is drawn only from
   the eval pool, so none of its examples' historical replies can leak into the retrieval
   corpus that grounds the agent's own reply generation.
3. Bucketed the eval pool with a cheap keyword heuristic (`src/keyword_bucket.py`) into
   9 rough categories, then sampled up to 28 examples per bucket (`src/stratified_sample.py`,
   seed=7) -- 214 candidates total (some buckets had fewer than 28 members in the pool).
   The keyword bucketer is deliberately *not* an LLM: using the model under test to help pick
   its own exam questions would entangle sampling with evaluation.

## Labeling

Every one of the 214 candidates was read in full -- the customer message plus the entire
historical conversation thread -- and hand-labeled with:
- `intent_gold`: one of the 9 taxonomy intents (`src/taxonomy.py`)
- `escalate_gold`: would resolving this actually require touching the customer's real
  account/payment record, exposed PII, legal/self-harm/fraud language, or is it unintelligible
  spam? (Not a fixed mapping from intent -- see decision log item 5.)
- `escalate_reason_gold`: one-line justification
- `note`: populated for ~20 examples called out in the failure analysis (sarcasm, PII exposure,
  spam, language mismatch, low-signal intake, noisy historical labels)

Labels reflect what a support agent could reasonably decide **at first contact** -- i.e. from
the customer's message alone -- even though the full downstream conversation was read to
understand ground truth. Seeing the eventual resolution and then grading the first-contact
decision by it would be hindsight bias.

No LLM was involved in producing `intent_gold` / `escalate_gold` -- these are the actual human
labels the eval harness scores every system against.
