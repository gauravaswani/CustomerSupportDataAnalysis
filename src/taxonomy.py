"""Shared intent taxonomy + escalation policy for the SpotifyCares support agent.

Grounded in manual reading of ~50+ sampled threads from the dataset
(see report/decision_log.md for the derivation notes).
"""

INTENTS = {
    "playback_technical_issue": (
        "App/device bugs affecting playback: skipping, stuttering, stopping when "
        "backgrounded, autoplay behaving unexpectedly, shuffle/repeat broken, "
        "offline downloads not syncing, crashes."
    ),
    "account_billing": (
        "Anything involving money or subscription state: charges, refunds, plan "
        "changes, family/duo plan billing, student discount enrollment, promo "
        "pricing, payment method issues, cancellation requests."
    ),
    "login_account_access": (
        "Cannot log in, password reset, account locked/hacked, linked "
        "Facebook/Apple/Google account issues, verifying account ownership."
    ),
    "content_availability": (
        "A song, album, podcast, or the service itself is not available in the "
        "user's region or on the platform, or was removed/geo-blocked."
    ),
    "content_error_report": (
        "User is reporting incorrect catalog data: wrong artist/track "
        "attribution, mislabeled song, broken link, duplicate entries."
    ),
    "feature_request_or_limitation": (
        "User is complaining about or asking for a product limitation/feature "
        "that is a design decision, not a bug (e.g. the 10k song library cap, "
        "missing feature, UI complaint) -- no account-specific data needed."
    ),
    "redirect_other_team": (
        "Question is out of scope for consumer support: artist/label/developer "
        "platform questions, press inquiries, API questions, job inquiries."
    ),
    "praise_or_gratitude": (
        "Message is a thank-you, positive feedback, or closes out a resolved "
        "conversation with no new ask."
    ),
    "unclear_or_low_signal": (
        "Message is spam/unintelligible, or so vague ('I have a complaint', "
        "'something's wrong') that no intent can be assigned without a "
        "clarifying follow-up question."
    ),
}

INTENT_LIST = list(INTENTS.keys())

# Which intents CAN be safely auto-handled by a text-only agent with no access
# to the real account/billing systems, vs MUST be escalated to a human/authenticated
# system because resolving them requires touching real account or payment data,
# or because the brand's own historical behavior was to take it to a private
# channel (DM) rather than resolve publicly.
AUTO_HANDLE_INTENTS = {
    "playback_technical_issue",
    "content_availability",
    "content_error_report",
    "feature_request_or_limitation",
    "redirect_other_team",
    "praise_or_gratitude",
    "unclear_or_low_signal",  # default: ask a clarifying question, don't escalate on vagueness alone
}
ESCALATE_INTENTS = {
    "account_billing",
    "login_account_access",
}
# NOTE: escalation is NOT a pure function of intent bucket. The real policy
# (see agent.py / report/decision_log.md) escalates a message when resolving
# it requires touching the customer's real account/payment record, or the
# message contains exposed PII, legal/self-harm/fraud language, or is
# unintelligible spam -- not simply because it landed in a given intent
# bucket. account_billing/login_account_access are escalated *most* of the
# time in practice (hence their presence here as a prior), but general
# how-to/policy questions in those buckets are still auto-handleable.

ESCALATION_KEYWORDS = [
    "lawyer", "lawsuit", "sue", "legal action", "fraud", "unauthorized charge",
    "hacked", "suicide", "self harm", "kill myself", "gdpr", "data request",
    "delete my account", "delete my data", "chargeback",
]
