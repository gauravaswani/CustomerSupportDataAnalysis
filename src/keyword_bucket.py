"""Cheap, deterministic keyword-based intent bucketing. Used for two things:
1. stratified sampling of the golden eval set (so we don't hand-label 200
   random examples that are 90% "my song skips")
2. weak-supervision training labels for the simple TF-IDF+LogReg baseline

This is NOT the ground truth -- the golden set's real labels are assigned by
hand (see eval/golden_set.csv and report/decision_log.md).
"""
import re

BUCKET_KEYWORDS = {
    "login_account_access": [
        "log in", "login", "logged out", "password", "locked", "sign in",
        "signin", "verify my account", "linked account", "can't access my account",
        "hacked",
    ],
    "account_billing": [
        "charge", "charged", "bill", "billing", "refund", "subscription",
        "premium", "discount", "price", "pay", "payment", "cancel my",
        "cancel premium", "family plan", "duo", "student plan", "invoice",
        "renew",
    ],
    "content_availability": [
        "not available in", "available in my country", "region", "removed from spotify",
        "licensing", "geo", "blocked in", "can't find this country",
    ],
    "content_error_report": [
        "wrong artist", "wrong song", "mislabeled", "incorrect song", "duplicate",
        "wrong album", "not the right song", "credited to",
    ],
    "redirect_other_team": [
        "artist team", "for artists", "spotify for podcasters", "api ", "developer",
        "press inquiry", "job", "career", "internship", "label",
    ],
    "praise_or_gratitude": [
        "thank", "thanks", "appreciate", "you guys rock", "love spotify",
        "great job", "awesome support",
    ],
    "feature_request_or_limitation": [
        "please add", "wish spotify", "why can't i", "why doesn't spotify",
        "feature request", "10k", "limit", "should be able to", "would be nice",
    ],
    "playback_technical_issue": [
        "skip", "skipping", "stutter", "freeze", "frozen", "crash", "crashing",
        "stopped playing", "won't play", "wont play", "glitch", "buffering",
        "offline", "download", "sync", "shuffle", "repeat", "autoplay",
        "keeps pausing", "lagging", "won't load", "wont load", "app is broken",
    ],
}


def bucket(text: str) -> str:
    t = text.lower()
    scores = {}
    for intent, kws in BUCKET_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score:
            scores[intent] = score
    if not scores:
        return "unmatched"
    return max(scores, key=scores.get)
