"""Turn raw reconstructed threads into a clean per-conversation examples table.

For each thread we extract:
  - customer_message: the customer turn immediately preceding the FIRST brand reply
    (this is what an agent would need to classify + respond to)
  - context_before: any customer turns before that (rare, usually empty)
  - brand_reply: the brand's first reply (the historical "resolution start")
  - full_conversation: all turns concatenated, for grounding / judge reference
  - num_turns, resolved (bool: conversation ends with a customer turn that is not
    a further complaint -- heuristic: last turn is brand, or contains thanks)
"""
import json
import re
import pandas as pd

IN = "data/processed/SpotifyCares_threads.jsonl"
OUT = "data/processed/spotify_examples.csv"

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(t):
    if not isinstance(t, str):
        return ""
    return t.strip()


def strip_for_len_check(t):
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    return t.strip()


rows = []
with open(IN, encoding="utf-8") as f:
    for line in f:
        thread = json.loads(line)
        turns = thread["turns"]
        # find first brand turn index
        first_brand_idx = next((i for i, t in enumerate(turns) if t["author"] == "brand"), None)
        if first_brand_idx is None or first_brand_idx == 0:
            continue
        customer_turns_before = [t for t in turns[:first_brand_idx] if t["author"] == "customer"]
        if not customer_turns_before:
            continue
        customer_message = clean_text(customer_turns_before[-1]["text"])
        context_before = " ".join(clean_text(t["text"]) for t in customer_turns_before[:-1])
        brand_reply = clean_text(turns[first_brand_idx]["text"])

        if len(strip_for_len_check(customer_message)) < 8:
            continue  # too short / no real content once mentions+urls stripped

        full_conv = " || ".join(f"[{t['author']}] {clean_text(t['text'])}" for t in turns)
        last_turn = turns[-1]
        thanks_markers = ("thank", "thanks", "cheers", "appreciate", "sorted", "works fine", "resolved")
        resolved = (last_turn["author"] == "customer" and any(m in last_turn["text"].lower() for m in thanks_markers)) or \
                   (last_turn["author"] == "brand")

        rows.append({
            "thread_id": thread["thread_id"],
            "customer_message": customer_message,
            "context_before": context_before,
            "brand_reply": brand_reply,
            "full_conversation": full_conv,
            "num_turns": len(turns),
            "resolved_heuristic": resolved,
        })

df = pd.DataFrame(rows)
print(f"Examples before dedup: {len(df)}")
df = df.drop_duplicates(subset=["customer_message"]).reset_index(drop=True)
print(f"Examples after dedup: {len(df)}")
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"Wrote {OUT}")
print(df["num_turns"].describe())
