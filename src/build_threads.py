"""Reconstruct full customer<->brand conversation threads for one brand from the
Customer Support on Twitter dataset, and write them to a compact JSONL file.

Each output record is one thread:
{
  "thread_id": str,               # tweet_id of the root customer tweet
  "brand": "SpotifyCares",
  "turns": [
      {"author": "customer"|"brand", "tweet_id": ..., "text": ..., "created_at": ...},
      ...
  ]
}

A thread starts at a customer tweet that is NOT itself a reply to the brand
(i.e. the customer initiating contact), and follows the response_tweet_id /
in_response_to_tweet_id chain forward as far as it goes.
"""
import sys
import json
import pandas as pd

BRAND = sys.argv[1] if len(sys.argv) > 1 else "SpotifyCares"
RAW = "data/raw/twcs/twcs.csv"
OUT = f"data/processed/{BRAND}_threads.jsonl"

print(f"Loading raw CSV (this scans ~3M rows)...")
df = pd.read_csv(RAW, dtype=str)
df["inbound"] = df["inbound"].map({"True": True, "False": False, True: True, False: False})

# Index all tweets by id for O(1) lookup
by_id = df.set_index("tweet_id", drop=False)

# Tweets sent BY the brand
brand_tweets = df[(df["author_id"] == BRAND) & (~df["inbound"])]
print(f"Brand tweets for {BRAND}: {len(brand_tweets)}")

# Tweets sent TO the brand by customers, i.e. inbound tweets whose
# in_response_to_tweet_id is a brand tweet, OR any inbound tweet that
# mentions the brand and starts a thread (response_tweet_id points forward).
# Simplest robust approach: find root customer tweets = inbound tweets that
# are NOT a reply to anything (in_response_to_tweet_id is null) and that have
# at least one response eventually authored by the brand.

def get_children(tweet_id):
    """tweet ids that are replies to tweet_id"""
    rt = by_id.loc[tweet_id, "response_tweet_id"] if tweet_id in by_id.index else None
    if pd.isna(rt) or rt is None:
        return []
    return [t.strip() for t in str(rt).split(",") if t.strip()]

root_customer_tweets = df[(df["inbound"]) & (df["in_response_to_tweet_id"].isna())]
print(f"Root customer tweets (any brand): {len(root_customer_tweets)}")

threads = []
seen_roots = set()

for _, brow in brand_tweets.iterrows():
    parent_id = brow["in_response_to_tweet_id"]
    if pd.isna(parent_id):
        continue
    # walk up to the root of this conversation
    cur = parent_id
    chain_up = [brow["tweet_id"]]
    visited = set()
    while cur is not None and not pd.isna(cur) and cur in by_id.index and cur not in visited:
        visited.add(cur)
        chain_up.append(cur)
        parent = by_id.loc[cur, "in_response_to_tweet_id"]
        cur = parent if not pd.isna(parent) else None
    root_id = chain_up[-1]
    if root_id in seen_roots:
        continue
    seen_roots.add(root_id)

    # now walk DOWN from root, following response_tweet_id, preferring the
    # path that actually goes through this brand, to build the full turn list
    turns = []
    node = root_id
    guard = 0
    while node is not None and guard < 20:
        guard += 1
        if node not in by_id.index:
            break
        row = by_id.loc[node]
        turns.append({
            "tweet_id": row["tweet_id"],
            "author": "customer" if row["inbound"] else ("brand" if row["author_id"] == BRAND else "other_brand_or_agent"),
            "author_id": row["author_id"],
            "text": row["text"],
            "created_at": row["created_at"],
        })
        children = get_children(node)
        if not children:
            break
        # prefer the child that is inbound customer, else brand, else first
        next_node = None
        for c in children:
            if c in by_id.index and by_id.loc[c, "author_id"] == BRAND:
                next_node = c
                break
        if next_node is None:
            for c in children:
                if c in by_id.index and by_id.loc[c, "inbound"]:
                    next_node = c
                    break
        if next_node is None:
            next_node = children[0]
        node = next_node

    if any(t["author"] == "brand" for t in turns) and any(t["author"] == "customer" for t in turns):
        threads.append({"thread_id": root_id, "brand": BRAND, "turns": turns})

print(f"Reconstructed {len(threads)} threads for {BRAND}")

import os
os.makedirs("data/processed", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for t in threads:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"Wrote {OUT}")
