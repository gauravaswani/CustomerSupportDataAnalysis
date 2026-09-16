"""LLM call wrapper with three backends, auto-selected by whichever key is
present in the environment (checked in this order): Gemini, Anthropic API,
Anthropic CLI (dev-sandbox fallback only, see git history for why).

Override with LLM_BACKEND=gemini|api|cli.

Gemini is the default/primary path in this project (see decision log) --
Anthropic remains supported as an alternate backend.
"""
import os
import re
import json
import time
import subprocess

from dotenv import load_dotenv
load_dotenv()

if os.environ.get("LLM_BACKEND"):
    BACKEND = os.environ["LLM_BACKEND"]
elif os.environ.get("GEMINI_API_KEY"):
    BACKEND = "gemini"
elif os.environ.get("ANTHROPIC_API_KEY"):
    BACKEND = "api"
else:
    BACKEND = "cli"

# Anthropic model constants (used when BACKEND in {"api", "cli"})
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
CLI_MODEL_ALIAS = {HAIKU: "haiku", SONNET: "sonnet"}

# Gemini model constants (used when BACKEND == "gemini").
# Two distinct flash snapshots for agent vs. judge -- this account's free tier
# has zero quota for pro-tier models, so full model-family separation between
# agent and judge isn't available here; using different snapshots is a
# partial mitigation (see decision log).
GEMINI_AGENT_MODEL = "gemini-3.6-flash"
GEMINI_JUDGE_MODEL = "gemini-3.5-flash"
# This account's free tier caps each Gemini model at ~5 req/min, ~20 req/day
# (observed via RESOURCE_EXHAUSTED errors). Each model name is its own quota
# bucket, so once GEMINI_AGENT_MODEL is exhausted for the day we round-robin
# through same-family flash snapshots rather than blocking on one model's
# reset. Documented as a limitation in report/decision_log.md -- results mix
# outputs from a small ensemble of Gemini flash snapshots, not one fixed model.
GEMINI_AGENT_MODEL_POOL = [
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash",
    "gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.7-flash",
]
# Gemini 3.x flash models always spend some tokens on internal "thinking"
# before the visible answer; budget generously or output gets silently
# truncated to "".
GEMINI_MAX_TOKENS_FLOOR = 2048

# Per-role default model, resolved by whatever backend is active.
DEFAULT_MODEL = GEMINI_AGENT_MODEL if BACKEND == "gemini" else SONNET
JUDGE_MODEL = GEMINI_JUDGE_MODEL if BACKEND == "gemini" else SONNET

if BACKEND == "api":
    import anthropic
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
elif BACKEND == "gemini":
    from google import genai
    from google.genai import types as genai_types
    _gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _call_api(model, system, user, max_tokens, retries):
    last_err = None
    for attempt in range(retries):
        try:
            resp = _client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err


def _call_gemini(model, system, user, max_tokens, retries, json_mode):
    budget = max(max_tokens, GEMINI_MAX_TOKENS_FLOOR)
    config = genai_types.GenerateContentConfig(
        system_instruction=system or None,
        max_output_tokens=budget,
        response_mime_type="application/json" if json_mode else None,
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = _gclient.models.generate_content(model=model, contents=user, config=config)
            text = resp.text
            if not text:
                fr = resp.candidates[0].finish_reason if resp.candidates else None
                raise RuntimeError(f"Empty Gemini response (finish_reason={fr})")
            return text
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err


CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", r"C:\Users\raswani\AppData\Roaming\npm\claude.cmd")


def _call_cli(model, system, user, max_tokens, retries):
    alias = CLI_MODEL_ALIAS.get(model, model)
    full_prompt = f"{system}\n\n{user}" if system else user
    last_err = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                [CLI_PATH, "-p", full_prompt, "--model", alias],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, shell=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:500]}")
            return proc.stdout.strip()
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 15))
    raise last_err


def call(model, system, user, max_tokens=600, retries=5, json_mode=False):
    if BACKEND == "gemini":
        return _call_gemini(model, system, user, max_tokens, retries, json_mode)
    if BACKEND == "api":
        return _call_api(model, system, user, max_tokens, retries)
    return _call_cli(model, system, user, max_tokens, retries)


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Could not parse JSON from model output: {text[:300]!r}")


def call_json(model, system, user, max_tokens=600, retries=5):
    text = call(model, system, user, max_tokens=max_tokens, retries=retries, json_mode=True)
    return _parse_json(text)


def call_json_pool(models, system, user, max_tokens=600, retries_per_model=1):
    """Try each model in `models` in order; on a quota (429/RESOURCE_EXHAUSTED)
    error, move to the next model immediately instead of backing off on an
    exhausted bucket. Other errors still get `retries_per_model` attempts on
    the same model before moving on. Raises the last error if every model
    in the pool fails."""
    last_err = None
    for model in models:
        for attempt in range(retries_per_model):
            try:
                text = call(model, system, user, max_tokens=max_tokens, retries=1, json_mode=True)
                return _parse_json(text), model
            except Exception as e:
                last_err = e
                is_quota = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                if is_quota:
                    break  # don't waste retries on an exhausted model, try next
                time.sleep(min(2 ** attempt, 10))
    raise last_err
