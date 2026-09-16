"""The support agent: classify intent, draft a grounded reply, decide
auto-handle vs escalate -- in a single model call per message (keeps cost and
latency down, and avoids the classifier and the reply-writer disagreeing
with each other about what the message is even about).
"""
import json
from taxonomy import INTENTS, ESCALATION_KEYWORDS
from llm_client import call_json, call_json_pool, DEFAULT_MODEL, GEMINI_AGENT_MODEL_POOL, BACKEND
from retrieval import Retriever

SYSTEM_PROMPT = f"""You are a support-ticket triage assistant for SpotifyCares, \
Spotify's customer support Twitter account. You are shown one incoming \
customer message plus a few examples of how SpotifyCares has actually \
replied to similar historical messages. Your job:

1. Classify the message into exactly one of these intents:
{json.dumps(INTENTS, indent=2)}

2. Draft a reply in SpotifyCares' voice (friendly, casual, uses "Hey!", \
signs off with helpfulness, no corporate stiffness) that is CONSISTENT with \
how the historical examples resolved similar issues. Do not invent facts \
about Spotify's product (limits, prices, policies) beyond what the examples \
show or what is safely generic (e.g. asking for device/OS info).

3. Decide escalate=true if, and only if, resolving this message requires \
touching the customer's real account or payment record (a personalized \
lookup), OR the message exposes personal info (email, password, card \
digits) that needs to move to a private channel, OR it contains legal \
threats, self-harm, or fraud language, OR the message is unintelligible \
spam with no real support content. Otherwise escalate=false -- general \
troubleshooting, informational/policy questions, content availability, \
redirects to another team, feature feedback, and gratitude should NOT be \
escalated, even if the customer is angry. If the message is too vague to \
classify confidently, escalate=false and draft a clarifying question \
instead of guessing.

Respond with ONLY a JSON object, no markdown fences:
{{"intent": "<one of the intent keys>", "confidence": <0-1 float>, \
"escalate": <true|false>, "escalate_reason": "<one sentence, empty string if not escalating>", \
"draft_reply": "<the reply text>"}}
"""


class Agent:
    def __init__(self, corpus_csv="data/processed/corpus.csv", model=DEFAULT_MODEL, k=3,
                 use_model_pool=None):
        self.retriever = Retriever(corpus_csv)
        self.model = model
        self.k = k
        # round-robin across same-family Gemini flash snapshots when one hits
        # its per-model free-tier quota -- see llm_client.GEMINI_AGENT_MODEL_POOL
        self.use_model_pool = use_model_pool if use_model_pool is not None else (BACKEND == "gemini")

    def _build_user_prompt(self, customer_message, context_before=""):
        examples = self.retriever.top_k(customer_message, k=self.k)
        ex_block = "\n\n".join(
            f"Example {i+1} (similarity={e['similarity']:.2f}):\n"
            f"  Customer: {e['customer_message']}\n"
            f"  SpotifyCares replied: {e['brand_reply']}"
            for i, e in enumerate(examples)
        )
        ctx = f"Earlier customer messages in this thread: {context_before}\n" if context_before else ""
        return (
            f"{ctx}Customer message to handle now: {customer_message}\n\n"
            f"Historical examples of similar issues and how they were resolved:\n{ex_block}"
        ), examples

    def handle(self, customer_message, context_before=""):
        user_prompt, examples = self._build_user_prompt(customer_message, context_before)
        if self.use_model_pool:
            result, used_model = call_json_pool(GEMINI_AGENT_MODEL_POOL, SYSTEM_PROMPT, user_prompt, max_tokens=700)
            result["_model_used"] = used_model
        else:
            result = call_json(self.model, SYSTEM_PROMPT, user_prompt, max_tokens=700)
            result["_model_used"] = self.model
        result["retrieved_examples"] = examples
        result.setdefault("escalate_reason", "")
        # safety-net keyword override: never let the model auto-handle past an
        # explicit legal/self-harm/fraud signal, regardless of what it returned
        low = customer_message.lower()
        if any(kw in low for kw in ESCALATION_KEYWORDS) and not result.get("escalate"):
            result["escalate"] = True
            result["escalate_reason"] = (result.get("escalate_reason") or "") + " [keyword safety-net triggered]"
        return result
