"""
debug_parse_intent.py
Isolates just the parse_intent LLM call so we can see exactly what
Gemini returns, without the rest of the graph in the way.
Run: python debug_parse_intent.py
"""
from graph.llm_client import call_json, LLMError

system = (
    "You extract structured facts from a user's outdoor-activity-safety question. "
    "Return ONLY JSON with keys: location (string or null), activity (short string "
    "describing the activity/context, or null), question_summary (one sentence). "
    "If the user's message is a follow-up (e.g. 'what about this evening?') and doesn't "
    "restate a location, infer it from the conversation history if possible; otherwise "
    "return null for location."
)
user = (
    "Conversation so far:\n\n\n"
    "Latest user message: is it safe to cycle in Bhopal today?\n\n"
    "Previously resolved location (if any): None"
)

print("Calling Gemini...")
try:
    result = call_json(system, user)
    print("Parsed result:", result)
except LLMError as e:
    print("LLMError:", e)