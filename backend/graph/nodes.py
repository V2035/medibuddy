"""
nodes.py

Each function here is one LangGraph node. Keep the division of labour
explicit:
  - parse_intent_node    -> LLM (natural language understanding needed)
  - fetch_weather_node    -> pure code, zero LLM (facts must be real)
  - match_sop_node        -> LLM, but constrained to only ever return a
                              real SOP id from the loaded list, or "none"
  - compose_answer_node   -> LLM, but constrained to only rephrase the
                              matched SOP's advice + real weather numbers
  - fail_honestly_node    -> pure code, templated, zero LLM
  - no_sop_node            -> pure code, templated, zero LLM
  - match_error_node       -> pure code, templated, zero LLM
"""
from __future__ import annotations

import json

from weather import get_weather_for_city, WeatherLookupError
from .sop_loader import load_sops, sops_as_prompt_block, get_sop_by_id
from .llm_client import call_json, call_text, LLMError
from .state import BotState

_SOPS = load_sops()  # loaded once at import time; edit sops.yaml + restart to pick up changes


# ---------------------------------------------------------------------------
# 1. Parse intent (LLM)
# ---------------------------------------------------------------------------

def parse_intent_node(state: BotState) -> BotState:
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    system = (
        "You extract structured facts from a user's outdoor-activity-safety question. "
        "Return ONLY JSON with keys: location (string or null), activity (short string "
        "describing the activity/context, or null), question_summary (one sentence). "
        "If the user's message is a follow-up (e.g. 'what about this evening?') and doesn't "
        "restate a location, infer it from the conversation history if possible; otherwise "
        "return null for location."
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message: {state['user_message']}\n\n"
        f"Previously resolved location (if any): {state.get('last_location')}"
    )

    try:
        result = call_json(system, user)
    except LLMError as e:
        # If intent parsing itself fails, we can't proceed meaningfully.
        # Treat as an honest failure rather than guessing location/activity.
        state["weather_error"] = f"Could not understand the request: {e}"
        state["weather_error_stage"] = "intent_parsing"
        return state

    location = result.get("location") or state.get("last_location")
    state["location_text"] = location
    state["activity"] = result.get("activity")
    state["question_summary"] = result.get("question_summary")
    return state


# ---------------------------------------------------------------------------
# 2. Fetch weather (pure code, no LLM)
# ---------------------------------------------------------------------------

def fetch_weather_node(state: BotState) -> BotState:
    # If parse_intent already failed (e.g. LLM/API error), don't overwrite
    # its real error with a misleading "no location provided" message --
    # just pass the existing failure through untouched.
    if state.get("weather_error"):
        return state

    location_text = state.get("location_text")

    if not location_text:
        state["weather_error"] = "No location was provided or could be inferred."
        state["weather_error_stage"] = "geocoding"
        return state

    try:
        snapshot = get_weather_for_city(location_text)
    except WeatherLookupError as e:
        state["weather_error"] = e.reason
        state["weather_error_stage"] = e.stage
        return state

    state["weather_current"] = snapshot.current
    state["weather_daily"] = snapshot.daily
    state["resolved_location_name"] = snapshot.location.name
    state["last_location"] = snapshot.location.name
    # Explicitly clear any previous error so branching is unambiguous
    state["weather_error"] = None
    state["weather_error_stage"] = None
    return state


def weather_branch(state: BotState) -> str:
    """Conditional edge: route based on whether weather fetch succeeded."""
    return "failed" if state.get("weather_error") else "ok"


# ---------------------------------------------------------------------------
# 3. Match against SOPs (LLM, constrained to real SOP ids only)
# ---------------------------------------------------------------------------

def match_sop_node(state: BotState) -> BotState:
    sop_block = sops_as_prompt_block(_SOPS)
    valid_ids = {s.id for s in _SOPS}

    system = (
        "You are a strict policy-matching engine. You are given a list of Standard "
        "Operating Procedures (SOPs), live weather facts, and a user's question. "
        "Your ONLY job is to decide which SOP (if any) applies -- you do NOT decide "
        "what advice to give; that's a separate step. Consider the combined situation, "
        "not just single numeric fields in isolation -- an active severe weather system "
        "can apply even if no single number looks extreme. If multiple SOPs could apply, "
        "pick the single most severe/relevant one and explain why in reasoning. "
        "If genuinely no SOP applies, return sop_id: null. "
        "You MUST return ONLY JSON: {\"sop_id\": \"<one of the listed ids>\" or null, "
        "\"reasoning\": \"<one or two sentences>\"}. "
        "Never invent a sop_id that isn't in the list below."
    )
    user = (
        f"SOPs:\n{sop_block}\n\n"
        f"Live weather at {state.get('resolved_location_name')}:\n"
        f"Current: {json.dumps(state.get('weather_current'))}\n"
        f"Daily forecast: {json.dumps(state.get('weather_daily'))}\n\n"
        f"User's activity/context: {state.get('activity')}\n"
        f"User's question: {state.get('question_summary') or state.get('user_message')}"
    )

    try:
        result = call_json(system, user)
    except LLMError as e:
        # IMPORTANT: a broken matching call is NOT the same thing as "no SOP
        # applies". One is a system failure, the other is a genuine policy
        # gap -- conflating them would mislead the user about why they're
        # not getting advice. Route these to a distinct branch/message.
        state["matched_sop_id"] = None
        state["match_reasoning"] = None
        state["match_error"] = str(e)
        return state

    sop_id = result.get("sop_id")
    if sop_id and sop_id not in valid_ids:
        # LLM hallucinated an id that doesn't exist -- hard-block it.
        state["matched_sop_id"] = None
        state["match_reasoning"] = f"Model returned an invalid SOP id ({sop_id}); treated as no match."
        state["match_error"] = None
        return state

    state["matched_sop_id"] = sop_id
    state["match_reasoning"] = result.get("reasoning")
    state["match_error"] = None
    return state


def sop_branch(state: BotState) -> str:
    """Conditional edge: route based on match outcome. Three distinct
    outcomes, each with its own honest message downstream:
      - "matched"    -> a real SOP applies, compose grounded advice
      - "no_match"   -> genuinely no policy covers this, say so
      - "error"      -> the matching step itself broke (e.g. LLM outage),
                        say THAT plainly rather than implying no policy exists
    """
    if state.get("match_error"):
        return "error"
    return "matched" if state.get("matched_sop_id") else "no_match"


# ---------------------------------------------------------------------------
# 4a. Compose answer from matched SOP (LLM, constrained)
# ---------------------------------------------------------------------------

def compose_answer_node(state: BotState) -> BotState:
    sop = get_sop_by_id(_SOPS, state["matched_sop_id"])

    system = (
        "You write the final reply to the user. You MUST base your advice strictly on "
        "the SOP advice text given below -- you may rephrase it conversationally, but "
        "must not add advice, caveats, or facts beyond what's written there. "
        "Any weather numbers you mention (temperature, wind, rain, etc.) must come "
        "exactly from the live weather data given below -- never estimate or recall a "
        "number from memory. Mention the SOP id for traceability. Keep it concise and warm."
    )
    user = (
        f"Matched SOP: {sop.id} (severity: {sop.severity})\n"
        f"SOP advice to relay: {sop.advice}\n\n"
        f"Live weather at {state.get('resolved_location_name')}:\n"
        f"{json.dumps(state.get('weather_current'))}\n\n"
        f"User's question: {state['user_message']}"
    )

    try:
        answer = call_text(system, user)
    except LLMError as e:
        # Even composition failing must not produce a guess -- fall back to
        # the raw SOP advice text verbatim rather than inventing anything.
        answer = (
            f"(Note: reply composition had an issue, showing policy advice directly.)\n"
            f"Per policy {sop.id}: {sop.advice}"
        )

    state["final_answer"] = answer
    return state


# ---------------------------------------------------------------------------
# 4b. No SOP applies (pure code, templated, zero LLM)
# ---------------------------------------------------------------------------

def no_sop_node(state: BotState) -> BotState:
    state["final_answer"] = (
        f"I don't have guidance for that specific situation yet — I checked the current "
        f"policies and none of them clearly cover this case, so I'd rather be honest than "
        f"guess. (For reference: {state.get('match_reasoning', 'no matching policy found.')})"
    )
    return state


# ---------------------------------------------------------------------------
# 4c. System error during matching (pure code, templated, zero LLM)
# ---------------------------------------------------------------------------

def match_error_node(state: BotState) -> BotState:
    """
    Distinct from no_sop_node on purpose: this is NOT "no policy covers this",
    it's "the system had a technical failure while checking policy". Telling
    the user the true reason avoids implying a policy gap that doesn't
    actually exist.
    """
    state["final_answer"] = (
        f"I've got real weather data for this, but I hit a technical error while "
        f"checking it against policy (a temporary issue with the matching service: "
        f"{state.get('match_error')}). Rather than guess, please try asking again "
        f"in a moment."
    )
    return state


# ---------------------------------------------------------------------------
# 5. Fail honestly (pure code, templated, zero LLM)
# ---------------------------------------------------------------------------

def fail_honestly_node(state: BotState) -> BotState:
    stage = state.get("weather_error_stage")
    reason = state.get("weather_error")

    if stage == "geocoding":
        msg = (
            f"I couldn't figure out the location you mean ({reason}). "
            f"Could you clarify the city or place name?"
        )
    elif stage == "forecast":
        msg = (
            f"I found the location, but couldn't get live weather data right now "
            f"({reason}). I don't want to guess, so please try again in a moment."
        )
    else:
        msg = f"I couldn't process that request ({reason}). Please try rephrasing."

    state["final_answer"] = msg
    return state