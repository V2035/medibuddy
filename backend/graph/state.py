"""
state.py

The state object LangGraph threads through every node. One instance of
this represents "everything known so far about the current turn", plus
the running conversation history for session memory.
"""
from __future__ import annotations

from typing import TypedDict, Optional, List, Dict, Any


class BotState(TypedDict, total=False):
    # --- session memory (persists across turns within one chat session) ---
    messages: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    last_location: Optional[str]    # last resolved location, for follow-up questions

    # --- current turn inputs ---
    user_message: str

    # --- parse_intent output ---
    location_text: Optional[str]
    activity: Optional[str]
    question_summary: Optional[str]

    # --- fetch_weather output ---
    weather_current: Optional[Dict[str, Any]]
    weather_daily: Optional[Dict[str, Any]]
    resolved_location_name: Optional[str]
    weather_error: Optional[str]       # set only on failure
    weather_error_stage: Optional[str]  # "geocoding" | "forecast"

       # --- match_sop output ---
    matched_sop_id: Optional[str]
    match_reasoning: Optional[str]
    match_error: Optional[str]  # set only when the matching LLM call itself fails

    # --- final output ---
    final_answer: str
