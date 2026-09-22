"""
llm_client.py

All calls to the LLM go through here. Two things this file guarantees:
  1. Structured calls (parse_intent, match_sop) always ask for JSON and
     parse it safely, so callers get Python dicts, not raw text to
     regex out.
  2. This is the ONLY file that knows which model/provider we're using.
     Swapping providers later (Gemini -> OpenAI/Anthropic) means editing
     this file only -- graph nodes never call the API directly.
"""
from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
_api_key = os.environ.get("GOOGLE_API_KEY")

_client = genai.Client(api_key=_api_key) if _api_key else None


class LLMError(Exception):
    """Raised when the LLM call itself fails (network, auth, etc.) or
    returns something that can't be parsed as the JSON we asked for."""
    pass


def call_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
    """
    Calls the model with a system+user prompt, forcing JSON output.
    Returns a parsed Python dict. Raises LLMError on any failure --
    callers must not silently substitute a default; a broken LLM call
    should be treated as a real failure, not papered over.
    """
    if _client is None:
        raise LLMError("GOOGLE_API_KEY is not set.")

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        resp = _client.models.generate_content(
            model=_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        raise LLMError(f"LLM call failed: {e}") from e

    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM did not return valid JSON: {e}\nRaw output: {text}") from e


def call_text(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """
    Calls the model for a plain natural-language reply (used only for the
    final answer composition step, where we WANT prose, not JSON).
    """
    if _client is None:
        raise LLMError("GOOGLE_API_KEY is not set.")

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        resp = _client.models.generate_content(
            model=_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
    except Exception as e:
        raise LLMError(f"LLM call failed: {e}") from e

    return (resp.text or "").strip()
