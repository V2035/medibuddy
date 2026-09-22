"""
Quick sanity check that your Google Gemini API key is set up correctly.
Uses the current `google-genai` SDK (the older `google-generativeai`
package is fully deprecated as of 2026).
Run: python check_gemini_key.py
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads the .env file and loads it into the environment

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY is not set. Put it in your .env file, e.g.:")
    print("  GOOGLE_API_KEY=AIza...")
    raise SystemExit(1)

client = genai.Client(api_key=api_key)

try:
    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Say 'key works' and nothing else.",
    )
    print("Success:", resp.text.strip())
except Exception as e:
    print("Key check failed:", e)
