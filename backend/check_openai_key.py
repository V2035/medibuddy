"""
Quick sanity check that your OpenAI API key is set up correctly.
Run: python check_openai_key.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads the .env file in this directory and loads it into os.environ

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY is not set. Put it in a .env file or export it, e.g.:")
    print('  export OPENAI_API_KEY="sk-..."')
    raise SystemExit(1)

client = OpenAI(api_key=api_key)

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'key works' and nothing else."}],
        max_tokens=10,
    )
    print("Success:", resp.choices[0].message.content)
except Exception as e:
    print("Key check failed:", e)
