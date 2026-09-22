# Weather-Advisory Support Bot

A chatbot that answers questions like "is it safe to cycle today?" using
real, live weather data. It never makes up safety advice — every answer
either comes from a written rule (called an SOP) or the bot honestly says
it doesn't have guidance for that situation.

## How it works

1. You ask a question (e.g. "is it safe to bike in Bhopal today?")
2. The bot figures out the location and activity from your question
3. It fetches real, live weather for that location
4. It checks the weather against a list of rules (SOPs) we've written
5. If a rule matches, it gives you that rule's advice, grounded in the
   real weather numbers
6. If no rule matches, it says so honestly instead of guessing

All of this runs as a **LangGraph** agent — a flowchart-like pipeline
with branching, not just one big prompt.

## What's in this project

- `backend/` — the actual bot: the LangGraph agent, weather-fetching
  code, and the SOP rules
- `frontend/` — a simple webpage to chat with the bot
- `backend/evals/` — automated tests that check the bot behaves correctly

## Setup (do this once)

```bash
cd backend
python -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Then set up your API key:
```bash
cp .env.example .env
```
Open `backend/.env` and paste in your Gemini API key on this line: