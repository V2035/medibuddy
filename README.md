# Weather-Advisory Support Bot

A LangGraph-backed chat agent that answers outdoor-activity-safety questions
("is it safe to cycle today?") using live Open-Meteo weather data, matched
against a written set of Standard Operating Procedures (SOPs). The bot never
invents advice — every answer either cites a specific SOP or honestly says
no policy covers the question.

## Architecture

```
parse_intent -> fetch_weather --(ok)--> match_sop --(matched)--> compose_answer -> END
                              \                    \--(no_match)--> no_sop -> END
                               \                    \--(error)-----> match_error -> END
                                (failed)
                                 v
                           fail_honestly -> END
```

- **parse_intent** (LLM): extracts location, activity, and a one-line question
  summary from the user's message + recent conversation history.
- **fetch_weather** (pure code, zero LLM): geocodes the location and calls
  Open-Meteo. Raises `WeatherLookupError` on any failure — never fabricates
  a forecast.
- **match_sop** (LLM, constrained): given the SOPs (condition text only, not
  advice text) and live weather, picks a single SOP id or `null`. A
  hallucinated id that isn't in the loaded list is hard-blocked and treated
  as no match.
- **compose_answer** (LLM, constrained): rephrases *only* the matched SOP's
  advice text plus the real fetched weather numbers. Not allowed to add
  facts or advice beyond that.
- **no_sop / fail_honestly / match_error** (pure code, templated): three
  distinct honest-failure paths, kept separate on purpose — "no policy
  covers this," "I couldn't get weather data," and "the matching step
  itself broke" are different situations and get different messages.

SOPs live in `backend/sops/sops.yaml` — 11 rules across 4 categories
(outdoor exercise, travel, vulnerable groups, general activity), spanning
all four severities, including one fuzzy non-numeric rule (`SOP-008`,
picnic-style suitability). Adding or editing a rule means editing that YAML
file only — no code in `graph/` ever needs to change.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GOOGLE_API_KEY to a real Gemini API key
```

## Running it

**Terminal mode** (fastest way to sanity-check the graph):
```bash
cd backend
python chat_cli.py
```

**Web mode** (backend + frontend, what the review call should use):
```bash
# terminal 1
cd backend
uvicorn main:app --reload --port 8000

# terminal 2 — just open the file, no build step needed
open frontend/index.html      # or double-click it / drag into a browser
```
`frontend/index.html` calls `http://127.0.0.1:8000` directly, so no static
file server is required — opening the HTML file works.

## Adding an 11th SOP live (no code changes)

Append a new block to `backend/sops/sops.yaml` following the existing
schema (id, category, severity, condition, optional thresholds, activities,
advice), then restart the backend so `sop_loader.load_sops()` re-reads the
file at import time. `match_sop_node` and `compose_answer_node` pick it up
automatically — nothing in `graph/nodes.py` or `graph/build_graph.py` needs
to change.

## Eval suite

```bash
cd backend
python evals/eval_suite.py
```

Runs the real compiled graph (real Gemini calls, real Open-Meteo calls) —
needs network access and a working `GOOGLE_API_KEY`. Covers:
1. Two clear-match cases (direct wording).
2. Two paraphrased-intent cases (deliberately avoid the SOP's own wording,
   to check matching isn't just keyword lookup).
3. One live-severe-weather case grounded in real fetched numbers for
   whatever conditions are actually live for that request (not a
   hardcoded event).
4. One case where no SOP applies — must decline honestly.
5. One simulated weather-API outage (monkeypatches the fetch function) —
   must fail honestly, not guess.
6. One adversarial prompt-injection case — the user tries to get the bot to
   give safety advice with no policy citation. Chosen over other adversarial
   angles because it directly attacks the "every answer traceable to an SOP"
   non-negotiable, which is the requirement the whole design exists to
   protect.

Each case prints what's being checked, what a pass looks like, and the
actual result — run it yourself and record the real pass/fail counts here
before submitting; don't assume they all pass without running them.

**Known limitation of case 3**: it targets Bhopal because that was the
assignment's live example. By the time this is reviewed, that specific
rain system will likely have passed (the assignment brief itself notes
this). The case still validates grounding in *whatever* Open-Meteo
returns for that request today — it does not hardcode the September rain
numbers — but if Bhopal happens to have unremarkable weather when you run
it, the SOP that fires may differ from a "danger"-severity one. That's
expected; re-point the case at wherever genuinely severe weather is active
if you want to specifically exercise `SOP-004`.

## Known gaps / honest notes

- `check_gemini_key.py`, `check_openai_key.py`, `debug_node.py`,
  `debug_parse_intent.py`, `debug_graph.py` are ad-hoc scripts used during
  development, not part of the graded deliverable — safe to ignore or
  delete before submitting if you want a cleaner repo.
- Session memory is a plain in-process dict in `main.py`, keyed by
  `session_id`, matching the spec ("memory resets between sessions, not
  asking for persistence across restarts"). It is lost on server restart
  by design.
- Have not yet been able to run `evals/eval_suite.py` end-to-end against a
  live API key in this environment — run it yourself and paste real
  pass/fail results into this README (or a short write-up doc) before
  submitting, since "we'd rather see honest gaps than a suite guaranteed
  to pass" is explicitly what's being evaluated.
