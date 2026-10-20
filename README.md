# Insurance Claims SOP Agent

Phase-gated claims support agent: `VERIFY_ID` → `RESOLVE_INTENT` → `PROCESS_CASE` → `POST_PROCESS`.

SOP (phase order, 3-of-5 identity match, tool gates, email consent) is enforced in code. The model extracts slots from messy/transcript text and writes replies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Optional: `OPENROUTER_MODEL` (any OpenRouter chat model).

```bash
export PYTHONPATH=.
uvicorn apps.insurance_claims.api:app --reload --port 8000
```

Open http://127.0.0.1:8000 — chat on the left, session debug (phase, verified, slots) on the right.

```bash
pytest apps/insurance_claims/tests -q
```

## Docker

```bash
docker build -t insurance-sop .
docker run --rm -p 8000:8000 --env-file .env insurance-sop
```
