# GetClaimed

Insurance claims support agent with a fixed SOP: **VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS**.

The model talks naturally and chooses tools. The harness owns phase order, identity gates, and what tools are allowed. Claim facts come from fixture tools, not from the model inventing a file.

Repo: [Hyperion101010/getclaimed_harness](https://github.com/Hyperion101010/getclaimed_harness)

## How it works

1. **VERIFY_ID** — Tools: `verify_identity`, `remember`, `escalate_to_human`. Needs **3 of 5**: name, DOB, phone, email, SSN/ID last 4. Policy number is lookup only. No claim details until verified.
2. **RESOLVE_INTENT** — `list_claims` / `get_claim`. Hints stored during ID (type, status, month) are reused. Unique match selects the case.
3. **PROCESS_CASE** — Answers from `get_claim` and `get_document_guidance` only.
4. **POST_PROCESS** — Offer an email summary. Send only after an explicit yes (`send_email_summary` with `confirmed=true`). Skip closes the session.

`remember` can store a hint like “denied healthcare January” while still in VERIFY_ID.

## Setup (local)

Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Optional: `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`).

```bash
export PYTHONPATH=.
uvicorn apps.insurance_claims.api:app --reload --port 8000
```

Open http://127.0.0.1:8000

```bash
pytest apps/insurance_claims/tests -q
```

Live OpenRouter tests need a key and network. Offline tool/SOP tests do not.

## Demo script

Reload the page for a new session. Paste:

```text
I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.
```

Expect: verify first (no CL-2048 before ID), then **CL-2048 only**. Ask why it was denied, then wrap up and send or skip email.

More scripts: [`apps/insurance_claims/sample_test_cases.md`](apps/insurance_claims/sample_test_cases.md)

Fixture callers (3 of 5):

| Caller | Policy | DOB | Last 4 |
|---|---|---|---|
| Margaret Chen | POL-9921 | 1985-03-15 | 4472 |
| Ma Tian | POL-8836 | 1964-09-10 | 6688 |
| Ava Lopez | POL-1044 | 1990-08-21 | 9180 |
| Ya Wen Li / Yaven Li | POL-7742 | 1989-12-03 | 5317 |

Ava and Ya Wen have no claims on file.

## Docker

```bash
docker build -t getclaimed .
docker run --rm -p 8000:8000 --env-file .env getclaimed
```

Then http://127.0.0.1:8000

## Railway

The app is one FastAPI process (UI + `/chat`). Use a **single replica**. In-memory sessions are per process; that is fine on one container, not on multi-instance serverless.

1. Connect this GitHub repo in Railway (Dockerfile at repo root).
2. Variables:
   - `OPENROUTER_API_KEY` (required)
   - `OPENROUTER_MODEL=openai/gpt-4o-mini` (optional)
   - `PORT=8000` (matches the Dockerfile listen port)
3. Networking → Generate domain.
4. Check `https://<your-domain>/healthz` then `/`.

If the service never becomes healthy, set custom start command:

```text
uvicorn apps.insurance_claims.api:app --host 0.0.0.0 --port $PORT
```

and drop the `PORT=8000` variable so Railway’s `$PORT` is used.

Do not commit `.env`. Use Railway / `.env` locally only.
