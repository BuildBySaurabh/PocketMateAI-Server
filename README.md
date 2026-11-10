
# PocketMateAI-Server
# PocketMate backend

Small FastAPI server between the phone app and the AI provider. The AI key stays here.

```
python -m venv .venv
.venv\Scripts\activate           (Windows)   or   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
copy .env.example .env           (Windows)   or   cp .env.example .env
# edit .env: set APP_TOKEN to a long random value
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
python -m pytest -q
```

Endpoints: `GET /health` (no token), `POST /v1/chat` (header `Authorization: Bearer <APP_TOKEN>`,
body `{"messages":[{"role":"user","content":"hi"}],"memory":["optional facts"]}`).
Errors are JSON `{"code","message"}` with codes `unauthorized`, `rate_limited`, `model_unavailable`,
`bad_request`, `server_error`.

Run it from this folder so `.env` is found. See `../docs/AI_PROVIDER_AND_API_KEY.md`.


## Optional request fields (added with the app upgrades)

`POST /v1/chat` still accepts just `messages` (and `memory`). It also accepts, all optional:

| Field | Meaning | Limits |
|---|---|---|
| `persona` | Short style hint chosen in the app | 400 characters |
| `context` | Passages from the user's own documents, picked on the phone | 6 passages, 1500 characters each |
| `web_search` | The user allowed web search for this question | boolean |
| `image` | `{ "media_type": "image/jpeg\|png\|webp", "data": "<base64>" }` one picture the user chose to send | about 2 MB |

The reply is `{ "reply": "...", "web_search_used": false }`. Document passages and web results are given to the model
as reference text with a warning not to follow instructions inside them. `GET /health` also reports whether web
search is configured. To enable search set `SEARCH_PROVIDER=brave` and `SEARCH_API_KEY=...` in `.env`; the user's
question text is then sent to that search service when the app asks for it. If search fails the answer is still
returned without it. Nothing here has been run against the real Anthropic or Brave services, only against fakes.