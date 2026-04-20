# TalentSync AI — Backend

Multi-agent AI talent intelligence platform powered by Google Gemini. Processes resumes through a 3-agent pipeline (Parser → Normalizer → Matcher), stores results persistently, and exposes a complete REST API.

---

## Quick Start

### 1. Prerequisites 
- Python 3.10+
- An Gemini API key (`AIza-..`)

### 2. Setup

```bash
# Clone / unzip the project, then:
cd talentsync-backend

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=sk-ant-your-key-here
```

### 3. Run

```bash
bash start.sh
# or directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app

- **Frontend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs (interactive Swagger UI)
- **Health check:** http://localhost:8000/health

---

## Architecture

```
talentsync-backend/
├── main.py              ← FastAPI app, all route definitions
├── agents.py            ← AI agents (Parser, Normalizer, Matcher)
├── database.py          ← SQLite persistence layer
├── models.py            ← Pydantic request/response schemas
├── utils.py             ← File extraction (PDF, DOCX, TXT)
├── requirements.txt
├── start.sh
├── .env.example
└── static/
    ├── index.html       ← Full integrated frontend (with backend hooks)
    └── backend-integration.js  ← Backend API layer (auto-loaded)
```

---

## API Reference

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/pipeline/run` | Full 3-agent pipeline — parse, normalize, match |

**Request** (`multipart/form-data`):
- `file` — resume file (PDF, TXT, DOCX) — OR —
- `resume_text` — pasted resume text
- `job_description` — optional JD to match against
- `filename` — optional label

**Response:**
```json
{
  "candidate_id": "uuid",
  "parsed": { "name": "...", "title": "...", "skills": [...], ... },
  "normalized_skills": [{ "raw": "...", "normalized": "...", "category": "...", "level": "..." }],
  "match": { "match_score": 87, "verdict": "Strong Match", "strengths": [...], "gaps": [...] },
  "pipeline_logs": [...],
  "pipeline_time_ms": 4200
}
```

---

### Candidates

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/candidates` | List all candidates (paginated) |
| `GET` | `/api/candidates/{id}` | Get full candidate profile |
| `DELETE` | `/api/candidates/{id}` | Delete a candidate |

---

### Job Descriptions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/jobs` | Save a job description |
| `GET` | `/api/jobs` | List all saved jobs |
| `GET` | `/api/jobs/{id}` | Get a job |

---

### Skills

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/skills/normalize` | Normalize a single skill with full taxonomy |
| `POST` | `/api/skills/batch-normalize` | Normalize a list of skills |
| `POST` | `/api/skills/report` | Generate AI prose skill report |

---

### Matching

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/match/quick` | Quick candidate vs JD match |
| `POST` | `/api/match/semantic` | Full semantic match with seniority/technical fit, offer percentile |
| `POST` | `/api/match/batch` | Rank multiple candidates against one JD |

---

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/analytics/dashboard` | Aggregate stats: score distribution, skill categories, top skills |
| `GET` | `/api/analytics/leaderboard` | Top candidates ranked by match score |

---

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Context-aware chat with session state + conversation history |

---

### Claude Proxy

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/gemini/proxy` (or `/api/claude/proxy` for backward compatibility) | Secure Google API proxy — keeps key server-side |

---

## How the Frontend Connects

The `static/index.html` file is the original TalentSync AI frontend with the backend integration layer injected. It works in two modes:

**Backend mode** (default, `USE_BACKEND = true`):
- All AI calls go through `/api/gemini/proxy` (or `/api/claude/proxy` for backward compatibility) — API key stays on server
- Pipeline results are stored in SQLite and persist across sessions
- Candidate list sidebar loads from DB on page open
- Dashboard analytics sync with historical data
- Leaderboard shows top candidates from all sessions.

**Standalone mode** (`USE_BACKEND = false` or backend unreachable):
- Falls back to direct Google API calls (requires Claude artifact runner environment)
- State is in-memory only, resets on refresh

To switch modes, edit the top of `static/index.html`:
```js
window.USE_BACKEND = false; // standalone mode
```

---

## Database

SQLite database (`talentsync.db`) is created automatically on first run. Tables:

- **candidates** — parsed profiles, normalized skills, match results
- **jobs** — saved job descriptions
- **skill_cache** — normalized skill lookup cache (speeds up repeated normalizations)
- **chat_sessions** — chat history (reserved for future multi-turn support)

The DB file is local to the project directory. Back it up before deploying to production — it contains all your candidate data.

---

## Deployment (Production)

For production, we recommend:

1. **Use a proper ASGI server:** `gunicorn -k uvicorn.workers.UvicornWorker main:app`
2. **Put Nginx in front** for HTTPS, static file serving, and reverse proxy
3. **Migrate to PostgreSQL** — swap `database.py` to use `asyncpg` or `SQLAlchemy`
4. **Set `allow_origins`** to your specific domain in the CORS config in `main.py`

Basic Nginx config:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Agent Details

### Agent 1 — Parser
Sends raw resume text to `gemini-2.0-flash` and extracts a structured JSON candidate profile:
name, title, location, email, summary, experience (array), skills (array), education, certifications, years of experience, and confidence scores.

### Agent 2 — Normalizer
Takes the raw skill strings from the Parser and maps them to canonical industry names with:
- Standard name (e.g., `ReactJS` → `React`)
- Category (`Frontend | Backend | AI/ML | DevOps | Cloud | Data | Mobile | Other`)
- Proficiency level (`Foundational | Intermediate | Advanced | Expert`)

### Agent 3 — Matcher
Compares the candidate profile against the job description and returns:
- Numerical match score (0–100)
- Verdict (`Strong/Good/Moderate/Weak Match`)
- Specific strengths with evidence
- Skill gaps with severity
- 2–3 sentence recruiter recommendation
- 3 interview questions
- Hiring signal (`proceed | consider | pass`)

---

## Known Limitations

- **PDF parsing:** `pypdf` handles most PDFs well. Scanned/image-only PDFs won't extract text — users should paste text instead.
- **1000 token cap:** Each agent call uses `max_tokens=1000`. Very long resumes may get truncated. Increase in `agents.py` if needed (costs more).
- **SQLite concurrency:** Fine for a hackathon or small team. Use PostgreSQL for high concurrency.
- **No auth:** The API has no authentication. Add an API key header or session-based auth before exposing publicly.

---

## Hackathon Tips

- Run the demo mode first (`Try Live Demo` button) to verify everything works end-to-end
- The `/docs` endpoint gives you a live Swagger UI to test all API endpoints
- Use `/api/match/batch` to rank multiple uploaded candidates against a single JD — great demo
- The leaderboard section updates automatically as you process resumes.
