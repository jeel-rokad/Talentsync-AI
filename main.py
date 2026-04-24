"""
TalentSync AI — FastAPI Backend v3.0
100% Problem Statement Coverage:
  ✓ Multi-format resume parsing (PDF multi-column, DOCX, TXT)
  ✓ Skill taxonomy + normalization + emerging skill detection + hierarchy inference
  ✓ Semantic + vector embedding matching with upskilling paths
  ✓ Orchestrator with per-agent metrics and graceful degradation
  ✓ API key authentication, rate limiting
  ✓ Async batch pipeline with job tracking + webhook callbacks
  ✓ Full REST API: taxonomy, leaderboard, analytics, chat, SDKs
"""

import os, json, uuid, time, re, asyncio
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Optional, List
from collections import Counter

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form,
    Request, Depends, Security, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security.api_key import APIKeyHeader

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import google.genai as genai
from google.genai import types
import httpx

from database import db
from models import (
    ParseRequest, NormalizeRequest, MatchRequest,
    ChatRequest, SkillQueryRequest, SemanticMatchRequest,
    JobDescriptionCreate, BatchPipelineRequest, BatchPipelineItem,
    ThresholdMatchRequest, WebhookTestRequest, EmergingSkillReviewRequest,
)
from agents import (
    ParserAgent, NormalizerAgent, MatcherAgent,
    EmbeddingAgent, Orchestrator, MODEL, _call_gemini, _parse_json,
    SKILL_HIERARCHY,
)
from utils import extract_text_from_file, sanitize_text, compute_quality_score

# ── App & Middleware ──────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app = FastAPI(
    title="TalentSync AI API",
    version="3.0.0",
    description=(
        "Multi-agent AI talent intelligence platform. "
        "Parse resumes, normalize skills, match candidates to jobs. "
        "Powered by Google Gemini + vector embeddings."
    ),
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini Client ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "placeholder")
client = genai.Client(api_key=GEMINI_API_KEY)

# ── API Key Authentication ────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_KEYS = set(
    k.strip() for k in os.environ.get("TALENTSYNC_API_KEYS", "").split(",") if k.strip()
)


async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)):
    """
    If TALENTSYNC_API_KEYS is set in .env, all requests must carry a valid
    X-API-Key header. If the env var is empty, auth is disabled (dev mode).
    """
    if not _VALID_KEYS:
        return "dev-mode"
    if api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Add X-API-Key header.",
        )
    return api_key


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(start: float) -> int:
    return int((time.time() - start) * 1000)


DEFAULT_JD = (
    "Senior Software Engineer — Python, distributed systems, "
    "5+ years experience, cloud platforms, REST APIs."
)


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    stats = db.get_stats()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model": MODEL,
        "ai_provider": "Google Gemini",
        "auth_enabled": bool(_VALID_KEYS),
        **stats,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE — Single resume
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/pipeline/run", tags=["Pipeline"])
@limiter.limit("20/minute")
async def run_pipeline(
    request: Request,
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    filename: Optional[str] = Form("resume"),
    _key=Depends(verify_api_key),
):
    """
    Full 3-agent pipeline: parse → normalize → embed → match.
    Upload a file (PDF/DOCX/TXT) or paste resume_text.
    """
    if file:
        try:
            content = await file.read()
            text = extract_text_from_file(content, file.filename)
            filename = file.filename
        except Exception as e:
            raise HTTPException(400, f"Could not read file: {e}")
    elif resume_text:
        text = resume_text.strip()
        if len(text) < 30:
            raise HTTPException(400, "Resume text too short (minimum 30 characters)")
    else:
        raise HTTPException(400, "Provide either a file upload or resume_text")

    text = sanitize_text(text)
    jd = job_description or DEFAULT_JD

    orchestrator = Orchestrator(client)
    try:
        parsed, normalized, emerging, match_data, logs, total_ms = \
            await orchestrator.run_pipeline(text, jd, filename or "resume")
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    # Persist emerging skills for review
    for skill in emerging:
        db.record_emerging_skill(skill)

    # Cache normalized skills
    for sk in normalized:
        if sk.get("raw"):
            db.cache_skill(sk["raw"], sk)

    candidate_id = str(uuid.uuid4())
    db.save_candidate({
        "id": candidate_id,
        "filename": filename,
        "parsed": parsed,
        "normalized_skills": normalized,
        "match_data": match_data,
        "job_description": jd,
        "pipeline_time_ms": total_ms,
        "created_at": datetime.utcnow().isoformat(),
    })

    return {
        "candidate_id": candidate_id,
        "parsed": parsed,
        "normalized_skills": normalized,
        "match": match_data,
        "pipeline_logs": logs,
        "pipeline_time_ms": total_ms,
        "emerging_skills_detected": emerging,
        "summary": {
            "name": parsed.get("name", "Unknown"),
            "title": parsed.get("title", ""),
            "skills_count": len(normalized),
            "match_score": match_data.get("match_score", 0),
            "verdict": match_data.get("verdict", ""),
            "hiring_signal": match_data.get("hiring_signal", "consider"),
            "vector_similarity_score": match_data.get("vector_similarity_score"),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE — Async batch with job tracking
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/pipeline/batch", tags=["Pipeline"])
@limiter.limit("5/minute")
async def batch_pipeline(
    request: Request,
    req: BatchPipelineRequest,
    background_tasks: BackgroundTasks,
    _key=Depends(verify_api_key),
):
    """
    Submit a batch of up to 50 resumes for async processing.
    Returns a job_id immediately. Poll GET /api/pipeline/batch/{job_id} for status.
    Optionally provide webhook_url to receive a POST callback on completion.
    """
    if not req.items:
        raise HTTPException(400, "Provide at least one resume item")

    job_id = str(uuid.uuid4())
    db.save_batch_job({
        "id": job_id,
        "status": "pending",
        "total": len(req.items),
        "webhook_url": req.webhook_url or "",
        "created_at": datetime.utcnow().isoformat(),
    })
    background_tasks.add_task(_process_batch, job_id, req)

    return {
        "job_id": job_id,
        "status": "pending",
        "total": len(req.items),
        "message": (
            f"Batch job queued. Poll GET /api/pipeline/batch/{job_id} for status. "
            + (f"Webhook will POST to {req.webhook_url} on completion." if req.webhook_url else "")
        ),
    }


async def _process_batch(job_id: str, req: BatchPipelineRequest):
    """Background task: process each resume sequentially, update job record."""
    db.update_batch_job(job_id, {"status": "processing"})
    results = []
    orchestrator = Orchestrator(client)
    shared_jd = req.job_description or DEFAULT_JD

    for i, item in enumerate(req.items):
        try:
            text = sanitize_text(item.resume_text)
            jd = item.job_description or shared_jd
            parsed, normalized, emerging, match_data, logs, total_ms = \
                await orchestrator.run_pipeline(text, jd, item.filename or "resume")

            for skill in emerging:
                db.record_emerging_skill(skill)
            for sk in normalized:
                if sk.get("raw"):
                    db.cache_skill(sk["raw"], sk)

            candidate_id = str(uuid.uuid4())
            db.save_candidate({
                "id": candidate_id,
                "filename": item.filename or "resume",
                "parsed": parsed,
                "normalized_skills": normalized,
                "match_data": match_data,
                "job_description": jd,
                "pipeline_time_ms": total_ms,
                "created_at": datetime.utcnow().isoformat(),
            })

            results.append({
                "index": i,
                "candidate_id": candidate_id,
                "name": parsed.get("name", "Unknown"),
                "title": parsed.get("title", ""),
                "match_score": match_data.get("match_score", 0),
                "verdict": match_data.get("verdict", ""),
                "hiring_signal": match_data.get("hiring_signal", "consider"),
                "vector_similarity_score": match_data.get("vector_similarity_score"),
                "pipeline_time_ms": total_ms,
                "status": "success",
            })
        except Exception as e:
            results.append({"index": i, "filename": item.filename, "status": "failed", "error": str(e)})

        db.update_batch_job(job_id, {
            "completed_count": i + 1,
            "results_json": json.dumps(results),
        })

    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    db.update_batch_job(job_id, {
        "status": "completed",
        "results_json": json.dumps(results),
    })

    # Webhook callback
    job = db.get_batch_job(job_id)
    webhook_url = job.get("webhook_url", "")
    if webhook_url:
        try:
            async with httpx.AsyncClient(timeout=15) as hc:
                await hc.post(webhook_url, json={
                    "job_id": job_id,
                    "status": "completed",
                    "total": len(req.items),
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            db.update_batch_job(job_id, {"webhook_delivered": 1})
        except Exception as e:
            db.update_batch_job(job_id, {"error": f"Webhook delivery failed: {e}"})


@app.get("/api/pipeline/batch/{job_id}", tags=["Pipeline"])
def get_batch_job_status(job_id: str, _key=Depends(verify_api_key)):
    """Poll the status and results of an async batch pipeline job."""
    job = db.get_batch_job(job_id)
    if not job:
        raise HTTPException(404, "Batch job not found")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "total": job["total"],
        "completed": job["completed_count"],
        "progress_pct": round(job["completed_count"] / max(1, job["total"]) * 100),
        "results": job.get("results", []),
        "webhook_url": job.get("webhook_url", ""),
        "webhook_delivered": bool(job.get("webhook_delivered", 0)),
        "error": job.get("error", ""),
        "created_at": job["created_at"],
        "updated_at": job.get("updated_at", ""),
    }


@app.get("/api/pipeline/batch", tags=["Pipeline"])
def list_batch_jobs(_key=Depends(verify_api_key)):
    """List recent batch pipeline jobs."""
    return {"batch_jobs": db.list_batch_jobs(limit=20)}


# ═════════════════════════════════════════════════════════════════════════════
# CANDIDATES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/candidates", tags=["Candidates"])
def list_candidates(limit: int = 50, skip: int = 0, _key=Depends(verify_api_key)):
    return {
        "candidates": db.list_candidates(limit=limit, skip=skip),
        "total": db.get_stats()["total_candidates"],
    }


@app.get("/api/candidates/{candidate_id}", tags=["Candidates"])
def get_candidate(candidate_id: str, _key=Depends(verify_api_key)):
    c = db.get_candidate(candidate_id)
    if not c:
        raise HTTPException(404, "Candidate not found")
    return c


@app.get("/api/candidates/{candidate_id}/skills", tags=["Candidates"])
def get_candidate_skills(candidate_id: str, _key=Depends(verify_api_key)):
    """Return only the normalized skill profile for a candidate."""
    c = db.get_candidate(candidate_id)
    if not c:
        raise HTTPException(404, "Candidate not found")
    skills = c.get("normalized_skills", [])
    categories = {}
    for sk in skills:
        cat = sk.get("category", "Other")
        categories.setdefault(cat, []).append(sk.get("normalized", sk.get("raw", "")))
    return {
        "candidate_id": candidate_id,
        "name": c.get("name", ""),
        "title": c.get("title", ""),
        "skills": skills,
        "skill_count": len(skills),
        "by_category": categories,
        "inferred_skills": [s for s in skills if s.get("inferred")],
    }


@app.delete("/api/candidates/{candidate_id}", tags=["Candidates"])
def delete_candidate(candidate_id: str, _key=Depends(verify_api_key)):
    if not db.delete_candidate(candidate_id):
        raise HTTPException(404, "Candidate not found")
    return {"deleted": True, "candidate_id": candidate_id}


# ═════════════════════════════════════════════════════════════════════════════
# JOBS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/jobs", tags=["Jobs"])
def create_job(job: JobDescriptionCreate, _key=Depends(verify_api_key)):
    job_id = db.save_job({
        "id": str(uuid.uuid4()),
        "title": job.title,
        "description": job.description,
        "company": job.company or "",
        "created_at": datetime.utcnow().isoformat(),
    })
    return {"job_id": job_id, "message": "Job saved"}


@app.get("/api/jobs", tags=["Jobs"])
def list_jobs(_key=Depends(verify_api_key)):
    return {"jobs": db.list_jobs()}


@app.get("/api/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str, _key=Depends(verify_api_key)):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ═════════════════════════════════════════════════════════════════════════════
# SKILLS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/skills/normalize", tags=["Skills"])
@limiter.limit("30/minute")
async def normalize_skill(request: Request, req: NormalizeRequest, _key=Depends(verify_api_key)):
    """Normalize a single skill with full taxonomy, hierarchy, aliases, and demand info."""
    if not req.skill:
        raise HTTPException(400, "Provide a skill")
    cached = db.get_cached_skill(req.skill)
    if cached:
        cached["from_cache"] = True
        return cached

    prompt = f"""Normalize this tech skill with full taxonomy info.

Skill: {req.skill}

Respond ONLY with valid JSON (no markdown):
{{
  "raw": "{req.skill}",
  "normalized": "Standard canonical name",
  "category": "Frontend|Backend|AI/ML|DevOps|Cloud|Data|Mobile|Other",
  "level": "Foundational|Intermediate|Advanced|Expert",
  "emerging": false,
  "aliases": ["other common names"],
  "related": ["related skills"],
  "hierarchy": ["Skill", "Subcategory", "Category", "Domain"],
  "demand": "High|Medium|Low",
  "description": "One-sentence industry description"
}}"""
    try:
        raw = _call_gemini(client, prompt)
        result = _parse_json(raw)
        result["from_cache"] = False
        db.cache_skill(req.skill, result)
        return result
    except Exception as e:
        raise HTTPException(500, f"Normalization failed: {e}")


@app.post("/api/skills/batch-normalize", tags=["Skills"])
@limiter.limit("10/minute")
async def batch_normalize(request: Request, req: NormalizeRequest, _key=Depends(verify_api_key)):
    """Normalize a list of skills with emerging skill detection."""
    if not req.skills:
        raise HTTPException(400, "Provide a list of skills")
    normalizer = NormalizerAgent(client)
    normalized, emerging = await normalizer.run(req.skills)
    for sk in normalized:
        if sk.get("raw"):
            db.cache_skill(sk["raw"], sk)
    for skill in emerging:
        db.record_emerging_skill(skill)
    return {
        "normalized": normalized,
        "emerging_skills": emerging,
        "inferred_skills": [s for s in normalized if s.get("inferred")],
    }


@app.post("/api/skills/report", tags=["Skills"])
async def generate_skill_report(req: SkillQueryRequest, _key=Depends(verify_api_key)):
    """Generate an AI prose skill intelligence report for a set of normalized skills."""
    if not req.skills:
        raise HTTPException(400, "No skills provided")
    prompt = f"""You are a senior technical recruiter. Analyze these normalized skills and generate a comprehensive report.

Skills:
{json.dumps(req.skills, indent=2)}

Write a 3-paragraph prose report covering:
1. The candidate's strongest technical domain and standout skills
2. Skill breadth vs. depth — specialist or generalist?
3. Market positioning, salary band estimate, and hiring recommendation

Be specific, insightful, and professional."""
    try:
        report = _call_gemini(client, prompt)
        return {"report": report.strip()}
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


@app.get("/api/skills/taxonomy", tags=["Skills"])
def get_taxonomy(_key=Depends(verify_api_key)):
    """
    Browse the full skill taxonomy tree derived from all normalized skills in the cache.
    Returns skills grouped by category with subcategory hierarchy.
    """
    cached = db.list_skill_cache()
    taxonomy: dict = {}

    for item in cached:
        cat = item.get("category", "Other")
        if cat not in taxonomy:
            taxonomy[cat] = {
                "category": cat,
                "hierarchy": SKILL_HIERARCHY.get(cat, SKILL_HIERARCHY["Other"]),
                "skills": [],
                "count": 0,
            }
        taxonomy[cat]["skills"].append({
            "name": item.get("normalized", item.get("raw", "")),
            "level": item.get("level", "Unknown"),
            "demand": item.get("demand", "Unknown"),
            "aliases": item.get("aliases", []),
            "hierarchy": item.get("hierarchy", []),
        })
        taxonomy[cat]["count"] += 1

    return {
        "taxonomy": sorted(taxonomy.values(), key=lambda x: -x["count"]),
        "total_skills": sum(t["count"] for t in taxonomy.values()),
        "categories": list(taxonomy.keys()),
        "full_hierarchy": SKILL_HIERARCHY,
    }


@app.get("/api/skills/emerging", tags=["Skills"])
def list_emerging_skills(reviewed: bool = False, _key=Depends(verify_api_key)):
    """
    List skills the normalizer couldn't confidently map to the taxonomy.
    These are flagged for human review and potential taxonomy expansion.
    """
    skills = db.list_emerging_skills(reviewed=reviewed)
    return {
        "emerging_skills": skills,
        "total": len(skills),
        "note": "These skills are not yet in the canonical taxonomy. Review and add them.",
    }


@app.post("/api/skills/emerging/review", tags=["Skills"])
def mark_skill_reviewed(req: EmergingSkillReviewRequest, _key=Depends(verify_api_key)):
    """Mark an emerging skill as reviewed (human-in-the-loop taxonomy expansion)."""
    ok = db.mark_emerging_skill_reviewed(req.skill_id)
    if not ok:
        raise HTTPException(404, "Emerging skill not found")
    return {"reviewed": True, "skill_id": req.skill_id}


# ═════════════════════════════════════════════════════════════════════════════
# MATCHING
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/match/quick", tags=["Matching"])
@limiter.limit("15/minute")
async def quick_match(request: Request, req: MatchRequest, _key=Depends(verify_api_key)):
    """Fast LLM-based candidate vs JD match."""
    if not req.candidate:
        raise HTTPException(400, "Candidate data required")
    result = await MatcherAgent(client).run(
        req.candidate, req.normalized_skills or [], req.job_description or DEFAULT_JD
    )
    if req.candidate_id:
        db.update_candidate_match(req.candidate_id, result)
    return result


@app.post("/api/match/semantic", tags=["Matching"])
@limiter.limit("10/minute")
async def semantic_match(request: Request, req: SemanticMatchRequest, _key=Depends(verify_api_key)):
    """
    Deep semantic match: vector embedding similarity + LLM analysis.
    Returns seniority_fit, technical_fit, upskilling paths, time-to-hire estimate.
    """
    if not req.candidate or not req.job_description:
        raise HTTPException(400, "Both candidate and job_description required")

    # Vector similarity first
    embedder = EmbeddingAgent(client)
    vector_score = None
    try:
        vector_score = await embedder.compute_match_score(
            req.candidate, req.normalized_skills or [], req.job_description
        )
    except Exception:
        pass

    result = await MatcherAgent(client).run(
        req.candidate, req.normalized_skills or [], req.job_description, vector_score
    )
    if req.candidate_id:
        db.update_candidate_match(req.candidate_id, result)
    return result


@app.post("/api/match/threshold", tags=["Matching"])
@limiter.limit("15/minute")
async def threshold_match(request: Request, req: ThresholdMatchRequest, _key=Depends(verify_api_key)):
    """
    Match a candidate and apply a configurable pass/fail threshold (0-100).
    Allows hiring managers to tune precision vs. recall.
    """
    embedder = EmbeddingAgent(client)
    vector_score = None
    try:
        vector_score = await embedder.compute_match_score(
            req.candidate, req.normalized_skills or [], req.job_description
        )
    except Exception:
        pass

    result = await MatcherAgent(client).run(
        req.candidate, req.normalized_skills or [], req.job_description, vector_score
    )
    if req.candidate_id:
        db.update_candidate_match(req.candidate_id, result)

    result["threshold"] = req.threshold
    result["meets_threshold"] = result.get("match_score", 0) >= req.threshold
    result["threshold_verdict"] = (
        "PASS — candidate meets or exceeds threshold"
        if result["meets_threshold"]
        else f"FAIL — score {result.get('match_score', 0)}% is below threshold {req.threshold}%"
    )
    return result


@app.post("/api/match/batch", tags=["Matching"])
@limiter.limit("5/minute")
async def batch_match(request: Request, req: MatchRequest, _key=Depends(verify_api_key)):
    """Rank multiple candidates against a single JD. Returns sorted leaderboard."""
    if not req.candidates or not req.job_description:
        raise HTTPException(400, "Provide candidates list and job_description")

    results = []
    for cand in req.candidates:
        try:
            m = await MatcherAgent(client).run(cand, [], req.job_description)
            results.append({
                "candidate_id": cand.get("id"),
                "name": cand.get("parsed", {}).get("name", cand.get("name", "Unknown")),
                "match_score": m.get("match_score", 0),
                "verdict": m.get("verdict"),
                "hiring_signal": m.get("hiring_signal", "consider"),
                "strengths": m.get("strengths", []),
                "gaps": m.get("gaps", []),
            })
        except Exception:
            results.append({"candidate_id": cand.get("id"), "match_score": 0, "status": "failed"})

    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return {
        "ranked_candidates": results,
        "top_candidate": results[0] if results else None,
        "total": len(results),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/analytics/dashboard", tags=["Analytics"])
def dashboard_analytics(_key=Depends(verify_api_key)):
    """Aggregate stats: score distribution, skill categories, top skills, trends."""
    candidates = db.list_candidates(limit=1000)
    stats = db.get_stats()

    if not candidates:
        return {
            "total_candidates": 0, "avg_match_score": 0,
            "total_skills": 0, "unique_skills": 0,
            "score_buckets": {"<60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90+": 0},
            "skill_categories": {}, "top_skills": [], "recent_candidates": [],
            "hiring_signals": {"proceed": 0, "consider": 0, "pass": 0},
            **stats,
        }

    scores = [c.get("match_score", 0) for c in candidates if c.get("match_score")]
    avg_score = round(sum(scores) / len(scores)) if scores else 0

    buckets = {"<60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90+": 0}
    for s in scores:
        if s < 60: buckets["<60"] += 1
        elif s < 70: buckets["60-70"] += 1
        elif s < 80: buckets["70-80"] += 1
        elif s < 90: buckets["80-90"] += 1
        else: buckets["90+"] += 1

    all_skills, skill_categories, hiring_signals = [], {}, {"proceed": 0, "consider": 0, "pass": 0}
    for c in candidates:
        for sk in c.get("normalized_skills", []):
            all_skills.append(sk.get("normalized", ""))
            cat = sk.get("category", "Other")
            skill_categories[cat] = skill_categories.get(cat, 0) + 1
        signal = c.get("match_data", {}).get("hiring_signal", "consider")
        if signal in hiring_signals:
            hiring_signals[signal] += 1

    top_skills = [{"skill": s, "count": c} for s, c in Counter(all_skills).most_common(10)]

    return {
        "total_candidates": len(candidates),
        "avg_match_score": avg_score,
        "total_skills": len(all_skills),
        "unique_skills": len(set(all_skills)),
        "score_buckets": buckets,
        "skill_categories": skill_categories,
        "top_skills": top_skills,
        "recent_candidates": candidates[:5],
        "hiring_signals": hiring_signals,
        **stats,
    }


@app.get("/api/analytics/leaderboard", tags=["Analytics"])
def leaderboard(_key=Depends(verify_api_key)):
    """Top 20 candidates ranked by match score."""
    candidates = db.list_candidates(limit=200)
    ranked = sorted(candidates, key=lambda c: c.get("match_score", 0), reverse=True)
    return {
        "leaderboard": ranked[:20],
        "total_ranked": len(ranked),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHAT
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat", tags=["Chat"])
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest, _key=Depends(verify_api_key)):
    """
    Context-aware recruiter assistant chat.
    Maintains conversation history for multi-turn sessions.
    """
    context_parts = []
    if req.current_candidate:
        c = req.current_candidate
        context_parts.append(
            f"Current candidate: {c.get('name','Unknown')}, {c.get('title','N/A')}. "
            f"Skills: {', '.join(c.get('skills',[])[:10])}. "
            f"Match score: {req.match_score or 'not scored'}%. "
            f"JD: {(req.job_description or '')[:200]}"
        )
    if req.resumes_parsed is not None:
        context_parts.append(
            f"Session stats: {req.resumes_parsed} resumes parsed, {req.total_skills} skills extracted."
        )
    context = " ".join(context_parts) or "No candidate loaded."

    system = (
        "You are TalentSync AI, an expert recruiting assistant powered by Google Gemini. "
        "Help HR teams analyze candidates, identify skill gaps, suggest interview questions, "
        "and make data-driven hiring decisions. Be concise (2-4 sentences), insightful, and professional."
    )

    history_text = ""
    if req.history:
        lines = [f"{t.role.upper()}: {t.content}" for t in req.history[-6:]]
        history_text = "\n".join(lines) + "\n"

    prompt = f"{system}\n\nContext: {context}\n\n{history_text}USER: {req.message}"
    try:
        reply = _call_gemini(client, prompt, max_tokens=512)
        return {"reply": reply.strip()}
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/webhooks/test", tags=["Webhooks"])
async def test_webhook(req: WebhookTestRequest, _key=Depends(verify_api_key)):
    """Send a test payload to a webhook URL to verify delivery."""
    payload = req.payload or {
        "event": "webhook.test",
        "message": "TalentSync webhook delivery test",
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            resp = await hc.post(req.url, json=payload)
        return {
            "delivered": True,
            "status_code": resp.status_code,
            "url": req.url,
        }
    except Exception as e:
        raise HTTPException(400, f"Webhook delivery failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# GEMINI PROXY (keeps API key server-side)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/gemini/proxy", tags=["Proxy"])
@limiter.limit("30/minute")
async def gemini_proxy(request: Request, _key=Depends(verify_api_key)):
    body = await request.json()
    try:
        prompt = body.get("prompt", "")
        system = body.get("system", "")
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=body.get("max_tokens", 1000), temperature=0.3
            ),
        )
        return {"content": [{"type": "text", "text": response.text}], "model": MODEL}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/claude/proxy", tags=["Proxy"])
@limiter.limit("30/minute")
async def claude_proxy_alias(request: Request, _key=Depends(verify_api_key)):
    """Backward-compatible alias for frontends using the Claude proxy format."""
    body = await request.json()
    try:
        messages = body.get("messages", [])
        system = body.get("system", "")
        parts = ([system] if system else []) + [m.get("content", "") for m in messages]
        full_prompt = "\n\n".join(parts)
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=body.get("max_tokens", 1000), temperature=0.3
            ),
        )
        return {
            "content": [{"type": "text", "text": response.text}],
            "usage": {"output_tokens": len(response.text.split())},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SDK EXAMPLES (integration docs endpoint)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/sdk/python", tags=["SDK"], response_class=HTMLResponse)
def sdk_python():
    """Python SDK usage example."""
    code = '''# TalentSync AI — Python Integration Example
import requests

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "your-api-key"}  # omit if auth not configured

# 1. Parse + match a resume
with open("resume.pdf", "rb") as f:
    resp = requests.post(f"{BASE}/api/pipeline/run",
        files={"file": f},
        data={"job_description": "Senior Python Engineer, FastAPI, 5yr exp"},
        headers=HEADERS)
result = resp.json()
print(f"Match Score: {result['match']['match_score']}%")
print(f"Verdict: {result['match']['verdict']}")

# 2. Async batch
batch_payload = {
    "items": [
        {"resume_text": "Alice Smith, Python dev ...", "filename": "alice.txt"},
        {"resume_text": "Bob Jones, Java dev ...", "filename": "bob.txt"},
    ],
    "job_description": "Senior Backend Engineer",
    "webhook_url": "https://your-app.com/webhook"
}
job = requests.post(f"{BASE}/api/pipeline/batch", json=batch_payload, headers=HEADERS).json()
print(f"Batch job: {job['job_id']}")

# 3. Check status
import time
while True:
    status = requests.get(f"{BASE}/api/pipeline/batch/{job['job_id']}", headers=HEADERS).json()
    print(f"Progress: {status['progress_pct']}%")
    if status["status"] == "completed":
        for r in status["results"]:
            print(f"  {r['name']}: {r['match_score']}%")
        break
    time.sleep(3)

# 4. Browse skill taxonomy
taxonomy = requests.get(f"{BASE}/api/skills/taxonomy", headers=HEADERS).json()
for cat in taxonomy["taxonomy"]:
    print(f"{cat['category']}: {cat['count']} skills")
'''
    return f"<pre>{code}</pre>"


@app.get("/api/sdk/javascript", tags=["SDK"], response_class=HTMLResponse)
def sdk_javascript():
    """JavaScript SDK usage example."""
    code = '''// TalentSync AI — JavaScript Integration Example
const BASE = "http://localhost:8000";
const HEADERS = { "Content-Type": "application/json", "X-API-Key": "your-api-key" };

// 1. Parse a resume from text
async function parseResume(resumeText, jobDescription) {
  const form = new FormData();
  form.append("resume_text", resumeText);
  form.append("job_description", jobDescription);
  const resp = await fetch(`${BASE}/api/pipeline/run`, { method: "POST", body: form,
    headers: { "X-API-Key": "your-api-key" } });
  return resp.json();
}

// 2. Async batch pipeline
async function runBatch(items, sharedJD, webhookUrl) {
  const resp = await fetch(`${BASE}/api/pipeline/batch`, {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({ items, job_description: sharedJD, webhook_url: webhookUrl })
  });
  const { job_id } = await resp.json();

  // Poll for completion
  while (true) {
    const status = await fetch(`${BASE}/api/pipeline/batch/${job_id}`,
      { headers: HEADERS }).then(r => r.json());
    console.log(`Progress: ${status.progress_pct}%`);
    if (status.status === "completed") return status.results;
    await new Promise(r => setTimeout(r, 3000));
  }
}

// 3. Threshold matching
async function thresholdMatch(candidate, jd, threshold = 75) {
  const resp = await fetch(`${BASE}/api/match/threshold`, {
    method: "POST", headers: HEADERS,
    body: JSON.stringify({ candidate, job_description: jd, threshold })
  });
  const result = await resp.json();
  console.log(`Meets threshold: ${result.meets_threshold} (${result.match_score}% vs ${threshold}%)`);
  return result;
}
'''
    return f"<pre>{code}</pre>"


# ═════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(
        "<h1>TalentSync AI v3.0</h1>"
        "<p>Place frontend in /static/index.html</p>"
        "<p><a href='/docs'>→ API Documentation (Swagger)</a></p>"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
