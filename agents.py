"""
TalentSync AI — Agent Definitions v3.0
Agents: Parser, Normalizer, Matcher, EmbeddingAgent
Orchestrator: manages agent lifecycle, metrics, graceful degradation.
"""

import json, re, time, asyncio
from typing import Dict, Any, List, Tuple, Optional
import google.genai as genai
from google.genai import types

MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "text-embedding-004"

# ─────────────────────────────────────────────────────────────────────────────
# Skill Inference Map — explicit skills → implied higher-order competencies
# ─────────────────────────────────────────────────────────────────────────────
SKILL_INFERENCES = {
    frozenset(["TensorFlow", "PyTorch"]): ["Deep Learning"],
    frozenset(["TensorFlow", "Keras"]): ["Deep Learning"],
    frozenset(["PyTorch", "Hugging Face"]): ["LLM Engineering", "Deep Learning"],
    frozenset(["Docker", "Kubernetes"]): ["Container Orchestration"],
    frozenset(["React", "TypeScript"]): ["Modern Frontend Development"],
    frozenset(["AWS", "Azure"]): ["Multi-Cloud Architecture"],
    frozenset(["AWS", "GCP"]): ["Multi-Cloud Architecture"],
    frozenset(["Pandas", "NumPy", "Scikit-learn"]): ["Data Science", "Machine Learning"],
    frozenset(["PostgreSQL", "MySQL"]): ["Relational Database Design"],
    frozenset(["MongoDB", "Redis"]): ["NoSQL & Caching"],
    frozenset(["FastAPI", "Django"]): ["Python Web Development"],
    frozenset(["Kafka", "RabbitMQ"]): ["Event-Driven Architecture"],
    frozenset(["Terraform", "Ansible"]): ["Infrastructure as Code"],
    frozenset(["CI/CD", "GitHub Actions"]): ["DevOps Automation"],
}

# Canonical hierarchy: skill → [subcategory, category, domain]
SKILL_HIERARCHY = {
    "Frontend":  ["Frontend Development", "Web Engineering", "Software Engineering"],
    "Backend":   ["Backend Development", "Web Engineering", "Software Engineering"],
    "AI/ML":     ["Machine Learning", "Data Science", "Computer Science"],
    "DevOps":    ["DevOps", "Infrastructure", "Software Engineering"],
    "Cloud":     ["Cloud Computing", "Infrastructure", "Technology"],
    "Data":      ["Data Engineering", "Data Science", "Analytics"],
    "Mobile":    ["Mobile Development", "Software Engineering", "Technology"],
    "Other":     ["General Technology", "Software Engineering", "Technology"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Any:
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    # Try direct parse first
    try:
        return json.loads(clean)
    except Exception:
        pass
    # Extract first JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', clean, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON in response")


def _call_gemini(client: genai.Client, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens, temperature=0.2
                ),
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt + 1
                time.sleep(wait)
            else:
                raise e
    raise Exception("Failed after 3 retries due to rate limiting")


async def _call_gemini_async(client: genai.Client, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    """Async wrapper — runs sync Gemini call in thread pool so event loop stays free."""
    return await asyncio.to_thread(_call_gemini, client, prompt, system, max_tokens)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Falls back to pure-Python if numpy absent."""
    try:
        import numpy as np
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Parser
# ─────────────────────────────────────────────────────────────────────────────

class ParserAgent:
    SYSTEM = (
        "You are an expert resume parser. Extract ALL information accurately. "
        "Always respond with valid JSON only — no markdown, no explanation, no extra text."
    )

    def __init__(self, client: genai.Client):
        self.client = client

    async def run(self, resume_text: str) -> Dict[str, Any]:
        prompt = f"""Parse this resume and extract all information.

Resume text:
{resume_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "name": "Full Name",
  "title": "Current/Most Recent Job Title",
  "location": "City, State/Country",
  "email": "email if found or empty string",
  "phone": "phone if found or empty string",
  "linkedin": "linkedin url if found or empty string",
  "summary": "2-3 sentence professional summary",
  "experience": [
    {{"role": "Job Title", "company": "Company Name", "period": "2021-Present",
      "description": "one-line description", "years": 2}}
  ],
  "skills": ["Skill1", "Skill2"],
  "education": [
    {{"degree": "Degree Name", "institution": "School Name", "year": "Year", "field": "Field"}}
  ],
  "certifications": ["Cert1"],
  "languages": ["English"],
  "projects": ["Brief project description"],
  "years_of_experience": 5,
  "confidence": {{"skills": 95, "experience": 92, "education": 98}}
}}"""
        raw = await _call_gemini_async(self.client, prompt, self.SYSTEM)
        return _parse_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class NormalizerAgent:
    SYSTEM = (
        "You are a tech skill taxonomy expert. Always respond with valid JSON only — "
        "no markdown, no explanation."
    )

    def __init__(self, client: genai.Client):
        self.client = client

    def _infer_additional_skills(self, normalized_names: List[str]) -> List[Dict[str, Any]]:
        """Add implicitly-held skills based on skill co-occurrence rules."""
        inferred = []
        name_set = set(normalized_names)
        for required_skills, implied_skills in SKILL_INFERENCES.items():
            if required_skills.issubset(name_set):
                for implied in implied_skills:
                    if implied not in name_set:
                        category = "AI/ML" if any(k in implied for k in ["Learning", "LLM"]) \
                            else "DevOps" if any(k in implied for k in ["Container", "Infra", "Code", "DevOps"]) \
                            else "Frontend" if "Frontend" in implied \
                            else "Cloud" if "Cloud" in implied \
                            else "Backend"
                        inferred.append({
                            "raw": implied,
                            "normalized": implied,
                            "category": category,
                            "level": "Intermediate",
                            "emerging": False,
                            "inferred": True,
                            "hierarchy": SKILL_HIERARCHY.get(category, SKILL_HIERARCHY["Other"]),
                        })
                        name_set.add(implied)
        return inferred

    async def run(self, skills: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Returns (normalized_skills, emerging_skill_names).
        Emerging skills are those the model cannot confidently map to known taxonomy.
        """
        if not skills:
            return [], []

        prompt = f"""Normalize these tech skills to standard industry names.

Raw skills: {', '.join(skills)}

Rules:
- React.js/ReactJS → React | JS → JavaScript | Node → Node.js | K8s → Kubernetes
- TF → TensorFlow | ML → Machine Learning | DL → Deep Learning
- If a skill is brand-new, highly niche, or you cannot confidently classify it, set emerging=true

Category must be one of: Frontend | Backend | AI/ML | DevOps | Cloud | Data | Mobile | Other
Level must be one of: Foundational | Intermediate | Advanced | Expert

Respond ONLY with a valid JSON array (no markdown):
[{{
  "raw": "original",
  "normalized": "Standard Name",
  "category": "Backend",
  "level": "Advanced",
  "emerging": false,
  "hierarchy": ["Standard Name", "Subcategory", "Category", "Domain"]
}}]"""

        raw = await _call_gemini_async(self.client, prompt, self.SYSTEM)
        result = _parse_json(raw)
        if isinstance(result, dict) and "skills" in result:
            result = result["skills"]
        if not isinstance(result, list):
            result = []

        normalized, emerging_names = [], []
        for item in result:
            if item.get("emerging"):
                emerging_names.append(item.get("raw", ""))
            else:
                # Attach full hierarchy if model didn't provide one
                if not item.get("hierarchy"):
                    cat = item.get("category", "Other")
                    item["hierarchy"] = [item.get("normalized", ""), *SKILL_HIERARCHY.get(cat, SKILL_HIERARCHY["Other"])]
                item["inferred"] = False
                normalized.append(item)

        # Add inferred skills from co-occurrence rules
        norm_names = [s.get("normalized", "") for s in normalized]
        inferred = self._infer_additional_skills(norm_names)
        normalized.extend(inferred)

        return normalized, emerging_names


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Matcher
# ─────────────────────────────────────────────────────────────────────────────

class MatcherAgent:
    SYSTEM = (
        "You are a senior technical recruiter with 15 years of experience. "
        "Always respond with valid JSON only — no markdown, no explanation."
    )

    def __init__(self, client: genai.Client):
        self.client = client

    async def run(
        self,
        candidate: Dict[str, Any],
        normalized_skills: List[Dict[str, Any]],
        job_description: str,
        vector_similarity_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        skill_names = [s.get("normalized", s.get("raw", "")) for s in normalized_skills]
        all_skills = list(set(candidate.get("skills", []) + skill_names))
        exp_summary = "; ".join(
            f"{e.get('role')} at {e.get('company')} ({e.get('period')})"
            for e in candidate.get("experience", [])[:4]
        )
        vector_hint = (
            f"\nVector embedding similarity pre-score: {vector_similarity_score}% "
            f"(use this as one data point, not the sole determinant)."
            if vector_similarity_score is not None else ""
        )

        prompt = f"""Analyze this candidate against the job description.{vector_hint}

CANDIDATE:
Name: {candidate.get('name', 'Unknown')}
Title: {candidate.get('title', 'N/A')}
Years of Experience: {candidate.get('years_of_experience', 'Unknown')}
Skills: {', '.join(all_skills[:35])}
Experience: {exp_summary}
Education: {'; '.join(e.get('degree','') + ' from ' + e.get('institution','') for e in candidate.get('education', []))}
Certifications: {', '.join(candidate.get('certifications', []))}
Projects: {'; '.join((candidate.get('projects', []))[:3])}

JOB DESCRIPTION:
{job_description}

Respond ONLY with valid JSON (no markdown):
{{
  "match_score": 87,
  "verdict": "Strong Match|Good Match|Moderate Match|Weak Match",
  "seniority_fit": "Excellent|Good|Moderate|Poor",
  "technical_fit": "Excellent|Good|Moderate|Poor",
  "cultural_fit_signals": ["signal 1", "signal 2"],
  "strengths": ["strength 1 with evidence", "strength 2", "strength 3"],
  "matched_skills": ["skills present in both candidate and JD"],
  "missing_skills": ["skills in JD not on candidate profile"],
  "gaps": [{{
    "skill": "Missing Skill",
    "severity": "required|preferred|optional",
    "context": "why it matters for this role",
    "upskilling_path": "Specific course, certification, or project to close this gap"
  }}],
  "recommendation": "2-3 sentence recruiter recommendation with concrete next steps.",
  "interview_questions": [
    "Technical depth question?",
    "Behavioral / culture-fit question?",
    "Gap-probing question?"
  ],
  "hiring_signal": "proceed|consider|pass",
  "time_to_hire_estimate": "1-2 weeks",
  "offer_range_percentile": "Top 25%"
}}"""
        raw = await _call_gemini_async(self.client, prompt, self.SYSTEM)
        result = _parse_json(raw)
        if vector_similarity_score is not None:
            result["vector_similarity_score"] = vector_similarity_score
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Embedding (Vector Similarity)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingAgent:
    """
    Uses Gemini's text-embedding-004 model to compute vector representations
    and cosine similarity between candidate profiles and job descriptions.
    Satisfies the 'vector embedding-based skill matching' requirement.
    """

    def __init__(self, client: genai.Client):
        self.client = client

    def _embed(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text[:3000],
        )
        # google-genai SDK returns EmbedContentResponse
        return response.embeddings[0].values

    async def compute_match_score(
        self,
        candidate: Dict[str, Any],
        normalized_skills: List[Dict[str, Any]],
        job_description: str,
    ) -> int:
        """
        Returns a 0-100 vector similarity score between the candidate profile and JD.
        Runs embedding in thread pool to keep event loop free.
        """
        skill_names = [s.get("normalized", s.get("raw", "")) for s in normalized_skills]
        exp_parts = [
            f"{e.get('role', '')} {e.get('company', '')} {e.get('description', '')}"
            for e in candidate.get("experience", [])[:4]
        ]
        candidate_text = " ".join([
            candidate.get("title", ""),
            candidate.get("summary", ""),
            " ".join(skill_names),
            " ".join(exp_parts),
        ]).strip()

        def _do_embed():
            cand_emb = self._embed(candidate_text)
            jd_emb = self._embed(job_description)
            similarity = _cosine_similarity(cand_emb, jd_emb)
            # Cosine similarity ranges -1..1; normalize to 0..100
            score = int((similarity + 1) / 2 * 100)
            return max(0, min(100, score))

        return await asyncio.to_thread(_do_embed)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — manages agent lifecycle, metrics, graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates Parser → Normalizer → EmbeddingAgent → Matcher.
    Records per-agent: status, latency_ms, quality_score.
    Implements retry logic and graceful degradation:
      - If Normalizer fails → fallback raw skill mapping (pipeline continues)
      - If EmbeddingAgent fails → proceed with LLM-only scoring
      - If Matcher fails → return fallback score based on vector similarity
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.parser = ParserAgent(client)
        self.normalizer = NormalizerAgent(client)
        self.matcher = MatcherAgent(client)
        self.embedder = EmbeddingAgent(client)

    async def run_pipeline(
        self,
        resume_text: str,
        job_description: str,
        filename: str = "resume",
    ) -> Tuple[Dict, List[Dict], List[str], Dict, List[Dict], int]:
        """
        Returns:
            parsed, normalized_skills, emerging_skills,
            match_data, pipeline_logs, total_ms
        """
        logs = []
        start = time.time()

        # ── Agent 1: Parser ──────────────────────────────────────────────────
        a_start = time.time()
        try:
            parsed = await self.parser.run(resume_text)
            quality = self._parser_quality(parsed)
            logs.append(self._log("parser", "success", a_start, quality,
                f"Extracted {len(parsed.get('skills',[]))} skills, "
                f"{len(parsed.get('experience',[]))} roles, "
                f"{len(parsed.get('education',[]))} education entries"))
        except Exception as e:
            logs.append(self._log("parser", "failed", a_start, 0, f"Parser failed: {e}"))
            raise RuntimeError(f"Parser agent failed: {e}")

        # ── Agent 2: Normalizer ──────────────────────────────────────────────
        a_start = time.time()
        emerging_skills: List[str] = []
        try:
            normalized_skills, emerging_skills = await self.normalizer.run(
                parsed.get("skills", [])
            )
            mapped = [s for s in normalized_skills if not s.get("inferred")]
            inferred = [s for s in normalized_skills if s.get("inferred")]
            quality = min(100, int(
                len([s for s in mapped if s.get("category") != "Other"]) /
                max(1, len(mapped)) * 100
            ))
            logs.append(self._log("normalizer", "success", a_start, quality,
                f"{len(mapped)} skills normalized, {len(inferred)} inferred, "
                f"{len(emerging_skills)} emerging flagged"))
        except Exception as e:
            normalized_skills = [
                {"raw": s, "normalized": s, "category": "Other",
                 "level": "Unknown", "emerging": False, "inferred": False,
                 "hierarchy": SKILL_HIERARCHY["Other"]}
                for s in parsed.get("skills", [])
            ]
            logs.append(self._log("normalizer", "degraded", a_start, 30,
                f"Fallback normalization — {e}"))

        # ── Agent 3: EmbeddingAgent (vector similarity) ───────────────────────
        a_start = time.time()
        vector_score: Optional[int] = None
        try:
            vector_score = await self.embedder.compute_match_score(
                parsed, normalized_skills, job_description
            )
            logs.append(self._log("embedder", "success", a_start, 95,
                f"Vector similarity score: {vector_score}%"))
        except Exception as e:
            logs.append(self._log("embedder", "degraded", a_start, 0,
                f"Embedding skipped (LLM-only scoring) — {e}"))

        # ── Agent 4: Matcher ─────────────────────────────────────────────────
        a_start = time.time()
        try:
            match_data = await self.matcher.run(
                parsed, normalized_skills, job_description, vector_score
            )
            logs.append(self._log("matcher", "success", a_start, 90,
                f"Score: {match_data.get('match_score')}% — {match_data.get('verdict')} "
                f"| Signal: {match_data.get('hiring_signal', 'n/a')}"))
        except Exception as e:
            fallback_score = vector_score or 65
            verdict = (
                "Strong Match" if fallback_score >= 85 else
                "Good Match" if fallback_score >= 70 else
                "Moderate Match" if fallback_score >= 55 else
                "Weak Match"
            )
            match_data = {
                "match_score": fallback_score,
                "verdict": verdict,
                "seniority_fit": "Unknown",
                "technical_fit": "Unknown",
                "strengths": ["Technical background present"],
                "gaps": [],
                "matched_skills": [],
                "missing_skills": [],
                "recommendation": "Full analysis unavailable. Review candidate manually.",
                "interview_questions": [],
                "hiring_signal": "consider",
                "vector_similarity_score": vector_score,
            }
            logs.append(self._log("matcher", "degraded", a_start, 30,
                f"Fallback match score: {fallback_score}% — {e}"))

        total_ms = int((time.time() - start) * 1000)
        logs.append({
            "agent": "orchestrator",
            "status": "complete",
            "latency_ms": total_ms,
            "quality_score": 100,
            "message": f"Pipeline complete in {total_ms}ms | "
                       f"Agents: {sum(1 for l in logs if l['status']=='success')} success, "
                       f"{sum(1 for l in logs if l['status']=='degraded')} degraded",
        })

        return parsed, normalized_skills, emerging_skills, match_data, logs, total_ms

    @staticmethod
    def _log(agent: str, status: str, start: float, quality: int, message: str) -> Dict:
        return {
            "agent": agent,
            "status": status,
            "latency_ms": int((time.time() - start) * 1000),
            "quality_score": quality,
            "message": f"[{agent.upper()}] {message}",
        }

    @staticmethod
    def _parser_quality(parsed: Dict) -> int:
        from utils import compute_quality_score
        return compute_quality_score(parsed)
