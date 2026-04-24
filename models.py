"""
TalentSync AI — Pydantic Models / Request Schemas v3.0
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    resume_text: str
    filename: Optional[str] = "resume"


class NormalizeRequest(BaseModel):
    skill: Optional[str] = None
    skills: Optional[List[str]] = None


class MatchRequest(BaseModel):
    candidate: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    normalized_skills: Optional[List[Dict[str, Any]]] = None
    job_description: Optional[str] = None
    candidate_id: Optional[str] = None


class SemanticMatchRequest(BaseModel):
    candidate: Dict[str, Any]
    normalized_skills: Optional[List[Dict[str, Any]]] = None
    job_description: str
    candidate_id: Optional[str] = None


class ThresholdMatchRequest(BaseModel):
    candidate: Dict[str, Any]
    job_description: str
    normalized_skills: Optional[List[Dict[str, Any]]] = None
    threshold: int = Field(default=70, ge=0, le=100, description="Minimum match score (0-100) to pass")
    candidate_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Batch Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BatchPipelineItem(BaseModel):
    resume_text: str
    filename: Optional[str] = "resume"
    job_description: Optional[str] = None  # overrides shared JD if provided


class BatchPipelineRequest(BaseModel):
    items: List[BatchPipelineItem] = Field(..., min_length=1, max_length=50)
    job_description: Optional[str] = None  # shared JD for all items
    webhook_url: Optional[str] = None      # POST here on completion


# ─────────────────────────────────────────────────────────────────────────────
# Skills
# ─────────────────────────────────────────────────────────────────────────────

class SkillQueryRequest(BaseModel):
    skills: List[Dict[str, Any]]


class EmergingSkillReviewRequest(BaseModel):
    skill_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    current_candidate: Optional[Dict[str, Any]] = None
    match_score: Optional[int] = None
    resumes_parsed: Optional[int] = None
    total_skills: Optional[int] = None
    job_description: Optional[str] = None
    history: Optional[List[ChatTurn]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

class JobDescriptionCreate(BaseModel):
    title: str
    description: str
    company: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Webhooks
# ─────────────────────────────────────────────────────────────────────────────

class WebhookTestRequest(BaseModel):
    url: str
    payload: Optional[Dict[str, Any]] = None
