"""Pydantic models for AuraBridge backend."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EmotionSnapshot(BaseModel):
    timestamp: float
    happy: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    fearful: float = 0.0
    disgusted: float = 0.0
    surprised: float = 0.0
    neutral: float = 0.0
    arousal: float = 0.0
    valence: float = 0.0


class VocalMetrics(BaseModel):
    timestamp: float
    jitter: float = 0.0
    shimmer: float = 0.0
    mean_f0: float = 0.0
    energy: float = 0.0


class HRVSnapshot(BaseModel):
    """Simulated Heart Rate Variability snapshot."""
    timestamp: float
    heart_rate: float = 72.0  # bpm
    hrv_rmssd: float = 42.0  # ms — Root Mean Square of Successive Differences


class OculomotorSnapshot(BaseModel):
    """Eye-tracking metrics from MediaPipe FaceLandmarker."""
    timestamp: float
    blink_rate: float = 0.0  # blinks per minute
    gaze_avoidance_pct: float = 0.0  # 0-100%


class CognitiveDistortion(BaseModel):
    """A single detected cognitive distortion category."""
    name: str
    count: int = 0
    examples: list[str] = []


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: float


class SessionCreate(BaseModel):
    patient_name: Optional[str] = "Anonymous"


class Session(BaseModel):
    id: str
    patient_name: str = "Anonymous"
    created_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    is_active: bool = True
    conversation: list[ConversationTurn] = []
    emotion_snapshots: list[EmotionSnapshot] = []
    vocal_metrics: list[VocalMetrics] = []
    hrv_snapshots: list[HRVSnapshot] = []
    oculomotor_snapshots: list[OculomotorSnapshot] = []
    risk_level: RiskLevel = RiskLevel.NONE


class SessionReport(BaseModel):
    session_id: str
    patient_name: str
    session_date: datetime
    duration_minutes: float
    overall_mood: str
    emotional_trajectory: str
    key_themes: list[str]
    risk_assessment: str
    risk_level: RiskLevel
    recommendations: list[str]
    emotion_summary: dict
    vocal_summary: dict
    hrv_summary: dict = {}
    oculomotor_summary: dict = {}
    cognitive_distortions: list[CognitiveDistortion] = []
    cognitive_distortions_summary: str = ""
    full_transcript: list[ConversationTurn]


class ChatRequest(BaseModel):
    message: str
    emotion_snapshot: Optional[EmotionSnapshot] = None
    vocal_metrics: Optional[VocalMetrics] = None
    hrv_snapshot: Optional[HRVSnapshot] = None
    oculomotor_snapshot: Optional[OculomotorSnapshot] = None


class ChatResponse(BaseModel):
    reply: str
    risk_level: RiskLevel = RiskLevel.NONE
    risk_alert: Optional[str] = None
