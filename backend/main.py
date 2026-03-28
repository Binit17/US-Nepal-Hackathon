"""AuraBridge FastAPI Backend — Main application with village growth + trigger-aware AI."""
from __future__ import annotations

import json as json_lib
import uuid
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from models import (
    Session, SessionCreate, ChatRequest, ChatResponse,
    ConversationTurn, EmotionSnapshot, VocalMetrics, RiskLevel,
    HRVSnapshot, OculomotorSnapshot,
)
from gemini_client import GeminiTherapist
from risk_monitor import assess_risk
from report_generator import generate_report
from trigger_analyzer import analyze_triggers

load_dotenv()

# In-memory stores
sessions: dict[str, Session] = {}
therapist: GeminiTherapist | None = None

# Village state (persists across sessions for the demo)
village_state: dict = {
    "level": 1,
    "sessions_completed": 0,
    "streak": 0,
    "last_session_date": None,
}

# User trigger profile (accumulates across sessions)
user_profile: dict = {
    "triggers": [],
    "themes": [],
    "avoidance_topics": [],
    "growth_areas": [],
    "conversation_hooks": [],
}

# Level thresholds (sessions needed for each level)
LEVEL_THRESHOLDS = [0, 1, 2, 3, 5, 7, 10, 12, 15, 18, 22, 25, 28, 32, 36, 40, 45, 50, 55, 60]


def _compute_level(sessions_completed: int) -> int:
    """Compute village level from total sessions completed."""
    for i in range(len(LEVEL_THRESHOLDS) - 1, -1, -1):
        if sessions_completed >= LEVEL_THRESHOLDS[i]:
            return min(i + 1, 20)
    return 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global therapist
    therapist = GeminiTherapist()
    yield


app = FastAPI(
    title="AuraBridge API",
    description="AI-powered Gamified Mental Health Companion — Backend API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Session Endpoints ─────────────────────────────────────────────

@app.post("/api/sessions", response_model=Session)
async def create_session(data: SessionCreate):
    """Create a new therapy session."""
    session_id = str(uuid.uuid4())[:8]
    session = Session(
        id=session_id,
        patient_name=data.patient_name or "Anonymous",
    )
    sessions[session_id] = session
    return session


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    return list(sessions.values())


@app.get("/api/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    """Get a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a session — triggers village growth + trigger analysis."""
    global village_state, user_profile
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.is_active = False
    session.ended_at = datetime.now()

    # Grow the village
    village_state["sessions_completed"] += 1
    old_level = village_state["level"]
    village_state["level"] = _compute_level(village_state["sessions_completed"])
    new_level = village_state["level"]
    village_state["last_session_date"] = datetime.now().isoformat()

    # Update streak
    village_state["streak"] = village_state.get("streak", 0) + 1

    # Run trigger analysis on this session's transcript
    if session.conversation:
        transcript = "\n".join(
            f"[{turn.role.upper()}]: {turn.content}"
            for turn in session.conversation
        )
        try:
            user_profile = await analyze_triggers(transcript, user_profile)
        except Exception as e:
            print(f"Trigger analysis failed (non-blocking): {e}")

    leveled_up = new_level > old_level

    return {
        "status": "ended",
        "session_id": session_id,
        "village": {
            "level": new_level,
            "sessions_completed": village_state["sessions_completed"],
            "leveled_up": leveled_up,
            "old_level": old_level,
        },
    }


# ─── Chat Endpoint ─────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    """Send a message in a therapy session and get AI response."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session has ended")

    # Store user message
    user_turn = ConversationTurn(
        role="user",
        content=req.message,
        timestamp=time.time(),
    )
    session.conversation.append(user_turn)

    # Store biomarker data if provided
    if req.emotion_snapshot:
        session.emotion_snapshots.append(req.emotion_snapshot)
    if req.vocal_metrics:
        session.vocal_metrics.append(req.vocal_metrics)
    if req.hrv_snapshot:
        session.hrv_snapshots.append(req.hrv_snapshot)
    if req.oculomotor_snapshot:
        session.oculomotor_snapshots.append(req.oculomotor_snapshot)

    # Check for crisis indicators
    risk_level, risk_alert = assess_risk(req.message)
    if risk_level.value > session.risk_level.value:
        session.risk_level = risk_level

    # Build conversation history for Gemini
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in session.conversation[:-1]  # Exclude the current message
    ]

    # Get AI response with trigger-aware context
    emotion_ctx = req.emotion_snapshot.model_dump() if req.emotion_snapshot else None
    vocal_ctx = req.vocal_metrics.model_dump() if req.vocal_metrics else None
    oculomotor_ctx = req.oculomotor_snapshot.model_dump() if req.oculomotor_snapshot else None

    reply = await therapist.chat(
        message=req.message,
        conversation_history=history,
        emotion_context=emotion_ctx,
        vocal_context=vocal_ctx,
        oculomotor_context=oculomotor_ctx,
        user_profile=user_profile if user_profile.get("triggers") else None,
    )

    # Store AI response
    ai_turn = ConversationTurn(
        role="model",
        content=reply,
        timestamp=time.time(),
    )
    session.conversation.append(ai_turn)

    return ChatResponse(
        reply=reply,
        risk_level=risk_level,
        risk_alert=risk_alert,
    )


# ─── Biomarker Data Endpoints ──────────────────────────────────────

@app.post("/api/sessions/{session_id}/emotions")
async def submit_emotions(session_id: str, snapshot: EmotionSnapshot):
    """Submit a facial emotion snapshot for the session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id].emotion_snapshots.append(snapshot)
    return {"status": "recorded"}


@app.post("/api/sessions/{session_id}/vocals")
async def submit_vocals(session_id: str, metrics: VocalMetrics):
    """Submit vocal biomarker data for the session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id].vocal_metrics.append(metrics)
    return {"status": "recorded"}


# ─── Report Endpoint ───────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/report")
async def get_report(session_id: str):
    """Generate a clinical report for the session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if session.is_active:
        raise HTTPException(
            status_code=400,
            detail="Session must be ended before generating a report"
        )

    report = await generate_report(session)
    return report


# ─── Village Endpoints ─────────────────────────────────────────────

@app.get("/api/village/state")
async def get_village_state():
    """Get current village state."""
    return village_state


class VillageGrowRequest(BaseModel):
    sessions_to_add: int = 1

@app.post("/api/village/grow")
async def grow_village(req: VillageGrowRequest):
    """Manually grow the village (for demo purposes)."""
    global village_state
    village_state["sessions_completed"] += req.sessions_to_add
    old_level = village_state["level"]
    village_state["level"] = _compute_level(village_state["sessions_completed"])
    return {
        "level": village_state["level"],
        "sessions_completed": village_state["sessions_completed"],
        "leveled_up": village_state["level"] > old_level,
    }


@app.post("/api/village/reset")
async def reset_village():
    """Reset village to starting state (for demo)."""
    global village_state
    village_state = {
        "level": 1,
        "sessions_completed": 0,
        "streak": 0,
        "last_session_date": None,
    }
    return village_state


# ─── Trigger Profile Endpoints ─────────────────────────────────────

@app.get("/api/profile/triggers")
async def get_trigger_profile():
    """Get the user's accumulated trigger profile."""
    return user_profile


# ─── Health Check ──────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "AuraBridge API v2.0"}


# ─── Daily App Check-In Analysis ───────────────────────────────────

class CheckInAnalyzeRequest(BaseModel):
    transcript: str = ""
    emotions: dict = {}
    vocals: dict = {}
    oculomotor: dict = {}


@app.post("/api/checkin/analyze")
async def analyze_checkin(req: CheckInAnalyzeRequest):
    """Single-call check-in analysis for the Daily app."""
    from google.genai import types

    prompt = f"""You are a clinical AI analyzing a mental health check-in. The user recorded a video/audio journal entry. Analyze all available multi-modal data and provide a structured assessment.

## Transcript
{req.transcript or "No speech detected."}

## Facial Emotion Averages (0-1 scale)
{json_lib.dumps(req.emotions, indent=2) if req.emotions else "No facial data captured."}

## Vocal Biomarkers
{json_lib.dumps(req.vocals, indent=2) if req.vocals else "No vocal data captured."}

## Eye Tracking Metrics
{json_lib.dumps(req.oculomotor, indent=2) if req.oculomotor else "No eye tracking data captured."}

Provide your analysis as JSON with this exact format:
{{
    "emotions": [
        {{"name": "emotion name", "emoji": "emoji", "score": 0.0 to 1.0}},
        (exactly 4 emotions, sorted by score descending)
    ],
    "patterns": [
        {{"label": "pattern name", "emoji": "emoji"}},
        (3-5 detected behavioral/emotional patterns)
    ],
    "journal_summary": "A 3-4 sentence personalized journal entry summarizing what the AI observed. Reference specific biomarker data points. Write in third person. Include one <strong>bolded recommendation</strong>.",
    "dominant_mood": "one-word mood descriptor",
    "risk_level": "none"
}}"""

    try:
        response = await therapist.client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            ),
        )

        text = response.text or "{}"
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json_lib.loads(text)
    except Exception as e:
        print(f"Check-in analysis error: {e}")
        return {
            "emotions": [
                {"name": "Stress", "emoji": "😤", "score": 0.6},
                {"name": "Tiredness", "emoji": "😩", "score": 0.5},
                {"name": "Sadness", "emoji": "😢", "score": 0.3},
                {"name": "Calm", "emoji": "😌", "score": 0.1},
            ],
            "patterns": [
                {"label": "Check-In Completed", "emoji": "✅"},
            ],
            "journal_summary": "The user completed a check-in entry. Full analysis could not be processed at this time.",
            "dominant_mood": "stressed",
            "risk_level": "low",
        }
