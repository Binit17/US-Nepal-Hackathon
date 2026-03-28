"""Report generation module for post-session clinical summaries."""
from __future__ import annotations

import os
from datetime import datetime
from google import genai
from google.genai import types

from models import (
    Session, SessionReport, RiskLevel, CognitiveDistortion,
    ConversationTurn, EmotionSnapshot, VocalMetrics,
    HRVSnapshot, OculomotorSnapshot,
)
from distortion_analyzer import analyze_cognitive_distortions


async def generate_report(session: Session) -> SessionReport:
    """Generate a clinical summary report for a completed session.
    
    Uses Gemini to analyze the conversation transcript along with
    emotion, vocal, HRV, and oculomotor biomarker data to produce
    a structured report. Also runs cognitive distortion analysis.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    # Compute duration
    ended = session.ended_at or datetime.now()
    duration = (ended - session.created_at).total_seconds() / 60.0

    # Build transcript text
    transcript_text = "\n".join(
        f"[{turn.role.upper()}]: {turn.content}"
        for turn in session.conversation
    )

    # Summarize all biomarker data
    emotion_summary = _summarize_emotions(session.emotion_snapshots)
    vocal_summary = _summarize_vocals(session.vocal_metrics)
    hrv_summary = _summarize_hrv(session.hrv_snapshots)
    oculomotor_summary = _summarize_oculomotor(session.oculomotor_snapshots)

    # Run cognitive distortion analysis in parallel with main report
    distortion_result = await analyze_cognitive_distortions(transcript_text)

    prompt = f"""You are a clinical AI assistant generating a post-session report for a therapist. 
Analyze the following therapy session data and generate a structured summary.

## Session Transcript
{transcript_text if transcript_text else "No conversation recorded."}

## Facial Emotion Data Summary
{emotion_summary}

## Vocal Biomarker Summary
{vocal_summary}

## Heart Rate Variability (HRV) Summary
{hrv_summary}

## Oculomotor (Eye Tracking) Summary
{oculomotor_summary}

## Session Duration
{duration:.1f} minutes

Please provide:
1. **Overall Mood**: A one-word or short phrase summary of the patient's dominant emotional state
2. **Emotional Trajectory**: How did the patient's emotions change throughout the session? (2-3 sentences)
3. **Key Themes**: List 3-5 key topics or themes discussed
4. **Risk Assessment**: Your assessment of any risk indicators observed (1-2 sentences)
5. **Recommendations**: 3-5 actionable recommendations for the treating therapist

Format your response as JSON with these exact keys:
{{
    "overall_mood": "...",
    "emotional_trajectory": "...",
    "key_themes": ["...", "..."],
    "risk_assessment": "...",
    "recommendations": ["...", "..."]
}}
"""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            ),
        )

        import json
        text = response.text or "{}"
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
    except Exception:
        data = {
            "overall_mood": "Unable to analyze",
            "emotional_trajectory": "Report generation encountered an error.",
            "key_themes": ["Session data available for manual review"],
            "risk_assessment": f"Risk level: {session.risk_level.value}",
            "recommendations": ["Review session transcript manually"],
        }

    # Build cognitive distortion models
    cognitive_distortions = [
        CognitiveDistortion(**d)
        for d in distortion_result.get("distortions", [])
    ]

    return SessionReport(
        session_id=session.id,
        patient_name=session.patient_name,
        session_date=session.created_at,
        duration_minutes=round(duration, 1),
        overall_mood=data.get("overall_mood", "Unknown"),
        emotional_trajectory=data.get("emotional_trajectory", ""),
        key_themes=data.get("key_themes", []),
        risk_assessment=data.get("risk_assessment", ""),
        risk_level=session.risk_level,
        recommendations=data.get("recommendations", []),
        emotion_summary=emotion_summary if isinstance(emotion_summary, dict) else {"raw": emotion_summary},
        vocal_summary=vocal_summary if isinstance(vocal_summary, dict) else {"raw": vocal_summary},
        hrv_summary=hrv_summary if isinstance(hrv_summary, dict) else {"raw": hrv_summary},
        oculomotor_summary=oculomotor_summary if isinstance(oculomotor_summary, dict) else {"raw": oculomotor_summary},
        cognitive_distortions=cognitive_distortions,
        cognitive_distortions_summary=distortion_result.get("summary", ""),
        full_transcript=session.conversation,
    )


def _summarize_emotions(snapshots: list[EmotionSnapshot]) -> dict:
    """Compute average emotion scores across session."""
    if not snapshots:
        return {"note": "No facial emotion data captured"}

    keys = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
    avgs = {}
    for key in keys:
        values = [getattr(s, key) for s in snapshots]
        avgs[key] = round(sum(values) / len(values), 3)

    avgs["avg_arousal"] = round(
        sum(s.arousal for s in snapshots) / len(snapshots), 3
    )
    avgs["avg_valence"] = round(
        sum(s.valence for s in snapshots) / len(snapshots), 3
    )
    avgs["sample_count"] = len(snapshots)
    return avgs


def _summarize_vocals(metrics: list[VocalMetrics]) -> dict:
    """Compute average vocal biomarkers across session."""
    if not metrics:
        return {"note": "No vocal data captured"}

    avgs = {
        "avg_jitter": round(sum(m.jitter for m in metrics) / len(metrics), 4),
        "avg_shimmer": round(sum(m.shimmer for m in metrics) / len(metrics), 4),
        "avg_f0": round(sum(m.mean_f0 for m in metrics) / len(metrics), 1),
        "avg_energy": round(sum(m.energy for m in metrics) / len(metrics), 3),
        "max_jitter": round(max(m.jitter for m in metrics), 4),
        "max_shimmer": round(max(m.shimmer for m in metrics), 4),
        "sample_count": len(metrics),
    }
    return avgs


def _summarize_hrv(snapshots: list[HRVSnapshot]) -> dict:
    """Compute average HRV metrics across session."""
    if not snapshots:
        return {"note": "No HRV data captured"}

    avg_hr = round(sum(s.heart_rate for s in snapshots) / len(snapshots), 1)
    avg_hrv = round(sum(s.hrv_rmssd for s in snapshots) / len(snapshots), 1)
    min_hrv = round(min(s.hrv_rmssd for s in snapshots), 1)
    max_hrv = round(max(s.hrv_rmssd for s in snapshots), 1)

    # Interpret HRV — lower HRV = higher stress
    stress_level = "low" if avg_hrv > 50 else "moderate" if avg_hrv > 30 else "elevated"

    return {
        "avg_heart_rate_bpm": avg_hr,
        "avg_hrv_rmssd_ms": avg_hrv,
        "min_hrv_rmssd_ms": min_hrv,
        "max_hrv_rmssd_ms": max_hrv,
        "physiological_stress": stress_level,
        "sample_count": len(snapshots),
    }


def _summarize_oculomotor(snapshots: list[OculomotorSnapshot]) -> dict:
    """Compute average oculomotor metrics across session."""
    if not snapshots:
        return {"note": "No oculomotor data captured"}

    avg_blink = round(sum(s.blink_rate for s in snapshots) / len(snapshots), 1)
    avg_gaze = round(sum(s.gaze_avoidance_pct for s in snapshots) / len(snapshots), 1)
    max_gaze = round(max(s.gaze_avoidance_pct for s in snapshots), 1)

    # Interpret
    blink_note = "normal" if 15 <= avg_blink <= 20 else "elevated (anxiety indicator)" if avg_blink > 20 else "suppressed (depression indicator)"
    gaze_note = "minimal" if avg_gaze < 20 else "moderate" if avg_gaze < 40 else "significant (avoidance behavior)"

    return {
        "avg_blink_rate_per_min": avg_blink,
        "avg_gaze_avoidance_pct": avg_gaze,
        "max_gaze_avoidance_pct": max_gaze,
        "blink_interpretation": blink_note,
        "gaze_interpretation": gaze_note,
        "sample_count": len(snapshots),
    }
