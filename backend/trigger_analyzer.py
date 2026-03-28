"""Trigger Analyzer — extracts emotional triggers and patterns from conversation history."""
from __future__ import annotations

import os
import json
from google import genai
from google.genai import types


async def analyze_triggers(transcript: str, existing_profile: dict | None = None) -> dict:
    """Analyze a session transcript to extract emotional triggers, recurring themes,
    avoidance patterns, and growth areas. Merges with existing profile.

    Returns:
        {
            "triggers": ["family_expectations", "academic_pressure", ...],
            "themes": ["loneliness", "self_worth", ...],
            "avoidance_topics": ["childhood", ...],
            "growth_areas": ["setting_boundaries", ...],
            "conversation_hooks": [
                "You mentioned feeling pressure about exams — how has that been lately?",
                ...
            ]
        }
    """
    if not transcript or not transcript.strip():
        return existing_profile or _empty_profile()

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    existing_context = ""
    if existing_profile:
        existing_context = f"""
## Existing User Profile (from previous sessions)
- Known triggers: {', '.join(existing_profile.get('triggers', []))}
- Recurring themes: {', '.join(existing_profile.get('themes', []))}
- Topics they avoid: {', '.join(existing_profile.get('avoidance_topics', []))}
- Growth areas: {', '.join(existing_profile.get('growth_areas', []))}
"""

    prompt = f"""You are a clinical pattern analyzer. Analyze this therapy session transcript to build/update a user psychological profile.

{existing_context}

## New Session Transcript
{transcript}

## Your Task
1. Identify emotional TRIGGERS — specific situations, topics, or memories that cause distress
2. Identify recurring THEMES — broader patterns (e.g., self-worth, family dynamics, work stress)
3. Identify AVOIDANCE topics — things the user deflects from, changes subject, or gives short answers to
4. Identify GROWTH areas — skills or insights the user is developing
5. Generate 3-5 CONVERSATION HOOKS — casual, natural questions that would gently guide the NEXT session toward their triggers WITHOUT being obvious. These should feel like friendly curiosity, NOT therapy questions.

## Important Rules for Conversation Hooks
- NEVER be direct like "How does that make you feel?" or "Tell me about your childhood"
- Instead use indirect, casual openers like:
  - "I was curious — do you get to see your family much these days?"
  - "What's been keeping you busy lately?" (if work stress is a trigger)
  - "Have you done anything fun for yourself recently?" (if self-neglect is a theme)
- The user should NEVER feel like they're being analyzed

Respond ONLY with valid JSON:
{{
    "triggers": ["trigger1", "trigger2"],
    "themes": ["theme1", "theme2"],
    "avoidance_topics": ["topic1"],
    "growth_areas": ["area1"],
    "conversation_hooks": ["hook1", "hook2", "hook3"]
}}"""

    try:
        response = await client.aio.models.generate_content(
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

        new_profile = json.loads(text)

        # Merge with existing profile (deduplicate)
        if existing_profile:
            for key in ["triggers", "themes", "avoidance_topics", "growth_areas"]:
                existing = set(existing_profile.get(key, []))
                new_items = set(new_profile.get(key, []))
                new_profile[key] = list(existing | new_items)

        return new_profile

    except Exception as e:
        print(f"Trigger analysis error: {e}")
        return existing_profile or _empty_profile()


def _empty_profile() -> dict:
    return {
        "triggers": [],
        "themes": [],
        "avoidance_topics": [],
        "growth_areas": [],
        "conversation_hooks": [],
    }
