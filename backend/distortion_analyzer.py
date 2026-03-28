"""Cognitive Distortion Analyzer — classifies transcript sentences into CBT distortions."""
from __future__ import annotations

import os
import json
from google import genai
from google.genai import types


# The 10 core CBT Cognitive Distortions
DISTORTION_NAMES = [
    "All-or-Nothing Thinking",
    "Overgeneralization",
    "Mental Filter",
    "Disqualifying the Positive",
    "Jumping to Conclusions",
    "Magnification/Minimization",
    "Emotional Reasoning",
    "Should Statements",
    "Labeling",
    "Personalization",
]


async def analyze_cognitive_distortions(transcript: str) -> dict:
    """Run a second-pass LLM analysis to classify patient sentences into
    the 10 core CBT Cognitive Distortions.

    Returns:
        {
            "distortions": [{"name": str, "count": int, "examples": [str]}],
            "summary": str
        }
    """
    if not transcript or not transcript.strip():
        return {
            "distortions": [{"name": d, "count": 0, "examples": []} for d in DISTORTION_NAMES],
            "summary": "No patient dialogue was available for cognitive distortion analysis.",
        }

    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""You are a clinical CBT analyst. Analyze the following therapy session transcript and identify instances of the 10 core Cognitive Distortions in the PATIENT's statements ONLY (ignore the AI assistant's statements).

## The 10 Cognitive Distortions
1. All-or-Nothing Thinking — Seeing things in black-and-white categories
2. Overgeneralization — Viewing a single negative event as a never-ending pattern
3. Mental Filter — Focusing exclusively on negatives, ignoring positives
4. Disqualifying the Positive — Rejecting positive experiences as "not counting"
5. Jumping to Conclusions — Making negative interpretations without evidence (mind reading, fortune telling)
6. Magnification/Minimization — Exaggerating negatives or shrinking positives
7. Emotional Reasoning — Assuming negative feelings reflect reality ("I feel it, so it must be true")
8. Should Statements — Using "should", "must", "ought to" creating guilt or frustration
9. Labeling — Attaching labels to oneself or others instead of describing behavior
10. Personalization — Taking responsibility for events outside one's control

## Transcript
{transcript}

## Instructions
For each distortion, count how many distinct instances you found and provide up to 3 direct quote examples from the patient's speech. If no instances found, set count to 0 and examples to empty array.

Also provide a 2-3 sentence clinical summary of the overall distortion patterns observed.

Respond ONLY with valid JSON in this exact format:
{{
    "distortions": [
        {{"name": "All-or-Nothing Thinking", "count": 0, "examples": []}},
        {{"name": "Overgeneralization", "count": 0, "examples": []}},
        {{"name": "Mental Filter", "count": 0, "examples": []}},
        {{"name": "Disqualifying the Positive", "count": 0, "examples": []}},
        {{"name": "Jumping to Conclusions", "count": 0, "examples": []}},
        {{"name": "Magnification/Minimization", "count": 0, "examples": []}},
        {{"name": "Emotional Reasoning", "count": 0, "examples": []}},
        {{"name": "Should Statements", "count": 0, "examples": []}},
        {{"name": "Labeling", "count": 0, "examples": []}},
        {{"name": "Personalization", "count": 0, "examples": []}}
    ],
    "summary": "..."
}}"""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )

        text = response.text or "{}"
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        return data

    except Exception as e:
        print(f"Cognitive distortion analysis error: {e}")
        return {
            "distortions": [{"name": d, "count": 0, "examples": []} for d in DISTORTION_NAMES],
            "summary": "Cognitive distortion analysis could not be completed.",
        }
