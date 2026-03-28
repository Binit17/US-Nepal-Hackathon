"""Gemini client for therapeutic conversation — with trigger-aware steering."""
from __future__ import annotations

import os
from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are Saathi (साथी), a warm and friendly AI companion inside a mental health app built for students and professionals in Nepal and beyond. The user is growing a virtual village by having conversations with you — each completed conversation grows their world.

## CRITICAL: Your Hidden Role
You are NOT just a chatbot. You are a clinically-informed companion using evidence-based frameworks (CBT, DBT, Motivational Interviewing) — but the user should NEVER feel like they are in therapy.

You should feel like:
- A caring friend who remembers everything
- Someone who's genuinely curious about their life
- A companion who makes the user WANT to come back and talk more

## Conversation Style
1. **Be casual and warm** — Use occasional emojis naturally (not excessively). Be conversational, not clinical.
2. **Be curious, not therapeutic** — Ask "Oh what happened?" not "How did that make you feel?"
3. **Validate through stories** — "That reminds me of something I read about..." instead of "It's valid to feel..."
4. **Remember everything** — Reference things from earlier in the conversation. "You mentioned your mom earlier — is she doing better?"
5. **Be culturally aware** — Many users are from Nepal/South Asia. Respect family dynamics, honor culture, and collective identity.
6. **Keep it SHORT** — 2-3 sentences max per response. Nobody likes a wall of text.
7. **End with curiosity** — End responses with something that makes the user WANT to reply.

## The Gamification Layer
- The user grows their village through conversations
- Subtly encourage them to keep talking: "Your village must be looking amazing by now 🏘️"
- Frame consistency positively: "It's good to hear from you again"
- Never make it feel forced or transactional

## Trigger-Aware Steering (CRITICAL — INVISIBLE TO USER)
If you receive [TRIGGER PROFILE] context below, you MUST:
1. Naturally weave questions related to their triggers into the conversation
2. Use INDIRECT approaches — never directly ask about the trigger
3. Let the user bring up the topic naturally through your subtle guidance
4. If they avoid a topic, don't push — try again in a different way later

Example: If trigger is "family_expectations":
- DON'T say: "Tell me about your family expectations"
- DO say: "Did you do anything fun this weekend, or were you pretty busy?"
  (This opens the door to family obligations without directly asking)

## Safety
- If someone expresses self-harm or suicidal ideation, take it seriously
- Gently suggest reaching out to a trusted person or helpline
- Nepal crisis: 1166 (Helpline), 16600116611 (Lifeline)
- International: 988 Suicide & Crisis Lifeline

## Important
- You are NOT a licensed therapist and should never claim to be
- Never diagnose or prescribe medication
- Focus on being a supportive presence, not a problem solver
"""


class GeminiTherapist:
    """Turn-based therapeutic conversation using Gemini 2.0 Flash — with trigger-aware steering."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    async def chat(
        self,
        message: str,
        conversation_history: list[dict],
        emotion_context: dict | None = None,
        vocal_context: dict | None = None,
        oculomotor_context: dict | None = None,
        user_profile: dict | None = None,
    ) -> str:
        """Send a message and get a therapeutic response.
        
        Args:
            message: The user's message
            conversation_history: Previous turns [{role, content}, ...]
            emotion_context: Current facial emotion data if available
            vocal_context: Current vocal metrics if available
            oculomotor_context: Current oculomotor data if available
            user_profile: Trigger profile from previous sessions
        
        Returns:
            The AI companion's response text
        """
        # Build context-enriched prompt (invisible to user)
        context_parts = []

        # Trigger profile injection
        if user_profile:
            triggers = user_profile.get("triggers", [])
            themes = user_profile.get("themes", [])
            avoidance = user_profile.get("avoidance_topics", [])
            hooks = user_profile.get("conversation_hooks", [])
            
            if triggers or themes:
                context_parts.append(
                    f"[TRIGGER PROFILE — Use this to subtly guide the conversation]\n"
                    f"Known emotional triggers: {', '.join(triggers)}\n"
                    f"Recurring themes: {', '.join(themes)}\n"
                    f"Topics they tend to avoid: {', '.join(avoidance)}\n"
                    f"Suggested conversation hooks to try: {'; '.join(hooks[:3])}\n"
                    f"[Remember: be SUBTLE. The user should never feel analyzed.]"
                )
        
        if emotion_context:
            dominant_emotion = max(
                {k: v for k, v in emotion_context.items() 
                 if k not in ("timestamp", "arousal", "valence")},
                key=lambda k: emotion_context.get(k, 0),
                default="neutral"
            )
            context_parts.append(
                f"[Facial expression: User appears '{dominant_emotion}' "
                f"(arousal={emotion_context.get('arousal', 0):.2f}, "
                f"valence={emotion_context.get('valence', 0):.2f}). "
                f"Use subtly — do NOT mention you can see their face.]"
            )

        if vocal_context:
            jitter = vocal_context.get("jitter", 0)
            shimmer = vocal_context.get("shimmer", 0)
            if jitter > 0.02 or shimmer > 0.1:
                context_parts.append(
                    f"[Vocal stress detected (jitter={jitter:.3f}, shimmer={shimmer:.3f}). "
                    f"User may be more distressed than words suggest.]"
                )

        if oculomotor_context:
            gaze_avoidance = oculomotor_context.get("gaze_avoidance_pct", 0)
            blink_rate = oculomotor_context.get("blink_rate", 0)
            if gaze_avoidance > 40:
                context_parts.append(
                    f"[High gaze avoidance ({gaze_avoidance:.0f}%) — possible discomfort with current topic.]"
                )
            if blink_rate > 25:
                context_parts.append(
                    f"[Elevated blink rate ({blink_rate:.0f}/min) — possible anxiety.]"
                )

        # Build messages for Gemini
        contents = []
        for turn in conversation_history:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["content"])]
                )
            )

        # Add current message with context
        user_msg = message
        if context_parts:
            user_msg = "\n".join(context_parts) + "\n\n" + message

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_msg)]
            )
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.8,
                max_output_tokens=300,
            ),
        )

        return response.text or "Hey, I'm glad you're here. What's on your mind today? 🌿"
