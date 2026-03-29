# Saathi (साथी)

**Your AI companion that actually listens — not just to your words, but to how you're really doing.**

Built for the US-Nepal Hackathon 2026.

---

## What is Saathi?

Saathi means "companion" in Nepali, and that's exactly what this is. It's a mental health app prototype that tries to understand how you're feeling using real signals — your face, your voice, your eyes — not just what you type.

A lot of mental health apps feel like filling out a form. Saathi is meant to feel more like talking to a friend who genuinely notices when something's off, even if you don't say it directly.

We built this specifically with students and young professionals from Nepal in mind — the kind of stress that comes from family expectations, academic pressure, and trying to figure things out while being far from home.

---

## What it does

**Multi-modal check-in (Scenario 2 — this one actually works)**

You open the app, hit record, and just talk. While you're talking, the app is:

- Watching your face (MediaPipe reads 52 facial landmarks and turns them into 7 emotion scores in real-time)
- Listening to your voice (Web Audio API picks up jitter, shimmer, and pitch — subtle stress markers you can't fake)
- Tracking your eyes (blink rate and gaze avoidance, which often show discomfort before words do)
- Transcribing what you say (browser SpeechRecognition)

All of that gets sent to Gemini 2.0 Flash, which writes a personalized journal entry summarizing what it observed and flags any patterns worth noting.

**The other scenarios (demo flows)**

- Lock screen HRV alert — simulates a smartwatch detecting a stress spike and reaching out before you even open the app
- Pre-meeting check-in — uses calendar context + HRV data to offer a quick grounding exercise before something stressful
- Crisis safety net — detects distress in language and surfaces emergency resources (Nepal helpline: 1166)
- Weekly insights — mood trends, a burnout score, and a breakdown of cognitive patterns over time

**Saathi the AI companion**

There's a built-in chat companion named Saathi who uses CBT and motivational interviewing techniques but talks like a friend. It remembers things you've said earlier in the conversation and nudges you toward topics that matter to you — without making it feel like therapy.

**The plant**

Every time you check in, your plant grows. It's a small thing, but it makes consistency feel rewarding. The plant goes through 5 stages from seedling to full bloom, and the pot has a Dhaka textile pattern on it because we wanted Nepal to be visible in the design, not just mentioned in the description.

**Nepal cultural context**

When you tap certain stressor cards (family pressure, academic stress, etc.), the AI gets extra context about what that kind of stress actually means in a South Asian context. It's a small touch but it makes the responses feel less generic.

---

## How to run it

You'll need Python 3.10+, a Google API key, and Chrome.

**Get a Google API key:** https://aistudio.google.com/app/apikey

```bash
# Clone the repo
git clone https://github.com/Poudel-Sanskriti/US-Nepal-Hackathon.git
cd US-Nepal-Hackathon

# Set up the backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Open .env and paste your GOOGLE_API_KEY

# Start the backend
uvicorn main:app --port 8000 --reload
```

In a new terminal:

```bash
# Serve the frontend
npx -y serve . -l 3456
```

Open Chrome and go to `http://localhost:3456`

To see the live check-in: go to Scenario 2, allow camera and mic when prompted, hit record, talk for a bit, then tap **Analyze with AI**.

---

## Project structure

```
US-Nepal-Hackathon/
├── index.html              # All screens in one file (iPhone frame demo)
├── app.js                  # All the frontend logic — camera, audio, plant, AI calls
├── styles.css              # Design system with Dhaka pattern accents
├── assets/
│   └── icon.png
└── backend/
    ├── main.py             # FastAPI server
    ├── gemini_client.py    # Saathi AI companion (trigger-aware, culturally informed)
    ├── models.py           # Data models
    ├── risk_monitor.py     # Crisis detection
    ├── report_generator.py # Session summaries
    ├── trigger_analyzer.py # Tracks emotional patterns across sessions
    ├── distortion_analyzer.py  # CBT cognitive distortion detection
    └── requirements.txt
```

---

## Tech stack

- **Google Gemini 2.0 Flash** — the AI brain behind both the check-in analysis and the companion chat
- **MediaPipe FaceLandmarker** — facial emotion detection, runs in-browser
- **Web Audio API** — extracts vocal stress biomarkers from mic input
- **SpeechRecognition API** — live transcription (works best in Chrome)
- **FastAPI** — lightweight async Python backend
- **Vanilla JS** — no frameworks, just plain JavaScript

---

## The demo

The prototype is designed to run as a phone screen inside a browser — you'll see an iPhone-style frame. Use Chrome for the full experience (Firefox doesn't support SpeechRecognition, Safari has partial MediaPipe support).

The camera and mic need browser permission. If you see a "Camera access required" message, click the lock icon in your browser's address bar and allow both.

---

## Team

**Binit KC** and **Sanskriti Poudel**

US-Nepal Hackathon 2026
