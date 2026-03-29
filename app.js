/* ============================================
   Saathi – AI Mental Health Companion App Logic
   ============================================ */

// ====== MULTI-MODAL CHECK-IN ENGINE ======
const CheckInEngine = {
  // State
  mediaStream: null,
  audioContext: null,
  analyserNode: null,
  audioSource: null,
  faceLandmarker: null,
  isMediaPipeReady: false,
  detectionRafId: null,
  audioRafId: null,
  bioUpdateInterval: null,
  sampleInterval: null,
  cameraReady: false,
  checkinMode: 'video', // 'video' | 'audio'

  // Current data
  emotions: { happy:0, sad:0, angry:0, fearful:0, disgusted:0, surprised:0, neutral:1, arousal:0, valence:0 },
  vocalData: { jitter:0, shimmer:0, meanF0:0, energy:0 },
  oculomotorData: { blinkRate:0, gazeAvoidancePct:0 },

  // Accumulation for averages
  emotionSamples: [],
  vocalSamples: [],

  // Blink detection
  blinkTimestamps: [],
  wasBlinking: false,
  gazeFrameCount: 0,
  gazeAvertFrameCount: 0,

  // Vocal analysis
  prevPitches: [],
  prevAmplitudes: [],

  // Speech
  recognition: null,
  isListening: false,
  transcript: '',

  API_BASE: 'http://localhost:8000',

  // ─── Camera & Mic Init ───
  async initCamera() {
    try {
      const constraints = this.checkinMode === 'video'
        ? { video: { facingMode: 'user', width: 640, height: 480 }, audio: true }
        : { audio: true };

      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

      if (this.checkinMode === 'video') {
        const video = document.getElementById('camera-video');
        if (video) {
          video.srcObject = this.mediaStream;
          video.style.display = 'block';
          const placeholder = document.getElementById('camera-placeholder');
          if (placeholder) placeholder.style.display = 'none';
        }
      }

      // Audio analysis setup
      this.audioContext = new AudioContext();
      this.audioSource = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 2048;
      this.audioSource.connect(this.analyserNode);

      this.cameraReady = true;

      // Init MediaPipe for video mode
      if (this.checkinMode === 'video') {
        this.initMediaPipe();
      }

      return true;
    } catch (err) {
      console.warn('Media access denied:', err);
      const initText = document.querySelector('.camera-init-text');
      if (initText) initText.textContent = 'Camera access denied. Try Audio mode.';
      return false;
    }
  },

  // ─── MediaPipe FaceLandmarker Init ───
  async initMediaPipe() {
    try {
      const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs');
      const { FaceLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
      );

      this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        outputFaceBlendshapes: true,
        runningMode: 'VIDEO',
        numFaces: 1,
      });

      this.isMediaPipeReady = true;
      console.log('✅ MediaPipe FaceLandmarker ready');
    } catch (err) {
      console.warn('MediaPipe init failed (non-critical):', err);
    }
  },

  // ─── Face Detection Loop ───
  startDetection() {
    const video = document.getElementById('camera-video');
    if (!this.faceLandmarker || !video) return;

    let lastTimestamp = -1;

    const detect = () => {
      if (!this.faceLandmarker || !video) return;

      if (video.readyState >= 2 && video.currentTime !== lastTimestamp) {
        lastTimestamp = video.currentTime;
        try {
          const results = this.faceLandmarker.detectForVideo(video, performance.now());
          if (results?.faceBlendshapes && results.faceBlendshapes.length > 0) {
            const bs = results.faceBlendshapes[0].categories;
            this.emotions = this.mapBlendshapesToEmotions(bs);
            this.updateOculomotor(bs);
          }
        } catch (e) { /* skip frame */ }
      }

      this.detectionRafId = requestAnimationFrame(detect);
    };

    detect();
  },

  stopDetection() {
    if (this.detectionRafId) { cancelAnimationFrame(this.detectionRafId); this.detectionRafId = null; }
  },

  // ─── Blendshape → Emotion Mapping (from MVP useMediaPipe.ts) ───
  mapBlendshapesToEmotions(blendshapes) {
    const bs = {};
    for (const b of blendshapes) bs[b.categoryName] = b.score;

    const happy = Math.min(1, (bs['mouthSmileLeft']||0)*0.5 + (bs['mouthSmileRight']||0)*0.5 + (bs['cheekSquintLeft']||0)*0.3 + (bs['cheekSquintRight']||0)*0.3);
    const sad = Math.min(1, (bs['mouthFrownLeft']||0)*0.5 + (bs['mouthFrownRight']||0)*0.5 + (bs['browInnerUp']||0)*0.4);
    const angry = Math.min(1, (bs['browDownLeft']||0)*0.5 + (bs['browDownRight']||0)*0.5 + (bs['mouthPressLeft']||0)*0.3 + (bs['mouthPressRight']||0)*0.3);
    const fearful = Math.min(1, (bs['browInnerUp']||0)*0.4 + (bs['browOuterUpLeft']||0)*0.3 + (bs['browOuterUpRight']||0)*0.3 + (bs['eyeWideLeft']||0)*0.3 + (bs['eyeWideRight']||0)*0.3);
    const disgusted = Math.min(1, (bs['noseSneerLeft']||0)*0.5 + (bs['noseSneerRight']||0)*0.5 + (bs['mouthShrugUpper']||0)*0.3);
    const surprised = Math.min(1, (bs['browOuterUpLeft']||0)*0.4 + (bs['browOuterUpRight']||0)*0.4 + (bs['jawOpen']||0)*0.4 + (bs['eyeWideLeft']||0)*0.2 + (bs['eyeWideRight']||0)*0.2);

    const total = happy + sad + angry + fearful + disgusted + surprised + 0.01;
    const neutral = Math.max(0, 1 - total);
    const sum = total + neutral;

    const emotions = {
      happy: happy/sum, sad: sad/sum, angry: angry/sum,
      fearful: fearful/sum, disgusted: disgusted/sum,
      surprised: surprised/sum, neutral: neutral/sum,
    };

    emotions.valence = emotions.happy*0.8 + emotions.surprised*0.2 - emotions.sad*0.6 - emotions.angry*0.4 - emotions.fearful*0.3 - emotions.disgusted*0.5;
    emotions.arousal = emotions.angry*0.8 + emotions.fearful*0.7 + emotions.surprised*0.6 + emotions.happy*0.3 - emotions.sad*0.3 - emotions.neutral*0.5;
    emotions.valence = Math.max(-1, Math.min(1, emotions.valence));
    emotions.arousal = Math.max(-1, Math.min(1, emotions.arousal));

    return emotions;
  },

  // ─── Oculomotor Tracking (from MVP useOculomotor.ts) ───
  updateOculomotor(blendshapes) {
    const bs = {};
    for (const b of blendshapes) bs[b.categoryName] = b.score;

    const avgBlink = ((bs['eyeBlinkLeft']||0) + (bs['eyeBlinkRight']||0)) / 2;
    const isBlinking = avgBlink > 0.4;
    if (isBlinking && !this.wasBlinking) this.blinkTimestamps.push(Date.now());
    this.wasBlinking = isBlinking;

    const now = Date.now();
    this.blinkTimestamps = this.blinkTimestamps.filter(t => t > now - 10000);
    const blinkRate = Math.round((this.blinkTimestamps.length / 10000) * 60000);

    const maxDev = Math.max(
      bs['eyeLookOutLeft']||0, bs['eyeLookOutRight']||0,
      bs['eyeLookInLeft']||0, bs['eyeLookInRight']||0,
      bs['eyeLookDownLeft']||0, bs['eyeLookDownRight']||0,
      bs['eyeLookUpLeft']||0, bs['eyeLookUpRight']||0
    );
    const isAverting = maxDev > 0.35;
    this.gazeFrameCount++;
    if (isAverting) this.gazeAvertFrameCount++;
    const gazeAvoidancePct = this.gazeFrameCount > 0 ? Math.round((this.gazeAvertFrameCount / this.gazeFrameCount) * 100) : 0;

    this.oculomotorData = { blinkRate, gazeAvoidancePct };
  },

  // ─── Audio Analysis (from MVP useAudioAnalysis.ts) ───
  startAudioAnalysis() {
    if (!this.analyserNode) return;

    const timeData = new Float32Array(this.analyserNode.fftSize);

    const loop = () => {
      this.analyserNode.getFloatTimeDomainData(timeData);

      let sumSq = 0;
      for (let i = 0; i < timeData.length; i++) sumSq += timeData[i] * timeData[i];
      const energy = Math.sqrt(sumSq / timeData.length);

      const sampleRate = this.analyserNode.context.sampleRate;
      const pitch = this.autoCorrelate(timeData, sampleRate);

      if (pitch > 0) {
        this.prevPitches.push(pitch);
        this.prevAmplitudes.push(energy);
        if (this.prevPitches.length > 30) this.prevPitches.shift();
        if (this.prevAmplitudes.length > 30) this.prevAmplitudes.shift();
      }

      let jitter = 0;
      if (this.prevPitches.length > 2) {
        let diffs = 0;
        for (let i = 1; i < this.prevPitches.length; i++) diffs += Math.abs(this.prevPitches[i] - this.prevPitches[i-1]);
        const avgP = this.prevPitches.reduce((a,b) => a+b, 0) / this.prevPitches.length;
        jitter = avgP > 0 ? diffs / (this.prevPitches.length - 1) / avgP : 0;
      }

      let shimmer = 0;
      if (this.prevAmplitudes.length > 2) {
        let diffs = 0;
        for (let i = 1; i < this.prevAmplitudes.length; i++) diffs += Math.abs(this.prevAmplitudes[i] - this.prevAmplitudes[i-1]);
        const avgA = this.prevAmplitudes.reduce((a,b) => a+b, 0) / this.prevAmplitudes.length;
        shimmer = avgA > 0 ? diffs / (this.prevAmplitudes.length - 1) / avgA : 0;
      }

      const meanF0 = this.prevPitches.length > 0 ? this.prevPitches.reduce((a,b) => a+b, 0) / this.prevPitches.length : 0;
      this.vocalData = { jitter, shimmer, meanF0, energy };

      this.audioRafId = requestAnimationFrame(loop);
    };

    loop();
  },

  stopAudioAnalysis() {
    if (this.audioRafId) { cancelAnimationFrame(this.audioRafId); this.audioRafId = null; }
  },

  autoCorrelate(buf, sampleRate) {
    let rms = 0;
    for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / buf.length);
    if (rms < 0.01) return -1;

    const SIZE = buf.length;
    const corr = new Float32Array(SIZE);
    for (let lag = 0; lag < SIZE; lag++) {
      let sum = 0;
      for (let i = 0; i < SIZE - lag; i++) sum += buf[i] * buf[i + lag];
      corr[lag] = sum;
    }

    let d = 0;
    while (d < SIZE && corr[d] > 0) d++;
    let maxVal = -1, maxPos = -1;
    for (let i = d; i < SIZE; i++) {
      if (corr[i] > maxVal) { maxVal = corr[i]; maxPos = i; }
    }
    return maxPos === -1 ? -1 : sampleRate / maxPos;
  },

  // ─── Waveform Drawing (Audio mode) ───
  drawWaveform() {
    const canvas = document.getElementById('audio-waveform-canvas');
    if (!canvas || !this.analyserNode) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const bufLen = this.analyserNode.frequencyBinCount;
    const data = new Uint8Array(bufLen);

    const draw = () => {
      if (!this.analyserNode) return;
      this.analyserNode.getByteTimeDomainData(data);
      ctx.fillStyle = 'rgba(28,28,30,0.3)';
      ctx.fillRect(0, 0, W, H);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#7C5CFC';
      ctx.beginPath();
      const sliceWidth = W / bufLen;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = v * H / 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      if (app.isRecording) requestAnimationFrame(draw);
    };
    draw();
  },

  // ─── Speech Recognition ───
  initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { console.warn('SpeechRecognition not supported'); return; }

    this.recognition = new SR();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';

    this.recognition.onresult = (event) => {
      let finalT = '', interimT = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalT += event.results[i][0].transcript;
        else interimT += event.results[i][0].transcript;
      }
      if (finalT) this.transcript = (this.transcript + ' ' + finalT).trim();

      const display = interimT ? this.transcript + ' ' + interimT : this.transcript;
      const el = document.getElementById(this.checkinMode === 'audio' ? 'audio-transcript-text' : 'transcript-text');
      if (el) el.textContent = display || 'Listening...';
    };

    this.recognition.onend = () => {
      if (this.isListening) {
        setTimeout(() => { if (this.isListening) try { this.recognition.start(); } catch(e) {} }, 100);
      }
    };

    this.recognition.onerror = (event) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      this.isListening = false;
    };
  },

  startListening() {
    if (!this.recognition) this.initSpeechRecognition();
    if (!this.recognition || this.isListening) return;
    this.transcript = '';
    try { this.recognition.start(); this.isListening = true; } catch(e) {}
  },

  stopListening() {
    this.isListening = false;
    if (this.recognition) try { this.recognition.stop(); } catch(e) {}
  },

  // ─── Biomarker UI Update ───
  updateBiomarkerUI() {
    const emoMap = { happy:'😊', sad:'😢', angry:'😠', fearful:'😨', disgusted:'🤢', surprised:'😮', neutral:'😐' };
    const emoEmoji = document.getElementById('bio-emotion-emoji');
    const emoLabel = document.getElementById('bio-emotion-label');
    if (emoEmoji && emoLabel) {
      const dominant = Object.entries(this.emotions)
        .filter(([k]) => !['arousal','valence'].includes(k))
        .sort((a,b) => b[1] - a[1])[0];
      emoEmoji.textContent = emoMap[dominant[0]] || '😐';
      emoLabel.textContent = dominant[0].charAt(0).toUpperCase() + dominant[0].slice(1) + ' ' + Math.round(dominant[1]*100) + '%';
    }

    const vocLabel = document.getElementById('bio-vocal-label');
    if (vocLabel) {
      const stressed = this.vocalData.jitter > 0.02 || this.vocalData.shimmer > 0.1;
      vocLabel.textContent = (stressed ? 'Stressed' : 'Normal') + (this.vocalData.meanF0 > 0 ? ' · ' + Math.round(this.vocalData.meanF0) + 'Hz' : '');
    }

    const eyeLabel = document.getElementById('bio-eye-label');
    if (eyeLabel) {
      eyeLabel.textContent = this.oculomotorData.blinkRate + '/min · ' + this.oculomotorData.gazeAvoidancePct + '% off';
    }
  },

  // ─── Start All Real-Time Processing ───
  startProcessing() {
    // Start face detection if MediaPipe is ready
    if (this.isMediaPipeReady) this.startDetection();

    // Start audio analysis
    this.startAudioAnalysis();

    // Start speech recognition
    this.startListening();

    // Show biomarker overlay
    const overlay = document.getElementById('biomarker-overlay');
    if (overlay && this.checkinMode === 'video') overlay.classList.remove('hidden');

    // Show live transcript
    const lt = document.getElementById(this.checkinMode === 'audio' ? 'audio-live-transcript' : 'live-transcript');
    if (lt) lt.classList.remove('hidden');

    // Periodic UI updates + sampling
    this.bioUpdateInterval = setInterval(() => this.updateBiomarkerUI(), 250);
    this.sampleInterval = setInterval(() => {
      this.emotionSamples.push({...this.emotions});
      this.vocalSamples.push({...this.vocalData});
    }, 1000);

    // Waveform for audio mode
    if (this.checkinMode === 'audio') this.drawWaveform();
  },

  // ─── Stop All Processing ───
  stopProcessing() {
    this.stopDetection();
    this.stopAudioAnalysis();
    this.stopListening();
    if (this.bioUpdateInterval) { clearInterval(this.bioUpdateInterval); this.bioUpdateInterval = null; }
    if (this.sampleInterval) { clearInterval(this.sampleInterval); this.sampleInterval = null; }
  },

  // ─── Backend API Call ───
  async analyzeWithBackend() {
    const avgEmo = this.emotionSamples.length > 0
      ? Object.keys(this.emotions).reduce((acc, key) => {
          acc[key] = +(this.emotionSamples.reduce((s, e) => s + (e[key]||0), 0) / this.emotionSamples.length).toFixed(3);
          return acc;
        }, {})
      : this.emotions;

    const avgVoc = this.vocalSamples.length > 0
      ? {
          jitter: +(this.vocalSamples.reduce((s,v) => s+v.jitter, 0) / this.vocalSamples.length).toFixed(4),
          shimmer: +(this.vocalSamples.reduce((s,v) => s+v.shimmer, 0) / this.vocalSamples.length).toFixed(4),
          meanF0: +(this.vocalSamples.reduce((s,v) => s+v.meanF0, 0) / this.vocalSamples.length).toFixed(1),
          energy: +(this.vocalSamples.reduce((s,v) => s+v.energy, 0) / this.vocalSamples.length).toFixed(4),
        }
      : this.vocalData;

    const resp = await fetch(this.API_BASE + '/api/checkin/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: this.transcript,
        emotions: avgEmo,
        vocals: avgVoc,
        oculomotor: this.oculomotorData,
        cycle_context: CycleEngine.getContextString(),
        stressor_context: app.getStressorContext(),
      }),
    });

    if (!resp.ok) throw new Error('Backend returned ' + resp.status);
    return await resp.json();
  },

  // ─── Display Real Results ───
  displayResults(data) {
    // Emotion bars
    const container = document.getElementById('emotion-bars-container');
    if (container && data.emotions && data.emotions.length) {
      const colors = ['#FF6B8A', '#007AFF', '#FF9500', '#34C759'];
      container.innerHTML = data.emotions.map((e, i) => `
        <div class="emotion-bar">
          <span class="emotion-label">${e.emoji} ${e.name}</span>
          <div class="emotion-track"><div class="emotion-fill" style="width:${Math.round(e.score*100)}%; background:${colors[i%colors.length]}"></div></div>
          <span class="emotion-pct">${Math.round(e.score*100)}%</span>
        </div>
      `).join('');
    }

    // Patterns
    const patternsEl = document.getElementById('patterns-container');
    if (patternsEl && data.patterns && data.patterns.length) {
      patternsEl.innerHTML = data.patterns.map(p => `<span class="tag tag-burnout">${p.emoji} ${p.label}</span>`).join('');
    }

    // Journal
    const ts = document.getElementById('journal-timestamp');
    if (ts) {
      const now = new Date();
      ts.textContent = `${now.toLocaleDateString('en-US',{month:'long',day:'numeric',year:'numeric'})} · ${now.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit'})} · ${this.checkinMode === 'video' ? 'Video' : 'Audio'} Entry`;
    }
    const summary = document.getElementById('journal-summary');
    if (summary) summary.innerHTML = data.journal_summary || 'Analysis complete.';

    const moodTags = document.getElementById('journal-mood-tags');
    if (moodTags && data.dominant_mood) {
      moodTags.innerHTML = `
        <span class="tag tag-burnout" style="font-size:11px">🎯 ${data.dominant_mood}</span>
        <span class="tag tag-fatigue" style="font-size:11px">AI-analyzed</span>
      `;
    }
  },

  // ─── Full Cleanup ───
  cleanup() {
    this.stopProcessing();
    if (this.faceLandmarker) { try { this.faceLandmarker.close(); } catch(e) {} this.faceLandmarker = null; }
    if (this.audioContext) { try { this.audioContext.close(); } catch(e) {} this.audioContext = null; }
    if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }

    const video = document.getElementById('camera-video');
    if (video) { video.srcObject = null; video.style.display = 'none'; }
    const placeholder = document.getElementById('camera-placeholder');
    if (placeholder) placeholder.style.display = '';

    // Reset all data
    this.emotions = { happy:0, sad:0, angry:0, fearful:0, disgusted:0, surprised:0, neutral:1, arousal:0, valence:0 };
    this.vocalData = { jitter:0, shimmer:0, meanF0:0, energy:0 };
    this.oculomotorData = { blinkRate:0, gazeAvoidancePct:0 };
    this.transcript = '';
    this.emotionSamples = [];
    this.vocalSamples = [];
    this.prevPitches = [];
    this.prevAmplitudes = [];
    this.blinkTimestamps = [];
    this.wasBlinking = false;
    this.gazeFrameCount = 0;
    this.gazeAvertFrameCount = 0;
    this.isMediaPipeReady = false;
    this.cameraReady = false;
  },
};


// ====== CYCLE-AWARE ENGINE ======
const CYCLE_PHASES = {
  menstrual: {
    name: 'Menstrual Phase', emoji: '🌙',
    color: '#FF6B8A', bg: 'rgba(255,107,138,0.09)', border: 'rgba(255,107,138,0.22)',
    headline: 'Rest & Restore',
    body: 'Your body is doing important work. Lower energy is completely normal — this is a time for gentleness, not pushing through.',
    workTip: 'Avoid high-stakes decisions today. Focus on completing existing tasks, not starting new ones.',
    checkinPrompt: 'How is your body feeling today? Be honest — there\'s no "should" here.',
    tags: ['Lower energy', 'Need rest', 'Reflective'],
  },
  follicular: {
    name: 'Follicular Phase', emoji: '🌱',
    color: '#34D399', bg: 'rgba(52,211,153,0.09)', border: 'rgba(52,211,153,0.22)',
    headline: 'Rising Energy',
    body: 'Estrogen is rising. You may feel more optimistic, creative, and ready to take on new challenges.',
    workTip: 'Great time to start projects, brainstorm ideas, or have important conversations.',
    checkinPrompt: 'What are you feeling motivated or excited about right now?',
    tags: ['Rising energy', 'Creative', 'Optimistic'],
  },
  ovulatory: {
    name: 'Ovulatory Phase', emoji: '✨',
    color: '#FBBF24', bg: 'rgba(251,191,36,0.09)', border: 'rgba(251,191,36,0.28)',
    headline: 'Peak Power',
    body: 'Peak estrogen means peak confidence and social energy. Your verbal fluency and communication are at their strongest.',
    workTip: 'Schedule presentations, interviews, or difficult conversations for this window — you\'re at your best.',
    checkinPrompt: 'What\'s feeling possible for you right now?',
    tags: ['Peak energy', 'Social', 'Confident'],
  },
  luteal: {
    name: 'Luteal Phase', emoji: '🍂',
    color: '#FF9500', bg: 'rgba(255,149,0,0.09)', border: 'rgba(255,149,0,0.22)',
    headline: 'Turn Inward',
    body: 'Progesterone rises then falls. Sensitivity, irritability, or fatigue are hormonal — not a character flaw.',
    workTip: 'Wrap up projects and review rather than start new ones. Setting limits right now is valid and wise.',
    checkinPrompt: 'What\'s feeling heavy or hard right now? Let\'s name it.',
    tags: ['Sensitive', 'Inward', 'PMS possible'],
  },
};

const CycleEngine = {
  lastPeriodDate: null,
  cycleLength: 28,
  dismissed: false,

  setData(lastPeriod, cycleLength) {
    this.lastPeriodDate = new Date(lastPeriod);
    this.cycleLength = parseInt(cycleLength) || 28;
  },

  getDayInCycle() {
    if (!this.lastPeriodDate) return null;
    const daysSince = Math.floor((new Date() - this.lastPeriodDate) / 86400000);
    return (daysSince % this.cycleLength) + 1;
  },

  getCurrentPhase() {
    const day = this.getDayInCycle();
    if (day === null) return null;
    let key = 'luteal';
    if (day <= 5) key = 'menstrual';
    else if (day <= 13) key = 'follicular';
    else if (day <= 16) key = 'ovulatory';
    return { ...CYCLE_PHASES[key], day };
  },

  getContextString() {
    const phase = this.getCurrentPhase();
    if (!phase) return '';
    return `[Hormonal Context: Day ${phase.day} of cycle — ${phase.name}. ` +
      `Emotional tendencies: ${phase.tags.join(', ')}. ` +
      `Acknowledge if relevant to mood/energy, but don't over-explain.]`;
  },
};

// ====== RECOMMENDATIONS DATA ======
const RECS = [
  // ── BURNOUT ──
  {
    category: 'burnout', type: 'course',
    title: 'Burnout: How to Avoid It',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '1h 2m', free: true, audience: ['student', 'professional'],
    desc: 'Recognise the warning signs of burnout and build sustainable work habits.',
  },
  {
    category: 'burnout', type: 'course',
    title: 'Managing Stress for Positive Change',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '58m', free: true, audience: ['student', 'professional'],
    desc: 'Turn stress into a driver for growth rather than a source of damage.',
  },
  {
    category: 'burnout', type: 'article',
    title: '6 Causes of Burnout, and How to Avoid Them',
    source: 'Harvard Business Review', sourceColor: '#C41E3A',
    duration: '6 min read', free: true, audience: ['professional'],
    desc: 'Research-backed breakdown of why professionals burn out and what actually helps.',
  },
  {
    category: 'burnout', type: 'article',
    title: 'Burnout Is About Your Workplace, Not Your People',
    source: 'Harvard Business Review', sourceColor: '#C41E3A',
    duration: '8 min read', free: true, audience: ['professional'],
    desc: 'Reframes burnout as a systemic issue — useful for advocating for yourself.',
  },
  // ── ANXIETY / STRESS ──
  {
    category: 'anxiety', type: 'course',
    title: 'Building Resilience',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '1h 14m', free: true, audience: ['student', 'professional'],
    desc: 'Practical frameworks for bouncing back from setbacks and pressure.',
  },
  {
    category: 'anxiety', type: 'course',
    title: 'Mindfulness Practices',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '45m', free: true, audience: ['student', 'professional'],
    desc: 'Short evidence-based mindfulness exercises for high-pressure environments.',
  },
  {
    category: 'anxiety', type: 'article',
    title: 'How to Recover from Work Stress, According to Science',
    source: 'Harvard Business Review', sourceColor: '#C41E3A',
    duration: '5 min read', free: true, audience: ['professional'],
    desc: 'Science-backed recovery strategies that actually move the needle on stress.',
  },
  {
    category: 'anxiety', type: 'course',
    title: 'The Science of Well-Being',
    source: 'Coursera — Yale University', sourceColor: '#0056D2',
    duration: '19 hours', free: true, audience: ['student'],
    desc: 'Yale\'s most popular course ever. Understand what actually makes humans flourish.',
  },
  // ── FATIGUE / LOW ENERGY ──
  {
    category: 'fatigue', type: 'course',
    title: 'Time Management Fundamentals',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '2h 55m', free: true, audience: ['student', 'professional'],
    desc: 'Stop feeling behind. Build a system that works with your energy, not against it.',
  },
  {
    category: 'fatigue', type: 'course',
    title: 'Overcoming Procrastination',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '1h 23m', free: true, audience: ['student', 'professional'],
    desc: 'Understand the emotional roots of procrastination and break the cycle.',
  },
  {
    category: 'fatigue', type: 'video',
    title: 'How to Study Effectively — Evidence-Based Tips',
    source: 'YouTube', sourceColor: '#FF0000',
    duration: '18 min', free: true, audience: ['student'],
    desc: 'Cognitive science-backed study techniques used by top university students.',
  },
  // ── CAREER / GENERAL ──
  {
    category: 'general', type: 'course',
    title: 'Work Smarter, Not Harder: Time Management',
    source: 'Coursera — UC Irvine', sourceColor: '#0056D2',
    duration: '10 hours', free: true, audience: ['professional'],
    desc: 'Reduce cognitive overload and get more done with less pressure.',
  },
  {
    category: 'general', type: 'course',
    title: 'Learning How to Learn',
    source: 'Coursera — UC San Diego', sourceColor: '#0056D2',
    duration: '15 hours', free: true, audience: ['student'],
    desc: 'The most enrolled online course in history. Master how your brain retains knowledge.',
  },
  {
    category: 'general', type: 'course',
    title: 'Managing Your Career: Early Career',
    source: 'LinkedIn Learning', sourceColor: '#0A66C2',
    duration: '1h 10m', free: true, audience: ['student', 'professional'],
    desc: 'Reduce career uncertainty with a clear framework for professional growth.',
  },
  {
    category: 'general', type: 'video',
    title: 'How to Stop Being Tired All the Time',
    source: 'YouTube', sourceColor: '#FF0000',
    duration: '12 min', free: true, audience: ['student', 'professional'],
    desc: 'Evidence-based tips for recovering energy during high-demand periods.',
  },
];

const CATEGORY_META = {
  burnout:  { label: 'For Burnout Recovery',   emoji: '🔥', color: '#FF6B8A' },
  anxiety:  { label: 'For Stress & Anxiety',    emoji: '🌀', color: '#7C5CFC' },
  fatigue:  { label: 'For Low Energy & Focus',  emoji: '⚡', color: '#FF9500' },
  general:  { label: 'Career & Learning',        emoji: '🚀', color: '#34D399' },
};

// ====== MAIN APPLICATION ======
const app = {
  currentScreen: 'screen-home',
  currentScenario: 1,
  isRecording: false,
  recordTimer: null,
  recordSeconds: 0,
  breathingInterval: null,
  breathingPhase: 0,
  breathingCycle: 0,
  chartsDrawn: false,

  // ====== SCREEN NAVIGATION ======
  showScreen(screenId) {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(s => {
      s.classList.remove('active', 'slide-out-left');
    });

    requestAnimationFrame(() => {
      const target = document.getElementById(screenId);
      if (target) {
        target.classList.add('active');
        target.scrollTop = 0;
      }
      this.currentScreen = screenId;
    });
  },

  goHome() {
    this.hideNotification();
    this.stopBreathing();
    this.stopRecording();
    this.resetStressors();
    CheckInEngine.cleanup();
    this.showScreen('screen-home');
    this.updateScenarioNav(1);
    this.updateInfoCard(1);
  },

  // ====== SCENARIO ROUTING ======
  goToScenario(num) {
    this.currentScenario = num;
    this.updateScenarioNav(num);
    this.hideNotification();
    CheckInEngine.cleanup();
    // If user navigates via sidebar during onboarding, treat as skip
    if (!this.cycleOnboardingDone) {
      this.cycleOnboardingDone = true;
      CycleEngine.dismissed = true;
      this.saveCycleToStorage();
    }

    switch(num) {
      case 0: this.startScenario0(); break;
      case 1: this.startScenario1(); break;
      case 2: this.startScenario2(); break;
      case 3: this.startScenario3(); break;
      case 4: this.startScenario4(); break;
    }
  },

  updateScenarioNav(num) {
    document.querySelectorAll('.scenario-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`nav-s${num}`);
    if (activeBtn) activeBtn.classList.add('active');
  },

  // ====== SCENARIO 0: LOCK SCREEN HRV ALERT ======
  startScenario0() {
    const dailyNotif = document.getElementById('ls-daily-notif');
    dailyNotif.classList.add('hidden');
    this.showScreen('screen-lockscreen');
    this.updateInfoCard(0);

    setTimeout(() => {
      dailyNotif.classList.remove('hidden');
      dailyNotif.style.animation = 'none';
      requestAnimationFrame(() => { dailyNotif.style.animation = ''; });
    }, 2000);
  },

  // ====== SCENARIO 1: PROACTIVE SUPPORT ======
  startScenario1() {
    this.showScreen('screen-home');
    this.updateInfoCard(1);
    setTimeout(() => { this.showNotification(); }, 800);
  },

  showNotification() {
    const overlay = document.getElementById('notif-overlay');
    overlay.classList.add('show');
  },

  hideNotification() {
    const overlay = document.getElementById('notif-overlay');
    overlay.classList.remove('show');
  },

  showBreathing() {
    this.hideNotification();
    this.showScreen('screen-breathing');
    this.resetBreathing();
    setTimeout(() => { this.startBreathing(); }, 1000);
  },

  resetBreathing() {
    this.stopBreathing();
    this.breathingPhase = 0;
    this.breathingCycle = 0;
    const circle = document.getElementById('breathing-circle');
    const label = document.getElementById('breathing-label');
    const timer = document.getElementById('breathing-timer');
    const instruction = document.getElementById('breathing-instruction');

    circle.className = 'breathing-circle';
    label.textContent = 'Ready';
    timer.textContent = '4';
    instruction.textContent = 'Starting in a moment...';

    document.querySelectorAll('.breath-dot').forEach(d => { d.className = 'breath-dot'; });
  },

  startBreathing() {
    const phases = [
      { name: 'Inhale', duration: 4, className: 'inhale' },
      { name: 'Hold', duration: 4, className: 'hold' },
      { name: 'Exhale', duration: 4, className: 'exhale' },
      { name: 'Hold', duration: 4, className: 'hold' },
    ];

    let phaseIndex = 0;
    let countdown = phases[0].duration;
    const circle = document.getElementById('breathing-circle');
    const label = document.getElementById('breathing-label');
    const timer = document.getElementById('breathing-timer');
    const instruction = document.getElementById('breathing-instruction');
    const dots = document.querySelectorAll('.breath-dot');

    const updatePhase = () => {
      const phase = phases[phaseIndex];
      circle.className = `breathing-circle ${phase.className}`;
      label.textContent = phase.name;
      countdown = phase.duration;
      timer.textContent = countdown;
      if (phase.name === 'Inhale') instruction.textContent = 'Breathe in slowly through your nose';
      else if (phase.name === 'Hold') instruction.textContent = 'Hold your breath gently';
      else instruction.textContent = 'Breathe out slowly through your mouth';
    };

    updatePhase();

    this.breathingInterval = setInterval(() => {
      countdown--;
      timer.textContent = countdown;
      if (countdown <= 0) {
        phaseIndex++;
        if (phaseIndex >= phases.length) {
          if (this.breathingCycle < dots.length) dots[this.breathingCycle].classList.add('completed');
          this.breathingCycle++;
          phaseIndex = 0;
          if (this.breathingCycle >= 4) {
            this.stopBreathing();
            label.textContent = 'Done ✨';
            timer.textContent = '';
            instruction.textContent = 'Great job! You completed the exercise.';
            circle.className = 'breathing-circle';
            return;
          }
          if (this.breathingCycle < dots.length) dots[this.breathingCycle].classList.add('current');
        }
        updatePhase();
      }
    }, 1000);

    if (dots.length > 0) dots[0].classList.add('current');
  },

  stopBreathing() {
    if (this.breathingInterval) { clearInterval(this.breathingInterval); this.breathingInterval = null; }
  },

  // ====== SCENARIO 2: MULTI-MODAL CHECK-IN (REAL) ======
  async startScenario2() {
    this.showScreen('screen-checkin');
    this.updateInfoCard(2);
    this.resetRecording();
    this.updateCheckinPromptForPhase();

    // Reset mode to video
    this.switchCheckinMode('video');

    // Initialize camera + mic
    const ok = await CheckInEngine.initCamera();
    if (!ok) {
      const caption = document.getElementById('video-caption');
      if (caption) caption.textContent = 'Camera access required. Please allow access and try again.';
    }
  },

  switchCheckinMode(mode) {
    CheckInEngine.checkinMode = mode;

    // Update tabs
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    const tab = document.getElementById('tab-' + mode);
    if (tab) tab.classList.add('active');

    // Toggle views
    const videoView = document.getElementById('checkin-video-view');
    const audioView = document.getElementById('checkin-audio-view');

    if (mode === 'video') {
      if (videoView) videoView.classList.remove('hidden');
      if (audioView) audioView.classList.add('hidden');
    } else if (mode === 'audio') {
      if (videoView) videoView.classList.add('hidden');
      if (audioView) audioView.classList.remove('hidden');
    } else if (mode === 'text') {
      // Switch to text input screen (Scenario 3)
      this.startScenario3();
    }
  },

  resetRecording() {
    this.isRecording = false;
    this.recordSeconds = 0;
    if (this.recordTimer) clearInterval(this.recordTimer);

    const btn = document.getElementById('record-btn');
    const audioBtn = document.getElementById('audio-record-btn');
    const indicator = document.getElementById('rec-indicator');
    const recTime = document.getElementById('rec-time');
    const caption = document.getElementById('video-caption');
    const submitBtn = document.getElementById('submit-video-btn');
    const overlay = document.getElementById('biomarker-overlay');
    const transcript = document.getElementById('live-transcript');
    const audioTranscript = document.getElementById('audio-live-transcript');

    if (btn) btn.classList.remove('recording');
    if (audioBtn) audioBtn.classList.remove('recording');
    if (indicator) indicator.classList.remove('show');
    if (recTime) recTime.textContent = '';
    if (caption) caption.textContent = 'Tap to start recording';
    if (submitBtn) submitBtn.classList.add('hidden');
    if (overlay) overlay.classList.add('hidden');
    if (transcript) transcript.classList.add('hidden');
    if (audioTranscript) audioTranscript.classList.add('hidden');
  },

  toggleRecording() {
    if (this.isRecording) this.stopRecording();
    else this.startRecording();
  },

  async startRecording() {
    // Ensure camera/mic is ready
    if (!CheckInEngine.cameraReady) {
      const ok = await CheckInEngine.initCamera();
      if (!ok) return;
    }

    this.isRecording = true;
    this.recordSeconds = 0;

    const isVideo = CheckInEngine.checkinMode === 'video';
    const btn = document.getElementById(isVideo ? 'record-btn' : 'audio-record-btn');
    const indicator = document.getElementById('rec-indicator');
    const recTime = document.getElementById('rec-time');
    const caption = document.getElementById('video-caption');
    const audioStatus = document.getElementById('audio-status');

    if (btn) btn.classList.add('recording');
    if (indicator && isVideo) indicator.classList.add('show');
    if (caption) caption.textContent = 'Recording... tap to stop';
    if (audioStatus && !isVideo) audioStatus.textContent = 'Recording...';

    // Start all real-time processing
    CheckInEngine.startProcessing();

    this.recordTimer = setInterval(() => {
      this.recordSeconds++;
      const mins = Math.floor(this.recordSeconds / 60).toString().padStart(2, '0');
      const secs = (this.recordSeconds % 60).toString().padStart(2, '0');
      if (recTime) recTime.textContent = `${mins}:${secs}`;

      // Auto-stop after 60 seconds
      if (this.recordSeconds >= 60) this.stopRecording();
    }, 1000);
  },

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    if (this.recordTimer) clearInterval(this.recordTimer);

    // Stop real-time processing but keep data
    CheckInEngine.stopProcessing();

    const isVideo = CheckInEngine.checkinMode === 'video';
    const btn = document.getElementById(isVideo ? 'record-btn' : 'audio-record-btn');
    const indicator = document.getElementById('rec-indicator');
    const caption = document.getElementById('video-caption');
    const audioStatus = document.getElementById('audio-status');
    const submitBtn = document.getElementById('submit-video-btn');

    if (btn) btn.classList.remove('recording');
    if (indicator) indicator.classList.remove('show');
    if (caption) caption.textContent = 'Recording complete ✓';
    if (audioStatus && !isVideo) audioStatus.textContent = 'Recording complete ✓';

    // Show the analyze button
    setTimeout(() => {
      if (submitBtn) { submitBtn.classList.remove('hidden'); submitBtn.classList.add('fade-in'); }
    }, 400);
  },

  // ====== AI ANALYSIS (REAL BACKEND CALL) ======
  async analyzeVideo() {
    this.showScreen('screen-analysis');
    await this.runRealAnalysis();
  },

  async runRealAnalysis() {
    const steps = ['step-facial', 'step-voice', 'step-sentiment', 'step-journal'];
    const progressBar = document.getElementById('analysis-progress');

    steps.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove('active', 'done');
    });
    if (progressBar) progressBar.style.width = '0%';

    // Animate first two steps while waiting for backend
    const animateStep = (index) => {
      return new Promise(resolve => {
        const el = document.getElementById(steps[index]);
        if (el) el.classList.add('active');
        if (progressBar) progressBar.style.width = `${((index + 1) / steps.length) * 100}%`;
        setTimeout(() => {
          if (el) { el.classList.remove('active'); el.classList.add('done'); }
          resolve();
        }, 800);
      });
    };

    // Start backend call in parallel with animation
    const backendPromise = CheckInEngine.analyzeWithBackend().catch(err => {
      console.error('Backend analysis failed:', err);
      return null; // Will fall through to show results screen with defaults
    });

    await animateStep(0); // Facial
    await animateStep(1); // Voice

    // Wait for backend result
    const result = await backendPromise;

    await animateStep(2); // Sentiment
    await animateStep(3); // Journal

    // Populate results
    if (result) {
      CheckInEngine.displayResults(result);
      // Map AI-detected patterns to recommendation categories
      if (result.patterns) {
        const patternMap = { burnout: 'burnout', overwork: 'burnout', stress: 'anxiety', anxiety: 'anxiety', fatigue: 'fatigue', tired: 'fatigue', 'low energy': 'fatigue' };
        const detected = result.patterns.map(p => patternMap[p.label?.toLowerCase()] || null).filter(Boolean);
        if (detected.length) app.lastDetectedCategories = [...new Set(detected)];
      }
    }

    // Cleanup camera/mic
    CheckInEngine.cleanup();

    // Show results after brief delay
    setTimeout(() => {
      this.showScreen('screen-results');
      this.autoWaterPlant(); // check-in complete = plant watered
    }, 600);
  },

  // ====== SCENARIO 3: SAFETY NET & RECOVERY (DUMMY DATA — UNCHANGED) ======
  startScenario3() {
    this.showScreen('screen-text-input');
    this.updateInfoCard(3);
    const ta = document.getElementById('journal-textarea');
    const cc = document.querySelector('.char-count');
    if (ta && cc) cc.textContent = ta.value.length + ' characters';
  },

  submitText() {
    const textarea = document.getElementById('journal-textarea');
    const text = textarea.value;

    const crisisKeywords = ['hopeless', 'no point', 'want it all to stop', 'nobody would notice', 'end it', 'give up', 'worthless'];
    const isCrisis = crisisKeywords.some(kw => text.toLowerCase().includes(kw));

    if (isCrisis) {
      this.showScreen('screen-emergency');
    } else {
      this.showRecovery();
    }
  },

  showRecovery() {
    this.showScreen('screen-recovery');
  },

  // ====== SCENARIO 4: WEEKLY REPORT (DUMMY DATA — UNCHANGED) ======
  startScenario4() {
    this.showScreen('screen-weekly');
    this.updateInfoCard(4);
    setTimeout(() => { this.drawCharts(); }, 500);
  },

  drawCharts() {
    this.drawBurnoutScore();
    this.drawMoodChart();
    this.drawHRVChart();
    this.drawEmotionChart();
  },

  drawBurnoutScore() {
    // Demo values derived from the hardcoded chart data
    // HRV avg 48ms vs baseline ~65ms → 26% decline → high contribution
    const hrvScore   = 0.80; // 30% weight
    const moodScore  = 0.75; // 30% weight — mood dropped 12%
    const eventScore = 0.60; // 25% weight — upcoming exam/deadline context
    // Cycle phase factor — luteal phase = higher vulnerability
    const phase = window.CycleEngine ? CycleEngine.getCurrentPhase() : null;
    const cycleScore = phase && phase.name === 'Luteal' ? 0.75 : 0.45;

    const total = Math.round(
      hrvScore   * 30 +
      moodScore  * 30 +
      eventScore * 25 +
      cycleScore * 15
    );

    // Update score display
    const scoreEl = document.getElementById('burnout-score');
    if (scoreEl) scoreEl.textContent = total;

    // Update threshold message
    const threshEl = document.getElementById('burnout-threshold');
    if (threshEl) {
      if (total >= 75)      threshEl.textContent = '🚨 Critical — burnout event likely within 2–3 days';
      else if (total >= 60) threshEl.textContent = '⚠️ High risk — est. 4–5 days to critical threshold';
      else if (total >= 40) threshEl.textContent = '🟡 Moderate — monitor trends this week';
      else                  threshEl.textContent = '✅ Low risk — keep up your current habits';
    }

    // Update factor bars
    const bars = {
      'bf-hrv':    hrvScore,
      'bf-mood':   moodScore,
      'bf-events': eventScore,
      'bf-cycle':  cycleScore,
    };
    Object.entries(bars).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el) el.style.width = Math.round(val * 100) + '%';
    });

    // Draw arc gauge
    const canvas = document.getElementById('burnout-gauge');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = 60, cy = 65, r = 50;
    const startAngle = Math.PI;
    const endAngle   = startAngle + Math.PI * (total / 100);

    ctx.clearRect(0, 0, 120, 70);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Filled arc
    const grad = ctx.createLinearGradient(10, 0, 110, 0);
    grad.addColorStop(0, '#34C759');
    grad.addColorStop(0.5, '#FF9500');
    grad.addColorStop(1, '#FF3B30');
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Score label inside gauge
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('/100', cx, cy - 8);
  },

  drawMoodChart() {
    const canvas = document.getElementById('chart-mood');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const moodData = [7.2, 5.1, 6.8, 3.5, 4.0, null, null];
    const maxMood = 10;
    const padL = 30, padR = 10, padT = 15, padB = 25;
    const chartW = W - padL - padR;
    const chartH = H - padT - padB;

    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padT + (chartH / 5) * i;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    }

    ctx.fillStyle = '#AEAEB2'; ctx.font = '500 10px Inter, system-ui'; ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const val = maxMood - (maxMood / 5) * i;
      ctx.fillText(val.toFixed(0), padL - 6, padT + (chartH / 5) * i + 3);
    }

    ctx.textAlign = 'center';
    days.forEach((day, i) => {
      const x = padL + (chartW / (days.length - 1)) * i;
      ctx.fillStyle = i === 4 ? '#7C5CFC' : '#AEAEB2';
      ctx.font = i === 4 ? '700 10px Inter, system-ui' : '500 10px Inter, system-ui';
      ctx.fillText(day, x, H - 5);
    });

    const points = [];
    moodData.forEach((val, i) => {
      if (val !== null) {
        points.push({ x: padL + (chartW / (days.length - 1)) * i, y: padT + chartH - (val / maxMood) * chartH });
      }
    });
    if (points.length < 2) return;

    const grad = ctx.createLinearGradient(0, padT, 0, H - padB);
    grad.addColorStop(0, 'rgba(124,92,252,0.25)'); grad.addColorStop(1, 'rgba(124,92,252,0.02)');
    ctx.beginPath(); ctx.moveTo(points[0].x, H - padB);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, H - padB); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const cp1x = points[i-1].x + (points[i].x - points[i-1].x) / 3;
      const cp2x = points[i].x - (points[i].x - points[i-1].x) / 3;
      ctx.bezierCurveTo(cp1x, points[i-1].y, cp2x, points[i].y, points[i].x, points[i].y);
    }
    ctx.strokeStyle = '#7C5CFC'; ctx.lineWidth = 2.5; ctx.stroke();

    points.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = i === points.length - 1 ? '#FF6B8A' : '#7C5CFC'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    });
  },

  drawHRVChart() {
    const canvas = document.getElementById('chart-hrv');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
    const hrvData = [55, 48, 52, 38, 42];
    const baseline = 58, maxVal = 80;
    const padL = 30, padR = 10, padT = 15, padB = 25;
    const chartW = W - padL - padR, chartH = H - padT - padB;

    ctx.strokeStyle = 'rgba(0,0,0,0.06)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (chartH / 4) * i;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    }

    const baseY = padT + chartH - (baseline / maxVal) * chartH;
    ctx.strokeStyle = 'rgba(52,199,89,0.4)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(W - padR, baseY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#34C759'; ctx.font = '600 9px Inter, system-ui'; ctx.textAlign = 'left';
    ctx.fillText('Baseline 58ms', padL + 4, baseY - 5);

    ctx.fillStyle = '#AEAEB2'; ctx.font = '500 10px Inter, system-ui'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      ctx.fillText((maxVal - (maxVal / 4) * i).toFixed(0), padL - 6, padT + (chartH / 4) * i + 3);
    }

    ctx.textAlign = 'center';
    const barW = 24;
    days.forEach((day, i) => {
      const x = padL + (chartW / (days.length - 1)) * i;
      ctx.fillStyle = '#AEAEB2'; ctx.font = '500 10px Inter, system-ui'; ctx.fillText(day, x, H - 5);

      const val = hrvData[i];
      const barH = (val / maxVal) * chartH;
      const y = padT + chartH - barH;
      const isLow = val < baseline;
      const grad2 = ctx.createLinearGradient(0, y, 0, padT + chartH);
      if (isLow) { grad2.addColorStop(0, '#FF6B8A'); grad2.addColorStop(1, '#FF3B30'); }
      else { grad2.addColorStop(0, '#34D399'); grad2.addColorStop(1, '#10B981'); }

      const r = 4;
      ctx.beginPath();
      ctx.moveTo(x - barW/2 + r, y); ctx.lineTo(x + barW/2 - r, y);
      ctx.quadraticCurveTo(x + barW/2, y, x + barW/2, y + r);
      ctx.lineTo(x + barW/2, padT + chartH); ctx.lineTo(x - barW/2, padT + chartH);
      ctx.lineTo(x - barW/2, y + r); ctx.quadraticCurveTo(x - barW/2, y, x - barW/2 + r, y);
      ctx.closePath(); ctx.fillStyle = grad2; ctx.fill();

      ctx.fillStyle = isLow ? '#FF3B30' : '#10B981'; ctx.font = '700 10px Inter, system-ui';
      ctx.textAlign = 'center'; ctx.fillText(val, x, y - 5);
    });
  },

  drawEmotionChart() {
    const canvas = document.getElementById('chart-emotions');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const data = [
      { label: 'Anxious', value: 32, color: '#FF6B8A', emoji: '😰' },
      { label: 'Tired', value: 28, color: '#FF9500', emoji: '😴' },
      { label: 'Sad', value: 18, color: '#007AFF', emoji: '😢' },
      { label: 'Calm', value: 12, color: '#34C759', emoji: '😌' },
      { label: 'Happy', value: 10, color: '#AF52DE', emoji: '😊' },
    ];

    const total = data.reduce((s, d) => s + d.value, 0);
    const cx = 90, cy = H / 2, r = 65;
    let startAngle = -Math.PI / 2;

    data.forEach(d => {
      const sliceAngle = (d.value / total) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
      ctx.arc(cx, cy, r - 22, startAngle + sliceAngle, startAngle, true);
      ctx.closePath(); ctx.fillStyle = d.color; ctx.fill();
      startAngle += sliceAngle;
    });

    ctx.fillStyle = '#1C1C1E'; ctx.font = '800 20px Inter, system-ui'; ctx.textAlign = 'center';
    ctx.fillText('12', cx, cy + 2);
    ctx.fillStyle = '#8E8E93'; ctx.font = '600 9px Inter, system-ui'; ctx.fillText('entries', cx, cy + 14);

    const legendX = 190;
    data.forEach((d, i) => {
      const y = 20 + i * 30;
      ctx.fillStyle = d.color; ctx.beginPath(); ctx.arc(legendX, y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#1C1C1E'; ctx.font = '600 12px Inter, system-ui'; ctx.textAlign = 'left';
      ctx.fillText(`${d.emoji} ${d.label}`, legendX + 12, y + 4);
      ctx.fillStyle = '#8E8E93'; ctx.font = '700 12px Inter, system-ui'; ctx.textAlign = 'right';
      ctx.fillText(`${d.value}%`, W - 10, y + 4);
    });
  },

  // ====== CYCLE FEATURE ======
  cycleOnboardingDone: false,

  initCycle() {
    // Restore persisted cycle data
    try {
      const saved = localStorage.getItem('saathi_cycle');
      if (saved) {
        const data = JSON.parse(saved);
        this.cycleOnboardingDone = true;
        CycleEngine.dismissed = data.dismissed || false;
        if (data.lastPeriodDate) {
          CycleEngine.lastPeriodDate = new Date(data.lastPeriodDate);
          CycleEngine.cycleLength = data.cycleLength || 28;
        }
        this.renderCycleCard();
        return;
      }
    } catch(e) {}
    // First launch — show onboarding
    this.showScreen('screen-cycle-onboarding');
  },

  saveCycleToStorage() {
    try {
      localStorage.setItem('saathi_cycle', JSON.stringify({
        dismissed: CycleEngine.dismissed,
        lastPeriodDate: CycleEngine.lastPeriodDate ? CycleEngine.lastPeriodDate.toISOString() : null,
        cycleLength: CycleEngine.cycleLength,
      }));
    } catch(e) {}
  },

  acceptCycleOnboarding() {
    this.cycleOnboardingDone = true;
    this.showScreen('screen-home');
    this.showCycleSetup();
  },

  skipCycleOnboarding() {
    this.cycleOnboardingDone = true;
    CycleEngine.dismissed = true;
    this.saveCycleToStorage();
    this.showScreen('screen-home');
  },

  renderCycleCard() {
    const container = document.getElementById('cycle-card-container');
    if (!container) return;
    const phase = CycleEngine.getCurrentPhase();

    if (phase) {
      container.innerHTML = `
        <div class="cycle-card" style="--cycle-color:${phase.color};--cycle-bg:${phase.bg};--cycle-border:${phase.border};background:${phase.bg};border-color:${phase.border}">
          <div class="cycle-card-header">
            <div class="cycle-phase-label">
              <span class="cycle-phase-emoji">${phase.emoji}</span>
              <span class="cycle-phase-name" style="color:${phase.color}">${phase.name}</span>
            </div>
            <span class="cycle-day-badge" style="background:${phase.color}">Day ${phase.day}</span>
          </div>
          <div class="cycle-headline">${phase.headline}</div>
          <div class="cycle-body">${phase.body}</div>
          <div class="cycle-work-tip"><span>💼</span><span>${phase.workTip}</span></div>
          <div class="cycle-tags">
            ${phase.tags.map(t => `<span class="cycle-tag" style="color:${phase.color};border-color:${phase.border}">${t}</span>`).join('')}
          </div>
        </div>`;
    } else {
      container.innerHTML = '';
    }
  },

  showCycleSetup() {
    const modal = document.getElementById('cycle-setup-modal');
    const overlay = document.getElementById('cycle-modal-overlay');
    if (modal) modal.classList.add('show');
    if (overlay) overlay.classList.add('show');
    const dateInput = document.getElementById('cycle-date-input');
    if (dateInput && !dateInput.value) {
      dateInput.value = new Date().toISOString().split('T')[0];
    }
  },

  hideCycleSetup() {
    const modal = document.getElementById('cycle-setup-modal');
    const overlay = document.getElementById('cycle-modal-overlay');
    if (modal) modal.classList.remove('show');
    if (overlay) overlay.classList.remove('show');
  },

  saveCycleData() {
    const dateInput = document.getElementById('cycle-date-input');
    const slider = document.getElementById('cycle-length-slider');
    if (!dateInput || !dateInput.value) return;
    CycleEngine.setData(dateInput.value, slider ? slider.value : 28);
    this.saveCycleToStorage();
    this.hideCycleSetup();
    this.renderCycleCard();
  },

  dismissCyclePrompt() {
    CycleEngine.dismissed = true;
    this.hideCycleSetup();
    this.renderCycleCard();
  },

  updateCheckinPromptForPhase() {
    const phase = CycleEngine.getCurrentPhase();
    const subtitle = document.getElementById('checkin-subtitle');
    if (subtitle && phase) subtitle.textContent = phase.checkinPrompt;
  },

  // ====== PLANT GROWTH SYSTEM ======
  PLANT_STAGES: [
    { cssStage: 1, name: 'Seedling',      label: 'Just planted 🌱',               stemH: '10px', next: 'Water 2 more days to sprout',      minStreak: 0  },
    { cssStage: 2, name: 'Sprouting',     label: 'First leaves appearing 🌿',     stemH: '24px', next: 'Water 4 more days to grow',         minStreak: 2  },
    { cssStage: 3, name: 'Growing',       label: 'Growing strong 💪',             stemH: '40px', next: 'Water 3 more days for buds',        minStreak: 4  },
    { cssStage: 4, name: 'Budding',       label: 'Buds forming 🌸',               stemH: '54px', next: 'Water 7 more days to bloom fully',  minStreak: 7  },
    { cssStage: 5, name: 'Blooming',      label: 'In full bloom 🌺 You made it!', stemH: '66px', next: 'You\'re thriving! Keep it up 🌳',   minStreak: 14 },
  ],

  getPlantData() {
    try {
      const raw = localStorage.getItem('saathi_plant');
      return raw ? JSON.parse(raw) : { streak: 4, lastWatered: null, wateredToday: false };
    } catch { return { streak: 4, lastWatered: null, wateredToday: false }; }
  },

  savePlantData(data) {
    try { localStorage.setItem('saathi_plant', JSON.stringify(data)); } catch {}
  },

  getPlantStage(streak) {
    let stage = this.PLANT_STAGES[0];
    for (const s of this.PLANT_STAGES) {
      if (streak >= s.minStreak) stage = s;
    }
    return stage;
  },

  initPlant() {
    const data = this.getPlantData();
    const today = new Date().toDateString();
    data.wateredToday = data.lastWatered === today;
    this.savePlantData(data);
    this.renderPlant(data);
  },

  renderPlant(data) {
    const stage   = this.getPlantStage(data.streak);
    const illus   = document.getElementById('plant-illustration');
    const stem    = document.getElementById('p-stem');
    const nameEl  = document.getElementById('plant-name');
    const stageEl = document.getElementById('plant-stage');
    const nextEl  = document.getElementById('plant-next');
    const streakEl= document.getElementById('plant-streak');
    const btnEl   = document.getElementById('plant-water-btn');

    // Update CSS stage class on illustration
    if (illus) {
      illus.className = `plant-illustration stage-${stage.cssStage}`;
    }
    // Grow stem height
    if (stem) stem.style.setProperty('--stem-h', stage.stemH);

    if (nameEl)   nameEl.textContent   = stage.name;
    if (stageEl)  stageEl.textContent  = stage.label;
    if (nextEl)   nextEl.textContent   = stage.next;
    if (streakEl) streakEl.textContent = `🔥 ${data.streak} day streak`;

    if (btnEl) {
      if (data.wateredToday) {
        btnEl.textContent = '✅ Watered today';
        btnEl.disabled = true;
        btnEl.classList.add('watered');
      } else {
        btnEl.textContent = '💧 Water Today';
        btnEl.disabled = false;
        btnEl.classList.remove('watered');
      }
    }
  },

  waterPlant() {
    const data = this.getPlantData();
    const today = new Date().toDateString();
    if (data.wateredToday) return;

    const oldStage = this.getPlantStage(data.streak);
    data.streak += 1;
    data.lastWatered = today;
    data.wateredToday = true;
    this.savePlantData(data);

    const newStage = this.getPlantStage(data.streak);
    const leveledUp = newStage.cssStage !== oldStage.cssStage;

    // Water ripple
    const ripple = document.getElementById('plant-ripple');
    if (ripple) {
      ripple.classList.add('splash');
      setTimeout(() => ripple.classList.remove('splash'), 700);
    }

    // Level up animation
    if (leveledUp) {
      const illus = document.getElementById('plant-illustration');
      if (illus) {
        illus.classList.add('leveling-up');
        setTimeout(() => illus.classList.remove('leveling-up'), 800);
      }
    }

    this.renderPlant(data);
  },

  // Called after check-in analysis completes — auto-waters the plant
  autoWaterPlant() {
    const data = this.getPlantData();
    const today = new Date().toDateString();
    if (data.wateredToday) return;
    data.streak += 1;
    data.lastWatered = today;
    data.wateredToday = true;
    this.savePlantData(data);
    this.renderPlant(data);
  },

  // ====== NEPAL STRESSOR CARDS ======
  activeStressors: new Set(),

  STRESSOR_CONTEXT: {
    abroad:     'User selected "Abroad ko Tension" — they are dealing with the stress of living or deciding to move abroad. Be aware of: visa anxiety, brain drain guilt, pressure to send remittances, loneliness of being far from family, and the cultural weight of leaving Nepal.',
    ghar:       'User selected "Ghar ko Pressure" — they are experiencing family pressure common in Nepali households. Be aware of: expectations to become a doctor/engineer/government officer, arranged marriage pressure alongside career, collective family decision-making overriding personal choices.',
    exam:       'User selected "Board Exam Pressure" — they are preparing for or recovering from high-stakes Nepali academic exams (SEE, +2 boards, IOE/IOM/Loksewa entrance). Be aware of: extreme cultural weight placed on these results, family honour tied to scores, fear of disappointing parents.',
    remittance: 'User selected "Remittance Burden" — they are financially supporting family back in Nepal while working abroad or in a demanding job. Be aware of: financial anxiety, guilt around personal spending, sacrificing mental health for family stability, isolation.',
  },

  toggleStressor(key) {
    const btn = document.getElementById('stressor-' + key);
    if (this.activeStressors.has(key)) {
      this.activeStressors.delete(key);
      if (btn) btn.classList.remove('active');
    } else {
      this.activeStressors.add(key);
      if (btn) btn.classList.add('active');
    }
  },

  getStressorContext() {
    if (!this.activeStressors.size) return '';
    const lines = [...this.activeStressors].map(k => this.STRESSOR_CONTEXT[k]).filter(Boolean);
    return '[Cultural Stressor Context]\n' + lines.join('\n');
  },

  resetStressors() {
    this.activeStressors.clear();
    document.querySelectorAll('.stressor-card').forEach(c => c.classList.remove('active'));
  },

  // ====== RECOMMENDATIONS ======
  currentAudience: 'student',
  lastDetectedCategories: ['burnout', 'fatigue'],

  goToRecommendations() {
    this.showScreen('screen-recommendations');
    this.renderMoodBanner();
    this.renderPhaseBanner();
    this.renderRecs();
  },

  switchAudience(type) {
    this.currentAudience = type;
    document.querySelectorAll('.recs-toggle-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('toggle-' + type);
    if (btn) btn.classList.add('active');
    this.renderRecs();
  },

  renderMoodBanner() {
    const el = document.getElementById('recs-mood-banner');
    if (!el) return;
    const cats = this.lastDetectedCategories;
    const labels = cats.map(c => CATEGORY_META[c] ? `${CATEGORY_META[c].emoji} ${CATEGORY_META[c].label}` : '').filter(Boolean);
    if (!labels.length) { el.innerHTML = ''; return; }
    el.innerHTML = `
      <div class="recs-mood-banner-inner">
        <span class="recs-mood-label">Based on your last check-in:</span>
        <div class="recs-mood-tags">${cats.map(c => {
          const m = CATEGORY_META[c];
          return m ? `<span class="recs-mood-tag" style="background:${m.color}20;color:${m.color}">${m.emoji} ${m.label}</span>` : '';
        }).join('')}</div>
      </div>`;
  },

  renderPhaseBanner() {
    const el = document.getElementById('recs-phase-banner');
    if (!el) return;
    const phase = CycleEngine.getCurrentPhase();
    if (!phase) { el.innerHTML = ''; return; }
    const phaseMessages = {
      menstrual:  'In your menstrual phase — we\'ve prioritised rest and recovery resources.',
      follicular: 'In your follicular phase — energy is rising. Great time for new learning.',
      ovulatory:  'At peak energy — we\'ve highlighted resources to make the most of it.',
      luteal:     'In your luteal phase — we\'ve surfaced gentler, lower-pressure resources.',
    };
    const key = Object.keys(CYCLE_PHASES).find(k => CYCLE_PHASES[k].name === phase.name) || 'luteal';
    el.innerHTML = `
      <div class="recs-phase-banner-inner" style="background:${phase.bg};border-color:${phase.border}">
        <span>${phase.emoji}</span>
        <span style="color:${phase.color}">${phaseMessages[key]}</span>
      </div>`;
  },

  renderRecs() {
    const container = document.getElementById('recs-container');
    if (!container) return;

    const audience = this.currentAudience;
    const detected = this.lastDetectedCategories;

    // Show detected categories first, then general
    const orderedCategories = [...new Set([...detected, 'general'])];

    let html = '';
    orderedCategories.forEach(cat => {
      const meta = CATEGORY_META[cat];
      if (!meta) return;
      const items = RECS.filter(r => r.category === cat && r.audience.includes(audience));
      if (!items.length) return;

      html += `<div class="recs-section">
        <div class="recs-section-title">
          <span>${meta.emoji}</span>
          <span>${meta.label}</span>
        </div>`;

      items.forEach(r => {
        const typeIcon = r.type === 'course' ? '🎓' : r.type === 'article' ? '📄' : '▶️';
        const typeLabel = r.type === 'course' ? 'Course' : r.type === 'article' ? 'Article' : 'Video';
        html += `
        <div class="rec-card">
          <div class="rec-card-top">
            <div class="rec-type-badge" style="background:${r.sourceColor}18;color:${r.sourceColor}">
              ${typeIcon} ${typeLabel}
            </div>
            ${r.free ? '<span class="rec-free-badge">Free</span>' : ''}
          </div>
          <div class="rec-title">${r.title}</div>
          <div class="rec-source" style="color:${r.sourceColor}">${r.source}</div>
          <div class="rec-desc">${r.desc}</div>
          <div class="rec-footer">
            <span class="rec-duration">⏱ ${r.duration}</span>
            <button class="rec-view-btn" style="background:${r.sourceColor}">View →</button>
          </div>
        </div>`;
      });

      html += `</div>`;
    });

    container.innerHTML = html;
  },

  // ====== INFO CARD PANEL ======
  updateInfoCard(scenario) {
    const container = document.getElementById('scenario-info');
    const infos = {
      0: {
        title: '🔒 Pitch Point',
        body: 'When the smartwatch detects an <strong>abnormal HRV spike</strong>, our app immediately sends a compassionate notification — even on the lock screen. The user doesn\'t need to open the app.',
        pitch: '"Your watch notices. We respond."'
      },
      1: {
        title: '🛡️ Pitch Point',
        body: 'The app doesn\'t wait for the user to be burnt out; it <strong>intervenes the moment the body shows signs of pressure</strong>, using Calendar + HRV data.',
        pitch: '"Proactive, not reactive."'
      },
      2: {
        title: '🎙️ Pitch Point',
        body: 'We capture the <strong>nuance of a user\'s voice and expression</strong> that text alone misses. Multi-modal AI analysis detects burnout signals from facial affect, voice tone, and sentiment.',
        pitch: '"See what text can\'t tell you."'
      },
      3: {
        title: '🚨 Pitch Point',
        body: 'When it matters most, the app becomes a <strong>safety net</strong>. It triggers emergency contacts, provides crisis resources, and creates <strong>personalized recovery plans</strong> pushed directly to iOS Reminders.',
        pitch: '"Always there when you need it most."'
      },
      4: {
        title: '📊 Pitch Point',
        body: 'Background <strong>behavioral analysis</strong> tracks patterns across sleep, exercise, screen time, and meetings. CBT framework detects <strong>cognitive distortions</strong> automatically. All insights are actionable.',
        pitch: '"Data-driven mental health, without the data burden."'
      }
    };

    const info = infos[scenario] || infos[1];
    container.innerHTML = `
      <div class="scenario-info-card show" id="info-card-${scenario}">
        <h4>${info.title}</h4>
        <p>${info.body}</p>
        <p class="pitch-point">${info.pitch}</p>
      </div>
    `;
  }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  app.initCycle();
  app.initPlant();

  const breathCircle = document.getElementById('breathing-circle');
  if (breathCircle) {
    breathCircle.addEventListener('click', () => {
      if (!app.breathingInterval) app.startBreathing();
    });
  }
});
