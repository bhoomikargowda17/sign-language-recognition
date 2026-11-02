/**
 * SignAI Vision - Frontend Controller
 * Enterprise-grade real-time recognition, landmark canvas renderer,
 * tokenized sentence builder, Web Speech API, and accessible accordion controls.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Camera
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('landmarkCanvas');
  const ctx = canvas.getContext('2d');
  const stagePlaceholder = document.getElementById('stagePlaceholder');
  const activeCameraControls = document.getElementById('activeCameraControls');
  const headerStartBtn = document.getElementById('headerStartBtn');
  const cameraToggleBtn = document.getElementById('cameraToggleBtn');
  const placeholderStartBtn = document.getElementById('placeholderStartBtn');
  const flipCameraBtn = document.getElementById('flipCameraBtn');
  const switchCameraBtn = document.getElementById('switchCameraBtn');
  const mobileHudPill = document.getElementById('mobileHudPill');
  const hudSign = document.getElementById('hudSign');
  const hudHoldFill = document.getElementById('hudHoldFill');
  const hudConf = document.getElementById('hudConf');
  const camDot = document.getElementById('camDot');
  const camStatusText = document.getElementById('camStatusText');
  const cameraErrorState = document.getElementById('cameraErrorState');
  const cameraRetryBtn = document.getElementById('cameraRetryBtn');
  const inactivityNotice = document.getElementById('inactivityNotice');
  const trackingBadge = document.getElementById('trackingBadge');
  const handednessTag = document.getElementById('handednessTag');
  const landmarkCountTag = document.getElementById('landmarkCountTag');
  const toggleMesh = document.getElementById('toggleMesh');
  const fpsCounter = document.getElementById('fpsCounter');
  const inferenceLatency = document.getElementById('inferenceLatency');

  // DOM Elements - Recognition
  const detectionDot = document.getElementById('detectionDot');
  const detectionStatus = document.getElementById('detectionStatus');
  const predictedSign = document.getElementById('predictedSign');
  const predictedSignHelper = document.getElementById('predictedSignHelper');
  const holdProgressBar = document.getElementById('holdProgressBar');
  const confidenceValue = document.getElementById('confidenceValue');
  const confidenceBar = document.getElementById('confidenceBar');
  const probabilityList = document.getElementById('probabilityList');
  const speakCurrentBtn = document.getElementById('speakCurrentBtn');

  // DOM Elements - Sentence Builder
  const sentenceOutput = document.getElementById('sentenceOutput');
  const tokenContainer = document.getElementById('tokenContainer');
  const speakSentenceBtn = document.getElementById('speakSentenceBtn');
  const copyTextBtn = document.getElementById('copyTextBtn');
  const spaceBtn = document.getElementById('spaceBtn');
  const backspaceBtn = document.getElementById('backspaceBtn');
  const clearTextBtn = document.getElementById('clearTextBtn');

  // DOM Elements - Speech Output
  const speechCard = document.getElementById('speechCard');
  const speechAccordionHeader = document.getElementById('speechAccordionHeader');
  const autoSpeakToggle = document.getElementById('autoSpeakToggle');
  const voiceSelect = document.getElementById('voiceSelect');
  const rateSlider = document.getElementById('rateSlider');
  const pitchSlider = document.getElementById('pitchSlider');
  const rateVal = document.getElementById('rateVal');
  const pitchVal = document.getElementById('pitchVal');
  const testVoiceBtn = document.getElementById('testVoiceBtn');

  // DOM Elements - Model & Accordion
  const modelCard = document.getElementById('modelCard');
  const modelAccordionHeader = document.getElementById('modelAccordionHeader');
  const classTagsList = document.getElementById('classTagsList');
  const openCollectorBtn = document.getElementById('openCollectorBtn');
  const retrainBtn = document.getElementById('retrainBtn');
  const retrainProgressWrap = document.getElementById('retrainProgressWrap');
  const retrainProgressFill = document.getElementById('retrainProgressFill');
  const retrainProgressText = document.getElementById('retrainProgressText');

  // DOM Elements - Modals
  const guideBtn = document.getElementById('guideBtn');
  const guideModal = document.getElementById('guideModal');
  const closeGuideBtn = document.getElementById('closeGuideBtn');
  const closeGuideBottomBtn = document.getElementById('closeGuideBottomBtn');
  const collectorModal = document.getElementById('collectorModal');
  const closeCollectorBtn = document.getElementById('closeCollectorBtn');
  const quickRetrainBtn = document.getElementById('quickRetrainBtn');
  const startRecordingBtn = document.getElementById('startRecordingBtn');
  const newSignLabel = document.getElementById('newSignLabel');
  const sampleFill = document.getElementById('sampleFill');
  const sampleCountBadge = document.getElementById('sampleCountBadge');
  const collectorStatusText = document.getElementById('collectorStatusText');

  // Retrain Confirm Modal
  const retrainConfirmModal = document.getElementById('retrainConfirmModal');
  const closeRetrainConfirmBtn = document.getElementById('closeRetrainConfirmBtn');
  const cancelRetrainBtn = document.getElementById('cancelRetrainBtn');
  const confirmRetrainBtn = document.getElementById('confirmRetrainBtn');

  // Toast Container
  const toastContainer = document.getElementById('toastContainer');

  // App State
  let stream = null;
  let isStreaming = false;
  let isFlipped = true;
  let currentFacingMode = "user"; // "user" (front) or "environment" (rear)
  let animationFrameId = null;
  let isPredicting = false;
  let frameCount = 0;
  let lastFpsUpdate = performance.now();
  let lastHandDetectedTime = performance.now();
  let clearedSentenceBackup = "";

  // MediaPipe standard 21 joints connections
  const defaultConnections = [
    [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        // Index
    [0, 9], [9, 10], [10, 11], [11, 12],   // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]              // Palm
  ];

  // Sentence Accumulator State
  let currentStableSign = null;
  let stableSignStartTime = 0;
  let lastAppendedSign = null;
  let handEnteredTime = 0;
  let isHandPresent = false;
  const SIGN_HOLD_DURATION_MS = 1200; // Hold sign 1.2s to commit to sentence

  // TTS State
  let synth = window.speechSynthesis;
  let voices = [];
  let lastSpokenSign = null;
  let lastSpokenTime = 0;

  // Offscreen capture canvas
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  // Mobile Audio & Speech Unlock on First User Touch
  function unlockMobileAudio() {
    if ('speechSynthesis' in window) {
      try {
        const silent = new SpeechSynthesisUtterance('');
        silent.volume = 0;
        window.speechSynthesis.speak(silent);
      } catch (e) {
        // Silently catch if not allowed yet
      }
    }
    document.removeEventListener('touchstart', unlockMobileAudio);
    document.removeEventListener('click', unlockMobileAudio);
  }
  document.addEventListener('touchstart', unlockMobileAudio, { once: true, passive: true });
  document.addEventListener('click', unlockMobileAudio, { once: true });

  // ==========================================
  // Camera Management
  // ==========================================
  async function startCamera() {
    try {
      cameraErrorState.style.display = 'none';
      stagePlaceholder.style.display = 'flex';
      stagePlaceholder.style.opacity = '1';

      // Check for camera support & secure context
      const hasMediaDevices = navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function';
      const hasLegacyGUM = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

      if (!hasMediaDevices && !hasLegacyGUM) {
        let msg = "Camera API is not supported in this browser.";
        if (window.isSecureContext === false) {
          msg = "Camera requires HTTPS. Browsers block camera on insecure HTTP IP addresses. Please open via localhost:5000 or your deployed HTTPS URL!";
        }
        throw new Error(msg);
      }

      // Ensure previous tracks stopped if switching
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }

      // 1. Try standard ideal constraints with facing mode
      let mediaStream = null;
      try {
        if (hasMediaDevices) {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 640 },
              height: { ideal: 480 },
              facingMode: currentFacingMode ? { ideal: currentFacingMode } : undefined
            },
            audio: false
          });
        }
      } catch (errConstraint) {
        console.warn("Primary getUserMedia constraint failed, attempting basic fallback:", errConstraint);
      }

      // 2. Try basic video: true fallback
      if (!mediaStream && hasMediaDevices) {
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
          });
        } catch (errBasic) {
          console.warn("Basic getUserMedia failed, attempting legacy fallback:", errBasic);
        }
      }

      // 3. Try legacy fallback if needed
      if (!mediaStream && hasLegacyGUM) {
        mediaStream = await new Promise((resolve, reject) => {
          const legacyFn = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
          legacyFn.call(navigator, { video: true }, resolve, reject);
        });
      }

      if (!mediaStream) {
        throw new Error("Unable to establish camera stream.");
      }

      stream = mediaStream;
      video.muted = true;
      video.defaultMuted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.setAttribute('webkit-playsinline', '');
      video.srcObject = stream;

      // Ensure video is ready before playback to prevent play() rejections
      await new Promise((resolve) => {
        if (video.readyState >= 2) {
          resolve();
        } else {
          video.onloadedmetadata = () => resolve();
          setTimeout(resolve, 800); // 800ms safety timeout
        }
      });

      try {
        await video.play();
      } catch (playErr) {
        console.warn("Initial video.play() warning:", playErr);
        video.muted = true;
        await video.play();
      }

      isStreaming = true;
      lastHandDetectedTime = performance.now();

      stagePlaceholder.style.opacity = '0';
      setTimeout(() => { stagePlaceholder.style.display = 'none'; }, 200);

      // Header status
      camDot.className = "status-dot dot-active";
      camStatusText.textContent = "Camera live";

      // Toggle UI button visibility
      if (headerStartBtn) headerStartBtn.style.display = 'none';
      activeCameraControls.style.display = 'flex';

      updateVideoFlip();
      resizeCanvas();
      startInferenceLoop();
      showToast("Camera started successfully", "info");
    } catch (err) {
      console.error("Camera access error:", err);
      stagePlaceholder.style.display = 'none';
      cameraErrorState.style.display = 'flex';

      let userMsg = "Camera access was blocked or unavailable.";
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        userMsg = "Camera permission was denied. Please click the camera icon in your browser address bar and choose 'Allow'.";
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        userMsg = "No camera was detected on this device.";
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        userMsg = "Camera is currently in use by another application (e.g. Zoom, Teams, or another browser window).";
      } else if (err.message && err.message.includes("HTTPS")) {
        userMsg = err.message;
      }

      const errorTitle = cameraErrorState.querySelector('.error-title');
      if (errorTitle) errorTitle.textContent = userMsg;
      showToast(userMsg, "error");
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    isStreaming = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stagePlaceholder.style.display = 'flex';
    stagePlaceholder.style.opacity = '1';
    if (headerStartBtn) headerStartBtn.style.display = 'inline-flex';
    activeCameraControls.style.display = 'none';
    cameraErrorState.style.display = 'none';
    inactivityNotice.style.display = 'none';
    trackingBadge.style.display = 'none';

    if (mobileHudPill) mobileHudPill.style.display = 'none';
    camDot.className = "status-dot dot-inactive";
    camStatusText.textContent = "Camera off";

    fpsCounter.textContent = "--";
    inferenceLatency.textContent = "--";

    updateDetectionStatus("Waiting for hand", "neutral");
    predictedSign.textContent = "--";
    predictedSign.classList.add('no-sign');
    predictedSignHelper.textContent = "Position hand clearly in frame";
    holdProgressBar.style.width = "0%";
    confidenceBar.style.width = "0%";
    confidenceValue.textContent = "0%";
    renderSkeletonProbabilities();
    highlightActiveChip(null);
  }

  cameraToggleBtn.addEventListener('click', stopCamera);
  placeholderStartBtn.addEventListener('click', startCamera);
  if (headerStartBtn) headerStartBtn.addEventListener('click', startCamera);
  if (cameraRetryBtn) cameraRetryBtn.addEventListener('click', startCamera);

  function updateVideoFlip() {
    if (isFlipped) {
      video.style.transform = "scaleX(-1)";
      canvas.style.transform = "scaleX(-1)";
    } else {
      video.style.transform = "scaleX(1)";
      canvas.style.transform = "scaleX(1)";
    }
  }

  flipCameraBtn.addEventListener('click', () => {
    isFlipped = !isFlipped;
    updateVideoFlip();
    showToast(isFlipped ? "Mirror view enabled" : "Mirror view disabled", "info");
  });

  if (switchCameraBtn) {
    switchCameraBtn.addEventListener('click', async () => {
      currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
      isFlipped = (currentFacingMode === 'user');
      updateVideoFlip();
      if (isStreaming) {
        await startCamera();
        showToast(currentFacingMode === 'user' ? 'Front camera active' : 'Rear camera active', 'info');
      }
    });
  }

  function resizeCanvas() {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      captureCanvas.width = 320;
      captureCanvas.height = 240;
    }
  }

  video.addEventListener('loadedmetadata', resizeCanvas);
  window.addEventListener('resize', resizeCanvas);

  // ==========================================
  // Inference & Loop
  // ==========================================
  function startInferenceLoop() {
    async function loop() {
      if (!isStreaming) return;

      const now = performance.now();
      frameCount++;
      if (now - lastFpsUpdate >= 1000) {
        fpsCounter.textContent = `${frameCount} fps`;
        frameCount = 0;
        lastFpsUpdate = now;
      }

      // Inactivity notice: if active > 5s with no hand
      if (now - lastHandDetectedTime > 5000) {
        inactivityNotice.style.display = 'block';
      } else {
        inactivityNotice.style.display = 'none';
      }

      if (!isPredicting && video.readyState === video.HAVE_ENOUGH_DATA) {
        dispatchFrame();
      }

      animationFrameId = requestAnimationFrame(loop);
    }
    loop();
  }

  async function dispatchFrame() {
    isPredicting = true;
    const startInference = performance.now();

    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const base64Image = captureCanvas.toDataURL('image/jpeg', 0.65);

      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Image,
          flip: false
        })
      });

      const elapsed = Math.round(performance.now() - startInference);
      inferenceLatency.textContent = `${elapsed} ms`;

      if (response.ok) {
        const data = await response.json();
        handleInferenceResult(data);
      }
    } catch (err) {
      console.warn("Prediction dispatch error:", err);
    } finally {
      isPredicting = false;
    }
  }

  function handleInferenceResult(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data.hand_detected || !data.landmarks) {
      isHandPresent = false;
      trackingBadge.style.display = "none";
      if (mobileHudPill) {
        mobileHudPill.style.display = "none";
        hudSign.textContent = "--";
        hudConf.textContent = "0%";
        if (hudHoldFill) hudHoldFill.style.width = "0%";
      }
      updateDetectionStatus("Waiting for hand", "neutral");
      predictedSign.textContent = "--";
      predictedSign.classList.add('no-sign');
      predictedSignHelper.textContent = "Position hand clearly in frame";
      holdProgressBar.style.width = "0%";
      confidenceBar.style.width = "0%";
      confidenceValue.textContent = "0%";
      renderSkeletonProbabilities();
      currentStableSign = null;
      highlightActiveChip(null);
      return;
    }

    // Hand detected
    if (!isHandPresent) {
      isHandPresent = true;
      handEnteredTime = performance.now();
    }
    lastHandDetectedTime = performance.now();
    trackingBadge.style.display = "flex";
    handednessTag.textContent = `Hand: ${data.handedness || 'Detected'}`;

    // Clean skeleton drawing
    if (toggleMesh.checked) {
      drawHandSkeleton(data.landmarks, data.connections || defaultConnections);
    }

    const sign = data.smoothed_sign || data.sign || "UNCERTAIN";
    const confidence = data.confidence || 0.0;
    const confPct = Math.round(confidence * 100);

    // Update Mobile HUD
    if (mobileHudPill) {
      mobileHudPill.style.display = "flex";
      hudSign.textContent = (sign === "UNCERTAIN") ? "--" : sign;
      hudConf.textContent = `${confPct}%`;
    }

    // Update typography
    if (sign === "UNCERTAIN") {
      updateDetectionStatus("Detecting gesture", "detecting");
      predictedSign.textContent = "--";
      predictedSign.classList.add('no-sign');
      predictedSignHelper.textContent = "Hold sign steady in front of camera";
      confidenceBar.classList.add('warn');
      highlightActiveChip(null);
    } else {
      updateDetectionStatus("Locked", "active");
      predictedSign.textContent = sign;
      predictedSign.classList.remove('no-sign');
      predictedSignHelper.textContent = `${confPct}% confidence`;
      if (confPct < 55) {
        confidenceBar.classList.add('warn');
      } else {
        confidenceBar.classList.remove('warn');
      }
      highlightActiveChip(sign);
    }

    confidenceBar.style.width = `${confPct}%`;
    confidenceValue.textContent = `${confPct}%`;

    // Render table rows
    renderProbabilities(data.probabilities, data.top_k);

    // Sentence accumulation & hold bar
    handleSentenceAccumulation(sign, confidence);

    // Auto-Speak
    handleAutoSpeak(sign, confidence);
  }

  function updateDetectionStatus(text, type) {
    detectionStatus.textContent = text;
    detectionDot.className = "status-dot";
    if (type === "active") detectionDot.classList.add("dot-active");
    else if (type === "detecting") detectionDot.classList.add("dot-detecting");
    else detectionDot.classList.add("dot-neutral");
  }

  function drawHandSkeleton(landmarks, connections) {
    const w = canvas.width;
    const h = canvas.height;

    // Clean lines, no glow
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#0F766E";
    ctx.lineCap = "round";

    connections.forEach(([i, j]) => {
      if (landmarks[i] && landmarks[j]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[i].x * w, landmarks[i].y * h);
        ctx.lineTo(landmarks[j].x * w, landmarks[j].y * h);
        ctx.stroke();
      }
    });

    // Clean circular joints
    landmarks.forEach((lm, idx) => {
      const px = lm.x * w;
      const py = lm.y * h;

      ctx.beginPath();
      // Fingertips: 4, 8, 12, 16, 20
      if ([4, 8, 12, 16, 20].includes(idx)) {
        ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#0F766E";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#FFFFFF";
        ctx.stroke();
      } else {
        ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#FFFFFF";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#0F766E";
        ctx.stroke();
      }
    });
  }

  function renderSkeletonProbabilities() {
    probabilityList.innerHTML = `
      <div class="prob-row prob-skeleton"><span class="skeleton-name"></span><div class="skeleton-bar"></div><span class="skeleton-val"></span></div>
      <div class="prob-row prob-skeleton"><span class="skeleton-name"></span><div class="skeleton-bar"></div><span class="skeleton-val"></span></div>
      <div class="prob-row prob-skeleton"><span class="skeleton-name"></span><div class="skeleton-bar"></div><span class="skeleton-val"></span></div>
      <div class="prob-row prob-skeleton"><span class="skeleton-name"></span><div class="skeleton-bar"></div><span class="skeleton-val"></span></div>
    `;
  }

  function renderProbabilities(probabilities, topK) {
    if (!probabilities || Object.keys(probabilities).length === 0) {
      renderSkeletonProbabilities();
      return;
    }

    let items = topK || [];
    if (!items.length) {
      items = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
        .map(([sign, prob]) => ({ sign, probability: prob }));
    } else {
      items = items.slice(0, 4);
    }

    probabilityList.innerHTML = items.map(item => {
      const pct = Math.round(item.probability * 100);
      return `
        <div class="prob-row">
          <span class="prob-name">${item.sign}</span>
          <div class="prob-bar-container">
            <div class="prob-bar" style="width: ${pct}%"></div>
          </div>
          <span class="prob-val">${pct}%</span>
        </div>
      `;
    }).join('');
  }

  // ==========================================
  // Sentence Accumulation & Tokenized Output
  // ==========================================
  function handleSentenceAccumulation(sign, confidence) {
    const now = performance.now();

    // Grace period when hand first enters view (500ms) to prevent false detection on motion
    if (now - handEnteredTime < 500) {
      currentStableSign = null;
      holdProgressBar.style.width = "0%";
      if (hudHoldFill) hudHoldFill.style.width = "0%";
      return;
    }

    if (!sign || sign === "UNCERTAIN" || confidence < 0.50) {
      currentStableSign = null;
      holdProgressBar.style.width = "0%";
      holdProgressBar.classList.remove('committed');
      if (hudHoldFill) hudHoldFill.style.width = "0%";
      return;
    }

    if (currentStableSign !== sign) {
      currentStableSign = sign;
      stableSignStartTime = now;
      holdProgressBar.style.width = "0%";
      holdProgressBar.classList.remove('committed');
      if (hudHoldFill) hudHoldFill.style.width = "0%";
    } else {
      const elapsed = now - stableSignStartTime;
      const pct = Math.min(100, Math.round((elapsed / SIGN_HOLD_DURATION_MS) * 100));
      holdProgressBar.style.width = `${pct}%`;
      if (hudHoldFill) hudHoldFill.style.width = `${pct}%`;

      if (elapsed >= SIGN_HOLD_DURATION_MS) {
        if (lastAppendedSign !== sign) {
          appendSignToSentence(sign);
          lastAppendedSign = sign;
          holdProgressBar.classList.add('committed');
          setTimeout(() => {
            holdProgressBar.style.width = "0%";
            holdProgressBar.classList.remove('committed');
            if (hudHoldFill) hudHoldFill.style.width = "0%";
          }, 300);
        }
      }
    }
  }

  function appendSignToSentence(sign) {
    let current = sentenceOutput.value.trim();
    if (current.length > 0) {
      sentenceOutput.value = current + " " + sign;
    } else {
      sentenceOutput.value = sign;
    }
    renderTokens();
  }

  function renderTokens() {
    const text = sentenceOutput.value.trim();
    if (!text) {
      tokenContainer.innerHTML = `<span class="token-placeholder" id="tokenPlaceholder">Recognized signs will accumulate here as you hold them steady.</span>`;
      return;
    }

    const tokens = text.split(/\s+/);
    tokenContainer.innerHTML = tokens.map(tok => {
      const isSingleLetter = tok.length === 1 && tok.match(/[A-Z0-9]/i);
      return `<span class="token-chip ${isSingleLetter ? 'single-letter' : ''}">${tok}</span>`;
    }).join('');
    // Auto-scroll token container to right/bottom
    tokenContainer.scrollTop = tokenContainer.scrollHeight;
  }

  // Sentence Toolbar actions
  spaceBtn.addEventListener('click', () => {
    sentenceOutput.value += " ";
    renderTokens();
  });

  backspaceBtn.addEventListener('click', () => {
    let words = sentenceOutput.value.trim().split(/\s+/);
    if (words.length > 0 && words[0] !== "") {
      words.pop();
      sentenceOutput.value = words.join(" ");
      lastAppendedSign = null;
      renderTokens();
    }
  });

  clearTextBtn.addEventListener('click', () => {
    const prev = sentenceOutput.value;
    if (prev.trim()) {
      clearedSentenceBackup = prev;
      sentenceOutput.value = "";
      lastAppendedSign = null;
      renderTokens();
      showToastWithUndo("Sentence cleared.", () => {
        sentenceOutput.value = clearedSentenceBackup;
        renderTokens();
      });
    }
  });

  copyTextBtn.addEventListener('click', () => {
    const val = sentenceOutput.value.trim();
    if (val) {
      navigator.clipboard.writeText(val);
      showToast("Copied to clipboard.");
    }
  });

  // Keyboard Shortcuts (Ctrl+Enter to speak, Backspace, Space)
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      speakSentenceBtn.click();
    }
  });

  tokenContainer.addEventListener('keydown', (e) => {
    if (e.key === 'Backspace') {
      e.preventDefault();
      backspaceBtn.click();
    } else if (e.key === ' ') {
      e.preventDefault();
      spaceBtn.click();
    }
  });

  // ==========================================
  // Text-To-Speech (TTS)
  // ==========================================
  function populateVoiceList() {
    if (!synth) return;
    voices = synth.getVoices().sort((a, b) => a.name.localeCompare(b.name));
    if (!voices.length) return;

    voiceSelect.innerHTML = voices.map((v, i) => {
      const isEnglish = v.lang.startsWith("en");
      return `<option value="${i}" ${isEnglish ? 'selected' : ''}>${v.name} (${v.lang})</option>`;
    }).join('');
  }

  if (synth) {
    populateVoiceList();
    if (synth.onvoiceschanged !== undefined) {
      synth.onvoiceschanged = populateVoiceList;
    }
  }

  rateSlider.addEventListener('input', () => {
    rateVal.textContent = rateSlider.value;
  });

  pitchSlider.addEventListener('input', () => {
    pitchVal.textContent = pitchSlider.value;
  });

  function speakText(text) {
    if (!synth || !text) return;
    synth.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    const selectedVoiceIdx = voiceSelect.value;
    if (voices[selectedVoiceIdx]) {
      utter.voice = voices[selectedVoiceIdx];
    }
    utter.rate = parseFloat(rateSlider.value) || 1.0;
    utter.pitch = parseFloat(pitchSlider.value) || 1.0;

    synth.speak(utter);
  }

  function handleAutoSpeak(sign, confidence) {
    if (!autoSpeakToggle.checked || !sign || sign === "UNCERTAIN" || confidence < 0.55) return;

    const now = performance.now();
    if (now - handEnteredTime < 500) return; // Ignore initial entry motion
    const cooldown = (sign === lastSpokenSign) ? 2500 : 1200;

    if (now - lastSpokenTime >= cooldown) {
      speakText(sign.toLowerCase());
      lastSpokenSign = sign;
      lastSpokenTime = now;
    }
  }

  testVoiceBtn.addEventListener('click', () => {
    speakText("Speech synthesis is operational.");
  });

  speakCurrentBtn.addEventListener('click', () => {
    const sign = predictedSign.textContent;
    if (sign && sign !== "--" && sign !== "UNCERTAIN") {
      speakText(sign.toLowerCase());
    } else {
      showToast("No active sign to speak.");
    }
  });

  speakSentenceBtn.addEventListener('click', () => {
    const text = sentenceOutput.value.trim();
    if (text) {
      speakText(text);
    } else {
      showToast("Sentence field is empty.");
    }
  });

  // ==========================================
  // Accordion Expand / Collapse
  // ==========================================
  speechAccordionHeader.addEventListener('click', () => {
    speechCard.classList.toggle('open');
    const isOpen = speechCard.classList.contains('open');
    speechAccordionHeader.setAttribute('aria-expanded', isOpen);
  });

  modelAccordionHeader.addEventListener('click', () => {
    modelCard.classList.toggle('open');
    const isOpen = modelCard.classList.contains('open');
    modelAccordionHeader.setAttribute('aria-expanded', isOpen);
  });

  // ==========================================
  // Model Management & Chips
  // ==========================================
  async function loadClasses() {
    try {
      const res = await fetch('/classes');
      if (res.ok) {
        const data = await res.json();
        renderClassTags(data.classes || []);
      }
    } catch (e) {
      console.error("Error loading classes:", e);
    }
  }

  function renderClassTags(classes) {
    classTagsList.innerHTML = classes.map(c => `
      <button type="button" class="class-tag" data-sign="${c}" title="View ${c} in sign guide">
        ${c}
      </button>
    `).join('');

    // Click chip -> open sign guide at that sign
    document.querySelectorAll('.class-tag').forEach(btn => {
      btn.addEventListener('click', () => {
        const sign = btn.getAttribute('data-sign');
        openGuideAtSign(sign);
      });
    });

    document.getElementById('modelStatusText').textContent = `${classes.length} signs loaded`;
  }

  function highlightActiveChip(sign) {
    document.querySelectorAll('.class-tag').forEach(el => {
      if (sign && el.getAttribute('data-sign') === sign) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });
  }

  loadClasses();

  // Retrain Confirm Modal & Pipeline
  retrainBtn.addEventListener('click', () => {
    retrainConfirmModal.style.display = 'flex';
  });

  closeRetrainConfirmBtn.addEventListener('click', () => {
    retrainConfirmModal.style.display = 'none';
  });

  cancelRetrainBtn.addEventListener('click', () => {
    retrainConfirmModal.style.display = 'none';
  });

  confirmRetrainBtn.addEventListener('click', async () => {
    retrainConfirmModal.style.display = 'none';
    await executeRetrain();
  });

  async function executeRetrain() {
    // Open model accordion to see progress
    if (!modelCard.classList.contains('open')) {
      modelCard.classList.add('open');
      modelAccordionHeader.setAttribute('aria-expanded', 'true');
    }

    retrainProgressWrap.style.display = 'flex';
    retrainProgressFill.style.width = '20%';
    retrainProgressText.textContent = "Training Random Forest classifier...";

    setTimeout(() => { retrainProgressFill.style.width = '60%'; }, 800);

    try {
      const res = await fetch('/retrain', { method: 'POST' });
      const data = await res.json();

      retrainProgressFill.style.width = '100%';

      setTimeout(() => {
        retrainProgressWrap.style.display = 'none';
        retrainProgressFill.style.width = '0%';
        if (data.success) {
          const acc = Math.round(data.accuracy * 100);
          showToast(`Model retrained. Accuracy: ${acc}%.`);
          renderClassTags(data.classes);
        } else {
          showToast(`Retraining failed: ${data.error}`, "error");
        }
      }, 500);
    } catch (e) {
      retrainProgressWrap.style.display = 'none';
      showToast(`Retraining failed: ${e.message}`, "error");
    }
  }

  quickRetrainBtn.addEventListener('click', async () => {
    collectorModal.classList.remove('active');
    await executeRetrain();
  });

  // ==========================================
  // In-Browser Sample Collector
  // ==========================================
  let isCollecting = false;

  startRecordingBtn.addEventListener('click', async () => {
    const label = newSignLabel.value.trim().toUpperCase();
    if (!label) {
      showToast("Enter a sign label first.", "error");
      return;
    }
    if (!isStreaming) {
      showToast("Start camera first to record samples.", "error");
      return;
    }

    if (isCollecting) return;
    isCollecting = true;
    startRecordingBtn.disabled = true;
    startRecordingBtn.textContent = "Recording...";

    const targetSamples = 30;
    let recorded = 0;
    sampleFill.style.width = "0%";
    sampleCountBadge.textContent = `0 / ${targetSamples} samples`;
    collectorStatusText.textContent = `Recording '${label}'. Hold pose steady.`;

    const interval = setInterval(async () => {
      if (recorded >= targetSamples || !isStreaming) {
        clearInterval(interval);
        isCollecting = false;
        startRecordingBtn.disabled = false;
        startRecordingBtn.textContent = "Record 30 samples";
        collectorStatusText.textContent = `Recorded ${recorded} samples for '${label}'. Click 'Retrain model now'.`;
        showToast(`Collected ${recorded} samples for ${label}.`);
        return;
      }

      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const base64Image = captureCanvas.toDataURL('image/jpeg', 0.7);

      try {
        const res = await fetch('/collect_sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, image: base64Image })
        });
        if (res.ok) {
          recorded++;
          const pct = Math.round((recorded / targetSamples) * 100);
          sampleFill.style.width = `${pct}%`;
          sampleCountBadge.textContent = `${recorded} / ${targetSamples} samples`;
        }
      } catch (e) {
        console.warn("Sample capture failed:", e);
      }
    }, 150);
  });

  // ==========================================
  // Modal Handlers
  // ==========================================
  function openGuideAtSign(sign) {
    guideModal.classList.add('active');
    if (sign) {
      setTimeout(() => {
        const item = document.getElementById(`guide-${sign}`);
        if (item) {
          item.scrollIntoView({ behavior: 'smooth', block: 'center' });
          item.style.backgroundColor = '#E6F4F1';
          setTimeout(() => { item.style.backgroundColor = ''; }, 1500);
        }
      }, 100);
    }
  }

  guideBtn.addEventListener('click', () => guideModal.classList.add('active'));
  closeGuideBtn.addEventListener('click', () => guideModal.classList.remove('active'));
  if (closeGuideBottomBtn) closeGuideBottomBtn.addEventListener('click', () => guideModal.classList.remove('active'));

  openCollectorBtn.addEventListener('click', () => collectorModal.classList.add('active'));
  closeCollectorBtn.addEventListener('click', () => collectorModal.classList.remove('active'));

  window.addEventListener('click', (e) => {
    if (e.target === guideModal) guideModal.classList.remove('active');
    if (e.target === collectorModal) collectorModal.classList.remove('active');
    if (e.target === retrainConfirmModal) retrainConfirmModal.style.display = 'none';
  });

  // ==========================================
  // Neutral Snackbar / Toast with 5s Undo
  // ==========================================
  function showToast(message, type = "normal") {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span class="toast-message">${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 150ms ease';
      setTimeout(() => toast.remove(), 150);
    }, 3200);
  }

  function showToastWithUndo(message, onUndo) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
      <span class="toast-message">${message}</span>
      <button type="button" class="toast-undo-btn">Undo</button>
    `;

    const undoBtn = toast.querySelector('.toast-undo-btn');
    let undone = false;

    undoBtn.addEventListener('click', () => {
      undone = true;
      onUndo();
      toast.remove();
    });

    toastContainer.appendChild(toast);

    setTimeout(() => {
      if (!undone && toast.parentNode) {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 150ms ease';
        setTimeout(() => toast.remove(), 150);
      }
    }, 5000);
  }
});
