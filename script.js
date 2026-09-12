/**
 * ARIYATHE THALAYATTIYATH — AI-Powered Understanding Detection System
 * Hackathon Web Application Logic (Computer Vision + Machine Learning)
 * 
 * Pipeline:
 * Webcam -> MediaPipe FaceMesh -> 9 Kinematic Features -> Trained ML Classifier -> Nod Behavior -> Understanding Score
 */

(function () {
  'use strict';

  // ==========================================================================
  // CONFIGURATION & CONSTANTS
  // ==========================================================================
  const CONFIG = {
    NOD_DOWN_THRESHOLD: 0.026,
    NOD_REBOUND_THRESHOLD: 0.010,
    NOD_MIN_DURATION_MS: 160,
    NOD_MAX_DURATION_MS: 900,
    NOD_DEBOUNCE_COOLDOWN_MS: 280,
    ROLLING_WINDOW_SEC: 10,
    EMA_ALPHA: 0.06,
    RECORDING_DURATION_MS: 6000,
    WAVEFORM_POINTS: 140,
    MEDIAPIPE_ASSETS: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/'
  };

  const FEATURE_NAMES = [
    'head_pitch',
    'head_yaw',
    'movement_velocity',
    'movement_acceleration',
    'nod_count',
    'nod_frequency',
    'nod_duration',
    'inter_nod_interval',
    'total_movement'
  ];

  const CLASSES = ['GENUINE', 'POLITE', 'DANGER', 'SURVIVAL'];
  const CLASS_MAP = { 'GENUINE': 0, 'POLITE': 1, 'DANGER': 2, 'SURVIVAL': 3 };

  // Realistic Seed Dataset: 40 balanced samples (10 per class)
  const SEED_DATASET = [
    // 🟢 Class 0: GENUINE (measured, thoughtful, slow nods, high duration, low velocity)
    { head_pitch: 0.508, head_yaw: 0.012, movement_velocity: 0.075, movement_acceleration: 0.142, nod_count: 1, nod_frequency: 10.0, nod_duration: 480.0, inter_nod_interval: 5200.0, total_movement: 0.115, label: 'GENUINE' },
    { head_pitch: 0.514, head_yaw: -0.008, movement_velocity: 0.082, movement_acceleration: 0.155, nod_count: 2, nod_frequency: 20.0, nod_duration: 460.0, inter_nod_interval: 4400.0, total_movement: 0.138, label: 'GENUINE' },
    { head_pitch: 0.505, head_yaw: 0.005, movement_velocity: 0.068, movement_acceleration: 0.130, nod_count: 1, nod_frequency: 10.0, nod_duration: 510.0, inter_nod_interval: 5600.0, total_movement: 0.098, label: 'GENUINE' },
    { head_pitch: 0.511, head_yaw: -0.015, movement_velocity: 0.089, movement_acceleration: 0.168, nod_count: 2, nod_frequency: 20.0, nod_duration: 440.0, inter_nod_interval: 4100.0, total_movement: 0.145, label: 'GENUINE' },
    { head_pitch: 0.502, head_yaw: 0.009, movement_velocity: 0.071, movement_acceleration: 0.138, nod_count: 1, nod_frequency: 10.0, nod_duration: 490.0, inter_nod_interval: 5400.0, total_movement: 0.108, label: 'GENUINE' },
    { head_pitch: 0.518, head_yaw: -0.003, movement_velocity: 0.084, movement_acceleration: 0.160, nod_count: 2, nod_frequency: 20.0, nod_duration: 450.0, inter_nod_interval: 4300.0, total_movement: 0.132, label: 'GENUINE' },
    { head_pitch: 0.507, head_yaw: 0.014, movement_velocity: 0.076, movement_acceleration: 0.145, nod_count: 1, nod_frequency: 10.0, nod_duration: 475.0, inter_nod_interval: 5100.0, total_movement: 0.119, label: 'GENUINE' },
    { head_pitch: 0.512, head_yaw: -0.011, movement_velocity: 0.091, movement_acceleration: 0.172, nod_count: 2, nod_frequency: 20.0, nod_duration: 435.0, inter_nod_interval: 3950.0, total_movement: 0.150, label: 'GENUINE' },
    { head_pitch: 0.504, head_yaw: 0.002, movement_velocity: 0.065, movement_acceleration: 0.125, nod_count: 1, nod_frequency: 10.0, nod_duration: 520.0, inter_nod_interval: 5800.0, total_movement: 0.092, label: 'GENUINE' },
    { head_pitch: 0.515, head_yaw: -0.007, movement_velocity: 0.086, movement_acceleration: 0.162, nod_count: 2, nod_frequency: 20.0, nod_duration: 455.0, inter_nod_interval: 4200.0, total_movement: 0.135, label: 'GENUINE' },

    // 🟡 Class 1: POLITE (social intervals, medium velocity, moderate interval)
    { head_pitch: 0.522, head_yaw: 0.022, movement_velocity: 0.135, movement_acceleration: 0.270, nod_count: 3, nod_frequency: 30.0, nod_duration: 380.0, inter_nod_interval: 2200.0, total_movement: 0.210, label: 'POLITE' },
    { head_pitch: 0.528, head_yaw: -0.018, movement_velocity: 0.148, movement_acceleration: 0.295, nod_count: 4, nod_frequency: 40.0, nod_duration: 360.0, inter_nod_interval: 1850.0, total_movement: 0.245, label: 'POLITE' },
    { head_pitch: 0.520, head_yaw: 0.015, movement_velocity: 0.128, movement_acceleration: 0.255, nod_count: 3, nod_frequency: 30.0, nod_duration: 395.0, inter_nod_interval: 2350.0, total_movement: 0.198, label: 'POLITE' },
    { head_pitch: 0.525, head_yaw: -0.024, movement_velocity: 0.152, movement_acceleration: 0.310, nod_count: 4, nod_frequency: 40.0, nod_duration: 350.0, inter_nod_interval: 1780.0, total_movement: 0.258, label: 'POLITE' },
    { head_pitch: 0.519, head_yaw: 0.019, movement_velocity: 0.132, movement_acceleration: 0.265, nod_count: 3, nod_frequency: 30.0, nod_duration: 385.0, inter_nod_interval: 2280.0, total_movement: 0.205, label: 'POLITE' },
    { head_pitch: 0.530, head_yaw: -0.014, movement_velocity: 0.155, movement_acceleration: 0.315, nod_count: 4, nod_frequency: 40.0, nod_duration: 345.0, inter_nod_interval: 1720.0, total_movement: 0.265, label: 'POLITE' },
    { head_pitch: 0.523, head_yaw: 0.025, movement_velocity: 0.140, movement_acceleration: 0.280, nod_count: 3, nod_frequency: 30.0, nod_duration: 375.0, inter_nod_interval: 2150.0, total_movement: 0.220, label: 'POLITE' },
    { head_pitch: 0.526, head_yaw: -0.020, movement_velocity: 0.145, movement_acceleration: 0.290, nod_count: 4, nod_frequency: 40.0, nod_duration: 355.0, inter_nod_interval: 1810.0, total_movement: 0.240, label: 'POLITE' },
    { head_pitch: 0.521, head_yaw: 0.016, movement_velocity: 0.130, movement_acceleration: 0.260, nod_count: 3, nod_frequency: 30.0, nod_duration: 390.0, inter_nod_interval: 2300.0, total_movement: 0.202, label: 'POLITE' },
    { head_pitch: 0.529, head_yaw: -0.022, movement_velocity: 0.150, movement_acceleration: 0.305, nod_count: 4, nod_frequency: 40.0, nod_duration: 348.0, inter_nod_interval: 1750.0, total_movement: 0.252, label: 'POLITE' },

    // 🔴 Class 2: DANGER (rapid repetition, high velocity, short duration, panic intervals)
    { head_pitch: 0.548, head_yaw: 0.038, movement_velocity: 0.315, movement_acceleration: 0.630, nod_count: 6, nod_frequency: 60.0, nod_duration: 260.0, inter_nod_interval: 1100.0, total_movement: 0.440, label: 'DANGER' },
    { head_pitch: 0.555, head_yaw: -0.032, movement_velocity: 0.345, movement_acceleration: 0.690, nod_count: 7, nod_frequency: 70.0, nod_duration: 235.0, inter_nod_interval: 920.0, total_movement: 0.485, label: 'DANGER' },
    { head_pitch: 0.542, head_yaw: 0.042, movement_velocity: 0.295, movement_acceleration: 0.590, nod_count: 5, nod_frequency: 50.0, nod_duration: 280.0, inter_nod_interval: 1250.0, total_movement: 0.405, label: 'DANGER' },
    { head_pitch: 0.558, head_yaw: -0.035, movement_velocity: 0.360, movement_acceleration: 0.720, nod_count: 7, nod_frequency: 70.0, nod_duration: 225.0, inter_nod_interval: 880.0, total_movement: 0.510, label: 'DANGER' },
    { head_pitch: 0.545, head_yaw: 0.036, movement_velocity: 0.308, movement_acceleration: 0.615, nod_count: 6, nod_frequency: 60.0, nod_duration: 265.0, inter_nod_interval: 1120.0, total_movement: 0.428, label: 'DANGER' },
    { head_pitch: 0.552, head_yaw: -0.040, movement_velocity: 0.338, movement_acceleration: 0.675, nod_count: 6, nod_frequency: 60.0, nod_duration: 245.0, inter_nod_interval: 980.0, total_movement: 0.470, label: 'DANGER' },
    { head_pitch: 0.544, head_yaw: 0.045, movement_velocity: 0.300, movement_acceleration: 0.600, nod_count: 5, nod_frequency: 50.0, nod_duration: 275.0, inter_nod_interval: 1210.0, total_movement: 0.415, label: 'DANGER' },
    { head_pitch: 0.556, head_yaw: -0.028, movement_velocity: 0.350, movement_acceleration: 0.700, nod_count: 7, nod_frequency: 70.0, nod_duration: 230.0, inter_nod_interval: 900.0, total_movement: 0.495, label: 'DANGER' },
    { head_pitch: 0.546, head_yaw: 0.034, movement_velocity: 0.312, movement_acceleration: 0.625, nod_count: 6, nod_frequency: 60.0, nod_duration: 258.0, inter_nod_interval: 1080.0, total_movement: 0.435, label: 'DANGER' },
    { head_pitch: 0.550, head_yaw: -0.038, movement_velocity: 0.330, movement_acceleration: 0.660, nod_count: 6, nod_frequency: 60.0, nod_duration: 250.0, inter_nod_interval: 1020.0, total_movement: 0.460, label: 'DANGER' },

    // 💀 Class 3: SURVIVAL (insane frequency, extreme velocity and jerk, tiny intervals)
    { head_pitch: 0.575, head_yaw: 0.065, movement_velocity: 0.590, movement_acceleration: 1.180, nod_count: 9, nod_frequency: 90.0, nod_duration: 180.0, inter_nod_interval: 580.0, total_movement: 0.810, label: 'SURVIVAL' },
    { head_pitch: 0.588, head_yaw: -0.072, movement_velocity: 0.660, movement_acceleration: 1.320, nod_count: 11, nod_frequency: 110.0, nod_duration: 155.0, inter_nod_interval: 460.0, total_movement: 0.920, label: 'SURVIVAL' },
    { head_pitch: 0.568, head_yaw: 0.058, movement_velocity: 0.540, movement_acceleration: 1.080, nod_count: 8, nod_frequency: 80.0, nod_duration: 195.0, inter_nod_interval: 650.0, total_movement: 0.740, label: 'SURVIVAL' },
    { head_pitch: 0.595, head_yaw: -0.080, movement_velocity: 0.710, movement_acceleration: 1.420, nod_count: 12, nod_frequency: 120.0, nod_duration: 145.0, inter_nod_interval: 410.0, total_movement: 0.980, label: 'SURVIVAL' },
    { head_pitch: 0.572, head_yaw: 0.062, movement_velocity: 0.575, movement_acceleration: 1.150, nod_count: 9, nod_frequency: 90.0, nod_duration: 185.0, inter_nod_interval: 600.0, total_movement: 0.790, label: 'SURVIVAL' },
    { head_pitch: 0.584, head_yaw: -0.075, movement_velocity: 0.640, movement_acceleration: 1.280, nod_count: 10, nod_frequency: 100.0, nod_duration: 165.0, inter_nod_interval: 510.0, total_movement: 0.880, label: 'SURVIVAL' },
    { head_pitch: 0.570, head_yaw: 0.068, movement_velocity: 0.560, movement_acceleration: 1.120, nod_count: 8, nod_frequency: 80.0, nod_duration: 190.0, inter_nod_interval: 640.0, total_movement: 0.765, label: 'SURVIVAL' },
    { head_pitch: 0.590, head_yaw: -0.085, movement_velocity: 0.680, movement_acceleration: 1.360, nod_count: 11, nod_frequency: 110.0, nod_duration: 150.0, inter_nod_interval: 440.0, total_movement: 0.945, label: 'SURVIVAL' },
    { head_pitch: 0.578, head_yaw: 0.060, movement_velocity: 0.605, movement_acceleration: 1.210, nod_count: 9, nod_frequency: 90.0, nod_duration: 175.0, inter_nod_interval: 560.0, total_movement: 0.830, label: 'SURVIVAL' },
    { head_pitch: 0.582, head_yaw: -0.078, movement_velocity: 0.625, movement_acceleration: 1.250, nod_count: 10, nod_frequency: 100.0, nod_duration: 168.0, inter_nod_interval: 525.0, total_movement: 0.865, label: 'SURVIVAL' }
  ];

  const QUIP_POOL = [
    { title: "Cranial Biometric Alert", msg: "Understanding not found. Nodding detected.", icon: "🧠" },
    { title: "Cognitive 404", msg: "Your head says yes. Your brain says 404 not found.", icon: "⚡" },
    { title: "Impostor Metric", msg: "Confidence: 98%. Knowledge: highly questionable.", icon: "📊" },
    { title: "Corporate Survival Mode", msg: "Professional nodder detected. Ready for senior management.", icon: "👔" },
    { title: "Ergonomic Warning", msg: "Please stop nodding before your neck disconnects from your spine.", icon: "⚠️" },
    { title: "Malayalam Neural Core", msg: "“Athe athe athe” frequency reaching supersonic limits.", icon: "🥥" },
    { title: "Academic Survival", msg: "Professor: 'Did you get the derivation?' You: *vigorous oscillation*", icon: "🎓" },
    { title: "Bandwidth Depleted", msg: "Cognitive buffer empty. Buffering with affirmative head bobbing.", icon: "🔄" }
  ];

  // ==========================================================================
  // APPLICATION STATE
  // ==========================================================================
  const state = {
    mode: 'standby', // 'standby' | 'webcam' | 'demo'
    isRunning: false,
    
    // Heuristic & Telemetry
    baselinePitch: null,
    currentPitch: 0,
    currentYaw: 0,
    pitchDelta: 0,
    nodState: 'IDLE',
    downstrokeStartTime: 0,
    lastNodTime: 0,
    totalNods: 0,
    nodTimestamps: [],
    
    // Kinematic Tracking Buffer (last 7 seconds of frames)
    lastFrameTime: 0,
    lastFramePitch: null,
    lastFrameYaw: null,
    lastFrameVelocity: null,
    kinematicFrameBuffer: [],
    nodCycleRecords: [],
    
    // ML Classifier Engine
    classifierMode: 'heuristic', // 'heuristic' | 'ml'
    activeMlModel: null,
    lastMlPrediction: null,
    
    // Dataset Recording
    dataset: [],
    isRecording: false,
    recordingLabel: null,
    recordingStartTime: 0,
    recordingIntervalId: null,
    recordingFrames: [],
    recordingNods: [],
    
    // Output Metrics
    nodFrequency: 0,
    understandingScore: 92,
    nodConfidence: 0,
    genuineProb: 88,
    fakeProb: 12,
    currentStatus: 'neutral',
    
    // UI & Settings
    soundEnabled: true,
    showWireframe: true,
    lastToastTime: 0,
    lastAlertStatus: null,
    fps: 0,
    frameCount: 0,
    lastFpsUpdate: performance.now(),
    
    // Demo simulation parameters
    demoScenario: 'genuine',
    demoTime: 0,
    demoNodInterval: 3500,
    lastDemoNodTime: 0,
    demoTourStartTime: 0,
    
    // Waveform Buffer
    waveformBuffer: new Array(CONFIG.WAVEFORM_POINTS).fill(0),
    nodEventMarkers: []
  };

  // ==========================================================================
  // DOM ELEMENT SELECTORS
  // ==========================================================================
  const DOM = {
    soundToggleBtn: document.getElementById('soundToggleBtn'),
    soundIconOn: document.getElementById('soundIconOn'),
    soundIconOff: document.getElementById('soundIconOff'),
    soundStatusText: document.getElementById('soundStatusText'),
    certificateBtn: document.getElementById('certificateBtn'),
    
    startWebcamBtn: document.getElementById('startWebcamBtn'),
    startDemoBtn: document.getElementById('startDemoBtn'),
    
    webcamVideo: document.getElementById('webcamVideo'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    waveformCanvas: document.getElementById('waveformCanvas'),
    viewportContainer: document.getElementById('viewportContainer'),
    standbyOverlay: document.getElementById('standbyOverlay'),
    cameraErrorOverlay: document.getElementById('cameraErrorOverlay'),
    cameraErrorMessage: document.getElementById('cameraErrorMessage'),
    errorFallbackDemoBtn: document.getElementById('errorFallbackDemoBtn'),
    standbyWebcamBtn: document.getElementById('standbyWebcamBtn'),
    standbyDemoBtn: document.getElementById('standbyDemoBtn'),
    feedStatusDot: document.getElementById('feedStatusDot'),
    feedTitleText: document.getElementById('feedTitleText'),
    fpsValue: document.getElementById('fpsValue'),
    pitchBar: document.getElementById('pitchBar'),
    currentPipelineTag: document.getElementById('currentPipelineTag'),
    toggleMeshBtn: document.getElementById('toggleMeshBtn'),
    meshToggleText: document.getElementById('meshToggleText'),
    stopSessionBtn: document.getElementById('stopSessionBtn'),
    
    // Dashboard & Inference Controls
    btnModeHeuristic: document.getElementById('btnModeHeuristic'),
    btnModeML: document.getElementById('btnModeML'),
    mlInferenceBadge: document.getElementById('mlInferenceBadge'),
    mlPredictedClass: document.getElementById('mlPredictedClass'),
    mlConfidenceVal: document.getElementById('mlConfidenceVal'),
    mlActiveModelChip: document.getElementById('mlActiveModelChip'),
    
    statusCard: document.getElementById('statusCard'),
    statusBadge: document.getElementById('statusBadge'),
    statusUrgency: document.getElementById('statusUrgency'),
    statusTitle: document.getElementById('statusTitle'),
    statusSubtitle: document.getElementById('statusSubtitle'),
    understandingScoreVal: document.getElementById('understandingScoreVal'),
    gaugeCircle: document.getElementById('gaugeCircle'),
    explanationText: document.getElementById('explanationText'),
    retentionVal: document.getElementById('retentionVal'),
    retentionBar: document.getElementById('retentionBar'),
    
    nodConfidenceVal: document.getElementById('nodConfidenceVal'),
    nodConfidenceBar: document.getElementById('nodConfidenceBar'),
    genuineProbVal: document.getElementById('genuineProbVal'),
    genuineProbBar: document.getElementById('genuineProbBar'),
    fakeProbVal: document.getElementById('fakeProbVal'),
    fakeProbBar: document.getElementById('fakeProbBar'),
    nodsDetectedVal: document.getElementById('nodsDetectedVal'),
    nodFreqVal: document.getElementById('nodFreqVal'),
    nodCadenceSubtext: document.getElementById('nodCadenceSubtext'),
    
    // Demo Suite
    toggleDemoRunBtn: document.getElementById('toggleDemoRunBtn'),
    demoRunBtnText: document.getElementById('demoRunBtnText'),
    resetAllBtn: document.getElementById('resetAllBtn'),
    presetCards: document.querySelectorAll('.preset-card'),
    triggerSingleNodBtn: document.getElementById('triggerSingleNodBtn'),
    triggerRapidBurstBtn: document.getElementById('triggerRapidBurstBtn'),
    triggerSurvivalPanicBtn: document.getElementById('triggerSurvivalPanicBtn'),
    triggerRandomQuoteBtn: document.getElementById('triggerRandomQuoteBtn'),
    
    // ML Training Suite Elements
    totalSamplesCount: document.getElementById('totalSamplesCount'),
    recordingStatusLabel: document.getElementById('recordingStatusLabel'),
    recordingProgressContainer: document.getElementById('recordingProgressContainer'),
    recordingTargetLabel: document.getElementById('recordingTargetLabel'),
    recordingCountdown: document.getElementById('recordingCountdown'),
    recordingProgressBar: document.getElementById('recordingProgressBar'),
    countGenuine: document.getElementById('countGenuine'),
    countPolite: document.getElementById('countPolite'),
    countDanger: document.getElementById('countDanger'),
    countSurvival: document.getElementById('countSurvival'),
    recordGenuineBtn: document.getElementById('recordGenuineBtn'),
    recordPoliteBtn: document.getElementById('recordPoliteBtn'),
    recordDangerBtn: document.getElementById('recordDangerBtn'),
    recordSurvivalBtn: document.getElementById('recordSurvivalBtn'),
    loadSeedDataBtn: document.getElementById('loadSeedDataBtn'),
    exportCsvBtn: document.getElementById('exportCsvBtn'),
    importCsvInput: document.getElementById('importCsvInput'),
    clearDatasetBtn: document.getElementById('clearDatasetBtn'),
    trainBrowserModelBtn: document.getElementById('trainBrowserModelBtn'),
    importJsonModelInput: document.getElementById('importJsonModelInput'),
    modelResultsCard: document.getElementById('modelResultsCard'),
    modelStatusTitle: document.getElementById('modelStatusTitle'),
    modelStatusSubtitle: document.getElementById('modelStatusSubtitle'),
    modelAccuracyVal: document.getElementById('modelAccuracyVal'),
    confusionTableBody: document.getElementById('confusionTableBody'),
    weightsGrid: document.getElementById('weightsGrid'),
    smallDatasetWarning: document.getElementById('smallDatasetWarning'),
    
    toastContainer: document.getElementById('toastContainer'),
    certificateModal: document.getElementById('certificateModal'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    dismissModalBtn: document.getElementById('dismissModalBtn'),
    printCertBtn: document.getElementById('printCertBtn'),
    
    certTotalNods: document.getElementById('certTotalNods'),
    certScore: document.getElementById('certScore'),
    certCadence: document.getElementById('certCadence'),
    certFakeQuotient: document.getElementById('certFakeQuotient'),
    certRankTitle: document.getElementById('certRankTitle'),
    certRankQuote: document.getElementById('certRankQuote')
  };

  const overlayCtx = DOM.overlayCanvas.getContext('2d');
  const waveformCtx = DOM.waveformCanvas.getContext('2d');

  let faceMeshInstance = null;
  let cameraStream = null;
  let cameraInstance = null;
  let animationFrameId = null;

  // ==========================================================================
  // PROCEDURAL WEB AUDIO SYNTHESIZER
  // ==========================================================================
  let audioCtx = null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) audioCtx = new AudioContextClass();
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  function playSound(type) {
    if (!state.soundEnabled) return;
    try {
      initAudioContext();
      if (!audioCtx) return;

      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (type === 'nod') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(620, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
        osc.start(now);
        osc.stop(now + 0.12);
      } else if (type === 'danger') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(480, now);
        osc.frequency.linearRampToValueAtTime(740, now + 0.1);
        osc.frequency.linearRampToValueAtTime(370, now + 0.25);
        gain.gain.setValueAtTime(0.16, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
        osc.start(now);
        osc.stop(now + 0.3);
      } else if (type === 'survival') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.setValueAtTime(1100, now + 0.06);
        osc.frequency.setValueAtTime(700, now + 0.12);
        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
        osc.start(now);
        osc.stop(now + 0.28);
      } else if (type === 'record_start') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(500, now);
        osc.frequency.setValueAtTime(800, now + 0.1);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.2);
      } else if (type === 'record_done') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.setValueAtTime(1200, now + 0.12);
        gain.gain.setValueAtTime(0.16, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.start(now);
        osc.stop(now + 0.25);
      } else if (type === 'click') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(400, now);
        gain.gain.setValueAtTime(0.05, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);
        osc.start(now);
        osc.stop(now + 0.04);
      }
    } catch (e) {
      console.warn("Audio playback issue:", e);
    }
  }

  // ==========================================================================
  // TOAST ALERT SYSTEM
  // ==========================================================================
  function showToast(title, msg, icon = "🚨", type = "toast-info", duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-icon">${icon}</div>
      <div class="toast-content">
        <div class="toast-title">${title}</div>
        <div class="toast-msg">${msg}</div>
      </div>
    `;

    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-out');
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    }, duration);
  }

  function triggerRandomRoast() {
    const quip = QUIP_POOL[Math.floor(Math.random() * QUIP_POOL.length)];
    showToast(quip.title, quip.msg, quip.icon, "toast-info", 4500);
    playSound('click');
  }

  // ==========================================================================
  // 9-DIMENSIONAL KINEMATIC FEATURE EXTRACTION
  // ==========================================================================
  /**
   * Extracts 9 numerical features from a window of frames and detected nod events.
   */
  function extractFeatures(frames, nods, windowDurationSec = 6) {
    if (!frames || frames.length === 0) {
      return {
        head_pitch: 0.51,
        head_yaw: 0.0,
        movement_velocity: 0.08,
        movement_acceleration: 0.15,
        nod_count: 0,
        nod_frequency: 0.0,
        nod_duration: 350.0,
        inter_nod_interval: 5000.0,
        total_movement: 0.0
      };
    }

    let sumPitch = 0, sumYaw = 0, sumVel = 0, sumAcc = 0;
    let totalDisp = 0;
    for (let i = 0; i < frames.length; i++) {
      sumPitch += (frames[i].pitch || 0.5);
      sumYaw += (frames[i].yaw || 0);
      sumVel += (frames[i].velocity || 0);
      sumAcc += (frames[i].acceleration || 0);
      if (i > 0) {
        const dp = frames[i].pitch - frames[i - 1].pitch;
        const dy = frames[i].yaw - frames[i - 1].yaw;
        totalDisp += Math.sqrt(dp * dp + dy * dy);
      }
    }

    const n = frames.length;
    const nodCount = (nods && nods.length) || 0;
    const nodFreq = (nodCount / Math.max(1, windowDurationSec)) * 60;

    let meanDuration = 350;
    if (nodCount > 0) {
      const sumDur = nods.reduce((acc, nod) => acc + (nod.duration || 350), 0);
      meanDuration = sumDur / nodCount;
    }

    let meanInterval = 5000;
    if (nodCount > 1) {
      let sumIntervals = 0;
      for (let i = 1; i < nods.length; i++) {
        sumIntervals += (nods[i].time - nods[i - 1].time);
      }
      meanInterval = sumIntervals / (nodCount - 1);
    }

    return {
      head_pitch: Math.round((sumPitch / n) * 1000) / 1000,
      head_yaw: Math.round((sumYaw / n) * 1000) / 1000,
      movement_velocity: Math.round((sumVel / n) * 1000) / 1000,
      movement_acceleration: Math.round((sumAcc / n) * 1000) / 1000,
      nod_count: nodCount,
      nod_frequency: Math.round(nodFreq * 10) / 10,
      nod_duration: Math.round(meanDuration * 10) / 10,
      inter_nod_interval: Math.round(meanInterval * 10) / 10,
      total_movement: Math.round(totalDisp * 1000) / 1000
    };
  }

  // ==========================================================================
  // NOD DETECTION & KINEMATICS ENGINE
  // ==========================================================================
  function processPitchSample(pitch, now, yaw = 0) {
    state.currentPitch = pitch;
    state.currentYaw = yaw;

    if (state.baselinePitch === null) {
      state.baselinePitch = pitch;
    } else {
      state.baselinePitch = state.baselinePitch * (1 - CONFIG.EMA_ALPHA) + pitch * CONFIG.EMA_ALPHA;
    }

    const delta = pitch - state.baselinePitch;
    state.pitchDelta = delta;

    // Head Pitch Meter in HUD
    if (DOM.pitchBar) {
      const clampDeflection = Math.max(-0.06, Math.min(0.06, delta));
      const percent = (clampDeflection / 0.06) * 50;
      if (percent >= 0) {
        DOM.pitchBar.style.top = '50%';
        DOM.pitchBar.style.height = `${percent}%`;
        DOM.pitchBar.style.background = percent > 35 ? 'var(--neon-rose)' : 'var(--neon-cyan)';
      } else {
        DOM.pitchBar.style.top = `${50 + percent}%`;
        DOM.pitchBar.style.height = `${Math.abs(percent)}%`;
        DOM.pitchBar.style.background = 'var(--neon-emerald)';
      }
    }

    // Kinematic displacement, velocity, and acceleration
    const dt = state.lastFrameTime ? Math.max(0.005, (now - state.lastFrameTime) / 1000) : 0.033;
    const dp = state.lastFramePitch !== null ? (pitch - state.lastFramePitch) : 0;
    const dy = state.lastFrameYaw !== null ? (yaw - state.lastFrameYaw) : 0;
    const velocity = Math.sqrt(dp * dp + dy * dy) / dt;
    const acceleration = state.lastFrameVelocity !== null ? Math.abs(velocity - state.lastFrameVelocity) / dt : 0;

    state.lastFrameTime = now;
    state.lastFramePitch = pitch;
    state.lastFrameYaw = yaw;
    state.lastFrameVelocity = velocity;

    const frameRecord = { time: now, pitch, yaw, delta, velocity, acceleration };
    state.kinematicFrameBuffer.push(frameRecord);
    state.kinematicFrameBuffer = state.kinematicFrameBuffer.filter(f => f.time >= now - 7500);

    if (state.isRecording) {
      state.recordingFrames.push(frameRecord);
    }

    // Waveform plot buffer
    state.waveformBuffer.shift();
    state.waveformBuffer.push(delta);

    // Nod State Machine
    if (state.nodState === 'IDLE') {
      if (delta > CONFIG.NOD_DOWN_THRESHOLD && (now - state.lastNodTime) > CONFIG.NOD_DEBOUNCE_COOLDOWN_MS) {
        state.nodState = 'DOWNSTROKE';
        state.downstrokeStartTime = now;
      }
    } else if (state.nodState === 'DOWNSTROKE') {
      const duration = now - state.downstrokeStartTime;
      if (delta < CONFIG.NOD_REBOUND_THRESHOLD) {
        if (duration >= CONFIG.NOD_MIN_DURATION_MS && duration <= CONFIG.NOD_MAX_DURATION_MS) {
          registerNod(now, duration);
          state.nodState = 'COOLDOWN';
        } else {
          state.nodState = 'IDLE';
        }
      } else if (duration > CONFIG.NOD_MAX_DURATION_MS) {
        state.nodState = 'IDLE';
      }
    } else if (state.nodState === 'COOLDOWN') {
      if (now - state.lastNodTime > CONFIG.NOD_DEBOUNCE_COOLDOWN_MS) {
        state.nodState = 'IDLE';
      }
    }

    updateRollingMetrics(now);
  }

  function registerNod(timestamp, duration = 380) {
    state.totalNods++;
    state.lastNodTime = timestamp;
    state.nodTimestamps.push(timestamp);

    const nodRecord = { time: timestamp, duration };
    state.nodCycleRecords.push(nodRecord);
    state.nodCycleRecords = state.nodCycleRecords.filter(r => r.time >= timestamp - 10000);

    if (state.isRecording) {
      state.recordingNods.push(nodRecord);
    }

    state.nodEventMarkers.push({
      index: CONFIG.WAVEFORM_POINTS - 1,
      delta: state.pitchDelta,
      time: timestamp
    });

    playSound('nod');

    if (DOM.nodsDetectedVal) {
      DOM.nodsDetectedVal.textContent = state.totalNods;
      DOM.nodsDetectedVal.classList.remove('bump');
      void DOM.nodsDetectedVal.offsetWidth;
      DOM.nodsDetectedVal.classList.add('bump');
    }

    updateRollingMetrics(timestamp, true);
  }

  // ==========================================================================
  // METRICS & DUAL-MODE INFERENCE (HEURISTIC VS TRAINED ML)
  // ==========================================================================
  function updateRollingMetrics(now, isNodEvent = false) {
    const cutoff = now - (CONFIG.ROLLING_WINDOW_SEC * 1000);
    state.nodTimestamps = state.nodTimestamps.filter(t => t >= cutoff);
    const recentNods = state.nodTimestamps.length;
    state.nodFrequency = Math.round((recentNods / CONFIG.ROLLING_WINDOW_SEC) * 60 * 10) / 10;

    let status = 'neutral';
    let statusTitle = "Awaiting Cranial Gesture";
    let statusSubtitle = "“The system is observing your head movement…”";
    let score = 90;
    let genuineProb = 85;
    let fakeProb = 15;
    let confidence = recentNods > 0 ? 82 : 40;
    let retention = 85;

    // Check if Trained ML Mode is active
    if (state.classifierMode === 'ml' && state.activeMlModel && state.kinematicFrameBuffer.length >= 15) {
      const currentFeatures = extractFeatures(state.kinematicFrameBuffer, state.nodCycleRecords, CONFIG.ROLLING_WINDOW_SEC);
      const prediction = predictWithMlModel(currentFeatures, state.activeMlModel);
      state.lastMlPrediction = prediction;

      if (DOM.mlInferenceBadge) DOM.mlInferenceBadge.classList.remove('hidden');
      if (DOM.mlPredictedClass) DOM.mlPredictedClass.textContent = prediction.predictedClass;
      if (DOM.mlConfidenceVal) DOM.mlConfidenceVal.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

      const pClass = prediction.predictedClass;
      confidence = Math.round(prediction.confidence * 100);

      if (pClass === 'GENUINE') {
        status = 'genuine';
        statusTitle = "🟢 Genuine Understanding (ML)";
        statusSubtitle = "“Supervised ML model predicts authentic comprehension.”";
        score = Math.min(96, Math.max(82, Math.round(92 * prediction.confidence)));
        genuineProb = Math.round(prediction.confidence * 100);
        fakeProb = 100 - genuineProb;
        retention = Math.round(88 * prediction.confidence);
      } else if (pClass === 'POLITE') {
        status = 'polite';
        statusTitle = "🟡 Polite Social Nodding (ML)";
        statusSubtitle = "“ML model detected social courtesy nod cadence.”";
        score = Math.min(74, Math.max(52, 65));
        genuineProb = 50;
        fakeProb = 50;
        retention = 58;
      } else if (pClass === 'DANGER') {
        status = 'danger';
        statusTitle = "🔴 Danger Nod Detected (ML)";
        statusSubtitle = "“ML model flags panic agreement: high confidence, zero retention.”";
        score = Math.max(18, 30);
        genuineProb = 18;
        fakeProb = 82;
        retention = 20;

        if (state.lastAlertStatus !== 'danger' && now - state.lastToastTime > 6000) {
          showToast("🚨 DANGER NOD DETECTED (ML)", "ML classifier detected panic agreement with " + confidence + "% confidence.", "🚨", "toast-danger", 5000);
          playSound('danger');
          state.lastToastTime = now;
        }
      } else if (pClass === 'SURVIVAL') {
        status = 'survival';
        statusTitle = "💀 Survival Mode Activated (ML)";
        statusSubtitle = "“ML model flags frantic survival bobblehead oscillations.”";
        score = 8;
        genuineProb = 4;
        fakeProb = 96;
        retention = 5;

        if (state.lastAlertStatus !== 'survival' && now - state.lastToastTime > 5000) {
          showToast("💀 SURVIVAL MODE ACTIVATED (ML)", "ML classifier detected survival nodding with " + confidence + "% confidence.", "💀", "toast-survival", 6000);
          playSound('survival');
          state.lastToastTime = now;
        }
      }
    } else {
      // Heuristic Rule-Based Pipeline
      if (DOM.mlInferenceBadge) DOM.mlInferenceBadge.classList.add('hidden');

      if (state.totalNods === 0 && recentNods === 0) {
        status = 'neutral';
        statusTitle = "Ready For Analysis";
        statusSubtitle = "“Nod naturally or start demo simulation to begin.”";
        score = 88;
        genuineProb = 75;
        fakeProb = 25;
        confidence = 50;
        retention = 80;
      } else if (recentNods <= 2) {
        status = 'genuine';
        statusTitle = "🟢 Genuine Understanding";
        statusSubtitle = "“You might actually understand this.”";
        score = Math.min(96, Math.max(82, 95 - recentNods * 4));
        genuineProb = Math.min(94, 86 + recentNods * 3);
        fakeProb = 100 - genuineProb;
        confidence = 88;
        retention = 88;
      } else if (recentNods <= 4) {
        status = 'polite';
        statusTitle = "🟡 Polite Social Nodding";
        statusSubtitle = "“You’re listening… probably.”";
        score = Math.min(74, Math.max(52, 75 - (recentNods - 2) * 9));
        genuineProb = 56;
        fakeProb = 44;
        confidence = 75;
        retention = 58;
      } else if (recentNods <= 7) {
        status = 'danger';
        statusTitle = "🔴 Danger Nod Detected";
        statusSubtitle = "“You are nodding with confidence despite understanding nothing.”";
        score = Math.max(18, 38 - (recentNods - 4) * 6);
        genuineProb = 18;
        fakeProb = 82;
        confidence = 94;
        retention = 22;

        if (state.lastAlertStatus !== 'danger' && now - state.lastToastTime > 6000) {
          showToast("🚨 DANGER NOD DETECTED", "You are nodding with confidence despite understanding nothing.", "🚨", "toast-danger", 5000);
          playSound('danger');
          state.lastToastTime = now;
        }
      } else {
        status = 'survival';
        statusTitle = "💀 Survival Mode Activated";
        statusSubtitle = "“You’re no longer understanding. You’re just agreeing.”";
        score = Math.max(4, 14 - (recentNods - 7) * 2);
        genuineProb = 4;
        fakeProb = 96;
        confidence = 99;
        retention = 6;

        if (state.lastAlertStatus !== 'survival' && now - state.lastToastTime > 5000) {
          showToast("💀 SURVIVAL MODE ACTIVATED", "At this point you’re not understanding. You’re just agreeing.", "💀", "toast-survival", 6000);
          playSound('survival');
          state.lastToastTime = now;
        }
      }
    }

    state.lastAlertStatus = status;
    state.currentStatus = status;
    state.understandingScore = score;
    state.genuineProb = genuineProb;
    state.fakeProb = fakeProb;
    state.nodConfidence = confidence;

    renderDashboardMetrics(status, statusTitle, statusSubtitle, score, genuineProb, fakeProb, confidence, retention, recentNods);
  }

  function renderDashboardMetrics(status, title, subtitle, score, genuine, fake, confidence, retention, recentNods) {
    if (DOM.statusTitle) DOM.statusTitle.textContent = title;
    if (DOM.statusSubtitle) DOM.statusSubtitle.textContent = subtitle;

    DOM.statusCard.className = 'glass-card status-card';
    if (status === 'genuine') DOM.statusCard.classList.add('status-genuine');
    else if (status === 'polite') DOM.statusCard.classList.add('status-polite');
    else if (status === 'danger') DOM.statusCard.classList.add('status-danger');
    else if (status === 'survival') DOM.statusCard.classList.add('status-survival');

    if (DOM.statusBadge) DOM.statusBadge.textContent = status.toUpperCase();
    if (DOM.statusUrgency) {
      if (status === 'danger') DOM.statusUrgency.textContent = "CRITICAL DISCONNECT";
      else if (status === 'survival') DOM.statusUrgency.textContent = "TOTAL BLACKOUT";
      else if (status === 'polite') DOM.statusUrgency.textContent = "MODERATE DRIFT";
      else if (status === 'genuine') DOM.statusUrgency.textContent = "OPTIMAL COMPREHENSION";
      else DOM.statusUrgency.textContent = "MONITORING ACTIVE";
    }

    if (DOM.understandingScoreVal) {
      DOM.understandingScoreVal.innerHTML = `${score}<span class="percent-sign">%</span>`;
    }
    if (DOM.gaugeCircle) {
      const circumference = 534;
      const offset = circumference - (score / 100) * circumference;
      DOM.gaugeCircle.style.strokeDashoffset = offset;
      if (score >= 80) DOM.gaugeCircle.style.stroke = 'var(--neon-emerald)';
      else if (score >= 50) DOM.gaugeCircle.style.stroke = 'var(--neon-amber)';
      else if (score >= 20) DOM.gaugeCircle.style.stroke = 'var(--neon-rose)';
      else DOM.gaugeCircle.style.stroke = 'var(--neon-purple)';
    }

    if (DOM.retentionVal) DOM.retentionVal.textContent = `${retention}%`;
    if (DOM.retentionBar) DOM.retentionBar.style.width = `${retention}%`;
    if (DOM.explanationText) {
      if (state.classifierMode === 'ml') {
        DOM.explanationText.textContent = `Supervised Machine Learning model is active. Classifying 9-dimensional head movement features via trained weights.`;
      } else {
        if (status === 'genuine') DOM.explanationText.textContent = "Measured cranial frequency indicates genuine semantic encoding. Subject displays low probability of polite duplicity.";
        else if (status === 'polite') DOM.explanationText.textContent = "Nod cadence exhibits social accommodation rhythms. Listening active, but actual memory retention is degrading.";
        else if (status === 'danger') DOM.explanationText.textContent = "Rapid affirmative oscillations detected. High certainty that subject has lost context and is nodding purely out of panic.";
        else if (status === 'survival') DOM.explanationText.textContent = "Continuous bobblehead response triggered. Cognitive faculties dormant; head is operating under autonomous survival reflexes.";
        else DOM.explanationText.textContent = "Position your face in the camera frame and nod naturally, or start Demo Mode to see the system in action.";
      }
    }

    if (DOM.nodConfidenceVal) DOM.nodConfidenceVal.textContent = confidence;
    if (DOM.nodConfidenceBar) DOM.nodConfidenceBar.style.width = `${confidence}%`;
    if (DOM.genuineProbVal) DOM.genuineProbVal.textContent = genuine;
    if (DOM.genuineProbBar) DOM.genuineProbBar.style.width = `${genuine}%`;
    if (DOM.fakeProbVal) DOM.fakeProbVal.textContent = fake;
    if (DOM.fakeProbBar) DOM.fakeProbBar.style.width = `${fake}%`;
    if (DOM.nodFreqVal) DOM.nodFreqVal.textContent = state.nodFrequency.toFixed(1);

    if (DOM.nodCadenceSubtext) {
      if (recentNods === 0) DOM.nodCadenceSubtext.textContent = "0 nods in last 10s (Dormant)";
      else DOM.nodCadenceSubtext.textContent = `${recentNods} nods in last 10s (${state.nodFrequency} nods/min)`;
    }
  }

  // ==========================================================================
  // DATASET COLLECTION & RECORDING ENGINE
  // ==========================================================================
  function startRecordingSample(label) {
    if (state.isRecording) return;
    initAudioContext();
    playSound('record_start');

    // If neither webcam nor demo is running, start demo mode automatically so frames exist!
    if (!state.isRunning) {
      startDemoMode();
    }

    state.isRecording = true;
    state.recordingLabel = label;
    state.recordingStartTime = performance.now();
    state.recordingFrames = [];
    state.recordingNods = [];

    DOM.recordingProgressContainer.classList.remove('hidden');
    DOM.recordingTargetLabel.textContent = label;
    DOM.recordingStatusLabel.textContent = `Recording ${label} sample...`;

    const startTime = performance.now();
    const duration = CONFIG.RECORDING_DURATION_MS;

    state.recordingIntervalId = setInterval(() => {
      const elapsed = performance.now() - startTime;
      const remaining = Math.max(0, (duration - elapsed) / 1000);
      const progressPercent = Math.min(100, (elapsed / duration) * 100);

      DOM.recordingCountdown.textContent = `${remaining.toFixed(1)}s remaining`;
      DOM.recordingProgressBar.style.width = `${progressPercent}%`;

      if (elapsed >= duration) {
        clearInterval(state.recordingIntervalId);
        finishRecordingSample();
      }
    }, 50);
  }

  function finishRecordingSample() {
    state.isRecording = false;
    DOM.recordingProgressContainer.classList.add('hidden');
    DOM.recordingStatusLabel.textContent = `Sample saved!`;
    playSound('record_done');

    const features = extractFeatures(state.recordingFrames, state.recordingNods, CONFIG.RECORDING_DURATION_MS / 1000);
    const sample = { ...features, label: state.recordingLabel };

    state.dataset.push(sample);
    saveDatasetToStorage();
    updateDatasetUI();

    showToast(`Sample Recorded!`, `Saved 1 ${state.recordingLabel} sample with 9 extracted features.`, "💾", "toast-info", 3500);
  }

  function updateDatasetUI() {
    const counts = { GENUINE: 0, POLITE: 0, DANGER: 0, SURVIVAL: 0 };
    state.dataset.forEach(s => {
      if (counts[s.label] !== undefined) counts[s.label]++;
    });

    if (DOM.totalSamplesCount) DOM.totalSamplesCount.textContent = `${state.dataset.length} samples`;
    if (DOM.countGenuine) DOM.countGenuine.textContent = `${counts.GENUINE} samples`;
    if (DOM.countPolite) DOM.countPolite.textContent = `${counts.POLITE} samples`;
    if (DOM.countDanger) DOM.countDanger.textContent = `${counts.DANGER} samples`;
    if (DOM.countSurvival) DOM.countSurvival.textContent = `${counts.SURVIVAL} samples`;
  }

  function saveDatasetToStorage() {
    try {
      localStorage.setItem('nod_dataset_samples', JSON.stringify(state.dataset));
    } catch (e) {
      console.warn("localStorage quota exceeded", e);
    }
  }

  function loadDatasetFromStorage() {
    try {
      const stored = localStorage.getItem('nod_dataset_samples');
      if (stored) {
        state.dataset = JSON.parse(stored);
      } else {
        // Automatically pre-load seed dataset so user has data immediately
        state.dataset = [...SEED_DATASET];
      }
    } catch (e) {
      state.dataset = [...SEED_DATASET];
    }
    updateDatasetUI();
  }

  function loadSeedData() {
    playSound('click');
    state.dataset = [...SEED_DATASET];
    saveDatasetToStorage();
    updateDatasetUI();
    showToast("Seed Dataset Loaded", "Loaded 40 realistic balanced samples (10 per class).", "📥", "toast-info", 3500);
  }

  function clearDataset() {
    playSound('click');
    if (confirm("Are you sure you want to clear all collected dataset samples?")) {
      state.dataset = [];
      saveDatasetToStorage();
      updateDatasetUI();
      showToast("Dataset Cleared", "All samples removed.", "↺", "toast-info", 2500);
    }
  }

  // ==========================================================================
  // CSV EXPORT & IMPORT ENGINE
  // ==========================================================================
  function exportDatasetCsv() {
    playSound('click');
    if (state.dataset.length === 0) {
      showToast("Dataset Empty", "Record some samples or load the seed dataset first!", "⚠️", "toast-info", 3000);
      return;
    }

    const headers = [...FEATURE_NAMES, 'label'];
    const rows = state.dataset.map(row => {
      return headers.map(h => row[h]).join(',');
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `nodding_dataset_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showToast("Dataset Exported", `Downloaded CSV with ${state.dataset.length} samples.`, "📤", "toast-info", 3500);
  }

  function importDatasetCsv(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
        if (lines.length < 2) throw new Error("CSV has no data rows.");

        const headers = lines[0].split(',').map(h => h.trim());
        const importedSamples = [];

        for (let i = 1; i < lines.length; i++) {
          const vals = lines[i].split(',').map(v => v.trim());
          if (vals.length !== headers.length) continue;

          const rowObj = {};
          headers.forEach((h, idx) => {
            if (h === 'label') {
              rowObj[h] = vals[idx].toUpperCase();
            } else {
              rowObj[h] = parseFloat(vals[idx]) || 0;
            }
          });

          if (CLASS_MAP[rowObj.label] !== undefined) {
            importedSamples.push(rowObj);
          }
        }

        if (importedSamples.length > 0) {
          state.dataset = importedSamples;
          saveDatasetToStorage();
          updateDatasetUI();
          showToast("CSV Imported", `Loaded ${importedSamples.length} samples successfully.`, "📂", "toast-info", 4000);
        } else {
          showToast("Import Failed", "No valid labeled rows found in CSV.", "⚠️", "toast-danger", 4000);
        }
      } catch (err) {
        showToast("Parse Error", `Failed to parse CSV: ${err.message}`, "❌", "toast-danger", 4000);
      }
    };
    reader.readAsText(file);
  }

  // ==========================================================================
  // IN-BROWSER SUPERVISED MACHINE LEARNING (MULTICLASS SOFTMAX LOGREG)
  // ==========================================================================
  function trainBrowserModel() {
    playSound('click');
    if (state.dataset.length < 4) {
      showToast("Need More Data", "Collect at least 1 sample of each class or load Seed Dataset!", "⚠️", "toast-danger", 4000);
      return;
    }

    // 1. Prepare Feature Matrix X and Labels y
    const X_raw = [];
    const y = [];
    state.dataset.forEach(s => {
      const row = FEATURE_NAMES.map(fname => s[fname] || 0);
      X_raw.push(row);
      y.push(CLASS_MAP[s.label] !== undefined ? CLASS_MAP[s.label] : 0);
    });

    const N = X_raw.length;
    const D = FEATURE_NAMES.length;
    const K = CLASSES.length;

    // 2. Compute Z-Score Normalization (StandardScaler)
    const means = new Array(D).fill(0);
    const stds = new Array(D).fill(0);

    for (let j = 0; j < D; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += X_raw[i][j];
      means[j] = sum / N;

      let varSum = 0;
      for (let i = 0; i < N; i++) varSum += Math.pow(X_raw[i][j] - means[j], 2);
      stds[j] = Math.sqrt(varSum / N) || 1.0;
    }

    const X = X_raw.map(row => row.map((val, j) => (val - means[j]) / stds[j]));

    // 3. Train / Test Split (80% / 20% with shuffle)
    const indices = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const splitIdx = Math.max(1, Math.floor(N * 0.8));
    const trainIndices = N >= 8 ? indices.slice(0, splitIdx) : indices;
    const testIndices = N >= 8 ? indices.slice(splitIdx) : indices;

    // 4. Initialize Weights (K x D) and Biases (K)
    const W = Array.from({ length: K }, () => Array.from({ length: D }, () => (Math.random() - 0.5) * 0.1));
    const b = new Array(K).fill(0);

    const epochs = 350;
    const lr = 0.08;
    const l2 = 0.002;

    // 5. Gradient Descent with Softmax Cross-Entropy Loss
    for (let ep = 0; ep < epochs; ep++) {
      const gradW = Array.from({ length: K }, () => new Array(D).fill(0));
      const gradB = new Array(K).fill(0);

      trainIndices.forEach(idx => {
        const x_i = X[idx];
        const target = y[idx];

        // Logits
        const logits = new Array(K).fill(0);
        let maxL = -Infinity;
        for (let k = 0; k < K; k++) {
          let dot = b[k];
          for (let j = 0; j < D; j++) dot += W[k][j] * x_i[j];
          logits[k] = dot;
          if (dot > maxL) maxL = dot;
        }

        // Softmax
        let sumExp = 0;
        const probs = new Array(K).fill(0);
        for (let k = 0; k < K; k++) {
          probs[k] = Math.exp(logits[k] - maxL);
          sumExp += probs[k];
        }
        for (let k = 0; k < K; k++) {
          probs[k] /= sumExp;
        }

        // Gradients
        for (let k = 0; k < K; k++) {
          const err = probs[k] - (k === target ? 1 : 0);
          gradB[k] += err;
          for (let j = 0; j < D; j++) {
            gradW[k][j] += err * x_i[j] + (l2 * W[k][j]);
          }
        }
      });

      const M = trainIndices.length;
      for (let k = 0; k < K; k++) {
        b[k] -= (lr * gradB[k]) / M;
        for (let j = 0; j < D; j++) {
          W[k][j] -= (lr * gradW[k][j]) / M;
        }
      }
    }

    // 6. Test Set Evaluation & Confusion Matrix
    let correct = 0;
    const confusionMatrix = Array.from({ length: K }, () => new Array(K).fill(0));

    testIndices.forEach(idx => {
      const x_i = X[idx];
      const actual = y[idx];

      let bestScore = -Infinity;
      let predicted = 0;

      for (let k = 0; k < K; k++) {
        let score = b[k];
        for (let j = 0; j < D; j++) score += W[k][j] * x_i[j];
        if (score > bestScore) {
          bestScore = score;
          predicted = k;
        }
      }

      confusionMatrix[actual][predicted]++;
      if (actual === predicted) correct++;
    });

    const accuracy = Math.round((correct / testIndices.length) * 1000) / 10;

    // 7. Save & Activate Model
    const model = {
      type: 'Multiclass Logistic Regression (Softmax)',
      weights: W,
      biases: b,
      scaler: { mean: means, std: stds },
      classes: CLASSES,
      featureNames: FEATURE_NAMES,
      accuracy: accuracy,
      confusionMatrix: confusionMatrix,
      trainedAt: new Date().toISOString()
    };

    state.activeMlModel = model;
    localStorage.setItem('active_nod_model', JSON.stringify(model));

    // Automatically switch to ML mode
    setClassifierMode('ml');

    // 8. Render Results in UI
    renderModelResults(model, N);
    showToast("Model Trained!", `Validation Accuracy: ${accuracy}% on test split.`, "🧠", "toast-info", 4500);
  }

  function predictWithMlModel(features, model) {
    const K = model.classes.length;
    const D = model.featureNames.length;
    const norm = model.featureNames.map((fname, j) => {
      const val = features[fname] || 0;
      const m = model.scaler.mean[j] || 0;
      const s = model.scaler.std[j] || 1;
      return (val - m) / s;
    });

    const logits = new Array(K).fill(0);
    let maxL = -Infinity;
    for (let k = 0; k < K; k++) {
      let score = model.biases[k];
      for (let j = 0; j < D; j++) score += model.weights[k][j] * norm[j];
      logits[k] = score;
      if (score > maxL) maxL = score;
    }

    let sumExp = 0;
    const probs = new Array(K).fill(0);
    for (let k = 0; k < K; k++) {
      probs[k] = Math.exp(logits[k] - maxL);
      sumExp += probs[k];
    }
    for (let k = 0; k < K; k++) probs[k] /= sumExp;

    let bestK = 0;
    let maxProb = 0;
    for (let k = 0; k < K; k++) {
      if (probs[k] > maxProb) {
        maxProb = probs[k];
        bestK = k;
      }
    }

    return {
      predictedClass: model.classes[bestK],
      confidence: maxProb,
      probabilities: probs
    };
  }

  function renderModelResults(model, sampleCount) {
    if (!DOM.modelResultsCard) return;
    DOM.modelResultsCard.classList.remove('hidden');

    if (DOM.modelAccuracyVal) DOM.modelAccuracyVal.textContent = `${model.accuracy.toFixed(1)}%`;
    if (DOM.modelStatusTitle) DOM.modelStatusTitle.textContent = `${model.type} Active`;
    if (DOM.modelStatusSubtitle) DOM.modelStatusSubtitle.textContent = `Trained on ${sampleCount} samples with 80/20 train-test validation.`;

    // Render Confusion Matrix
    if (DOM.confusionTableBody) {
      DOM.confusionTableBody.innerHTML = '';
      const matrix = model.confusionMatrix;
      CLASSES.forEach((actualLabel, i) => {
        const tr = document.createElement('tr');
        let rowHtml = `<td><strong>${actualLabel}</strong></td>`;
        CLASSES.forEach((predLabel, j) => {
          const val = (matrix && matrix[i] && matrix[i][j]) || 0;
          const isHit = i === j && val > 0;
          const isMiss = i !== j && val > 0;
          const cellClass = isHit ? 'cell-hit' : (isMiss ? 'cell-miss' : '');
          rowHtml += `<td class="${cellClass}">${val}</td>`;
        });
        tr.innerHTML = rowHtml;
        DOM.confusionTableBody.appendChild(tr);
      });
    }

    // Render Key Feature Weights
    if (DOM.weightsGrid) {
      DOM.weightsGrid.innerHTML = '';
      FEATURE_NAMES.forEach((fname, j) => {
        let weightSum = 0;
        for (let k = 0; k < model.classes.length; k++) {
          weightSum += Math.abs(model.weights[k][j] || 0);
        }
        const item = document.createElement('div');
        item.className = 'weight-item';
        item.innerHTML = `
          <span class="weight-name">${fname}</span>
          <span class="weight-val">|w|=${weightSum.toFixed(2)}</span>
        `;
        DOM.weightsGrid.appendChild(item);
      });
    }

    if (DOM.smallDatasetWarning) {
      DOM.smallDatasetWarning.classList.toggle('hidden', sampleCount >= 12);
    }
  }

  function importPythonModelJson(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        if (!json.weights || !json.biases || !json.scaler) {
          throw new Error("Invalid model JSON schema. Missing weights, biases, or scaler.");
        }

        state.activeMlModel = json;
        localStorage.setItem('active_nod_model', JSON.stringify(json));
        setClassifierMode('ml');
        renderModelResults(json, state.dataset.length || 40);
        showToast("Python Model Loaded", `Loaded ${json.type} with accuracy: ${(json.accuracy || 92).toFixed(1)}%.`, "⚙️", "toast-info", 4500);
      } catch (err) {
        showToast("JSON Error", `Failed to load model weights: ${err.message}`, "❌", "toast-danger", 4000);
      }
    };
    reader.readAsText(file);
  }

  function setClassifierMode(mode) {
    state.classifierMode = mode;
    playSound('click');

    if (mode === 'ml') {
      if (!state.activeMlModel) {
        // Automatically train on current dataset if no model exists yet
        trainBrowserModel();
      }
      DOM.btnModeML.classList.add('active');
      DOM.btnModeHeuristic.classList.remove('active');
      if (DOM.mlActiveModelChip) DOM.mlActiveModelChip.textContent = `${state.activeMlModel ? state.activeMlModel.type : 'ML'} (Active)`;
      showToast("Classifier Switched", "Now classifying with custom Supervised ML Model.", "🧠", "toast-info", 3000);
    } else {
      DOM.btnModeHeuristic.classList.add('active');
      DOM.btnModeML.classList.remove('active');
      if (DOM.mlInferenceBadge) DOM.mlInferenceBadge.classList.add('hidden');
      showToast("Classifier Switched", "Now classifying with Rule-Based Heuristics.", "📐", "toast-info", 3000);
    }
    updateRollingMetrics(performance.now());
  }

  // ==========================================================================
  // REAL-TIME OSCILLOSCOPE WAVEFORM RENDERER
  // ==========================================================================
  function renderWaveform() {
    const canvas = DOM.waveformCanvas;
    if (!canvas) return;

    const width = canvas.width = canvas.parentElement.clientWidth;
    const height = canvas.height = canvas.parentElement.clientHeight;

    waveformCtx.clearRect(0, 0, width, height);

    waveformCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    waveformCtx.lineWidth = 1;
    for (let y = 20; y < height; y += 30) {
      waveformCtx.beginPath();
      waveformCtx.moveTo(0, y);
      waveformCtx.lineTo(width, y);
      waveformCtx.stroke();
    }

    const centerY = height / 2;

    waveformCtx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
    waveformCtx.setLineDash([4, 4]);
    waveformCtx.beginPath();
    waveformCtx.moveTo(0, centerY);
    waveformCtx.lineTo(width, centerY);
    waveformCtx.stroke();
    waveformCtx.setLineDash([]);

    const thresholdY = centerY + (CONFIG.NOD_DOWN_THRESHOLD * 600);
    waveformCtx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
    waveformCtx.setLineDash([2, 4]);
    waveformCtx.beginPath();
    waveformCtx.moveTo(0, thresholdY);
    waveformCtx.lineTo(width, thresholdY);
    waveformCtx.stroke();

    const dangerY = centerY + (CONFIG.NOD_DOWN_THRESHOLD * 1000);
    waveformCtx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
    waveformCtx.beginPath();
    waveformCtx.moveTo(0, dangerY);
    waveformCtx.lineTo(width, dangerY);
    waveformCtx.stroke();
    waveformCtx.setLineDash([]);

    const buffer = state.waveformBuffer;
    const step = width / (CONFIG.WAVEFORM_POINTS - 1);

    waveformCtx.beginPath();
    for (let i = 0; i < buffer.length; i++) {
      const x = i * step;
      const y = centerY + (buffer[i] * 650);
      if (i === 0) waveformCtx.moveTo(x, y);
      else waveformCtx.lineTo(x, y);
    }

    waveformCtx.strokeStyle = state.currentStatus === 'danger' ? '#f43f5e' : 
                              state.currentStatus === 'survival' ? '#c084fc' : 
                              '#06b6d4';
    waveformCtx.lineWidth = 2.5;
    waveformCtx.shadowColor = waveformCtx.strokeStyle;
    waveformCtx.shadowBlur = 10;
    waveformCtx.stroke();
    waveformCtx.shadowBlur = 0;

    const now = performance.now();
    state.nodEventMarkers = state.nodEventMarkers.filter(marker => {
      const age = now - marker.time;
      if (age > 4000) return false;

      const x = (marker.index * step) - ((age / 1000) * 20);
      const y = centerY + (marker.delta * 650);

      if (x >= 0 && x <= width) {
        waveformCtx.fillStyle = '#10b981';
        waveformCtx.shadowColor = '#10b981';
        waveformCtx.shadowBlur = 12;
        waveformCtx.beginPath();
        waveformCtx.arc(x, y, 5, 0, Math.PI * 2);
        waveformCtx.fill();
        waveformCtx.shadowBlur = 0;
      }
      return true;
    });
  }

  // ==========================================================================
  // 3D HOLOGRAPHIC DEMO MODE
  // ==========================================================================
  const demoFaceModel = {
    vertices: [
      { x: 0, y: -0.9, z: 0.1 },
      { x: -0.5, y: -0.75, z: -0.2 },
      { x: 0.5, y: -0.75, z: -0.2 },
      { x: -0.7, y: -0.3, z: -0.3 },
      { x: 0.7, y: -0.3, z: -0.3 },
      { x: -0.35, y: -0.4, z: 0.2 },
      { x: -0.12, y: -0.42, z: 0.35 },
      { x: 0.12, y: -0.42, z: 0.35 },
      { x: 0.35, y: -0.4, z: 0.2 },
      { x: -0.25, y: -0.25, z: 0.25 },
      { x: 0.25, y: -0.25, z: 0.25 },
      { x: 0, y: -0.35, z: 0.4 },
      { x: 0, y: -0.05, z: 0.65 },
      { x: -0.14, y: 0.05, z: 0.45 },
      { x: 0.14, y: 0.05, z: 0.45 },
      { x: -0.22, y: 0.3, z: 0.3 },
      { x: 0, y: 0.25, z: 0.42 },
      { x: 0.22, y: 0.3, z: 0.3 },
      { x: 0, y: 0.38, z: 0.38 },
      { x: -0.6, y: 0.3, z: -0.2 },
      { x: 0.6, y: 0.3, z: -0.2 },
      { x: -0.35, y: 0.7, z: 0.05 },
      { x: 0.35, y: 0.7, z: 0.05 },
      { x: 0, y: 0.9, z: 0.25 }
    ],
    edges: [
      [0, 1], [0, 2], [1, 3], [2, 4],
      [5, 6], [7, 8], [6, 11], [7, 11],
      [11, 12], [12, 13], [12, 14], [13, 14],
      [15, 16], [16, 17], [17, 18], [18, 15],
      [3, 19], [4, 20], [19, 21], [20, 22], [21, 23], [22, 23],
      [13, 16], [14, 16], [16, 23], [9, 11], [10, 11],
      [9, 5], [10, 8], [3, 9], [4, 10]
    ]
  };

  function stepDemoSimulation(now) {
    state.demoTime += 0.025;
    const t = state.demoTime;

    if (state.demoScenario === 'tour') {
      const tourElapsed = (now - state.demoTourStartTime) / 1000;
      if (tourElapsed < 6) {
        state.demoNodInterval = 3400;
        updateActiveScenarioCard('genuine');
      } else if (tourElapsed < 14) {
        state.demoNodInterval = 2100;
        updateActiveScenarioCard('polite');
      } else if (tourElapsed < 22) {
        state.demoNodInterval = 1100;
        updateActiveScenarioCard('danger');
      } else if (tourElapsed < 30) {
        state.demoNodInterval = 650;
        updateActiveScenarioCard('survival');
      } else {
        state.demoTourStartTime = now;
      }
    }

    let ambientPitch = Math.sin(t * 1.4) * 0.006;
    let ambientYaw = Math.cos(t * 0.9) * 0.08;
    let nodDeflection = 0;

    if (now - state.lastDemoNodTime > state.demoNodInterval) {
      state.lastDemoNodTime = now;
    }

    const timeSinceNod = now - state.lastDemoNodTime;
    const nodDuration = 450;

    if (timeSinceNod < nodDuration) {
      const phase = (timeSinceNod / nodDuration) * Math.PI;
      nodDeflection = Math.sin(phase) * 0.048;
    }

    const simulatedPitch = 0.5 + ambientPitch + nodDeflection;
    const simulatedYaw = ambientYaw;

    processPitchSample(simulatedPitch, now, simulatedYaw);
    renderDemoHologram(ambientYaw, nodDeflection);
  }

  function renderDemoHologram(yaw, pitchNod) {
    const canvas = DOM.overlayCanvas;
    if (!canvas) return;

    const width = canvas.width = DOM.viewportContainer.clientWidth;
    const height = canvas.height = DOM.viewportContainer.clientHeight;

    overlayCtx.clearRect(0, 0, width, height);

    overlayCtx.strokeStyle = 'rgba(168, 85, 247, 0.08)';
    overlayCtx.lineWidth = 1;
    const gridStep = 40;
    for (let x = 0; x < width; x += gridStep) {
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, 0);
      overlayCtx.lineTo(x, height);
      overlayCtx.stroke();
    }
    for (let y = 0; y < height; y += gridStep) {
      overlayCtx.beginPath();
      overlayCtx.moveTo(0, y);
      overlayCtx.lineTo(width, y);
      overlayCtx.stroke();
    }

    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) * 0.38;

    const pitchAngle = (pitchNod * 8) + (Math.sin(state.demoTime * 1.5) * 0.08);
    const yawAngle = yaw;

    const cosP = Math.cos(pitchAngle), sinP = Math.sin(pitchAngle);
    const cosY = Math.cos(yawAngle), sinY = Math.sin(yawAngle);

    const projected = demoFaceModel.vertices.map(v => {
      let x1 = v.x * cosY + v.z * sinY;
      let y1 = v.y;
      let z1 = -v.x * sinY + v.z * cosY;

      let x2 = x1;
      let y2 = y1 * cosP - z1 * sinP;
      let z2 = y1 * sinP + z1 * cosP;

      const distance = 2.5;
      const fov = distance / (distance + z2);

      return {
        x: centerX + (x2 * scale * fov),
        y: centerY + (y2 * scale * fov),
        z: z2
      };
    });

    let meshColor = 'rgba(6, 182, 212, 0.7)';
    let nodeColor = '#38bdf8';

    if (state.currentStatus === 'genuine') {
      meshColor = 'rgba(16, 185, 129, 0.75)';
      nodeColor = '#34d399';
    } else if (state.currentStatus === 'polite') {
      meshColor = 'rgba(245, 158, 11, 0.75)';
      nodeColor = '#fbbf24';
    } else if (state.currentStatus === 'danger') {
      meshColor = 'rgba(244, 63, 94, 0.85)';
      nodeColor = '#f43f5e';
    } else if (state.currentStatus === 'survival') {
      meshColor = 'rgba(168, 85, 247, 0.9)';
      nodeColor = '#e879f9';
    }

    if (state.showWireframe) {
      overlayCtx.strokeStyle = meshColor;
      overlayCtx.lineWidth = 1.6;
      overlayCtx.shadowColor = nodeColor;
      overlayCtx.shadowBlur = 8;

      demoFaceModel.edges.forEach(([i, j]) => {
        const p1 = projected[i];
        const p2 = projected[j];
        overlayCtx.beginPath();
        overlayCtx.moveTo(p1.x, p1.y);
        overlayCtx.lineTo(p2.x, p2.y);
        overlayCtx.stroke();
      });

      projected.forEach((p, idx) => {
        overlayCtx.fillStyle = (idx === 12) ? '#ffffff' : nodeColor;
        overlayCtx.beginPath();
        overlayCtx.arc(p.x, p.y, (idx === 12) ? 4.5 : 2.5, 0, Math.PI * 2);
        overlayCtx.fill();
      });

      overlayCtx.shadowBlur = 0;
    }

    overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.35)';
    overlayCtx.font = '10px "JetBrains Mono", monospace';
    overlayCtx.fillText("SIMULATION CORE // SYNTHETIC CRANIAL MESH", 24, height - 20);
    overlayCtx.fillText(`YAW: ${(yawAngle * 180 / Math.PI).toFixed(1)}° | PITCH: ${(pitchAngle * 180 / Math.PI).toFixed(1)}°`, 24, height - 36);
  }

  // ==========================================================================
  // MEDIAPIPE WEBCAM INTEGRATION
  // ==========================================================================
  async function initFaceMesh() {
    if (faceMeshInstance) return faceMeshInstance;
    if (typeof window.FaceMesh === 'undefined') {
      throw new Error("MediaPipe FaceMesh script not loaded from CDN.");
    }

    const faceMesh = new window.FaceMesh({
      locateFile: (file) => `${CONFIG.MEDIAPIPE_ASSETS}${file}`
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onFaceMeshResults);
    faceMeshInstance = faceMesh;
    return faceMesh;
  }

  function onFaceMeshResults(results) {
    const canvas = DOM.overlayCanvas;
    if (!canvas) return;

    const width = canvas.width = DOM.viewportContainer.clientWidth;
    const height = canvas.height = DOM.viewportContainer.clientHeight;

    overlayCtx.clearRect(0, 0, width, height);

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      DOM.feedTitleText.textContent = "SEARCHING FOR FACE...";
      return;
    }

    DOM.feedTitleText.textContent = "FACE LOCKED — TRACKING CRANIAL KINEMATICS";

    const landmarks = results.multiFaceLandmarks[0];
    const now = performance.now();

    const nose = landmarks[1];
    const forehead = landmarks[10];
    const chin = landmarks[152];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];

    const faceHeight = Math.sqrt(
      Math.pow((chin.x - forehead.x) * width, 2) +
      Math.pow((chin.y - forehead.y) * height, 2)
    );

    let pitchMetric = 0.5;
    if (faceHeight > 20) {
      const noseY = nose.y * height;
      const foreheadY = forehead.y * height;
      pitchMetric = (noseY - foreheadY) / faceHeight;
    }

    const eyeSpan = Math.abs(rightEye.x - leftEye.x) * width;
    let yawMetric = 0;
    if (eyeSpan > 10) {
      const midEyeX = ((leftEye.x + rightEye.x) / 2) * width;
      yawMetric = ((nose.x * width) - midEyeX) / eyeSpan;
    }

    processPitchSample(pitchMetric, now, yawMetric);

    if (state.showWireframe) {
      drawLiveFaceWireframe(landmarks, width, height);
    }
  }

  function drawLiveFaceWireframe(landmarks, width, height) {
    overlayCtx.strokeStyle = (state.currentStatus === 'danger') ? 'rgba(244, 63, 94, 0.7)' :
                             (state.currentStatus === 'survival') ? 'rgba(168, 85, 247, 0.8)' :
                             'rgba(6, 182, 212, 0.6)';
    overlayCtx.lineWidth = 1.5;
    overlayCtx.fillStyle = 'rgba(6, 182, 212, 0.9)';

    const faceOval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
    const noseContour = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];
    const lipsContour = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61];

    function drawPath(indices) {
      overlayCtx.beginPath();
      for (let i = 0; i < indices.length; i++) {
        const pt = landmarks[indices[i]];
        const x = pt.x * width;
        const y = pt.y * height;
        if (i === 0) overlayCtx.moveTo(x, y);
        else overlayCtx.lineTo(x, y);
      }
      overlayCtx.stroke();
    }

    drawPath(faceOval);
    drawPath(noseContour);
    drawPath(lipsContour);

    const nose = landmarks[1];
    overlayCtx.fillStyle = '#ffffff';
    overlayCtx.shadowColor = '#06b6d4';
    overlayCtx.shadowBlur = 10;
    overlayCtx.beginPath();
    overlayCtx.arc(nose.x * width, nose.y * height, 4, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.shadowBlur = 0;
  }

  // ==========================================================================
  // SESSION LIFECYCLE
  // ==========================================================================
  async function startWebcamMode() {
    stopCurrentSession();
    initAudioContext();
    playSound('click');

    DOM.standbyOverlay.classList.add('hidden');
    DOM.cameraErrorOverlay.classList.add('hidden');
    DOM.feedStatusDot.className = 'feed-indicator live';
    DOM.feedTitleText.textContent = "INITIALIZING WEBCAM & NEURAL MODEL...";
    DOM.currentPipelineTag.textContent = "MEDIAPIPE FACEMESH (LOCAL)";
    DOM.stopSessionBtn.classList.remove('hidden');

    state.mode = 'webcam';
    state.isRunning = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
      });
      cameraStream = stream;
      DOM.webcamVideo.srcObject = stream;
      await DOM.webcamVideo.play();

      const faceMesh = await initFaceMesh();

      if (typeof window.Camera !== 'undefined') {
        cameraInstance = new window.Camera(DOM.webcamVideo, {
          onFrame: async () => {
            if (state.isRunning && state.mode === 'webcam') {
              await faceMesh.send({ image: DOM.webcamVideo });
            }
          },
          width: 640,
          height: 480
        });
        await cameraInstance.start();
      } else {
        const processFrame = async () => {
          if (state.isRunning && state.mode === 'webcam') {
            await faceMesh.send({ image: DOM.webcamVideo });
            animationFrameId = requestAnimationFrame(processFrame);
          }
        };
        animationFrameId = requestAnimationFrame(processFrame);
      }

      showToast("Camera Connected", "Face tracking active. Analyzing cranial nods locally.", "🟢", "toast-info", 3500);
      startMainRenderLoop();
    } catch (err) {
      console.error("Camera initialization error:", err);
      DOM.feedStatusDot.className = 'feed-indicator';
      DOM.cameraErrorOverlay.classList.remove('hidden');
      DOM.cameraErrorMessage.textContent = `Could not access camera (${err.name || 'Unavailable'}). Don't worry, Demo Mode works 100% offline!`;
      stopCurrentSession();
    }
  }

  function startDemoMode() {
    stopCurrentSession();
    initAudioContext();
    playSound('click');

    DOM.standbyOverlay.classList.add('hidden');
    DOM.cameraErrorOverlay.classList.add('hidden');
    DOM.feedStatusDot.className = 'feed-indicator simulated';
    DOM.feedTitleText.textContent = "DEMO SIMULATION ACTIVE (NO CAMERA)";
    DOM.currentPipelineTag.textContent = "SYNTHETIC 3D KINEMATICS";
    DOM.stopSessionBtn.classList.remove('hidden');
    DOM.demoRunBtnText.textContent = "⏸ PAUSE DEMO SIMULATION";

    state.mode = 'demo';
    state.isRunning = true;
    state.demoTourStartTime = performance.now();
    state.lastDemoNodTime = performance.now();

    showToast("Demo Mode Activated", "Simulating cranial nod physics. Zero webcam required.", "🚀", "toast-info", 3500);
    startMainRenderLoop();
  }

  function stopCurrentSession() {
    state.isRunning = false;
    
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (cameraInstance) {
      try { cameraInstance.stop(); } catch (e) {}
      cameraInstance = null;
    }

    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }

    if (DOM.webcamVideo) {
      DOM.webcamVideo.srcObject = null;
    }

    if (overlayCtx) overlayCtx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);

    DOM.feedStatusDot.className = 'feed-indicator';
    DOM.currentPipelineTag.textContent = "STANDBY";
    DOM.stopSessionBtn.classList.add('hidden');
    DOM.demoRunBtnText.textContent = "▶ START DEMO SIMULATION";
  }

  function resetAllState() {
    state.totalNods = 0;
    state.nodTimestamps = [];
    state.baselinePitch = null;
    state.currentPitch = 0;
    state.currentYaw = 0;
    state.nodFrequency = 0;
    state.understandingScore = 90;
    state.genuineProb = 85;
    state.fakeProb = 15;
    state.nodConfidence = 0;
    state.nodState = 'IDLE';
    state.waveformBuffer = new Array(CONFIG.WAVEFORM_POINTS).fill(0);
    state.nodEventMarkers = [];
    state.lastAlertStatus = null;
    state.kinematicFrameBuffer = [];
    state.nodCycleRecords = [];

    if (DOM.nodsDetectedVal) DOM.nodsDetectedVal.textContent = "0";
    if (DOM.nodFreqVal) DOM.nodFreqVal.textContent = "0.0";
    
    renderDashboardMetrics('neutral', 'Awaiting Nod Input', '“The system is observing your cranial inclinations…”', 90, 85, 15, 0, 80, 0);
    showToast("System Reset", "All nod counters and comprehension indices reset.", "↺", "toast-info", 2500);
    playSound('click');
  }

  function startMainRenderLoop() {
    function loop(timestamp) {
      if (!state.isRunning) return;

      state.frameCount++;
      if (timestamp - state.lastFpsUpdate >= 1000) {
        state.fps = state.frameCount;
        state.frameCount = 0;
        state.lastFpsUpdate = timestamp;
        if (DOM.fpsValue) DOM.fpsValue.textContent = state.fps;
      }

      if (state.mode === 'demo') {
        stepDemoSimulation(timestamp);
      }

      renderWaveform();
      animationFrameId = requestAnimationFrame(loop);
    }
    animationFrameId = requestAnimationFrame(loop);
  }

  // ==========================================================================
  // SCENARIO PRESETS & PRESENTER TOOLS
  // ==========================================================================
  function setScenario(scenarioKey) {
    state.demoScenario = scenarioKey;
    playSound('click');

    if (scenarioKey === 'genuine') state.demoNodInterval = 3500;
    else if (scenarioKey === 'polite') state.demoNodInterval = 2100;
    else if (scenarioKey === 'danger') state.demoNodInterval = 1100;
    else if (scenarioKey === 'survival') state.demoNodInterval = 650;
    else if (scenarioKey === 'tour') state.demoTourStartTime = performance.now();

    updateActiveScenarioCard(scenarioKey);

    if (!state.isRunning || state.mode !== 'demo') {
      startDemoMode();
    }
  }

  function updateActiveScenarioCard(scenarioKey) {
    DOM.presetCards.forEach(card => {
      card.classList.toggle('active', card.dataset.scenario === scenarioKey);
    });
  }

  function injectManualNod() {
    initAudioContext();
    registerNod(performance.now(), 400);
  }

  function injectRapidBurst() {
    initAudioContext();
    showToast("Rapid Burst Injected", "Simulating frantic panic agreement...", "⚡", "toast-danger", 3000);
    for (let i = 0; i < 4; i++) {
      setTimeout(() => registerNod(performance.now(), 250), i * 350);
    }
  }

  function triggerSurvivalModeInstant() {
    initAudioContext();
    showToast("Survival Mode Override", "Simulating total corporate survival nodding!", "💀", "toast-survival", 4000);
    for (let i = 0; i < 8; i++) {
      setTimeout(() => registerNod(performance.now(), 180), i * 220);
    }
  }

  // ==========================================================================
  // CERTIFICATE MODAL
  // ==========================================================================
  function generateCertificate() {
    playSound('click');
    
    DOM.certTotalNods.textContent = state.totalNods;
    DOM.certScore.textContent = `${state.understandingScore}%`;
    DOM.certCadence.textContent = `${state.nodFrequency} nods/min`;
    DOM.certFakeQuotient.textContent = `${state.fakeProb}%`;

    let rankTitle = "Dormant Spectator";
    let rankQuote = "“Zero cranial deflection recorded. Subject was either meditating or completely frozen in panic.”";

    if (state.totalNods > 0) {
      if (state.understandingScore >= 80) {
        rankTitle = "Authentic Comprehender (Rare Specimen)";
        rankQuote = "“Empirically confirmed: Subject actually absorbed the requirements without resorting to panic agreement.”";
      } else if (state.understandingScore >= 50) {
        rankTitle = "Polite Social Accomplice";
        rankQuote = "“Exhibited classic corporate courtesy. Nodded at respectful intervals with plausible deniability.”";
      } else if (state.understandingScore >= 20) {
        rankTitle = "S-Tier Impostor • Meeting Survivor";
        rankQuote = "“Nodded with relentless charismatic conviction despite internal cognition registering a total 404.”";
      } else {
        rankTitle = "Grandmaster of Bobblehead Agreement";
        rankQuote = "“Malayalam Athe-Athe Pro Max: Brain shut down completely while neck operated on sheer divine muscle memory.”";
      }
    }

    DOM.certRankTitle.textContent = rankTitle;
    DOM.certRankQuote.textContent = rankQuote;
    DOM.certificateModal.classList.remove('hidden');
  }

  // ==========================================================================
  // EVENT LISTENERS BINDING
  // ==========================================================================
  function bindEventListeners() {
    // Mode toggles
    DOM.btnModeHeuristic.addEventListener('click', () => setClassifierMode('heuristic'));
    DOM.btnModeML.addEventListener('click', () => setClassifierMode('ml'));

    // Hero & Standby
    DOM.startWebcamBtn.addEventListener('click', startWebcamMode);
    DOM.startDemoBtn.addEventListener('click', startDemoMode);
    DOM.standbyWebcamBtn.addEventListener('click', startWebcamMode);
    DOM.standbyDemoBtn.addEventListener('click', startDemoMode);
    DOM.errorFallbackDemoBtn.addEventListener('click', startDemoMode);

    DOM.stopSessionBtn.addEventListener('click', () => {
      stopCurrentSession();
      DOM.standbyOverlay.classList.remove('hidden');
      DOM.feedTitleText.textContent = "SYSTEM IDLE — SELECT MODE";
    });

    DOM.toggleMeshBtn.addEventListener('click', () => {
      state.showWireframe = !state.showWireframe;
      DOM.meshToggleText.textContent = state.showWireframe ? "ON" : "OFF";
      playSound('click');
    });

    DOM.soundToggleBtn.addEventListener('click', () => {
      state.soundEnabled = !state.soundEnabled;
      DOM.soundStatusText.textContent = state.soundEnabled ? "ON" : "OFF";
      DOM.soundIconOn.classList.toggle('hidden', !state.soundEnabled);
      DOM.soundIconOff.classList.toggle('hidden', state.soundEnabled);
      if (state.soundEnabled) {
        initAudioContext();
        playSound('nod');
      }
    });

    DOM.toggleDemoRunBtn.addEventListener('click', () => {
      if (state.isRunning && state.mode === 'demo') {
        stopCurrentSession();
        DOM.demoRunBtnText.textContent = "▶ RESUME DEMO SIMULATION";
      } else {
        startDemoMode();
      }
    });

    DOM.resetAllBtn.addEventListener('click', resetAllState);

    DOM.presetCards.forEach(card => {
      card.addEventListener('click', () => setScenario(card.dataset.scenario));
    });

    DOM.triggerSingleNodBtn.addEventListener('click', injectManualNod);
    DOM.triggerRapidBurstBtn.addEventListener('click', injectRapidBurst);
    DOM.triggerSurvivalPanicBtn.addEventListener('click', triggerSurvivalModeInstant);
    DOM.triggerRandomQuoteBtn.addEventListener('click', triggerRandomRoast);

    // Recording Controls
    DOM.recordGenuineBtn.addEventListener('click', () => startRecordingSample('GENUINE'));
    DOM.recordPoliteBtn.addEventListener('click', () => startRecordingSample('POLITE'));
    DOM.recordDangerBtn.addEventListener('click', () => startRecordingSample('DANGER'));
    DOM.recordSurvivalBtn.addEventListener('click', () => startRecordingSample('SURVIVAL'));

    DOM.loadSeedDataBtn.addEventListener('click', loadSeedData);
    DOM.exportCsvBtn.addEventListener('click', exportDatasetCsv);
    DOM.importCsvInput.addEventListener('change', importDatasetCsv);
    DOM.clearDatasetBtn.addEventListener('click', clearDataset);

    DOM.trainBrowserModelBtn.addEventListener('click', trainBrowserModel);
    DOM.importJsonModelInput.addEventListener('change', importPythonModelJson);

    // Certificate Modal
    DOM.certificateBtn.addEventListener('click', generateCertificate);
    DOM.closeModalBtn.addEventListener('click', () => DOM.certificateModal.classList.add('hidden'));
    DOM.dismissModalBtn.addEventListener('click', () => DOM.certificateModal.classList.add('hidden'));
    DOM.printCertBtn.addEventListener('click', () => window.print());

    window.addEventListener('resize', () => {
      if (state.isRunning) renderWaveform();
    });
  }

  // ==========================================================================
  // INITIALIZATION ON DOM READY
  // ==========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    bindEventListeners();
    loadDatasetFromStorage();
    renderWaveform();
    renderDashboardMetrics('neutral', 'Awaiting Nod Input', '“The system is observing your cranial inclinations…”', 90, 85, 15, 0, 80, 0);

    // Check if an active model was previously trained
    try {
      const savedModel = localStorage.getItem('active_nod_model');
      if (savedModel) {
        state.activeMlModel = JSON.parse(savedModel);
        renderModelResults(state.activeMlModel, state.dataset.length || 40);
      }
    } catch (e) {
      console.warn("Could not load saved model:", e);
    }

    console.log("ARIYATHE THALAYATTIYATH AI initialized with CV + ML pipeline.");
  });

})();
