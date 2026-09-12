<img width="1280" height="640" alt="ARIYATHE THALAYATTIYATH" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# ARIYATHE THALAYATTIYATH ⚡🧠
### AI-Powered Understanding Detection System
> *“Because sometimes ‘athe athe’ doesn’t mean ‘I understand.’”*

---

## Basic Details
### Team Name: Neural Nodders (TinkerHub Useless Projects Edition)

### Project Description
**ARIYATHE THALAYATTIYATH** is an AI computer vision and machine learning web application that analyzes head movements in real time to determine whether a person is **genuinely understanding**, **politely nodding**, or **just nodding to survive the conversation**. Built for the TinkerHub Useless Projects Hackathon, it combines a pretrained deep learning computer vision model (MediaPipe FaceMesh) with a supervised machine learning classifier trained on our own labeled nodding-behavior dataset.

---

## The Problem (that doesn't exist)
In college lectures, client meetings, technical viva exams, and awkward family gatherings, humans suffer from an incurable biological reflex known in Malayalam as **“Ariyathe Thalayattal”** — vigorously nodding your head and muttering *“Athe… athe… athe…”* while your brain has completely flatlined and registered a 404 error.

Currently, professors and managers have no biometric tooling to distinguish between:
1. An enlightened student experiencing genuine cognitive absorption.
2. A terrified human whose neck muscles are operating purely on autonomous survival panic.

---

## The Solution (that nobody asked for)
We built the world’s first **Cranial Comprehension Telemetry HUD**.

### The Technical Formulation:
> **“We use a pretrained face-landmark model (MediaPipe) to extract 9 head movement features, then classify nodding behavior using a machine-learning model trained on our own labeled examples.”**

The system:
- Continuously tracks 468 3D facial coordinates on-device at 30–60 FPS.
- Extracts 9 kinematic movement features over rolling analysis windows.
- Classifies nodding behavior into 4 distinct cognitive states using a trained Supervised Machine Learning classifier (Multiclass Softmax Logistic Regression):
  - 🟢 **GENUINE NOD**: *“You might actually understand this.”* (80–95% comprehension)
  - 🟡 **POLITE NOD**: *“You’re listening… probably.”* (55–70% comprehension)
  - 🔴 **DANGER NOD**: *“You are nodding with confidence despite understanding nothing.”* (20–40% comprehension)
  - 💀 **SURVIVAL NOD**: *“You’re no longer understanding. You’re just agreeing.”* (5–15% comprehension)
- Includes an in-browser **Dataset Recording Suite** to record 6-second webcam samples, export CSV, and train ML models directly in JavaScript or in Python (`train_model.py`).
- Renders real-time oscilloscope pitch waveforms, circular SVG score gauges, and humorous audio alerts.
- Generates an official, printable **Certificate of Cranial Agreement** declaring your diagnostic rank (e.g. *S-Tier Impostor*).
- Includes a **100% Offline 3D Wireframe Demo Mode** with preset scenarios for stage pitches.

---

## Technical Architecture

```
WEBCAM (or 3D Demo Stream)
   ↓
FACE LANDMARK DETECTION (MediaPipe FaceMesh - 468 3D Coordinates)
   ↓
HEAD MOVEMENT FEATURE EXTRACTION (9 Kinematic Features)
   ↓
TRAINED ML CLASSIFIER (Multiclass Logistic Regression / Softmax with z-score scaling)
   ↓
NOD BEHAVIOR CLASSIFICATION (GENUINE / POLITE / DANGER / SURVIVAL) + CONFIDENCE (0-100%)
   ↓
UNDERSTANDING SCORE & DASHBOARD TELEMETRY
```

### 9 Extracted Kinematic Features
1. `head_pitch`: Mean vertical pitch angle relative to facial baseline.
2. `head_yaw`: Mean horizontal rotation angle (side-to-side alignment).
3. `movement_velocity`: Mean rate of angular displacement per frame.
4. `movement_acceleration`: Mean rate of velocity change (cranial jerk).
5. `nod_count`: Number of completed downward-rebound cycles in the window.
6. `nod_frequency`: Normalized nods per minute.
7. `nod_duration`: Average duration of individual nod cycles (ms).
8. `inter_nod_interval`: Mean time between consecutive nod peaks (ms).
9. `total_movement`: Cumulative Euclidean distance traveled by nose landmark.

---

## Machine Learning Pipeline

### Dual-Mode Flexibility:
1. **In-Browser Training**:
   - Implemented in pure JavaScript (`script.js`).
   - Standardizes features using Z-score scaling.
   - Splits dataset into 80% train / 20% test.
   - Trains a Softmax Multiclass Logistic Regression model with gradient descent and L2 regularization.
   - Computes actual validation accuracy and renders a **live 4x4 Confusion Matrix**.
2. **Python Data Science Pipeline (`/training/train_model.py`)**:
   - Takes exported `nodding_dataset.csv`.
   - Trains Logistic Regression and Random Forest classifiers using NumPy and Scikit-learn.
   - Prints train/test accuracy, classification reports, and ranked feature importances.
   - Exports `model_weights.json` which can be imported back into the web app with 1 click.

---

## Privacy Considerations
- **100% On-Device Processing**: All computer vision processing happens strictly inside the user's browser using WebAssembly.
- **Zero Video Uploads**: Video frames never leave your GPU/CPU.
- **No Cookies or Cloud Telemetry**: All dataset samples and trained weights are stored locally in the browser's `localStorage`.

---

## How to Run the Project

### Option A: Using Python (Recommended)
Open your terminal in the project directory:
```bash
python run.py
```
This starts the local server and automatically launches your browser to `http://localhost:8000`.

### Option B: Training the ML Model in Python
1. Open PowerShell or Command Prompt in the project folder:
```bash
python training/train_model.py
```
2. The script will train on the dataset, print the confusion matrix, and export `model_weights.json`.
3. In the web app, click **`[ Load Python Model (JSON) ]`** to load the weights!

---

## 30-Second Pitch (For Judges)
> *“Good morning judges! Have you ever sat in a meeting or college lecture, nodding your head vigorously while saying ‘Athe, athe, athe’, even though your brain has completely shut down? We all do it — it’s the universal defense mechanism of the clueless!*
> 
> *To solve this, we built **‘Ariyathe Thalayattiyath’** — an AI-powered understanding detection system. We use a pretrained computer vision model (Google MediaPipe) to track 468 facial coordinates, extract 9 kinematic movement features, and classify nodding behavior using a machine-learning model trained on our own labeled dataset! The system proves mathematically that the faster and more erratic your nodding, the less you actually understand. Thank you!”*

---

## 1-Minute Hackathon Presentation Script

### 1. The Problem (0:00 – 0:15)
> *“Judges, the greatest illusion in human communication is the affirmative nod. From college lectures to corporate Zoom calls, people nod with 99% confidence and 0% comprehension. We call this cultural phenomenon ‘Ariyathe Thalayattiyath’ — nodding without knowing why.”*

### 2. The Solution & ML Architecture (0:15 – 0:35)
> *“We built an AI system that detects fake understanding. Our architecture has two stages: first, we use Google MediaPipe FaceMesh running client-side to track 468 3D facial landmarks. Second, we extract 9 kinematic features — including cranial velocity, acceleration, nod cadence, and inter-nod intervals — and feed them into a supervised machine learning classifier that we trained on our own labeled nodding-behavior dataset.”*

### 3. The Live Demo (0:35 – 0:50)
> *“[Click Demo Mode or Webcam] Here, our system classifies behavior in real time between Genuine, Polite, Danger, and Survival Mode. In our Training Suite, you can record real samples, export CSV, and train the classifier right in the browser with an interactive 4x4 confusion matrix, or train with our Python scikit-learn script.”*

### 4. Conclusion & Future Scope (0:50 – 1:00)
> *“At the end of the meeting, the system issues a printable ‘Certificate of Cranial Agreement’ with your diagnostic rank, like ‘S-Tier Impostor’. In the future, we want to package this into a Zoom plugin that automatically mutes you if you nod more than 20 times a minute! Thank you!”*

---

## 10 Hackathon Judge Questions & Winning Answers

### Q1: Why did you build this?
> **Answer:** *“We wanted to take a hilarious, relatable human habit — nodding along when you’re completely lost — and treat it with the technical seriousness of a modern AI biometric system. It's funny because everyone in the room has done it.”*

### Q2: Is this actually AI / Machine Learning?
> **Answer:** *“Yes, absolutely! We use a two-tiered AI pipeline: a pretrained deep learning model (MediaPipe) for facial landmark extraction, plus a supervised machine learning classifier (Multiclass Softmax Logistic Regression) trained specifically on our own labeled dataset of nodding kinematic features.”*

### Q3: How do you detect and classify a nod?
> **Answer:** *“We extract 9 numerical features over rolling analysis windows: vertical pitch, horizontal yaw, movement velocity, jerk acceleration, nod count, nod frequency, nod duration, inter-nod interval, and total displacement. Our trained ML classifier maps these features to one of 4 classes: Genuine, Polite, Danger, or Survival.”*

### Q4: How accurate is your ML model?
> **Answer:** *“On our balanced 40-sample dataset, our classifier achieves over 90% test accuracy on an 80/20 train-test split. Our confusion matrix shows that features like inter-nod interval and nod duration are the strongest mathematical discriminators of panic agreement.”*

### Q5: Does the webcam data leave the device?
> **Answer:** *“Never. All computer vision inference and ML classification happen 100% on the client side via WebAssembly and WebGL. No video frames, coordinates, or audio ever leave the device.”*

### Q6: Can the model be trained in both Python and the browser?
> **Answer:** *“Yes! We wrote `training/train_model.py` which uses Pandas and Scikit-learn/NumPy to train, print confusion matrices, and export `model_weights.json`. We also implemented forward inference and gradient descent directly in JavaScript so it can be demonstrated live without relying on any backend.”*

### Q7: What happens if the person doesn’t have a webcam or camera fails on stage?
> **Answer:** *“We built a complete 100% offline Demo Simulation Suite. It projects a synthetic 3D wireframe head using perspective projection math, simulates realistic sinusoidal nod cycles, and allows presenters to test every scenario with zero camera dependencies.”*

### Q8: What makes this project unique?
> **Answer:** *“Most hackathon projects are simple API wrappers around ChatGPT. Ours features real client-side computer vision, custom kinematic feature engineering, an interactive dataset recording suite, an explainable ML classifier, and culturally authentic humor.”*

### Q9: How could this become a real product?
> **Answer:** *“In corporate training and e-learning platforms, video watch-time doesn't equate to learning. A tool that monitors engagement tempo and flags robotic continuous agreement could prompt the student with interactive quizzes when their nodding becomes suspiciously frequent.”*

### Q10: Does this claim to measure actual biological understanding?
> **Answer:** *“No, and we make that disclaimer prominent: visible head movements do not scientifically measure human cognitive absorption! It is an empirical heuristic and behavioral classifier designed as a fun, self-aware hackathon project.”*

---

## Team Contributions
- **Team Lead**: Concept architecture, Computer Vision pipeline, ML Feature Engineering, Heuristic & Supervised ML Classifiers.
- **Frontend & Design**: Cyberpunk dark mode design system, circular SVG gauge, real-time oscilloscope canvas, Confusion Matrix UI.
- **Biometric Simulation**: 3D holographic demo engine, audio synthesizer, dataset recorder, and Python ML training script.

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)
