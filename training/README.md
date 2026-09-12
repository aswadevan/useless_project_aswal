# Machine Learning Training Pipeline 🧠
## ARIYATHE THALAYATTIYATH — Nodding Behavior Classifier

> **"അറിയില്ലെങ്കിലും തലയാട്ടും."**
> We detect head movement. We do NOT read minds.

---

## 🏗️ Architecture Overview

```
WEBCAM
  ↓
MediaPipe FaceMesh
  (pretrained deep learning model — Google)
  (detects 468 3D face landmarks per frame)
  ↓
Head Pose Extraction
  (nose tip, chin, forehead coordinates → pitch angle)
  ↓
9 Kinematic Features
  (computed over a 10-second rolling window)
  ↓
StandardScaler  (z-score normalization)
  ↓
Our Trained ML Classifier
  (Random Forest / SVM / Logistic Regression)
  (trained on our own labeled nodding dataset)
  ↓
Nod Class: GENUINE / POLITE / DANGER / SURVIVAL
  ↓
Understanding Score (0–100%)
```

**Two separate AI components:**
| Component | What it is | Who trained it |
|---|---|---|
| MediaPipe FaceMesh | Pretrained deep neural network (CNN) | Google |
| Our Nodding Classifier | Lightweight ML model (RF/SVM/LR) | **Us** (on our data) |

---

## 📐 9 Kinematic Features

Every 10-second recording window produces these 9 numbers:

| # | Feature | What it measures |
|---|---|---|
| 1 | `head_pitch` | Average downward tilt angle of the head |
| 2 | `head_yaw` | Average side-to-side rotation |
| 3 | `movement_velocity` | How fast the head is moving frame-to-frame |
| 4 | `movement_acceleration` | Rate of change of velocity (jerkiness) |
| 5 | `nod_count` | How many complete nod cycles happened |
| 6 | `nod_frequency` | Nods per minute (normalized) |
| 7 | `nod_duration` | Average length of one nod cycle (ms) |
| 8 | `inter_nod_interval` | Average time gap between nods (ms) |
| 9 | `total_movement` | Total Euclidean distance traveled by nose landmark |

> **Key insight for judges:** Short `inter_nod_interval` + high `movement_velocity` + low `nod_duration` = panic agreement. Long intervals + low velocity = genuine understanding. The math is real.

---

## 🚀 Step-by-Step: How to Collect Training Data

### Step 1 – Start the local server
```powershell
# In the project root folder:
python run.py
# Then open: http://localhost:8000
```

### Step 2 – Go to the ML Pipeline section
Scroll down to **"ട്രെയിനിംഗ് ഡാറ്റാ കളക്ഷൻ"** on the website.

Make sure your **webcam is ON** (click "തലയാട്ടി നോക്കാം" first).

### Step 3 – Record each class
Click one of these buttons, then nod for 6 seconds:

| Button | How to nod | What it represents |
|---|---|---|
| 🟢 Record Genuine Nod | Slowly, 1–2 deliberate nods | "Yes, I actually understand" |
| 🟡 Record Polite Nod | 3–4 nods at normal social pace | "I'm listening but not sure" |
| 🔴 Record Danger Nod | Fast repeated nodding (5–7 nods) | "I have no idea, nodding to survive" |
| 💀 Record Survival Nod | Extremely rapid continuous nodding | "Please stop asking me questions" |

### Step 4 – Export the dataset
Click **"📤 Export Dataset (CSV)"** → save the file → move it here:
```powershell
copy "%USERPROFILE%\Downloads\nodding_dataset.csv" "training\nodding_dataset.csv"
```

---

## 📊 How Much Data Should You Collect?

| Situation | Minimum | Recommended |
|---|---|---|
| Quick prototype demo | 10 per class (40 total) | — |
| Hackathon with credible accuracy | **50 per class (200 total)** | 75–100/class |
| Real-world deployment | 100+ per class | 200+/class, multiple people |

### Tips for better data quality:
- ✅ Record from **multiple people** (different head sizes, distances)
- ✅ Try **different lighting** (bright room, dim room)
- ✅ Try **different distances** from camera (60 cm vs 120 cm)
- ✅ Mix **slow and fast** versions of each nod type
- ✅ Try different **sitting postures**
- ❌ Don't record 50 identical nods from the same session — variety matters

### Avoiding data leakage:
If all your recordings are from one person in one session, the model may learn your specific head movement style rather than the general nod pattern. Collect from at least 2–3 people if possible.

---

## 🔬 How to Train the Model

### Prerequisites
```powershell
pip install -r training/requirements.txt
```

### Run training
```powershell
python training/train_model.py
```

### What the script does:
1. **Loads** `nodding_dataset.csv` and validates it (checks for NaN, bad labels)
2. **Shows class distribution** — warns if any class has too few samples
3. **Stratified 80/20 split** — preserves class balance in train/test sets
4. **5-fold cross-validation** on all 3 candidate models:
   - 🌲 Random Forest (100 trees, balanced class weights)
   - 🔷 SVM with RBF kernel (probability-calibrated)
   - 📈 Logistic Regression (multinomial softmax)
5. **Selects the best model** by CV F1-score (macro)
6. **Lightweight hyperparameter tuning** (RandomizedSearchCV, 20 iterations only)
7. **Evaluates on the held-out test set** — data the model never saw during training
8. **Checks for overfitting** — compares training vs validation vs test accuracy
9. **Exports** `model_weights.json` for the browser to load

---

## 📈 Understanding the Evaluation Output

### Sample output:
```
CROSS-VALIDATION (5-fold Stratified)
  Model             |  CV Acc (mean±std)    |  CV F1 (mean±std)
  Logistic Regression  82.5% ± 6.2%           81.3% ± 7.1%
  Random Forest        88.0% ± 4.5%           87.2% ± 5.0%   ← WINNER
  SVM (RBF)            85.0% ± 5.8%           84.1% ± 6.2%

OVERFITTING CHECK
  Training accuracy   : 96.2%
  CV accuracy (mean)  : 88.0%
  Test accuracy       : 85.0%
  [OK] Gap of 11.2%. Slight overfitting, but manageable.

MODEL PERFORMANCE ON HELD-OUT TEST SET
  Accuracy     : 85.0%
  Precision    : 84.6%
  Recall       : 85.1%
  F1 Score     : 84.8%
```

### Detecting overfitting:
| Gap (Train − Test accuracy) | Interpretation | Fix |
|---|---|---|
| < 8% | ✅ Good | — |
| 8–15% | ⚠️ Slight overfitting | Collect more varied data |
| > 15% | 🚨 Overfitting | More data, more people, more variety |

> If training accuracy is 100% but test accuracy is 60%, the model memorized training samples. This commonly happens with the 40-sample seed dataset because it was synthetically generated — all samples from the same class are nearly identical.

---

## 💾 Model Export Format

The script writes `model_weights.json` which the website loads directly:

```json
{
  "type": "Random Forest (trained by train_model.py v2.0)",
  "weights": [[...], [...], [...], [...]],  // LR coefficients for JS inference
  "biases": [...],
  "scaler": { "mean": [...], "std": [...] },
  "classes": ["GENUINE", "POLITE", "DANGER", "SURVIVAL"],
  "featureNames": ["head_pitch", "head_yaw", ...],
  "accuracy": 85.0,
  "precision": 84.6,
  "recall": 85.1,
  "f1": 84.8,
  "trainedSamples": 200,
  "cvResults": { ... }
}
```

> **Note:** When Random Forest or SVM is selected as the best model, a Logistic Regression approximation is exported for browser inference (since RF/SVM can't be easily run in vanilla JS without a 1 MB+ library). The Python training still uses the superior model; the browser gets the best practical approximation.

---

## 🌐 Loading the Model in the Website

1. Run `python training/train_model.py` — this generates `model_weights.json`
2. Open `http://localhost:8000`
3. Scroll to **"MACHINE LEARNING PIPELINE"**
4. Click **`[ ⚙️ Load Python Model (JSON) ]`**
5. Select `model_weights.json`
6. Set **CLASSIFIER ENGINE** → **"Trained ML Model"**
7. The dashboard now uses your trained weights for live prediction! ✅

---

## 🚀 Deploying to Vercel

The entire project is a static website — no server needed after training.

```bash
# 1. Install Vercel CLI (once)
npm install -g vercel

# 2. From project root:
vercel

# Follow prompts:
#   Framework: Other
#   Output directory: ./  (the project root)
#   No build step needed
```

**Before deploying**, make sure `model_weights.json` is in the project root. The website will auto-load it from the same directory.

> **Privacy note:** The website is fully client-side. No webcam footage is uploaded anywhere. Only the numerical feature CSV is exported when you explicitly click Export Dataset.

---

## 🎤 HOW TO EXPLAIN THIS TO JUDGES

### The one-sentence pitch:
> *"We use Google's pretrained MediaPipe face-landmark model for visual perception, extract 9 numerical head-movement features from it, and then classify nodding behavior using our own lightweight machine-learning model trained on labeled examples we recorded ourselves."*

### Why this is technically credible:
- **Two-stage AI pipeline**: pretrained CV model (Google MediaPipe) + our own trained classifier
- **Real feature engineering**: 9 kinematic features, not raw pixels
- **Multi-model comparison**: we tested Random Forest, SVM, and Logistic Regression
- **Proper evaluation**: cross-validation, overfitting check, confusion matrix
- **Our own labeled dataset**: we recorded it ourselves, it's not from the internet

### Why we didn't use a deep neural network:
> *"Our problem is tabular time-series classification on 9 features — not image recognition. A lightweight classifier (RF/SVM/LR) on engineered features trains in seconds, needs far less data, is interpretable, and runs on a CPU. A CNN or transformer would be computational overkill for this problem and would require 10,000+ samples instead of 200."*

### If judges ask "is this really AI?":
> *"Yes. MediaPipe is a production-grade deep learning model from Google. Our trained classifier is supervised ML — it learned the mapping from movement features to nod categories from labeled examples. The system literally gets better as you feed it more data."*

### If judges ask about accuracy:
> *"Our evaluation is honest. We show training accuracy vs validation accuracy vs test accuracy to detect overfitting. We report Precision, Recall, and F1, not just accuracy. With the current dataset of X samples, we achieved Y% on held-out test data. We could improve this by collecting more recordings from multiple people."*

### The Kerala angle:
> *"'Ariyathe Thalayattiyath' (അറിയാതെ തലയാട്ടിയത്) means 'nodding without understanding' — a universally relatable human behavior. We turned this joke into a genuinely functional AI system."*

---

## ⚡ Quick Reference Commands

```powershell
# Install dependencies
pip install -r training/requirements.txt

# Start local server
python run.py

# Run training (from project root)
python training/train_model.py

# Deploy to Vercel
vercel
```

---

*Built with 🥥 for TinkerHub Useless Projects | Absolutely no scientific claim about reading minds*
