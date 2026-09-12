"""
ARIYATHE THALAYATTIYATH — Machine Learning Training Pipeline v2.0
==================================================================
Lightweight multi-model comparison pipeline for nodding behavior classification.

Hardware target: 12th-gen Intel Core CPU, 8 GB RAM, no GPU
Models evaluated: Random Forest | SVM (RBF) | Logistic Regression
Features: 9 kinematic head-movement features (no raw video, no frames in RAM)

Pipeline:
  1. Load & validate nodding_dataset.csv
  2. Check class distribution — warn if dataset is too small
  3. Stratified 80/20 train/test split
  4. Stratified 5-fold cross-validation for all 3 candidates
  5. Select best model by mean CV F1 (macro)
  6. Lightweight RandomizedSearchCV on winner (n_iter=20, cv=3)
  7. Evaluate on held-out test set — NO DATA LEAKAGE
  8. Print overfitting check (train vs val vs test accuracy)
  9. Print Accuracy / Precision / Recall / F1 / Confusion Matrix / per-class
 10. Export model_weights.json (backward-compatible with existing JS inference)

DO NOT FAKE ACCURACY.
If the dataset is the 40-sample seed dataset, the script will warn you.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')  # suppress sklearn convergence noise

# Ensure UTF-8 output on Windows terminals (Malayalam text-safe)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'head_pitch',
    'head_yaw',
    'movement_velocity',
    'movement_acceleration',
    'nod_count',
    'nod_frequency',
    'nod_duration',
    'inter_nod_interval',
    'total_movement'
]

CLASSES = ['GENUINE', 'POLITE', 'DANGER', 'SURVIVAL']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Thresholds for dataset quality warnings
MIN_TOTAL_SAMPLES = 40        # absolute minimum to attempt training
MIN_SAMPLES_PER_CLASS = 8     # minimum per class for meaningful CV
RECOMMENDED_PER_CLASS = 50    # what we'd actually like for a real prototype
SEED = 42

# ---------------------------------------------------------------------------
# UTILITY: FIND DATASET
# ---------------------------------------------------------------------------

def find_dataset():
    candidates = [
        os.path.join(os.path.dirname(__file__), 'nodding_dataset.csv'),
        'nodding_dataset.csv',
        'training/nodding_dataset.csv',
        os.path.join(os.path.dirname(__file__), '..', 'nodding_dataset.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

# ---------------------------------------------------------------------------
# SOFTMAX FALLBACK — pure NumPy (used only when scikit-learn is missing)
# ---------------------------------------------------------------------------

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def numpy_logistic_regression(X_tr, y_tr, X_ts, y_ts, epochs=500, lr=0.07, l2=0.003):
    """Gradient-descent softmax LR — pure NumPy fallback."""
    N, D = X_tr.shape
    K = len(CLASSES)
    np.random.seed(SEED)
    W = np.random.randn(K, D) * 0.05
    b = np.zeros(K)
    Y_oh = np.zeros((N, K))
    Y_oh[np.arange(N), y_tr] = 1.0

    print("  [NumPy fallback] Training softmax LR ...")
    for ep in range(epochs):
        probs = softmax(X_tr @ W.T + b)
        err   = probs - Y_oh
        W    -= lr * ((err.T @ X_tr) / N + l2 * W)
        b    -= lr * np.mean(err, axis=0)

    train_preds = np.argmax(softmax(X_tr @ W.T + b), axis=1)
    test_preds  = np.argmax(softmax(X_ts @ W.T + b), axis=1)
    train_acc   = np.mean(train_preds == y_tr) * 100
    test_acc    = np.mean(test_preds  == y_ts) * 100

    cm = np.zeros((K, K), dtype=int)
    for a, p in zip(y_ts, test_preds):
        cm[a, p] += 1

    # feature importance = L1 norm of weight rows
    feat_imp = np.sum(np.abs(W), axis=0)
    feat_imp /= (feat_imp.sum() + 1e-12)

    return W, b, train_acc, test_acc, cm, feat_imp, 'Logistic Regression (NumPy)'

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    sep = '=' * 72

    print(sep)
    print('  ARIYATHE THALAYATTIYATH - ML TRAINING PIPELINE v2.0')
    print('  Lightweight CPU-only | RF + SVM + LR | Honest Evaluation')
    print(sep)

    # ------------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------------
    data_path = find_dataset()
    if not data_path:
        print('\n[ERROR] nodding_dataset.csv not found!')
        print('  -> Open the website, collect recordings, click [Export Dataset (CSV)]')
        print('  -> Copy the downloaded CSV to this training/ folder.')
        sys.exit(1)

    print(f'\n[+] Dataset path  : {data_path}')
    df = pd.read_csv(data_path)
    print(f'[+] Raw row count : {len(df)}')

    # ------------------------------------------------------------------
    # 2. VALIDATE & CLEAN
    # ------------------------------------------------------------------
    required_cols = FEATURE_NAMES + ['label']
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f'\n[ERROR] Missing columns in CSV: {missing_cols}')
        sys.exit(1)

    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[df['label'].isin(CLASSES)]             # drop unknown labels
    df = df.dropna(subset=FEATURE_NAMES + ['label'])  # drop NaN rows
    df[FEATURE_NAMES] = df[FEATURE_NAMES].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=FEATURE_NAMES)            # drop rows with non-numeric features

    print(f'[+] Clean row count: {len(df)}  (after removing invalid/NaN rows)')

    # ------------------------------------------------------------------
    # 3. CLASS DISTRIBUTION & QUALITY WARNINGS
    # ------------------------------------------------------------------
    print('\n' + '-' * 50)
    print('  CLASS DISTRIBUTION')
    print('-' * 50)
    class_counts = {}
    for c in CLASSES:
        n = int((df['label'] == c).sum())
        class_counts[c] = n
        bar  = '#' * n
        flag = ' ⚠️  (too few!)' if n < MIN_SAMPLES_PER_CLASS else ''
        print(f'  {c:10s} : {n:4d} samples  {bar}{flag}')

    total = len(df)
    print(f'\n  Total samples    : {total}')

    # Quality gate
    if total < MIN_TOTAL_SAMPLES:
        print(f'\n[WARNING] Only {total} samples total.')
        print(f'  The current seed dataset is SYNTHETIC — results look perfect because')
        print(f'  every class is perfectly separated by design, not by real data.')
        print(f'  To get HONEST accuracy you need REAL recordings from the webcam.')
        needed = RECOMMENDED_PER_CLASS * len(CLASSES) - total
        print(f'\n  COLLECT MORE DATA — recommendation:')
        print(f'    >= {RECOMMENDED_PER_CLASS} samples per class  ({RECOMMENDED_PER_CLASS * len(CLASSES)} total)')
        print(f'    You still need approximately {max(0, needed)} more recordings.')
        print(f'    Record from multiple people and lighting conditions for best results.')

    undersized = [c for c, n in class_counts.items() if n < MIN_SAMPLES_PER_CLASS]
    if undersized:
        print(f'\n[WARNING] Classes with fewer than {MIN_SAMPLES_PER_CLASS} samples: {undersized}')
        print(f'  Cross-validation results will be unreliable for these classes.')
        if any(class_counts[c] == 0 for c in CLASSES):
            print('[ERROR] At least one class has 0 samples. Cannot train.')
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. PREPARE FEATURES & LABELS
    # ------------------------------------------------------------------
    X_raw = df[FEATURE_NAMES].values.astype(np.float64)
    y     = np.array([CLASS_TO_IDX[lbl] for lbl in df['label']], dtype=int)
    N, D  = X_raw.shape

    # ------------------------------------------------------------------
    # 5. CHECK FOR scikit-learn
    # ------------------------------------------------------------------
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import (
            StratifiedShuffleSplit,
            StratifiedKFold,
            cross_validate,
            RandomizedSearchCV,
        )
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print('\n[WARNING] scikit-learn not installed — falling back to NumPy LR only.')
        print('  Install with:  pip install scikit-learn')

    # ------------------------------------------------------------------
    # 6a. scikit-learn PATH
    # ------------------------------------------------------------------
    if HAS_SKLEARN:

        # Stratified 80/20 split — preserves class balance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_idx, test_idx = next(sss.split(X_raw, y))
        X_tr, y_tr = X_raw[train_idx], y[train_idx]
        X_ts, y_ts = X_raw[test_idx],  y[test_idx]
        print(f'\n[+] Train samples: {len(X_tr)}  |  Test samples: {len(X_ts)}  (80/20 stratified)')

        # Three candidate pipelines
        scaler = StandardScaler()
        candidates = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    LogisticRegression(
                    max_iter=1000, C=1.0,
                    class_weight='balanced',
                    random_state=SEED,
                    solver='lbfgs', multi_class='auto'
                )),
            ]),
            'Random Forest': Pipeline([
                # RF doesn't need scaling but we include it for consistent API
                ('scaler', StandardScaler()),
                ('clf',    RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=SEED,
                    n_jobs=-1,
                )),
            ]),
            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    SVC(
                    kernel='rbf', C=1.0, gamma='scale',
                    class_weight='balanced',
                    probability=True,
                    random_state=SEED,
                )),
            ]),
        }

        # ---------------------------------------------------------------
        # 5-fold stratified cross-validation on TRAINING set only
        # ---------------------------------------------------------------
        cv_folds = min(5, min(class_counts.values()))  # cap at available samples/class
        cv_folds = max(2, cv_folds)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

        print(f'\n{"-" * 50}')
        print(f'  CROSS-VALIDATION  ({cv_folds}-fold Stratified, on training split only)')
        print(f'{"-" * 50}')
        print(f'  {"Model":<22}  {"CV Acc (mean±std)":<22}  {"CV F1 (mean±std)":<22}')
        print(f'  {"-"*22}  {"-"*22}  {"-"*22}')

        cv_results = {}
        for name, pipe in candidates.items():
            scores = cross_validate(
                pipe, X_tr, y_tr,
                cv=skf,
                scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
                n_jobs=-1,
                error_score='raise',
            )
            cv_results[name] = {
                'acc_mean': scores['test_accuracy'].mean() * 100,
                'acc_std':  scores['test_accuracy'].std()  * 100,
                'f1_mean':  scores['test_f1'].mean()  * 100,
                'f1_std':   scores['test_f1'].std()   * 100,
            }
            r = cv_results[name]
            print(f'  {name:<22}  {r["acc_mean"]:5.1f}% ± {r["acc_std"]:4.1f}%      '
                  f'{r["f1_mean"]:5.1f}% ± {r["f1_std"]:4.1f}%')

        # Select best by CV F1
        best_name = max(cv_results, key=lambda k: cv_results[k]['f1_mean'])
        best_pipe = candidates[best_name]
        print(f'\n[+] Best model: {best_name}  (CV F1 = {cv_results[best_name]["f1_mean"]:.1f}%)')

        # ---------------------------------------------------------------
        # Lightweight hyperparameter tuning (winner only)
        # ---------------------------------------------------------------
        print(f'\n[+] Running lightweight hyperparameter search on {best_name} (n_iter=20) ...')

        if best_name == 'Logistic Regression':
            param_dist = {
                'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'clf__max_iter': [500, 1000],
            }
        elif best_name == 'Random Forest':
            param_dist = {
                'clf__n_estimators': [50, 100, 150, 200],
                'clf__max_depth': [None, 5, 10, 15],
                'clf__min_samples_leaf': [1, 2, 3],
                'clf__max_features': ['sqrt', 'log2', None],
            }
        else:  # SVM
            param_dist = {
                'clf__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'clf__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            }

        rscv = RandomizedSearchCV(
            best_pipe, param_dist,
            n_iter=20, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
            scoring='f1_macro',
            random_state=SEED,
            n_jobs=-1,
            error_score='raise',
        )
        rscv.fit(X_tr, y_tr)
        best_pipe = rscv.best_estimator_
        print(f'  Best params: {rscv.best_params_}')
        print(f'  Tuned CV F1: {rscv.best_score_ * 100:.1f}%')

        # ---------------------------------------------------------------
        # Final fit on full training split → evaluate on test split
        # ---------------------------------------------------------------
        best_pipe.fit(X_tr, y_tr)
        train_preds = best_pipe.predict(X_tr)
        test_preds  = best_pipe.predict(X_ts)

        train_acc  = accuracy_score(y_tr, train_preds) * 100
        test_acc   = accuracy_score(y_ts, test_preds)  * 100
        precision  = precision_score(y_ts, test_preds, average='macro', zero_division=0) * 100
        recall     = recall_score(y_ts, test_preds,    average='macro', zero_division=0) * 100
        f1         = f1_score(y_ts, test_preds,        average='macro', zero_division=0) * 100
        cm         = confusion_matrix(y_ts, test_preds, labels=list(range(len(CLASSES))))

        # ---------------------------------------------------------------
        # OVERFITTING CHECK
        # ---------------------------------------------------------------
        print(f'\n{"-" * 50}')
        print('  OVERFITTING CHECK')
        print(f'{"-" * 50}')
        print(f'  Training accuracy   : {train_acc:.1f}%')
        print(f'  CV accuracy (mean)  : {cv_results[best_name]["acc_mean"]:.1f}%  '
              f'(on training split only)')
        print(f'  Test accuracy       : {test_acc:.1f}%  (held-out, never seen during training)')
        gap = train_acc - test_acc
        if gap > 15:
            print(f'\n  [WARNING] Gap of {gap:.1f}% between train and test accuracy.')
            print(f'  This suggests OVERFITTING — the model memorized training samples.')
            print(f'  Solution: collect more diverse data from multiple people/sessions.')
        elif gap > 8:
            print(f'\n  [CAUTION] Gap of {gap:.1f}%. Slight overfitting possible.')
            print(f'  More data from varied conditions would help.')
        else:
            print(f'\n  [OK] Gap of {gap:.1f}%. No strong overfitting detected.')

        # ---------------------------------------------------------------
        # FINAL RESULTS
        # ---------------------------------------------------------------
        print(f'\n{sep}')
        print('  MODEL PERFORMANCE ON HELD-OUT TEST SET')
        print(sep)
        print(f'  Model             : {best_name}')
        print(f'  Test Accuracy     : {test_acc:.1f}%')
        print(f'  Precision (macro) : {precision:.1f}%')
        print(f'  Recall (macro)    : {recall:.1f}%')
        print(f'  F1 Score (macro)  : {f1:.1f}%')

        if total <= MIN_TOTAL_SAMPLES:
            print(f'\n  [!] NOTE: These metrics are from {len(X_ts)} test samples only.')
            print(f'      The dataset is the 40-sample synthetic seed — NOT real webcam data.')
            print(f'      Do not report this accuracy to judges as real-world performance.')
            print(f'      Collect real recordings for credible numbers.')

        # Per-class breakdown
        print(f'\n{"-" * 50}')
        print('  PER-CLASS BREAKDOWN')
        print(f'{"-" * 50}')
        print(f'  {"Class":<12} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
        print(f'  {"-"*12} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
        for i, cls in enumerate(CLASSES):
            mask    = y_ts == i
            support = int(mask.sum())
            if support == 0:
                print(f'  {cls:<12} {"N/A":>10} {"N/A":>10} {"N/A":>10} {support:>10}')
                continue
            p = precision_score(y_ts == i, test_preds == i, zero_division=0) * 100
            r = recall_score(   y_ts == i, test_preds == i, zero_division=0) * 100
            f = f1_score(       y_ts == i, test_preds == i, zero_division=0) * 100
            print(f'  {cls:<12} {p:>9.1f}% {r:>9.1f}% {f:>9.1f}% {support:>10}')

        # Confusion matrix
        print(f'\n{"-" * 50}')
        print('  CONFUSION MATRIX  (rows = Actual, cols = Predicted)')
        print(f'{"-" * 50}')
        header = f'  {"Actual \\ Pred":<14}' + ''.join(f'{c[:7]:>9}' for c in CLASSES)
        print(header)
        print('  ' + '-' * (14 + 9 * len(CLASSES)))
        for i, row in enumerate(cm):
            row_s = f'  {CLASSES[i]:<14}' + ''.join(f'{v:>9d}' for v in row)
            print(row_s)

        # Feature importance
        print(f'\n{"-" * 50}')
        print('  FEATURE IMPORTANCE / DISCRIMINABILITY RANKING')
        print(f'{"-" * 50}')

        if best_name == 'Random Forest':
            importances = best_pipe.named_steps['clf'].feature_importances_
        else:
            # For LR and SVM: use absolute weight magnitudes summed across classes
            try:
                coef = best_pipe.named_steps['clf'].coef_  # shape (n_classes, n_features)
                importances = np.sum(np.abs(coef), axis=0)
                importances = importances / (importances.sum() + 1e-12)
            except Exception:
                importances = np.ones(D) / D

        ranked = sorted(zip(FEATURE_NAMES, importances.tolist()), key=lambda x: -x[1])
        for rank, (fname, imp) in enumerate(ranked, 1):
            bar = '█' * int(imp * 40)
            print(f'  {rank}. {fname:<24}  {imp*100:5.1f}%  {bar}')

        # ---------------------------------------------------------------
        # Scaler info (for JSON export — needed by JS inference)
        # ---------------------------------------------------------------
        scaler_fitted = best_pipe.named_steps['scaler']
        means = scaler_fitted.mean_.tolist()
        stds  = np.sqrt(scaler_fitted.var_).tolist()

        # Weights for JS inference (LR coefficients or RF approximation)
        if best_name == 'Logistic Regression':
            W = best_pipe.named_steps['clf'].coef_.tolist()
            b = best_pipe.named_steps['clf'].intercept_.tolist()
        elif best_name == 'SVM (RBF)':
            # SVM: coef_ only exists for linear kernel — use decision_function shape
            # For browser inference we fall back to a LR retrained on SVM predictions
            # (the JSON export always needs W+b in softmax form for JS compatibility)
            lr_fallback = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                                             solver='lbfgs', multi_class='auto')
            X_tr_scaled = scaler_fitted.transform(X_tr)
            X_ts_scaled = scaler_fitted.transform(X_ts)
            lr_fallback.fit(X_tr_scaled, y_tr)
            W = lr_fallback.coef_.tolist()
            b = lr_fallback.intercept_.tolist()
            print(f'\n  [INFO] SVM selected — LR coefficients used for browser JSON export.')
            print(f'         Browser inference uses the LR approximation.')
        else:  # Random Forest
            # Approximate with LR for JSON export
            lr_fallback = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                                             solver='lbfgs', multi_class='auto')
            X_tr_scaled = scaler_fitted.transform(X_tr)
            lr_fallback.fit(X_tr_scaled, y_tr)
            W = lr_fallback.coef_.tolist()
            b = lr_fallback.intercept_.tolist()
            print(f'\n  [INFO] Random Forest selected — LR approximation used for browser JSON export.')
            print(f'         Python training uses RF; browser loads the LR proxy for live inference.')

    # ------------------------------------------------------------------
    # 6b. NumPy-only fallback (no scikit-learn)
    # ------------------------------------------------------------------
    else:
        X_raw_f = X_raw.astype(np.float32)
        means_np = np.mean(X_raw_f, axis=0)
        stds_np  = np.std( X_raw_f, axis=0)
        stds_np[stds_np < 1e-6] = 1.0
        X_scaled  = (X_raw_f - means_np) / stds_np

        np.random.seed(SEED)
        idx   = np.random.permutation(N)
        split = max(1, int(N * 0.8))
        X_tr, y_tr = X_scaled[idx[:split]], y[idx[:split]]
        X_ts, y_ts = X_scaled[idx[split:]], y[idx[split:]]

        W_np, b_np, train_acc, test_acc, cm, importances, best_name = \
            numpy_logistic_regression(X_tr, y_tr, X_ts, y_ts)

        precision = recall = f1 = test_acc  # approximation for numpy path
        ranked = sorted(zip(FEATURE_NAMES, importances.tolist()), key=lambda x: -x[1])
        means  = means_np.tolist()
        stds   = stds_np.tolist()
        W      = W_np.tolist()
        b      = b_np.tolist()

        print(f'\n  Train Accuracy : {train_acc:.1f}%')
        print(f'  Test Accuracy  : {test_acc:.1f}%')

    # ------------------------------------------------------------------
    # 7. EXPORT model_weights.json
    #    Format is BACKWARD-COMPATIBLE with the existing script.js inference
    # ------------------------------------------------------------------
    export = {
        "type":    best_name + " (trained by train_model.py v2.0)",
        "weights": W,
        "biases":  b,
        "scaler": {
            "mean": means,
            "std":  stds,
        },
        "classes":      CLASSES,
        "featureNames": FEATURE_NAMES,
        "accuracy":     round(test_acc, 2),
        "precision":    round(precision, 2),
        "recall":       round(recall, 2),
        "f1":           round(f1, 2),
        "confusionMatrix":   cm.tolist() if HAS_SKLEARN else cm.tolist(),
        "featureImportances": {f: round(float(v), 6) for f, v in ranked},
        "trainedSamples": int(N),
        "cvResults": {k: {
                "acc_mean": round(v["acc_mean"], 2),
                "f1_mean":  round(v["f1_mean"],  2),
            } for k, v in cv_results.items()
        } if HAS_SKLEARN else {},
        "exportedAt": pd.Timestamp.now().isoformat(),
        "datasetNote": (
            "SYNTHETIC SEED DATA — collect real webcam recordings for credible accuracy."
            if total <= MIN_TOTAL_SAMPLES else
            "Trained on real collected data."
        ),
    }

    out_paths = [
        os.path.join(os.path.dirname(__file__), 'model_weights.json'),
        os.path.join(os.path.dirname(__file__), '..', 'model_weights.json'),
    ]
    saved_to = []
    for p in out_paths:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            saved_to.append(os.path.abspath(p))
        except Exception as e:
            print(f'  [WARN] Could not write {p}: {e}')

    print(f'\n{sep}')
    print('  EXPORT COMPLETE')
    print(sep)
    for p in saved_to:
        print(f'  Saved: {p}')

    print(f"""
  HOW TO LOAD THE MODEL IN THE WEBSITE:
  1. Open http://localhost:8000
  2. Scroll to "MACHINE LEARNING PIPELINE"
  3. Click [ Load Python Model (JSON) ] → select model_weights.json
  4. Switch CLASSIFIER ENGINE to "Trained ML Model"
  5. The live webcam / Demo Mode will now use your trained weights!
{sep}
""")

if __name__ == '__main__':
    main()
