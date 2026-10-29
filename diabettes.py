"""
Project 4 (ML): Diabetes Prediction
Predicts disease progression from 10 baseline health measurements.

REAL DATA: uses scikit-learn's bundled diabetes dataset (442 real patients,
Efron et al. 2004) — no download needed. The target is a quantitative
measure of disease progression one year after baseline.

Because the built-in target is continuous, this script does BOTH:
  1. Regression — predict the actual progression score.
  2. Classification — predict "high risk" (above median) vs "low risk",
     which is often what a screening tool actually needs.

Run: pip install scikit-learn numpy
     python 04_diabetes_prediction.py

Concepts practiced: turning a regression target into a classification
problem (thresholding), when each framing is appropriate, ROC-AUC for
a medical screening context.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, classification_report


def main():
    print("=== Diabetes Prediction (real sklearn dataset) ===")
    data = load_diabetes()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Features: {', '.join(data.feature_names)}")

    # ---------- Part 1: Regression ----------
    print("\n----- Part 1: Regression (predict progression score) -----")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = LinearRegression()
    reg.fit(X_train_s, y_train)
    r2 = r2_score(y_test, reg.predict(X_test_s))
    print(f"Linear Regression R2: {r2:.4f}")
    print("(Diabetes progression is genuinely hard to predict from baseline")
    print(" alone, so an R2 around 0.4-0.5 is expected and realistic here.)")

    # ---------- Part 2: Classification ----------
    print("\n----- Part 2: Classification (high vs low risk) -----")
    # Turn the continuous target into a binary label at the median.
    threshold = np.median(y)
    y_binary = (y > threshold).astype(int)

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xb_train, yb_train)

    predictions = clf.predict(Xb_test)
    probabilities = clf.predict_proba(Xb_test)[:, 1]

    acc = accuracy_score(yb_test, predictions)
    auc = roc_auc_score(yb_test, probabilities)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}  (0.5 = random, 1.0 = perfect)")
    print("\nClassification report:")
    print(classification_report(yb_test, predictions, target_names=["Low risk", "High risk"]))


if __name__ == "__main__":
    main()