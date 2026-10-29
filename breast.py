"""
Project 5 (ML): Breast Cancer Detection
Classifies tumors as malignant or benign from 30 features computed from
digitized images of fine-needle-aspirate biopsies.

REAL DATA: scikit-learn's bundled Wisconsin Breast Cancer dataset (569
real cases) — no download needed.

Important medical-ML point baked into this script: in cancer screening,
a FALSE NEGATIVE (missing a real malignancy) is far more dangerous than a
false positive. So we don't just report accuracy — we look at RECALL on the
malignant class specifically, and show the confusion matrix so those
errors are visible.

Run: pip install scikit-learn numpy
     python 05_breast_cancer_detection.py

Concepts practiced: why accuracy alone is misleading in medical contexts,
precision/recall tradeoff, confusion matrix, ROC-AUC.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)


def main():
<<<<<<< HEAD
    print("=====Breast Cancer Detection (real Wisconsin dataset) ===")
=======
    
>>>>>>> fix1
    data = load_breast_cancer()
    X, y = data.data, data.target
    # In this dataset: 0 = malignant, 1 = benign.
    print(f"Dataset: {X.shape[0]} cases, {X.shape[1]} features")
    print(f"Class balance: {np.sum(y==1)} benign, {np.sum(y==0)} malignant")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_s, y_train)

    predictions = model.predict(X_test_s)
    probabilities = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(f"                  Predicted malignant  Predicted benign")
    print(f"Actual malignant        {cm[0][0]:3d}                {cm[0][1]:3d}")
    print(f"Actual benign           {cm[1][0]:3d}                {cm[1][1]:3d}")

    missed_malignant = cm[0][1]
    print(f"\n>>> Missed malignancies (false negatives): {missed_malignant}")
    print("    This is the number that matters most in a screening tool —")
    print("    each one is a real cancer the model called benign.")

    print("\nFull report:")
    print(classification_report(y_test, predictions, target_names=["Malignant", "Benign"]))


if __name__ == "__main__":
    main()