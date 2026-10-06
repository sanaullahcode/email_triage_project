"""
Email Triage Classifier Prototype (Spam / Lead / Support)
Project: SafeX Week 2 Task Assignment
Prepared by: Sana Ullah
Group Leader: Shahid Mumtaz

This script trains a text classification model to route incoming emails
into three categories: spam, lead, support.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/emails.csv")
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

print(f"Total samples: {len(df)}")
print(df["label"].value_counts())

X = df["text"]
y = df["label"]

# ---------------------------------------------------------------------------
# 2. Held-out train/test split (stratified to keep class balance)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 3. Candidate models
# ---------------------------------------------------------------------------
candidates = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", MultinomialNB(alpha=0.3)),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ]),
}

# ---------------------------------------------------------------------------
# 4. Cross-validation on training set to pick the best model
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

print("\n--- Cross-validation (on training set) ---")
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:22s}  mean acc: {scores.mean():.3f}  (+/- {scores.std():.3f})")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_model = candidates[best_name]
print(f"\nSelected model based on CV: {best_name}")

# ---------------------------------------------------------------------------
# 5. Fit best model on training set, evaluate on held-out test set
# ---------------------------------------------------------------------------
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, digits=3)

print(f"\n--- Held-out Test Set Performance ({best_name}) ---")
print(f"Accuracy: {test_accuracy:.3f}")
print(f"Macro F1: {test_f1_macro:.3f}\n")
print(report)

# ---------------------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------------------
labels_order = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)

fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name}\nTest Accuracy: {test_accuracy:.1%}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Sample predictions on new, unseen emails (sanity check)
# ---------------------------------------------------------------------------
sample_emails = [
    "Claim your free lottery winnings now by clicking this link",
    "We would like to book a demo call to evaluate your platform for our team of 30",
    "My invoice this month is wrong please help me fix the billing",
]
sample_preds = best_model.predict(sample_emails)

print("\n--- Sample Predictions (unseen inputs) ---")
for text, pred in zip(sample_emails, sample_preds):
    print(f"[{pred:8s}] {text}")

# ---------------------------------------------------------------------------
# 8. Save accuracy report as JSON + text for documentation
# ---------------------------------------------------------------------------
results_summary = {
    "project": "Email Triage Classifier Prototype (Spam / Lead / Support)",
    "prepared_by": "Sana Ullah",
    "group_leader": "Shahid Mumtaz",
    "dataset_size": len(df),
    "class_distribution": df["label"].value_counts().to_dict(),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "cv_results": {k: {"mean_accuracy": float(v.mean()), "std": float(v.std())} for k, v in cv_results.items()},
    "selected_model": best_name,
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_f1_macro),
    "sample_predictions": [
        {"text": t, "predicted_label": p} for t, p in zip(sample_emails, sample_preds)
    ],
}

with open("outputs/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

with open("outputs/accuracy_report.txt", "w") as f:
    f.write("EMAIL TRIAGE CLASSIFIER - ACCURACY REPORT\n")
    f.write("=" * 55 + "\n\n")
    f.write("Project: Email Triage Classifier Prototype (Spam / Lead / Support)\n")
    f.write("Prepared by: Sana Ullah\n")
    f.write("Group Leader: Shahid Mumtaz\n\n")
    f.write(f"Dataset size: {len(df)} labeled emails\n")
    f.write(f"Class distribution: {df['label'].value_counts().to_dict()}\n")
    f.write(f"Train / Test split: {len(X_train)} / {len(X_test)} (stratified, 75/25)\n\n")
    f.write("Cross-validation results (4-fold, on training data):\n")
    for k, v in cv_results.items():
        f.write(f"  - {k:22s} mean accuracy: {v.mean():.3f} (+/- {v.std():.3f})\n")
    f.write(f"\nSelected model: {best_name}\n\n")
    f.write("Held-out test set performance:\n")
    f.write(f"  - Accuracy: {test_accuracy:.3f}\n")
    f.write(f"  - Macro F1: {test_f1_macro:.3f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nSample predictions on unseen emails:\n")
    for t, p in zip(sample_emails, sample_preds):
        f.write(f"  [{p}] {t}\n")

print("\nSaved: outputs/confusion_matrix.png")
print("Saved: outputs/results_summary.json")
print("Saved: outputs/accuracy_report.txt")
