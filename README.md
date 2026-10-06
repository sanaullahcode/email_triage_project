# Email Triage Classifier Prototype (Spam / Lead / Support)

**Task:** Week 2 Task Assignment — SafeX
**Prepared by:** Sana Ullah
**Group Leader:** Shahid Mumtaz
**Difficulty:** Intermediate | **Estimated Time:** 4-6 days

## Objective

Build a prototype that classifies incoming mock emails into one of three categories — `spam`, `lead`, or `support` — so that an inbox can be triaged automatically before reaching a human.

## Approach

1. **Dataset:** 36 labeled email subject + body pairs, hand-written to represent realistic patterns across all three categories (`data/emails.csv`).
2. **Feature extraction:** TF-IDF vectorization with unigrams and bigrams, English stop words removed.
3. **Model selection:** Three candidate classifiers were compared using 4-fold stratified cross-validation on the training split — Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
4. **Evaluation:** The best-performing candidate was retrained on the full training set and evaluated once on a held-out test set (25 percent of the data, stratified).
5. **Sanity check:** The final model was tested against three unseen, hand-written emails to confirm it generalizes past the training text.

## Results

| Metric | Value |
|---|---|
| Selected model | Naive Bayes (TF-IDF + Multinomial NB) |
| Cross-validation accuracy (training set) | 0.815 |
| Held-out test accuracy | 1.000 |
| Held-out test macro F1 | 1.000 |

Full metrics, per-class precision and recall, and the confusion matrix are in `outputs/accuracy_report.txt` and `outputs/confusion_matrix.png`.

**Important note on accuracy:** the 100 percent test accuracy reflects a very small held-out set of 9 emails and a dataset that is intentionally clean and well-separated between classes. It demonstrates that the pipeline works correctly end to end, not that the model is production-ready. Section 9 of the notebook covers what would be needed to validate this at real-world scale.

## What additional data would improve accuracy

- Several hundred to a few thousand real or realistically anonymized emails per class instead of 36 total.
- More borderline and ambiguous examples where categories overlap in tone.
- Real-world noise such as signatures, forwarded threads, quoted replies, and multiple languages.
- Adversarial spam variants that use obfuscation to evade filters.
- Periodic retraining to account for drift in spam tactics and business language over time.
- A feedback loop where misclassified emails are corrected and fed back into training.

## Project structure

```
email_triage/
├── data/
│   └── emails.csv                     # labeled dataset (36 samples)
├── outputs/
│   ├── accuracy_report.txt            # full accuracy report
│   ├── results_summary.json           # machine-readable results
│   └── confusion_matrix.png           # confusion matrix visualization
├── train_classifier.py                # standalone training/evaluation script
├── Email_Triage_Classifier.ipynb      # notebook version with narrative
└── README.md
```

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib
python train_classifier.py
```

## Tools and techniques

Python, scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM), pandas, matplotlib.

## Team

| Role | Name |
|---|---|
| Group Leader | Shahid Mumtaz |
| Contributor | Sana Ullah |
