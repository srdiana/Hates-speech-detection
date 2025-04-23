# Hates-speech-detection
Solving hate speech detection problem using BiLSTM with attention layer

Paper about GloVe https://nlp.stanford.edu/pubs/glove.pdf

---

###  Overview of Results

#### 1. **Logistic Regression (Unbalanced Data)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.62      | 0.55   | 0.58     | 2,921   |
| 1     | 0.45      | 0.15   | 0.22     | 3,780   |
| 2     | 0.77      | 0.94   | 0.85     | 13,050  |

- **Accuracy**: 73%
- **Macro Avg F1-score**: 0.55

>  **Class 1** is heavily underperforming, especially in recall. This suggests the model rarely predicts it correctly, likely due to the imbalance.

---

#### 2. **Balanced Logistic Regression (with `class_weight='balanced'`)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.50      | 0.72   | 0.59     | 2,921   |
| 1     | 0.35      | 0.49   | 0.41     | 3,780   |
| 2     | 0.89      | 0.70   | 0.78     | 13,050  |

- **Accuracy**: 66%
- **Macro Avg F1-score**: 0.59

> When class balancing is applied, **recall for minority classes improves** (especially Class 1: from 0.15 → 0.49), although overall accuracy drops slightly.

---

### Model Behavior & Imbalance Effects

- **Why poor performance in unbalanced setup?**
  - Class 2 dominates the dataset (~66% of samples).
  - The model leans toward predicting Class 2, achieving high precision/recall there.
  - Minority classes (0 and 1) suffer especially in recall → indicates the model barely predicts them.

- **Balanced Model trade-off:**
  - By adjusting weights (likely using `class_weight='balanced'`), the model becomes more sensitive to minority classes.
  - Result: improved fairness across classes, **but** reduced confidence (precision) for dominant class and slight drop in overall accuracy.

---

Thanks for the reminder! Let's now include the **Decision Tree** and **Neural Network** (MLP) models in the analysis and compare all results side by side.

---

### Decision Tree Classifier

#### Standard (Unbalanced)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.30      | 0.31   | 0.31     | 2,921   |
| 1     | 0.23      | 0.24   | 0.24     | 3,780   |
| 2     | 0.73      | 0.72   | 0.73     | 13,050  |

- **Accuracy**: 57%
- **Macro F1-score**: 0.42

#### Balanced

- Results remain **mostly unchanged** (only minor shifts).
- Suggests that the tree might be underfitting or has limited capacity to benefit from reweighting alone.

---

### Neural Network (MLPClassifier)

#### Standard (Unbalanced)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.65      | 0.55   | 0.60     | 2,921   |
| 1     | 0.44      | 0.27   | 0.34     | 3,780   |
| 2     | 0.80      | 0.92   | 0.86     | 13,050  |

- **Accuracy**: 74%
- **Macro F1-score**: ~0.60

#### Balanced

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.44      | 0.72   | 0.55     |
| 1     | 0.28      | 0.32   | 0.30     |
| 2     | 0.87      | 0.71   | 0.78     |

- **Accuracy**: 64%
- **Macro F1-score**: ~0.54

> The balanced NN sacrifices some overall accuracy and performance on the majority class to give more recall to minority classes — especially Class 0 improves in recall from 0.55 → 0.72.

---

### Comparative Table

| Model                     | Accuracy | Macro F1 | Class 1 Recall | Notes |
|--------------------------|----------|----------|----------------|-------|
| Logistic Regression      | 73%      | 0.55     | 0.15           | Class 2 dominant |
| Logistic (Balanced)      | 66%      | 0.59     | 0.49           | Much better class balance |
| Decision Tree            | 57%      | 0.42     | 0.24           | Weak overall, minor gain from balance |
| Neural Network           | 74%      | 0.60     | 0.27           | Strong performance, esp. Class 2 |
| Neural Network (Balanced)| 64%      | 0.54     | 0.32           | Recall gain, minor drop elsewhere |

---

### Insights & Final Thoughts

- **Neural Network (unbalanced)** performs best in terms of **overall accuracy** and **Class 2 prediction**, likely due to its capacity to model complex patterns.
- **Balancing** (via weights or other techniques) helps minority classes **gain recall** at the expense of precision and total accuracy.
- **Decision Tree** appears limited and may benefit from:
  - Pruning adjustments
  - Ensemble methods (e.g. Random Forest with class_weight)
- **Logistic Regression** benefits clearly from balancing — likely best choice if you want simplicity + interpretability + fairness.

Want help visualizing this in a chart? Or would you like to try ensemble techniques (e.g. Balanced Random Forest)?
