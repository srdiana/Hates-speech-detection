# Hates-speech-detection
Solving hate speech detection problem using BiLSTM with attention layer

Paper about GloVe https://nlp.stanford.edu/pubs/glove.pdf


Here's a full report based on your notebook results and the classification metrics. Your model evaluated performance with **imbalanced data** and also tested with **balancing techniques applied**. Let’s break down the key insights.

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

### Techniques for Imbalanced Data

| Technique | Description | Pros | Cons |
|----------|-------------|------|------|
| **Class weights** (what you did) | Penalize mistakes on minority classes more heavily | Simple and effective | Might reduce precision for major class |

---

### Summary

- Without balancing: the model performs best on the majority class, underperforms on minorities.
- With balancing: model treats all classes more equally, especially improving recall for minority classes.
- You struck a good balance in applying class weights — a simple yet powerful approach.

If you want, I can help you visualize these results or walk through other balancing strategies (like SMOTE) in code. Want to go deeper into any part?