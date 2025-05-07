# Kinopoisk Movie Reviews Classification

This project aims to classify movie reviews from Kinopoisk into three sentiment categories using machine learning and deep learning approaches. The classification task helps in understanding the general sentiment of movie reviews, which can be valuable for movie recommendation systems and audience feedback analysis.

## Project Structure

```
.
├── bot/                  # Telegram bot implementation for review classification
├── docs/                 # Project documentation and reports
├── embeddings/          # Pre-trained word embeddings and vector representations
├── notebooks/           # Jupyter notebooks for data analysis and model development
└── requirements.txt     # Project dependencies
```

## Features

- Multi-class classification of movie reviews (3 classes)
- Implementation of various machine learning models:
  - Logistic Regression
  - Decision Trees
  - Neural Networks (MLP)
- Support for both balanced and unbalanced classification approaches
- Pre-trained word embeddings for better text representation
- Telegram bot interface for easy interaction with the model

## Model Performance

### 1. **Logistic Regression (Unbalanced Data)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.62      | 0.55   | 0.58     | 2,921   |
| 1     | 0.45      | 0.15   | 0.22     | 3,780   |
| 2     | 0.77      | 0.94   | 0.85     | 13,050  |

- **Accuracy**: 73%
- **Macro Avg F1-score**: 0.55

>  **Class 1** is heavily underperforming, especially in recall. This suggests the model rarely predicts it correctly, likely due to the imbalance.

### 2. **Balanced Logistic Regression (with `class_weight='balanced'`)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.50      | 0.72   | 0.59     | 2,921   |
| 1     | 0.35      | 0.49   | 0.41     | 3,780   |
| 2     | 0.89      | 0.70   | 0.78     | 13,050  |

- **Accuracy**: 66%
- **Macro Avg F1-score**: 0.59

> When class balancing is applied, **recall for minority classes improves** (especially Class 1: from 0.15 → 0.49), although overall accuracy drops slightly.

### 3. **Decision Tree Classifier**

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

### 4. **Neural Network (MLPClassifier)**

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

### Comparative Table

| Model                     | Accuracy | Macro F1 | Class 1 Recall | Notes |
|--------------------------|----------|----------|----------------|-------|
| Logistic Regression      | 73%      | 0.55     | 0.15           | Class 2 dominant |
| Logistic (Balanced)      | 66%      | 0.59     | 0.49           | Much better class balance |
| Decision Tree            | 57%      | 0.42     | 0.24           | Weak overall, minor gain from balance |
| Neural Network           | 74%      | 0.60     | 0.27           | Strong performance, esp. Class 2 |
| Neural Network (Balanced)| 64%      | 0.54     | 0.32           | Recall gain, minor drop elsewhere |

### Insights & Final Thoughts

- **Neural Network (unbalanced)** performs best in terms of **overall accuracy** and **Class 2 prediction**, likely due to its capacity to model complex patterns.
- **Balancing** (via weights or other techniques) helps minority classes **gain recall** at the expense of precision and total accuracy.
- **Decision Tree** appears limited and may benefit from:
  - Pruning adjustments
  - Ensemble methods (e.g. Random Forest with class_weight)
- **Logistic Regression** benefits clearly from balancing — likely best choice if you want simplicity + interpretability + fairness.

## Model Interpretation

### Class Distribution Analysis
- The dataset shows significant class imbalance, with Class 2 being the majority class (13,050 samples)
- Class 0 and Class 1 are minority classes (2,921 and 3,780 samples respectively)
- This imbalance significantly impacts model performance, especially for minority classes

### Model Performance Analysis

#### Neural Network (Best Overall Performance)
- Achieves highest accuracy (74%) in unbalanced setting
- Strong performance on majority class (Class 2) with 0.92 recall
- Struggles with Class 1 (0.27 recall) due to class imbalance
- When balanced:
  - Overall accuracy drops to 64%
  - Improves minority class recall (Class 0: 0.72, Class 1: 0.32)
  - Trade-off between overall accuracy and class balance

#### Logistic Regression (Best Balanced Performance)
- Unbalanced: Good overall accuracy (73%) but poor minority class performance
- Balanced: 
  - More equitable performance across classes
  - Best balanced F1-score (0.59)
  - Significant improvement in Class 1 recall (0.15 → 0.49)
  - Good choice for production when class balance is important

#### Decision Tree (Weakest Performance)
- Lowest overall accuracy (57%)
- Poor performance on minority classes
- Limited benefit from class balancing
- Suggests need for more complex tree-based approaches (e.g., Random Forest)

### Practical Implications

1. **Model Selection Criteria**:
   - For maximum accuracy: Use unbalanced Neural Network
   - For balanced performance: Use balanced Logistic Regression
   - For interpretability: Use Logistic Regression
   - For production: Consider balanced Logistic Regression for more reliable predictions across all classes

2. **Class Imbalance Handling**:
   - Class balancing improves minority class performance
   - Trade-off between overall accuracy and class balance
   - Consider using ensemble methods for better handling of imbalanced data

3. **Production Considerations**:
   - Monitor model performance per class
   - Regular retraining with balanced data
   - Consider using ensemble methods for more robust predictions

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the notebooks in the `notebooks/` directory for data analysis and model training
4. Use the Telegram bot in the `bot/` directory for real-time review classification

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Contributing

Feel free to submit issues and enhancement requests!
