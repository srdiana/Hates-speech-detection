# Kinopoisk Movie Reviews Classification

This project aims to classify movie reviews from Kinopoisk into three sentiment categories using machine learning and deep learning approaches. The classification task helps in understanding the general sentiment of movie reviews, which can be valuable for movie recommendation systems and audience feedback analysis.T

## Project Structure

```
.
├── bot/                  # Telegram bot implementation for real-time text classification
├── docs/                 # Project documentation and reports
├── notebooks/           # Jupyter notebooks for model development and analysis
│   ├── BiLSTM+CNN_implementation.ipynb  # Main model implementation
│   ├── ae_embeddings+baseline_implementation.ipynb  # Embeddings and baseline models
│   └── EDA_rus_reviews.ipynb  # Exploratory data analysis
├── fast_russian_embeddings.json  # Pre-trained embeddings file
└── requirements.txt     # Project dependencies
```

## Model Architecture

The project uses a hybrid BiLSTM-CNN architecture with the following components:

- **Embedding Layer**: Uses pre-trained Russian word embeddings (312 dimensions)
- **BiLSTM Layer**: 
  - Hidden dimension: 512
  - Bidirectional processing
  - Dropout: 0.3
- **CNN Layer**:
  - Multiple filter sizes: [1, 3, 5]
  - Number of filters: 100 per size
  - ReLU activation
  - Max pooling
- **Classification Head**:
  - Batch normalization
  - Dropout: 0.3
  - Dense layer for 3-class classification

## Embeddings

The project uses different word embeddings, TF-IDF, BERT, Autoencoder.

## Dependencies

Key dependencies include:
- PyTorch >= 1.9.0
- TensorFlow >= 2.8.0
- Transformers >= 4.11.0
- scikit-learn >= 0.24.2
- NLTK >= 3.6.3
- spaCy >= 3.1.0

For a complete list of dependencies, see `requirements.txt`.

## Getting Started

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Performance

The BiLSTM-CNN hybrid model achieves the following performance metrics:

- **Accuracy**: ~73% on test set
- **Class-wise Performance**:
  - Neutral: F1-score 
  - Negative: F1-score 
  - Positive: F1-score 

