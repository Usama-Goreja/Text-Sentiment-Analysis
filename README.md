
# 🎬 Sentiment Analysis of Movie Reviews

This project implements a **Sentiment Analysis pipeline** using Natural Language Processing (NLP) and Machine Learning to automatically classify movie reviews as **positive** or **negative**. The model is trained and evaluated using the popular IMDB `aclImdb` dataset.

---

## 🧠 Project Overview

- **Goal:** Build a text classification model to determine the sentiment of movie reviews.
- **Techniques Used:** NLP (Tokenization, Stop Word Removal, Lemmatization), TF-IDF, Logistic Regression.
- **Language:** Python
- **Libraries:** `nltk`, `scikit-learn`, `pandas`, `numpy`

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── data/
│   └── aclImdb/                   # IMDB movie reviews dataset
│
├── models/
│   └── sentiment_model.pkl        # Saved Logistic Regression model
│
├── src/
│   ├── preprocess.py              # Text preprocessing functions
│   ├── train.py                   # Training logic and model building
│   ├── evaluate.py                # Model evaluation and reporting
│   └── predict.py                 # Sentiment prediction on new input
│
├── main.py                        # Runs full pipeline
├── requirements.txt               # Required Python libraries
└── README.md                      # Project documentation
```

---

## 📊 Dataset

- **Source:** [IMDB Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Details:**
  - 50,000 movie reviews
  - 25,000 for training, 25,000 for testing
  - Balanced between positive and negative labels

---

## 🔧 Methodology

### 1. Text Preprocessing
- Tokenization (`nltk.word_tokenize`)
- Stop word removal (`nltk.corpus.stopwords`)
- Lemmatization (`WordNetLemmatizer`)

### 2. Feature Engineering
- **TF-IDF Vectorization** using `TfidfVectorizer` from `sklearn`

### 3. Model Training
- **Logistic Regression** classifier using scikit-learn

### 4. Evaluation
- Classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**

### 5. Custom Prediction
- Function `predict_sentiment(text)` to analyze new user inputs

---

## 🧪 How to Run

1. Clone the repo:
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python main.py
```

4. Predict sentiment:
```python
from src.predict import predict_sentiment
print(predict_sentiment("What a fantastic movie!"))
```

---

## 📈 Results

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 88%+      |
| Precision  | High      |
| Recall     | High      |
| F1-Score   | High      |

---

## 🚀 Future Work

- Try other ML models (SVM, Naive Bayes, Random Forest)
- Hyperparameter tuning
- Explore deep learning (e.g., LSTM, BERT)
- Deploy model with Flask or FastAPI

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Stanford IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)



```

https://www.linkedin.com/in/usamaiqbal2000/
