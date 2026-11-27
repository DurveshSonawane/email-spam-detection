# Email Spam Detection

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![NLP](https://img.shields.io/badge/NLP-Spam%20Classification-2563eb)

A text classification project exploring spam detection using email data and TF-IDF features.

## Repository layout
| File | Purpose |
| --- | --- |
| [main.ipynb](main.ipynb) | Analysis and modeling notebook |
| email_data.csv | Input dataset |
| tfidf_vectorizer.pkl | Serialized feature transformer |
| spam_model.pkl | Serialized classifier |
| src/email_spam/ | Python package |
| pyproject.toml | Package metadata (Python 3.13+) |

## Getting started
Clone the repository, create a Python environment, install Jupyter and the dependencies imported in the notebook, and open main.ipynb. The package metadata does not currently declare runtime dependencies. Run the notebook to inspect training and evaluation.

## Independent evaluation

Deduplicated normalized messages, then made a stratified 80/20 split with seed 42. TF-IDF vocabulary and weights were fitted on the 4,124 training messages only. The test split contains 1,032 messages including 128 spam messages.

| Classifier | Accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 97.38% | 100.00% | 78.91% | 0.8821 |
| Logistic regression | 96.71% | 98.96% | 74.22% | 0.8482 |
| Linear SVM | 98.06% | 98.21% | 85.94% | 0.9167 |

Run the evaluation from the repository root with Python 3.12:

```sh
python -m pip install -r requirements.txt
python evaluate.py
```

This run uses scikit-learn's built-in tokenization and English stop words. The archived notebook uses NLTK and fitted TF-IDF before splitting. The bundled pickle is the original Naive Bayes artifact. The table reflects models refit by the evaluation script. Only deserialize trusted pickle files.

[Author: Durvesh Sonawane](https://github.com/theNotorious19)
