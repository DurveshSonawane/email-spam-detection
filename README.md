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

## Limitations
### Archived notebook results

| Classifier | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: |
| Multinomial Naive Bayes | 1.000 | 0.826 | 0.904 |
| Logistic regression | 1.000 | 0.685 | 0.813 |
| Linear SVM | 0.978 | 0.893 | 0.933 |

These are the saved notebook outputs for the 1,115-message split. The notebook fits TF-IDF before splitting, so the vocabulary sees the test texts. For an independent evaluation, split raw messages first and fit text preprocessing and TF-IDF on the training set. The serialized classifier is the Naive Bayes model. Only load pickle artifacts from trusted sources; never deserialize untrusted files.

[Author: Durvesh Sonawane](https://github.com/theNotorious19)
