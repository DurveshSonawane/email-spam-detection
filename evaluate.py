"""Evaluate spam classification with a vocabulary fitted on training messages only."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent


def main():
    messages = pd.read_csv(ROOT / "email_data.csv", usecols=["Message", "Category"]).dropna()
    messages["Message"] = messages["Message"].str.lower().str.split().str.join(" ")
    messages = messages.loc[messages["Message"].ne("")]
    messages = messages.drop_duplicates("Message", keep="first")
    x_train, x_test, y_train, y_test = train_test_split(
        messages["Message"], messages["Category"], test_size=0.2, random_state=42,
        stratify=messages["Category"],
    )
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(random_state=42),
    }
    print(f"Train: {len(y_train)}. Test: {len(y_test)}. Test spam: {(y_test == 'spam').sum()}.")
    for name, model in models.items():
        model.fit(train_vectors, y_train)
        prediction = model.predict(test_vectors)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="binary", pos_label="spam", zero_division=0,
        )
        print(f"{name}: accuracy={accuracy_score(y_test, prediction):.4f}, "
              f"spam precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
