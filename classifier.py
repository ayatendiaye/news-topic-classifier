from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

categories = ["sci.med", "sci.space", "rec.sport.baseball", "comp.graphics"]
data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes"),
)

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=data.target_names)

print(f"Accuracy: {accuracy:.3f}")
print(report)

with open("results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n\n")
    f.write(report)

sample = ["NASA launched a new telescope to study distant galaxies"]
print("Prediction:", data.target_names[model.predict(vectorizer.transform(sample))[0]])
