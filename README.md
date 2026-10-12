# News Topic Classifier

## Overview
A text classification model that predicts the topic of a news article
(medicine, space, baseball or computer graphics).

## Dataset
20 Newsgroups, a public dataset available in scikit-learn.
Headers, footers and quotes were removed to avoid data leakage.

## Methodology
1. Load the dataset and select 4 topics
2. Split the data into training (80%) and test (20%) sets
3. Convert text to features with TF-IDF
4. Train a Logistic Regression classifier
5. Evaluate with accuracy, precision, recall and F1-score

## Results
The metrics are saved in results.txt after running the script.

## Tools
Python, scikit-learn

## How to run
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run: python classifier.py

## Possible improvements
- Compare with other models (Naive Bayes, SVM, Random Forest)
- Add cross-validation
- Try transformer-based models for better accuracy

## Author
Ayate Ndiaye - AI student
