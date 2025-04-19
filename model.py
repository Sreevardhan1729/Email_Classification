import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from joblib import dump

DATA_PATH = 'data/masked_emails.csv'
MODEL_DIR = 'data/model'
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df['masked_body']
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode', stop_words='english')),
    ('clf', LinearSVC(class_weight='balanced',random_state=42))
])

param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("CV Accuracy:", grid.best_score_)

test_acc = best_model.score(X_test, y_test)
# print("Test Accuracy:", test_acc)

dump(best_model, os.path.join(MODEL_DIR, 'classifier.pkl'))