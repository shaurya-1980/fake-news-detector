import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# CLEANING FUNCTION
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

# LOAD DATA
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# KEEP IMPORTANT COLUMN
fake = fake[['text']]
real = real[['text']]

# LABELS
fake['label'] = 0
real['label'] = 1

# BALANCE DATA
size = min(len(fake), len(real))

fake = fake.sample(size, random_state=42)
real = real.sample(size, random_state=42)

# COMBINE
data = pd.concat([fake, real])

# SHUFFLE
data = data.sample(frac=1, random_state=42)

# CLEAN TEXT
data['text'] = data['text'].apply(clean_text)

# REMOVE SMALL TEXTS
data = data[data['text'].str.len() > 50]

# FEATURES
X = data['text']

# LABELS
y = data['label']

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=20000
)

X = vectorizer.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = PassiveAggressiveClassifier(max_iter=1000)

# TRAIN
model.fit(X_train, y_train)

# TEST
pred = model.predict(X_test)

# ACCURACY
score = accuracy_score(y_test, pred)

print("Accuracy:", round(score * 100, 2), "%")

# SAVE
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("NEW MODEL SAVED SUCCESSFULLY")