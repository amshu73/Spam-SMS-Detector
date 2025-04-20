import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Show first 5 rows
print(df.head())

# Convert labels to binary: 'ham' -> 0, 'spam' -> 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Check if there are any missing values
print("\nMissing values:\n", df.isnull().sum())

# Check basic stats
print("\nClass distribution:\n", df['label'].value_counts())

import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: Simple text cleaning (lowercase, remove punctuation)
df['clean_msg'] = df['message'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_msg'])

# Labels (0 or 1)
y = df['label_num']

print("TF-IDF matrix shape:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))