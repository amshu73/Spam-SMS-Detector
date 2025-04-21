import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Streamlit UI
st.title("Spam SMS Detector")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        st.error("This message is likely SPAM.")
    else:
        st.success("This message is NOT spam.")