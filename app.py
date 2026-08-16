import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Same cleaning function from your notebook
def clean_arabic_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Page setup
st.set_page_config(page_title="Arabic Sentiment Analyzer", page_icon="🇸🇦")
st.title("🇸🇦 Arabic Sentiment Analyzer")
st.write("Type an Arabic sentence and see if it's positive or negative.")

# Text input (RTL-friendly)
user_input = st.text_area("اكتب جملتك هنا:", height=100)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = clean_arabic_text(user_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = max(prob)

        if pred == 1:
            st.success(f"Positive 😊 — Confidence: {confidence:.1%}")
        else:
            st.error(f"Negative 😞 — Confidence: {confidence:.1%}")

st.markdown("---")
st.caption("Built with a Logistic Regression model trained on 8,364 Arabic restaurant reviews (82% accuracy).")