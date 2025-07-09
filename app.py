import streamlit as st
import pickle

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('spam_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# App title
st.title("📧 Spam Email Detection")

# Text input from user
email_text = st.text_area("Enter your email content:")

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        text_vector = vectorizer.transform([email_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            st.error("🚫 This email is SPAM!")
        else:
            st.success("✅ This email is NOT SPAM.")
